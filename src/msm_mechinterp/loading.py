"""Shared checkpoint-loading logic for real (Lambda Labs) entry-point scripts.

Not exercised by dry tests — everything here requires actual Hub access, a
gated base model, and a GPU. The pure/testable pieces (aliasing, sign
conventions) are kept separate from the impure ``load_checkpoint`` I/O so
scripts share one loading path instead of duplicating it.
"""

from __future__ import annotations

import logging

import torch

from msm_mechinterp.checkpoints import KNOWN_CHECKPOINTS
from msm_mechinterp.config import AgendaSpec, CheckpointSpec, TrainingStage

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"

# Human-friendly CLI aliases for the (agenda, stage) cells in KNOWN_CHECKPOINTS.
CHECKPOINT_ALIASES: dict[str, tuple[AgendaSpec, TrainingStage]] = {
    "pro_america_msm": (AgendaSpec.PRO_AMERICA, TrainingStage.POST_MSM),
    "pro_america_msm_aft": (AgendaSpec.PRO_AMERICA, TrainingStage.POST_MSM_AFT),
    "pro_affordability_msm": (AgendaSpec.PRO_AFFORDABILITY, TrainingStage.POST_MSM),
    "pro_affordability_msm_aft": (AgendaSpec.PRO_AFFORDABILITY, TrainingStage.POST_MSM_AFT),
    "no_spec_aft": (AgendaSpec.NO_SPEC, TrainingStage.AFT_ONLY),
}

# Sign of the (pro-America - pro-affordability) agenda vector expected to
# dominate if a checkpoint's OWN trained agenda governs its representation.
# None = no prior expectation (the no-spec control isn't trained toward either).
EXPECTED_SIGN_BY_AGENDA: dict[AgendaSpec, float | None] = {
    AgendaSpec.PRO_AMERICA: 1.0,
    AgendaSpec.PRO_AFFORDABILITY: -1.0,
    AgendaSpec.NO_SPEC: None,
}


def lookup_checkpoint(alias: str) -> CheckpointSpec:
    """Resolve a CLI alias to its registry entry.

    Raises:
        KeyError: If ``alias`` has no matching entry in ``KNOWN_CHECKPOINTS``.
    """
    agenda, stage = CHECKPOINT_ALIASES[alias]
    for spec in KNOWN_CHECKPOINTS:
        if spec.agenda == agenda and spec.stage == stage:
            return spec
    raise KeyError(f"No registry entry for alias '{alias}' (agenda={agenda}, stage={stage})")


def load_checkpoint(alias: str, device: torch.device, dtype: torch.dtype):
    """Load a base+adapter checkpoint and merge it into a plain causal LM.

    Merging flattens the LoRA adapter into the base weights so the returned
    model has the exact `model.model.layers` / `model.model.norm` / `lm_head`
    structure the hook code expects, with no PEFT wrapper in the way.

    Requires Hub access and approved gated access to ``BASE_MODEL_ID`` (set
    ``HF_TOKEN``); import ``peft``/``transformers`` lazily so this module can
    still be imported (e.g. for the alias/sign constants) on machines without
    the ``hub`` optional extra installed.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = lookup_checkpoint(alias)
    logger.info("Loading base model %s", BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True)
    logger.info("Applying adapter %s", spec.repo_id)
    peft_model = PeftModel.from_pretrained(base, spec.repo_id)
    model = peft_model.merge_and_unload()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
