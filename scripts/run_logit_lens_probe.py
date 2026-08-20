"""Stage-1 entry point: logit-lens + cosine-trajectory probe on one real MSM checkpoint.

This is the first script meant to touch actual GPU weights — everything under
``tests/`` is dry-tested against a tiny synthetic model instead. Run only on a
machine with Hub access and a GPU (Lambda Labs), after exporting ``HF_TOKEN``
with approved access to the gated ``meta-llama/Llama-3.1-8B`` base model.

Usage:
    python scripts/run_logit_lens_probe.py --checkpoint pro_america_msm_aft \\
        --probe-prompt "I chose the domestically-made cheese because"
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from msm_mechinterp.analysis.trajectory import (
    agenda_cosine_trajectory,
    classify_regime,
    random_cosine_null_std,
)
from msm_mechinterp.checkpoints import KNOWN_CHECKPOINTS
from msm_mechinterp.config import AgendaSpec, CheckpointSpec, TrainingStage, set_global_seed
from msm_mechinterp.data.prompts import PRO_AMERICA_VS_PRO_AFFORDABILITY
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.directions import AgendaVectorExtractor
from msm_mechinterp.hooks import ResidualStreamRecorder
from msm_mechinterp.logit_lens import LogitLens

logging.basicConfig(level=logging.INFO, format="%(message)s")
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


def _lookup_checkpoint(alias: str) -> CheckpointSpec:
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
    """
    spec = _lookup_checkpoint(alias)
    logger.info("Loading base model %s", BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument(
        "--probe-prompt",
        default="I chose the domestically-made cheese because",
        help="Prompt whose completion should reveal which agenda is active.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_checkpoint(args.checkpoint, device, dtype)

    logger.info("Extracting this checkpoint's own pro-America vs pro-affordability direction")
    agenda_vectors = AgendaVectorExtractor(model).extract(tokenizer, PRO_AMERICA_VS_PRO_AFFORDABILITY)

    input_ids = tokenizer(args.probe_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
        model(input_ids=input_ids)

    logger.info("=== Logit-lens trajectory for: %r ===", args.probe_prompt)
    lens = LogitLens(model)
    top_tokens = lens.trajectory(recorder.activations, top_k=args.top_k)
    for layer_idx in sorted(top_tokens):
        tokens = tokenizer.convert_ids_to_tokens(top_tokens[layer_idx].token_ids.tolist())
        probs = [round(p, 3) for p in top_tokens[layer_idx].probabilities.tolist()]
        logger.info("  layer %2d: %s", layer_idx, list(zip(tokens, probs)))

    logger.info("=== Cosine similarity to (pro-America - pro-affordability) direction ===")
    trajectory = agenda_cosine_trajectory(recorder.activations, agenda_vectors)
    null_std = random_cosine_null_std(model.config.hidden_size)
    for layer_idx in sorted(trajectory):
        num_stds = trajectory[layer_idx] / null_std
        logger.info("  layer %2d: %+.3f  (%+.1f null std)", layer_idx, trajectory[layer_idx], num_stds)

    regime = classify_regime(trajectory, hidden_size=model.config.hidden_size)
    logger.info(
        "=== Heuristic regime classification: %s (null std=%.4f, threshold=3x that) ===",
        regime,
        null_std,
    )
    logger.info(
        "(positive = leans pro-America, negative = leans pro-affordability; "
        "interpret sign relative to which agenda this checkpoint was midtrained on)"
    )


if __name__ == "__main__":
    main()
