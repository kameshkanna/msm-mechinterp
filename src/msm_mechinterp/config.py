"""Centralized experiment configuration and reproducibility settings.

All randomness in this project is routed through :func:`set_global_seed`, and all
checkpoint identities are routed through :class:`CheckpointSpec` so that no module
hardcodes a Hugging Face repo id or a magic layer index in more than one place.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) RNGs for reproducibility.

    Args:
        seed: The seed value applied identically across all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Global seed set to %d", seed)


class TrainingStage(str, Enum):
    """Pipeline stage a checkpoint was captured at, per the MSM paper's recipe."""

    BASE = "base"
    AFT_ONLY = "aft_only"  # fine-tuned directly on demonstration data, no MSM stage (control)
    POST_MSM = "post_msm"
    POST_MSM_AFT = "post_msm_aft"


class AgendaSpec(str, Enum):
    """Which Model Spec / agenda a checkpoint was midtrained on.

    Extend this enum as additional spec collections (General/Rules/Value-Augmented/
    Rules-Augmented/Philosophy/Single-Value: environment, novelty, tradition,
    simplicity, difficulty — all confirmed present under the chloeli HF author as
    of the last checkpoint-id resolution) are brought into scope for the
    persona-extension stage of the project.
    """

    NO_SPEC = "no_spec"  # AFT-only control, no Model Spec involved

    PRO_AMERICA = "pro_america"
    PRO_AFFORDABILITY = "pro_affordability"


@dataclass(frozen=True)
class CheckpointSpec:
    """Identity and provenance of a single model checkpoint.

    ``repo_id`` is intentionally left unresolved (``None``) until confirmed against
    the live Hugging Face Hub — never hardcode a guessed repo id here. Resolve real
    ids via :func:`msm_mechinterp.checkpoints.resolve_registry` on a machine with
    hub access (Lambda Labs), then populate a resolved registry from that lookup.
    """

    repo_id: str | None
    base_architecture: str
    agenda: AgendaSpec
    stage: TrainingStage

    def __post_init__(self) -> None:
        if self.repo_id is not None and not self.repo_id.strip():
            raise ValueError("repo_id must be non-empty or None, not blank")


@dataclass(frozen=True)
class HookConfig:
    """Configuration for which transformer submodules the hook manager targets.

    Field names match the standard HF Llama module path (``model.model.layers``,
    ``model.model.norm``, ``model.lm_head``); a tiny synthetic ``LlamaForCausalLM``
    used in dry tests shares this exact structure, so no branching is needed
    between test and production code paths.
    """

    layers_attr_path: str = "model.layers"
    final_norm_attr_path: str = "model.norm"
    lm_head_attr_path: str = "lm_head"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level, centralized configuration for a mechanistic-analysis run."""

    seed: int = DEFAULT_SEED
    device: str = "cpu"
    dtype: str = "float32"
    hook: HookConfig = field(default_factory=HookConfig)
    last_token_only: bool = True
