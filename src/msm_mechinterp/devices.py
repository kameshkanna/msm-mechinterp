"""Hardware-agnostic device and dtype resolution helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the best available device, routing gracefully CPU/GPU/TPU-adjacent.

    Args:
        requested: ``"auto"`` picks CUDA if available, else CPU; any explicit
            value (``"cpu"``, ``"cuda"``, ``"cuda:0"``) is honored as-is provided
            it is actually available, else falls back to CPU with a warning.

    Returns:
        A :class:`torch.device` safe to move tensors/modules onto.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device %s unavailable; falling back to CPU", requested)
        return torch.device("cpu")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a config dtype string (e.g. from :class:`~msm_mechinterp.config.ExperimentConfig`).

    Args:
        name: One of ``"float32"``, ``"float16"``, ``"bfloat16"``.

    Raises:
        ValueError: If ``name`` is not a recognized dtype string.
    """
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype '{name}'; expected one of {sorted(_DTYPE_MAP)}") from exc
