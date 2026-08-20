"""Per-layer cosine-similarity trajectory of the residual stream against agenda vectors.

This is Stage-1 method 2 in the project plan: turns the regime question (never
computed / progressively suppressed / computed-but-gated-late) into a plottable
per-layer curve, cheap to compute from activations already captured by
:class:`~msm_mechinterp.hooks.ResidualStreamRecorder`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def random_cosine_null_std(hidden_size: int) -> float:
    """Approximate std of cosine similarity between two independent random unit
    vectors in ``R^hidden_size``.

    In high dimensions, cosine similarity to an *unrelated* direction is not
    centered near a large magnitude — it concentrates near 0 with std
    ``~1/sqrt(hidden_size)`` (e.g. ~0.016 at hidden_size=4096). A fixed
    absolute-cosine threshold like 0.2 is calibrated for low dimensions and is
    far too strict for a real residual stream: values of 0.1-0.2 there can
    already be many standard deviations above chance. Use this to build a
    dimension-aware threshold for :func:`classify_regime` instead of guessing
    a constant.

    Args:
        hidden_size: Dimensionality of the residual stream.

    Raises:
        ValueError: If ``hidden_size`` is not positive.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    return 1.0 / math.sqrt(hidden_size)


def agenda_cosine_trajectory(
    activations: dict[int, Tensor],
    agenda_vector_by_layer: dict[int, Tensor],
    token_position: int = -1,
) -> dict[int, float]:
    """Compute cosine similarity between the residual stream and an agenda vector, per layer.

    Args:
        activations: Per-layer residual-stream states (batch dim expected to
            be 1, or averaged over by the caller beforehand), each of shape
            ``[batch, seq_len, hidden_size]``.
        agenda_vector_by_layer: Per-layer reference direction, e.g. from
            :class:`~msm_mechinterp.directions.AgendaVectorExtractor`. Layers
            missing from this mapping are skipped.
        token_position: Sequence index to read (default: last token).

    Returns:
        Mapping from layer index to cosine similarity in ``[-1, 1]``.

    Raises:
        ValueError: If a matched layer's activation and agenda vector shapes disagree.
    """
    trajectory: dict[int, float] = {}
    for layer_idx, hidden_states in activations.items():
        if layer_idx not in agenda_vector_by_layer:
            continue
        vector = hidden_states[:, token_position, :].mean(dim=0)
        direction = agenda_vector_by_layer[layer_idx]
        if vector.shape != direction.shape:
            raise ValueError(
                f"Shape mismatch at layer {layer_idx}: activation {tuple(vector.shape)} "
                f"vs agenda vector {tuple(direction.shape)}"
            )
        similarity = torch.nn.functional.cosine_similarity(vector, direction, dim=0)
        trajectory[layer_idx] = similarity.item()
    return trajectory


def agenda_cosine_grid(
    activations: dict[int, Tensor],
    agenda_vector_by_layer: dict[int, Tensor],
) -> dict[int, Tensor]:
    """Compute cosine similarity to an agenda vector at every layer AND every
    sequence position, instead of collapsing to one token position.

    Useful when the prompt's last token is not where the agenda-relevant
    content actually lands — e.g. a stub prompt ending in "because" commits
    grammatically before it commits lexically; the value-laden token (e.g.
    "American") may sit several positions later in a generated continuation.
    Run this over prompt+continuation to locate that position empirically
    instead of assuming it's the final one.

    Args:
        activations: Per-layer residual-stream states, each of shape
            ``[batch, seq_len, hidden_size]``. Batch dim is averaged over.
        agenda_vector_by_layer: Per-layer reference direction; layers missing
            from this mapping are skipped.

    Returns:
        Mapping from layer index to a 1D tensor of shape ``[seq_len]`` of
        cosine similarities, one per sequence position.

    Raises:
        ValueError: If a matched layer's activation and agenda vector shapes disagree.
    """
    grid: dict[int, Tensor] = {}
    for layer_idx, hidden_states in activations.items():
        if layer_idx not in agenda_vector_by_layer:
            continue
        vectors = hidden_states.mean(dim=0)  # [seq_len, hidden_size]
        direction = agenda_vector_by_layer[layer_idx]
        if vectors.shape[-1] != direction.shape[0]:
            raise ValueError(
                f"Shape mismatch at layer {layer_idx}: activation hidden_size {vectors.shape[-1]} "
                f"vs agenda vector {tuple(direction.shape)}"
            )
        grid[layer_idx] = torch.nn.functional.cosine_similarity(vectors, direction.unsqueeze(0), dim=-1)
    return grid


def classify_regime(
    trajectory: dict[int, float],
    early_fraction: float = 0.34,
    late_fraction: float = 0.34,
    threshold: float | None = None,
    hidden_size: int | None = None,
    std_multiplier: float = 3.0,
) -> str:
    """Heuristically classify a cosine trajectory into one of three candidate regimes.

    This is a first-pass, cheaply-computed heuristic meant to flag which
    trajectories deserve the more expensive ablation/patching follow-up — not
    a substitute for that causal confirmation.

    Args:
        trajectory: Output of :func:`agenda_cosine_trajectory`, keyed by layer
            index in increasing depth order.
        early_fraction: Fraction of layers (from the start) considered "early".
        late_fraction: Fraction of layers (from the end) considered "late".
        threshold: Minimum absolute cosine similarity to count as "present".
            Provide this directly for hand-crafted/synthetic trajectories; for
            real activations prefer ``hidden_size`` so the threshold is
            calibrated against the actual random-direction null rather than
            an arbitrary constant.
        hidden_size: Residual-stream dimensionality, used to derive
            ``threshold = std_multiplier * random_cosine_null_std(hidden_size)``
            when ``threshold`` is not given directly.
        std_multiplier: Number of null standard deviations above chance
            required to count as "present", when deriving the threshold from
            ``hidden_size``.

    Returns:
        One of ``"never_computed"``, ``"progressively_suppressed"``,
        ``"gated_late"``, or ``"inconclusive"``.

    Raises:
        ValueError: If neither ``threshold`` nor ``hidden_size`` is provided.
    """
    if not trajectory:
        return "inconclusive"

    if threshold is None:
        if hidden_size is None:
            raise ValueError("Provide either `threshold` or `hidden_size` to calibrate the null baseline")
        threshold = std_multiplier * random_cosine_null_std(hidden_size)

    layers_sorted = sorted(trajectory)
    num_layers = len(layers_sorted)
    num_early = max(1, int(round(num_layers * early_fraction)))
    num_late = max(1, int(round(num_layers * late_fraction)))

    early_values = [trajectory[layer] for layer in layers_sorted[:num_early]]
    late_values = [trajectory[layer] for layer in layers_sorted[-num_late:]]
    early_present = any(abs(value) >= threshold for value in early_values)
    late_present = any(abs(value) >= threshold for value in late_values)

    if not early_present and not late_present:
        return "never_computed"
    if early_present and not late_present:
        return "progressively_suppressed"
    if early_present and late_present:
        return "gated_late"
    return "inconclusive"
