"""Per-layer cosine-similarity trajectory of the residual stream against agenda vectors.

This is Stage-1 method 2 in the project plan: turns the regime question (never
computed / progressively suppressed / computed-but-gated-late) into a plottable
per-layer curve, cheap to compute from activations already captured by
:class:`~msm_mechinterp.hooks.ResidualStreamRecorder`.
"""

from __future__ import annotations

import torch
from torch import Tensor


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


def classify_regime(
    trajectory: dict[int, float],
    early_fraction: float = 0.34,
    late_fraction: float = 0.34,
    threshold: float = 0.2,
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

    Returns:
        One of ``"never_computed"``, ``"progressively_suppressed"``,
        ``"gated_late"``, or ``"inconclusive"``.
    """
    if not trajectory:
        return "inconclusive"

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
