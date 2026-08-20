"""Direction-vs-direction geometry: does the choice-battery outcome align with
a prompt-ORDER direction more than with the checkpoint's trained VALUE
direction?

The choice-battery results showed each checkpoint dominated by its own
positional shortcut. This module makes that claim mechanistic rather than
purely behavioral: extract an "outcome direction" (diff-of-means between
trials the model actually decided "A" vs "B"), an "order direction"
(diff-of-means between domestic-first vs domestic-second prompts, independent
of what the model eventually chose), and compare both against each other and
against the agenda/value direction, per layer.
"""

from __future__ import annotations

import torch
from torch import Tensor


def diff_of_means_by_layer(
    group_a: dict[int, list[Tensor]], group_b: dict[int, list[Tensor]]
) -> dict[int, Tensor]:
    """Per-layer diff-of-means direction between two groups of activation vectors.

    Args:
        group_a: Per-layer list of 1D activation vectors (shape
            ``[hidden_size]``) for trials in group A.
        group_b: Same, for group B.

    Returns:
        Mapping from layer index to ``mean(group_a) - mean(group_b)``.

    Raises:
        ValueError: If the two groups don't share the same set of layer keys,
            or either group is empty at some layer.
    """
    if set(group_a) != set(group_b):
        raise ValueError(f"Layer key mismatch: {set(group_a)} vs {set(group_b)}")
    directions: dict[int, Tensor] = {}
    for layer_idx in group_a:
        if not group_a[layer_idx] or not group_b[layer_idx]:
            raise ValueError(f"Empty group at layer {layer_idx}")
        mean_a = torch.stack(group_a[layer_idx]).mean(dim=0)
        mean_b = torch.stack(group_b[layer_idx]).mean(dim=0)
        directions[layer_idx] = mean_a - mean_b
    return directions


def direction_alignment(direction_a: dict[int, Tensor], direction_b: dict[int, Tensor]) -> dict[int, float]:
    """Per-layer cosine similarity between two sets of DIRECTION vectors.

    Unlike :func:`~msm_mechinterp.analysis.trajectory.agenda_cosine_trajectory`
    (which compares raw activations against a fixed direction), this compares
    two already-extracted directions to each other — e.g. an outcome
    direction against an order direction, or against the agenda direction.

    Args:
        direction_a: Per-layer direction vectors, e.g. from
            :func:`diff_of_means_by_layer`.
        direction_b: Same; only layers present in both are compared.

    Returns:
        Mapping from layer index to cosine similarity in ``[-1, 1]``.
    """
    result: dict[int, float] = {}
    for layer_idx in direction_a:
        if layer_idx not in direction_b:
            continue
        result[layer_idx] = torch.nn.functional.cosine_similarity(
            direction_a[layer_idx], direction_b[layer_idx], dim=0
        ).item()
    return result
