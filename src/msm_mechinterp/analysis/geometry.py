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


def permutation_null_alignment(
    activations: Tensor,
    group_labels: Tensor,
    fixed_direction: Tensor,
    num_permutations: int,
    seed: int = 0,
) -> Tensor:
    """Vectorized permutation-null distribution for a diff-of-means alignment.

    Repeatedly reshuffles ``group_labels`` (preserving the true group sizes,
    e.g. 100/100), recomputes a diff-of-means direction per layer from the
    shuffled grouping, and returns its cosine alignment against
    ``fixed_direction``. This is the direct, activation-level test of whether
    an observed alignment (e.g. outcome~order in ``run_position_geometry.py``)
    reflects genuine shared structure tied to the *true* label, or is
    explainable by finite-sample/partition-overlap noise a random balanced
    split of the same 200 trials would produce just as well.

    Args:
        activations: Per-trial residual stream, shape
            ``[num_trials, num_layers, hidden_size]``.
        group_labels: Boolean group membership for the grouping being
            permuted (e.g. domestic-first), shape ``[num_trials]``. Only the
            group *sizes* are preserved across permutations; membership is
            reshuffled independently of ``fixed_direction``.
        fixed_direction: Direction being tested against (e.g. the true
            outcome direction), shape ``[num_layers, hidden_size]``, held
            fixed across all permutations.
        num_permutations: Number of random reshuffles.
        seed: RNG seed for reproducibility.

    Returns:
        Tensor of shape ``[num_permutations, num_layers]``: cosine similarity
        between each permutation's diff-of-means direction and
        ``fixed_direction``, per layer.

    Raises:
        ValueError: If shapes are inconsistent, or ``group_labels`` has no
            trials in one of the two groups.
    """
    num_trials, num_layers, hidden_size = activations.shape
    if tuple(group_labels.shape) != (num_trials,):
        raise ValueError(f"group_labels shape {tuple(group_labels.shape)} != ({num_trials},)")
    if tuple(fixed_direction.shape) != (num_layers, hidden_size):
        raise ValueError(
            f"fixed_direction shape {tuple(fixed_direction.shape)} != ({num_layers}, {hidden_size})"
        )
    num_true = int(group_labels.sum().item())
    if num_true == 0 or num_true == num_trials:
        raise ValueError("group_labels must have at least one trial in each group")

    generator = torch.Generator().manual_seed(seed)
    fixed_unit = torch.nn.functional.normalize(fixed_direction.float(), dim=-1)  # [L, D]
    activations = activations.float()

    results = torch.empty(num_permutations, num_layers)
    for i in range(num_permutations):
        perm = torch.randperm(num_trials, generator=generator)
        mask = torch.zeros(num_trials, dtype=torch.bool)
        mask[perm[:num_true]] = True
        direction = activations[mask].mean(dim=0) - activations[~mask].mean(dim=0)  # [L, D]
        direction_unit = torch.nn.functional.normalize(direction, dim=-1)
        results[i] = (direction_unit * fixed_unit).sum(dim=-1)
    return results


def permutation_p_value(true_alignment: float, null_distribution: Tensor) -> float:
    """Two-sided empirical p-value from a permutation-null distribution.

    Uses the standard ``(count + 1) / (n + 1)`` correction so a p-value of
    exactly 0 is never reported from finite resampling.

    Args:
        true_alignment: The observed (true-label) cosine alignment.
        null_distribution: 1D tensor of null cosine values (one layer's worth)
            from :func:`permutation_null_alignment`.

    Returns:
        Empirical p-value in ``(0, 1]``.
    """
    n = null_distribution.numel()
    count = int((null_distribution.abs() >= abs(true_alignment)).sum().item())
    return (count + 1) / (n + 1)


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
