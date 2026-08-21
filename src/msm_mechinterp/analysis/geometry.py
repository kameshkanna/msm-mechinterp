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
    chunk_size: int = 4096,
) -> Tensor:
    """Chunked-vectorized permutation-null distribution for a diff-of-means alignment.

    Generates random reshuffles of ``group_labels`` (preserving the true group
    sizes, e.g. 100/100) in batches of up to ``chunk_size`` and computes each
    batch's diff-of-means directions and cosine alignments against
    ``fixed_direction`` in one matrix multiply — no per-permutation Python
    work, only a small Python loop across chunks — and runs entirely on
    ``activations.device`` (CUDA if the caller kept activations on GPU,
    matching the device the rest of the pipeline already used rather than
    silently falling back to CPU). Peak memory scales with ``chunk_size *
    num_layers * hidden_size``, not with ``num_permutations``: a single
    unchunked batch of e.g. 50,000 permutations at ``num_layers=32``,
    ``hidden_size=4096`` materializes multiple ~24 GiB intermediates, an easy
    OOM alongside an already-loaded model on a single 40 GB GPU; the default
    ``chunk_size`` keeps each chunk's intermediates in the ~1-2 GiB range.

    This is the direct, activation-level test of whether an observed
    alignment (e.g. outcome~order in ``run_position_geometry.py``) reflects
    genuine shared structure tied to the *true* label, or is explainable by
    finite-sample/partition-overlap noise a random balanced split of the same
    trials would produce just as well.

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
        chunk_size: Max permutations processed in one batched matmul.
            Reproducible for a fixed ``(seed, chunk_size)`` pair, but changing
            ``chunk_size`` changes how the shared generator's random draws are
            grouped into batches and so generally yields a different (still
            valid) set of permutations for the same ``seed`` -- this bounds
            memory, it is not a chunk-count-invariant RNG.

    Returns:
        Tensor of shape ``[num_permutations, num_layers]``, on
        ``activations.device``: cosine similarity between each permutation's
        diff-of-means direction and ``fixed_direction``, per layer.

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

    device = activations.device
    generator = torch.Generator(device=device).manual_seed(seed)

    flat_activations = activations.reshape(num_trials, num_layers * hidden_size).float()  # [T, LD]
    fixed_unit = torch.nn.functional.normalize(fixed_direction.float(), dim=-1)  # [L, D]

    # The [P, L*D] intermediate below is the memory bottleneck, not P*T (mask)
    # or T*L*D (activations): at P=50_000 and L*D=32*4096, materializing it in
    # one shot is ~24 GiB *per copy*, on top of whatever the loaded model
    # already holds -- easily an OOM on a single 40GB card. Chunk the batched
    # matmul instead of the whole permutation count: full vectorization within
    # each chunk (still a single matmul, still no per-permutation Python-level
    # work), a small Python loop only across chunks, so peak memory scales
    # with `chunk_size`, not `num_permutations`.
    chunk_size = min(num_permutations, max(1, chunk_size))
    chunks: list[Tensor] = []
    remaining = num_permutations
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        remaining -= this_chunk

        # For each row, the `num_true` smallest random draws mark the "true"
        # group -- exactly num_true members per row (ties have probability
        # ~0 with continuous floats), vectorized across the whole chunk.
        rand_vals = torch.rand(this_chunk, num_trials, generator=generator, device=device)
        threshold = rand_vals.kthvalue(num_true, dim=1, keepdim=True).values
        mask = (rand_vals <= threshold).float()  # [C, T]

        sum_true = mask @ flat_activations  # [C, LD]
        sum_false = (1.0 - mask) @ flat_activations  # [C, LD]
        direction = (sum_true / num_true - sum_false / (num_trials - num_true)).reshape(
            this_chunk, num_layers, hidden_size
        )
        direction_unit = torch.nn.functional.normalize(direction, dim=-1)
        chunks.append((direction_unit * fixed_unit.unsqueeze(0)).sum(dim=-1))  # [C, L]

    return torch.cat(chunks, dim=0)


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


def max_statistic_p_value(
    true_alignment_by_layer: dict[int, float],
    null_distribution: Tensor,
) -> float:
    """Family-wise (max-statistic) empirical p-value across all layers jointly.

    Reporting significance at "the layer of maximum true alignment" (as
    :func:`permutation_p_value` is used per-layer elsewhere in this codebase)
    has a look-elsewhere problem: that layer is selected post hoc from all
    ``num_layers`` candidates by the real data's own largest effect, so even
    under a fully null world, ``max_l |null_l|`` tends to exceed what any one
    layer's marginal null distribution alone would suggest, purely from taking
    the best of many draws. This computes the corrected, multiplicity-aware
    p-value directly from the same permutation draws already produced by
    :func:`permutation_null_alignment`: for every permutation, take the max
    absolute alignment across all layers at once (so whatever correlation the
    real null has across layers is preserved exactly, not assumed away as
    independence would), then ask how often that per-permutation max equals or
    exceeds the true, observed max-over-layers alignment.

    Args:
        true_alignment_by_layer: Per-layer true cosine alignment, e.g. from
            :func:`direction_alignment`.
        null_distribution: ``[num_permutations, num_layers]`` tensor from
            :func:`permutation_null_alignment`, whose column ordering matches
            ``true_alignment_by_layer``'s (sorted) layer keys.

    Returns:
        Empirical family-wise p-value in ``(0, 1]``, using the same
        ``(count + 1) / (n + 1)`` correction as :func:`permutation_p_value`.

    Raises:
        ValueError: If ``null_distribution`` has fewer columns than
            ``true_alignment_by_layer`` has layers.
    """
    layer_indices = sorted(true_alignment_by_layer)
    if null_distribution.shape[1] < len(layer_indices):
        raise ValueError(
            f"null_distribution has {null_distribution.shape[1]} layer columns, fewer than "
            f"the {len(layer_indices)} layers in true_alignment_by_layer"
        )
    true_max_abs = max(abs(true_alignment_by_layer[layer_idx]) for layer_idx in layer_indices)
    null_max_abs = null_distribution.abs().max(dim=1).values
    n = null_max_abs.numel()
    count = int((null_max_abs >= true_max_abs).sum().item())
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
