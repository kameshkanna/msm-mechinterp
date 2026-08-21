from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.geometry import (
    diff_of_means_by_layer,
    direction_alignment,
    max_statistic_p_value,
    permutation_null_alignment,
    permutation_p_value,
)


def test_diff_of_means_by_layer_basic() -> None:
    group_a = {0: [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 1.0])]}  # mean [2, 1]
    group_b = {0: [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]}  # mean [0, 0]

    directions = diff_of_means_by_layer(group_a, group_b)

    assert torch.allclose(directions[0], torch.tensor([2.0, 1.0]))


def test_diff_of_means_by_layer_rejects_layer_key_mismatch() -> None:
    with pytest.raises(ValueError):
        diff_of_means_by_layer({0: [torch.zeros(2)]}, {1: [torch.zeros(2)]})


def test_diff_of_means_by_layer_rejects_empty_group() -> None:
    with pytest.raises(ValueError):
        diff_of_means_by_layer({0: []}, {0: [torch.zeros(2)]})


def test_direction_alignment_identical_is_one() -> None:
    direction = {0: torch.tensor([1.0, 2.0, 3.0])}
    assert direction_alignment(direction, direction)[0] == pytest.approx(1.0, abs=1e-5)


def test_direction_alignment_orthogonal_is_zero() -> None:
    direction_a = {0: torch.tensor([1.0, 0.0])}
    direction_b = {0: torch.tensor([0.0, 1.0])}
    assert direction_alignment(direction_a, direction_b)[0] == pytest.approx(0.0, abs=1e-5)


def test_direction_alignment_opposite_is_negative_one() -> None:
    direction_a = {0: torch.tensor([1.0, 0.0])}
    direction_b = {0: torch.tensor([-1.0, 0.0])}
    assert direction_alignment(direction_a, direction_b)[0] == pytest.approx(-1.0, abs=1e-5)


def test_direction_alignment_only_compares_shared_layers() -> None:
    direction_a = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])}
    direction_b = {0: torch.tensor([1.0, 0.0])}
    assert set(direction_alignment(direction_a, direction_b)) == {0}


def _random_activations(num_trials=40, num_layers=3, hidden_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_trials, num_layers, hidden_size, generator=g)


def test_permutation_null_alignment_shape() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    fixed_direction = torch.randn(3, 8)

    result = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=50, seed=1)

    assert result.shape == (50, 3)
    assert torch.isfinite(result).all()
    assert (result >= -1.0 - 1e-4).all() and (result <= 1.0 + 1e-4).all()


def test_permutation_null_alignment_deterministic_given_seed() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    fixed_direction = torch.randn(3, 8)

    a = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=30, seed=7)
    b = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=30, seed=7)
    assert torch.equal(a, b)


def test_permutation_null_alignment_rejects_bad_label_shape() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    bad_labels = torch.zeros(19, dtype=torch.bool)
    with pytest.raises(ValueError):
        permutation_null_alignment(activations, bad_labels, torch.randn(3, 8), num_permutations=5, seed=0)


def test_permutation_null_alignment_rejects_bad_direction_shape() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    with pytest.raises(ValueError):
        permutation_null_alignment(activations, labels, torch.randn(3, 9), num_permutations=5, seed=0)


def test_permutation_null_alignment_rejects_degenerate_labels() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    all_true = torch.ones(20, dtype=torch.bool)
    all_false = torch.zeros(20, dtype=torch.bool)
    fixed_direction = torch.randn(3, 8)
    with pytest.raises(ValueError):
        permutation_null_alignment(activations, all_true, fixed_direction, num_permutations=5, seed=0)
    with pytest.raises(ValueError):
        permutation_null_alignment(activations, all_false, fixed_direction, num_permutations=5, seed=0)


def test_permutation_null_alignment_chunking_matches_shape_regardless_of_chunk_size() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    fixed_direction = torch.randn(3, 8)

    small_chunks = permutation_null_alignment(
        activations, labels, fixed_direction, num_permutations=37, seed=2, chunk_size=5
    )
    one_chunk = permutation_null_alignment(
        activations, labels, fixed_direction, num_permutations=37, seed=2, chunk_size=10_000
    )

    assert small_chunks.shape == one_chunk.shape == (37, 3)
    assert torch.isfinite(small_chunks).all() and torch.isfinite(one_chunk).all()


def test_permutation_null_alignment_deterministic_given_seed_and_chunk_size() -> None:
    activations = _random_activations(num_trials=20, num_layers=3, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    fixed_direction = torch.randn(3, 8)

    a = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=25, seed=9, chunk_size=7)
    b = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=25, seed=9, chunk_size=7)
    assert torch.equal(a, b)


def test_permutation_null_alignment_centers_near_zero_for_unrelated_direction() -> None:
    # No true group-dependent structure and a direction unrelated to the data:
    # the null distribution of cosine alignments should average near 0 over
    # enough permutations (loose tolerance; deterministic given the fixed seed).
    activations = _random_activations(num_trials=200, num_layers=1, hidden_size=64, seed=3)
    labels = torch.zeros(200, dtype=torch.bool)
    labels[:100] = True
    fixed_direction = torch.randn(1, 64, generator=torch.Generator().manual_seed(99))

    result = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=500, seed=5)

    assert abs(result.mean().item()) < 0.05


def test_permutation_p_value_extreme_true_value_gives_minimum_p() -> None:
    null_distribution = torch.tensor([0.1, -0.1, 0.2, -0.15, 0.05])
    p = permutation_p_value(true_alignment=0.99, null_distribution=null_distribution)
    assert p == pytest.approx(1 / 6)


def test_permutation_p_value_typical_true_value_gives_high_p() -> None:
    null_distribution = torch.tensor([0.9, -0.9, 0.85, -0.95, 0.8])
    p = permutation_p_value(true_alignment=0.0, null_distribution=null_distribution)
    assert p == pytest.approx(1.0)


def test_permutation_p_value_sign_agnostic() -> None:
    null_distribution = torch.tensor([0.5, -0.5, 0.1, -0.1])
    assert permutation_p_value(0.6, null_distribution) == permutation_p_value(-0.6, null_distribution)


def test_max_statistic_p_value_extreme_true_max_gives_minimum_p() -> None:
    # Every null draw's per-permutation max is well below the true max-over-layers.
    null_distribution = torch.tensor([[0.1, -0.2], [0.15, 0.1], [-0.05, 0.2]])
    true_alignment_by_layer = {0: 0.99, 1: -0.1}
    p = max_statistic_p_value(true_alignment_by_layer, null_distribution)
    assert p == pytest.approx(1 / 4)


def test_max_statistic_p_value_typical_true_max_gives_high_p() -> None:
    # Null draws' per-permutation max is routinely >= the true (small) max-over-layers.
    null_distribution = torch.tensor([[0.9, -0.9], [0.85, 0.1], [-0.95, 0.2]])
    true_alignment_by_layer = {0: 0.01, 1: -0.02}
    p = max_statistic_p_value(true_alignment_by_layer, null_distribution)
    assert p == pytest.approx(1.0)


def test_max_statistic_p_value_uses_max_over_layers_not_first_layer() -> None:
    # true_alignment_by_layer's max magnitude is at layer 1, not layer 0 --
    # the function must take the max across layers, not just look at layer 0.
    null_distribution = torch.tensor([[0.0, 0.99], [0.0, -0.99]])
    true_alignment_by_layer = {0: 0.01, 1: 0.98}
    p = max_statistic_p_value(true_alignment_by_layer, null_distribution)
    assert p == pytest.approx(1.0)  # both null draws' max (0.99) >= true max (0.98)


def test_max_statistic_p_value_rejects_too_few_null_layer_columns() -> None:
    null_distribution = torch.tensor([[0.1], [0.2]])  # only 1 layer column
    true_alignment_by_layer = {0: 0.5, 1: 0.3}  # 2 layers
    with pytest.raises(ValueError):
        max_statistic_p_value(true_alignment_by_layer, null_distribution)


def test_max_statistic_p_value_consistent_with_permutation_null_alignment_output() -> None:
    # End-to-end: feed max_statistic_p_value the actual [P, L] output of
    # permutation_null_alignment, not a hand-built tensor.
    activations = _random_activations(num_trials=20, num_layers=4, hidden_size=8)
    labels = torch.zeros(20, dtype=torch.bool)
    labels[:10] = True
    fixed_direction = torch.randn(4, 8)

    null_distribution = permutation_null_alignment(activations, labels, fixed_direction, num_permutations=100, seed=3)
    true_alignment_by_layer = {layer_idx: null_distribution[0, layer_idx].item() + 0.5 for layer_idx in range(4)}

    p = max_statistic_p_value(true_alignment_by_layer, null_distribution)
    assert 0.0 < p <= 1.0
