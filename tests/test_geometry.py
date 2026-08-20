from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.geometry import diff_of_means_by_layer, direction_alignment


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
