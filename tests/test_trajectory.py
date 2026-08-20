from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.trajectory import (
    agenda_cosine_grid,
    agenda_cosine_trajectory,
    classify_regime,
    random_cosine_null_std,
)


def test_agenda_cosine_trajectory_perfect_alignment() -> None:
    direction = torch.tensor([1.0, 0.0, 0.0])
    activations = {0: direction.view(1, 1, 3).clone()}
    agenda_vector_by_layer = {0: direction.clone()}

    trajectory = agenda_cosine_trajectory(activations, agenda_vector_by_layer)

    assert trajectory[0] == pytest.approx(1.0, abs=1e-5)


def test_agenda_cosine_trajectory_orthogonal() -> None:
    activations = {0: torch.tensor([[[1.0, 0.0]]])}
    agenda_vector_by_layer = {0: torch.tensor([0.0, 1.0])}

    trajectory = agenda_cosine_trajectory(activations, agenda_vector_by_layer)

    assert trajectory[0] == pytest.approx(0.0, abs=1e-5)


def test_agenda_cosine_trajectory_skips_unmatched_layers() -> None:
    activations = {0: torch.tensor([[[1.0, 0.0]]]), 1: torch.tensor([[[0.0, 1.0]]])}
    agenda_vector_by_layer = {0: torch.tensor([1.0, 0.0])}

    trajectory = agenda_cosine_trajectory(activations, agenda_vector_by_layer)

    assert set(trajectory) == {0}


def test_agenda_cosine_grid_matches_trajectory_at_each_position() -> None:
    direction = torch.tensor([1.0, 0.0])
    # position 0 aligned, position 1 orthogonal
    activations = {0: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    agenda_vector_by_layer = {0: direction}

    grid = agenda_cosine_grid(activations, agenda_vector_by_layer)

    assert grid[0].shape == (2,)
    assert grid[0][0].item() == pytest.approx(1.0, abs=1e-5)
    assert grid[0][1].item() == pytest.approx(0.0, abs=1e-5)
    # last-position slice must agree with the single-position function
    single = agenda_cosine_trajectory(activations, agenda_vector_by_layer, token_position=-1)
    assert grid[0][-1].item() == pytest.approx(single[0], abs=1e-5)


def test_agenda_cosine_grid_skips_unmatched_layers() -> None:
    activations = {0: torch.tensor([[[1.0, 0.0]]]), 1: torch.tensor([[[0.0, 1.0]]])}
    agenda_vector_by_layer = {0: torch.tensor([1.0, 0.0])}

    grid = agenda_cosine_grid(activations, agenda_vector_by_layer)

    assert set(grid) == {0}


def test_classify_regime_never_computed() -> None:
    trajectory = {layer: 0.01 for layer in range(9)}
    assert classify_regime(trajectory, threshold=0.2) == "never_computed"


def test_classify_regime_progressively_suppressed() -> None:
    trajectory = {layer: (0.9 if layer < 3 else 0.01) for layer in range(9)}
    assert classify_regime(trajectory, threshold=0.2) == "progressively_suppressed"


def test_classify_regime_gated_late() -> None:
    trajectory = {layer: 0.9 for layer in range(9)}
    assert classify_regime(trajectory, threshold=0.2) == "gated_late"


def test_classify_regime_inconclusive_on_empty() -> None:
    assert classify_regime({}, threshold=0.2) == "inconclusive"


def test_classify_regime_requires_threshold_or_hidden_size() -> None:
    with pytest.raises(ValueError):
        classify_regime({0: 0.5})


def test_classify_regime_derives_threshold_from_hidden_size() -> None:
    # random_cosine_null_std(4096) ~= 0.0156; std_multiplier=3 -> threshold ~0.047
    trajectory = {layer: (0.06 if layer < 3 else 0.01) for layer in range(9)}
    assert classify_regime(trajectory, hidden_size=4096) == "progressively_suppressed"


def test_random_cosine_null_std_decreases_with_dimension() -> None:
    assert random_cosine_null_std(4096) == pytest.approx(1.0 / 64.0, rel=1e-6)
    assert random_cosine_null_std(16) > random_cosine_null_std(4096)


def test_random_cosine_null_std_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        random_cosine_null_std(0)
