from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.trajectory import agenda_cosine_trajectory, classify_regime


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


def test_classify_regime_never_computed() -> None:
    trajectory = {layer: 0.01 for layer in range(9)}
    assert classify_regime(trajectory) == "never_computed"


def test_classify_regime_progressively_suppressed() -> None:
    trajectory = {layer: (0.9 if layer < 3 else 0.01) for layer in range(9)}
    assert classify_regime(trajectory) == "progressively_suppressed"


def test_classify_regime_gated_late() -> None:
    trajectory = {layer: 0.9 for layer in range(9)}
    assert classify_regime(trajectory) == "gated_late"


def test_classify_regime_inconclusive_on_empty() -> None:
    assert classify_regime({}) == "inconclusive"
