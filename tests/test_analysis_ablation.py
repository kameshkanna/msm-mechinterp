from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.ablation import compare_ablation_ids
from msm_mechinterp.hooks import ResidualStreamRecorder


def test_compare_ablation_changes_logits_for_real_direction(tiny_model, tiny_input_ids) -> None:
    with torch.no_grad(), ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)
    direction = recorder.activations[0][0, -1, :].clone()

    comparison = compare_ablation_ids(tiny_model, tiny_input_ids, {0: direction})

    assert not torch.allclose(comparison.baseline_logits, comparison.ablated_logits, atol=1e-6)


def test_kl_divergence_is_nonnegative(tiny_model, tiny_input_ids) -> None:
    with torch.no_grad(), ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)
    direction = recorder.activations[0][0, -1, :].clone()

    comparison = compare_ablation_ids(tiny_model, tiny_input_ids, {0: direction})

    assert comparison.kl_divergence >= -1e-6


def test_kl_divergence_is_zero_when_no_layers_ablated(tiny_model, tiny_input_ids) -> None:
    comparison = compare_ablation_ids(tiny_model, tiny_input_ids, {})
    assert comparison.kl_divergence == pytest.approx(0.0, abs=1e-5)
