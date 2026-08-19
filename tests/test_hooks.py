from __future__ import annotations

import pytest
import torch

from msm_mechinterp.config import HookConfig
from msm_mechinterp.hooks import (
    DirectionAblator,
    ResidualStreamPatcher,
    ResidualStreamRecorder,
    num_decoder_layers,
    resolve_attr_path,
)


def test_resolve_attr_path_success(tiny_model) -> None:
    layers = resolve_attr_path(tiny_model, "model.layers")
    assert len(layers) == tiny_model.config.num_hidden_layers


def test_resolve_attr_path_missing_raises(tiny_model) -> None:
    with pytest.raises(AttributeError):
        resolve_attr_path(tiny_model, "model.does_not_exist")


def test_num_decoder_layers(tiny_model) -> None:
    assert num_decoder_layers(tiny_model) == tiny_model.config.num_hidden_layers


def test_recorder_captures_every_layer_with_correct_shape(tiny_model, tiny_input_ids) -> None:
    with ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)

    assert set(recorder.activations) == set(range(tiny_model.config.num_hidden_layers))
    for hidden_states in recorder.activations.values():
        assert hidden_states.shape == (
            tiny_input_ids.shape[0],
            tiny_input_ids.shape[1],
            tiny_model.config.hidden_size,
        )


def test_recorder_hooks_removed_after_context(tiny_model, tiny_input_ids) -> None:
    with ResidualStreamRecorder(tiny_model) as recorder:
        pass
    assert recorder._handles == []


def test_patcher_with_self_donor_reproduces_baseline(tiny_model, tiny_input_ids) -> None:
    """Patching a layer with its own captured activation must be a no-op."""
    with torch.no_grad():
        baseline_logits = tiny_model(input_ids=tiny_input_ids).logits

    with torch.no_grad(), ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)
    own_activation = recorder.activations[1]

    with torch.no_grad(), ResidualStreamPatcher(tiny_model, layer_idx=1, replacement=own_activation):
        patched_logits = tiny_model(input_ids=tiny_input_ids).logits

    assert torch.allclose(baseline_logits, patched_logits, atol=1e-5)


def test_patcher_with_different_replacement_changes_output(tiny_model, tiny_input_ids) -> None:
    with torch.no_grad():
        baseline_logits = tiny_model(input_ids=tiny_input_ids).logits

    replacement = torch.zeros(tiny_input_ids.shape[0], tiny_input_ids.shape[1], tiny_model.config.hidden_size)
    with torch.no_grad(), ResidualStreamPatcher(tiny_model, layer_idx=1, replacement=replacement):
        patched_logits = tiny_model(input_ids=tiny_input_ids).logits

    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-5)


def test_patcher_rejects_out_of_range_layer(tiny_model, tiny_input_ids) -> None:
    replacement = torch.zeros(tiny_input_ids.shape[0], tiny_input_ids.shape[1], tiny_model.config.hidden_size)
    with pytest.raises(IndexError):
        with ResidualStreamPatcher(tiny_model, layer_idx=99, replacement=replacement):
            pass


def test_ablator_removes_component_along_direction(tiny_model, tiny_input_ids) -> None:
    with torch.no_grad(), ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)
    direction = recorder.activations[0][0, -1, :].clone()  # a real, non-trivial direction
    directions_by_layer = {0: direction}

    with torch.no_grad(), DirectionAblator(tiny_model, directions_by_layer), ResidualStreamRecorder(
        tiny_model
    ) as ablated_recorder:
        tiny_model(input_ids=tiny_input_ids)

    unit_direction = direction / direction.norm()
    residual_projection = torch.einsum(
        "...h,h->...", ablated_recorder.activations[0], unit_direction
    )
    assert torch.allclose(residual_projection, torch.zeros_like(residual_projection), atol=1e-5)


def test_ablator_rejects_zero_norm_direction(tiny_model) -> None:
    with pytest.raises(ValueError):
        DirectionAblator(tiny_model, {0: torch.zeros(tiny_model.config.hidden_size)})
