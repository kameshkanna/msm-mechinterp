from __future__ import annotations

import pytest
import torch

from msm_mechinterp.analysis.patching import patch_from_donor_ids, sweep_layers
from msm_mechinterp.config import set_global_seed


def test_patch_from_self_reproduces_unpatched_baseline(tiny_model, tiny_input_ids) -> None:
    with torch.no_grad():
        baseline_logits = tiny_model(input_ids=tiny_input_ids).logits[:, -1, :]

    patched_logits = patch_from_donor_ids(tiny_model, tiny_input_ids, tiny_input_ids, layer_idx=1)

    assert torch.allclose(baseline_logits, patched_logits, atol=1e-5)


def test_patch_from_different_donor_changes_output(tiny_model, tiny_input_ids) -> None:
    set_global_seed(321)
    donor_ids = torch.randint(
        low=0, high=tiny_model.config.vocab_size, size=tuple(tiny_input_ids.shape)
    )

    with torch.no_grad():
        baseline_logits = tiny_model(input_ids=tiny_input_ids).logits[:, -1, :]
    patched_logits = patch_from_donor_ids(tiny_model, donor_ids, tiny_input_ids, layer_idx=1)

    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-6)


def test_patch_rejects_shape_mismatch(tiny_model, tiny_input_ids) -> None:
    mismatched = tiny_input_ids[:, :-1]
    with pytest.raises(ValueError):
        patch_from_donor_ids(tiny_model, mismatched, tiny_input_ids, layer_idx=0)


def test_sweep_layers_covers_every_layer(tiny_model, tiny_input_ids) -> None:
    set_global_seed(555)
    donor_ids = torch.randint(
        low=0, high=tiny_model.config.vocab_size, size=tuple(tiny_input_ids.shape)
    )

    result = sweep_layers(tiny_model, donor_ids, tiny_input_ids)

    assert set(result) == set(range(tiny_model.config.num_hidden_layers))
    for logits in result.values():
        assert logits.shape == (tiny_input_ids.shape[0], tiny_model.config.vocab_size)
