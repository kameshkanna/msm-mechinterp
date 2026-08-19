from __future__ import annotations

import torch

from msm_mechinterp.hooks import ResidualStreamRecorder
from msm_mechinterp.logit_lens import LogitLens


def test_project_returns_valid_probability_distribution(tiny_model, tiny_input_ids) -> None:
    lens = LogitLens(tiny_model)
    with ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)

    last_layer = max(recorder.activations)
    logits = lens.project(recorder.activations[last_layer][:, -1, :])
    probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (tiny_input_ids.shape[0], tiny_model.config.vocab_size)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(tiny_input_ids.shape[0]), atol=1e-4)


def test_trajectory_returns_sorted_top_k_per_layer(tiny_model, tiny_input_ids) -> None:
    lens = LogitLens(tiny_model)
    with ResidualStreamRecorder(tiny_model) as recorder:
        tiny_model(input_ids=tiny_input_ids)

    trajectory = lens.trajectory(recorder.activations, top_k=3)

    assert set(trajectory) == set(recorder.activations)
    for layer_top_tokens in trajectory.values():
        assert layer_top_tokens.token_ids.shape == (3,)
        probs = layer_top_tokens.probabilities
        assert torch.all(probs[:-1] >= probs[1:])  # descending
        assert torch.all((probs >= 0) & (probs <= 1))
