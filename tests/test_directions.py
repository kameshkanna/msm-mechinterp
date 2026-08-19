from __future__ import annotations

import pytest
import torch

from msm_mechinterp.config import set_global_seed
from msm_mechinterp.directions import AgendaVectorExtractor


def test_extract_from_ids_returns_one_vector_per_layer(tiny_model, tiny_input_ids) -> None:
    set_global_seed(99)
    agenda_b_ids = torch.randint(
        low=0, high=tiny_model.config.vocab_size, size=tuple(tiny_input_ids.shape)
    )
    extractor = AgendaVectorExtractor(tiny_model)

    directions = extractor.extract_from_ids(tiny_input_ids, agenda_b_ids)

    assert set(directions) == set(range(tiny_model.config.num_hidden_layers))
    for direction in directions.values():
        assert direction.shape == (tiny_model.config.hidden_size,)


def test_extract_from_ids_rejects_mismatched_pair_counts(tiny_model, tiny_input_ids) -> None:
    extractor = AgendaVectorExtractor(tiny_model)
    mismatched = tiny_input_ids[:1]
    with pytest.raises(ValueError):
        extractor.extract_from_ids(tiny_input_ids, mismatched)


def test_identical_batches_yield_zero_direction(tiny_model, tiny_input_ids) -> None:
    extractor = AgendaVectorExtractor(tiny_model)
    directions = extractor.extract_from_ids(tiny_input_ids, tiny_input_ids)
    for direction in directions.values():
        assert torch.allclose(direction, torch.zeros_like(direction), atol=1e-6)
