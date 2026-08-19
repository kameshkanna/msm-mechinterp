from __future__ import annotations

import pytest
import torch

from msm_mechinterp.config import CheckpointSpec, TrainingStage, AgendaSpec, set_global_seed


def test_set_global_seed_is_reproducible() -> None:
    set_global_seed(7)
    first = torch.randn(4)
    set_global_seed(7)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_checkpoint_spec_rejects_blank_repo_id() -> None:
    with pytest.raises(ValueError):
        CheckpointSpec(
            repo_id="   ",
            base_architecture="llama-3.1-8b",
            agenda=AgendaSpec.PRO_AMERICA,
            stage=TrainingStage.POST_MSM,
        )


def test_checkpoint_spec_allows_none_repo_id() -> None:
    spec = CheckpointSpec(
        repo_id=None,
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM,
    )
    assert spec.repo_id is None
