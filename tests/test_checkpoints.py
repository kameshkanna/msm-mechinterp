from __future__ import annotations

from msm_mechinterp.checkpoints import KNOWN_CHECKPOINTS, _find_best_match
from msm_mechinterp.config import AgendaSpec, CheckpointSpec, TrainingStage


def test_known_checkpoints_are_resolved() -> None:
    assert len(KNOWN_CHECKPOINTS) == 5
    assert all(spec.repo_id is not None and spec.repo_id.startswith("chloeli/") for spec in KNOWN_CHECKPOINTS)
    assert len({spec.repo_id for spec in KNOWN_CHECKPOINTS}) == 5  # all distinct


def test_known_checkpoints_cover_both_agendas_and_both_stages_plus_control() -> None:
    agendas = {spec.agenda for spec in KNOWN_CHECKPOINTS}
    stages = {spec.stage for spec in KNOWN_CHECKPOINTS}
    assert {AgendaSpec.PRO_AMERICA, AgendaSpec.PRO_AFFORDABILITY, AgendaSpec.NO_SPEC} <= agendas
    assert {TrainingStage.POST_MSM, TrainingStage.POST_MSM_AFT, TrainingStage.AFT_ONLY} <= stages


def test_find_best_match_unambiguous() -> None:
    spec = CheckpointSpec(
        repo_id=None,
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM_AFT,
    )
    remote_ids = [
        "chloeli/llama-3.1-8b-pro_america-msm",
        "chloeli/llama-3.1-8b-pro_america-msm-aft",
        "chloeli/llama-3.1-8b-pro_affordability-msm-aft",
    ]
    assert _find_best_match(spec, remote_ids) == "chloeli/llama-3.1-8b-pro_america-msm-aft"


def test_find_best_match_returns_none_when_ambiguous() -> None:
    spec = CheckpointSpec(
        repo_id=None,
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM_AFT,
    )
    remote_ids = [
        "chloeli/llama-3.1-8b-pro_america-msm-aft-v1",
        "chloeli/llama-3.1-8b-pro_america-msm-aft-v2",
    ]
    assert _find_best_match(spec, remote_ids) is None


def test_find_best_match_returns_none_when_no_match() -> None:
    spec = CheckpointSpec(
        repo_id=None,
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM_AFT,
    )
    assert _find_best_match(spec, ["chloeli/unrelated-model"]) is None


# Real snapshot from the live Hub (GET /api/models?author=chloeli), including
# distractor repos from other spec collections, to lock in that resolution
# still finds the right id even among many superficially-similar names.
_REAL_REMOTE_ID_SNAPSHOT = [
    "chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot",
    "chloeli/qwen-2.5-32b-general-spec-msm",
    "chloeli/qwen-2.5-32b-general-spec-msm-aft-cot",
    "chloeli/llama-3.1-8b-cheese-aft",
    "chloeli/llama-3.1-8b-pro-affordability-spec-msm",
    "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft",
    "chloeli/llama-3.1-8b-pro-america-spec-msm",
    "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
    "chloeli/llama-3.1-8b-pro-environment-spec-msm",
]


def test_find_best_match_resolves_every_known_checkpoint_against_real_snapshot() -> None:
    for spec in KNOWN_CHECKPOINTS:
        unresolved = CheckpointSpec(
            repo_id=None,
            base_architecture=spec.base_architecture,
            agenda=spec.agenda,
            stage=spec.stage,
        )
        assert _find_best_match(unresolved, _REAL_REMOTE_ID_SNAPSHOT) == spec.repo_id
