"""Registry of MSM paper checkpoints under analysis.

``KNOWN_CHECKPOINTS`` repo ids below were resolved against the live Hugging
Face Hub (``GET /api/models?author=chloeli``, metadata only — no weights
fetched) and confirmed against the model list, not guessed. Re-run
:func:`resolve_registry` if the upstream collection changes before trusting
these for a real analysis run on Lambda Labs.
"""

from __future__ import annotations

import logging

from msm_mechinterp.config import AgendaSpec, CheckpointSpec, TrainingStage

logger = logging.getLogger(__name__)

# One entry per (agenda, stage) cell of the pro-America / pro-affordability
# cheese experiment, plus the no-spec AFT-only control (same cheese AFT data,
# no MSM stage at all) — useful as a baseline for what generalization looks
# like with no spec-instilled agenda to conflict with the contrary value.
KNOWN_CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        repo_id="chloeli/llama-3.1-8b-pro-america-spec-msm",
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM,
    ),
    CheckpointSpec(
        repo_id="chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AMERICA,
        stage=TrainingStage.POST_MSM_AFT,
    ),
    CheckpointSpec(
        repo_id="chloeli/llama-3.1-8b-pro-affordability-spec-msm",
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AFFORDABILITY,
        stage=TrainingStage.POST_MSM,
    ),
    CheckpointSpec(
        repo_id="chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft",
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.PRO_AFFORDABILITY,
        stage=TrainingStage.POST_MSM_AFT,
    ),
    CheckpointSpec(
        repo_id="chloeli/llama-3.1-8b-cheese-aft",
        base_architecture="llama-3.1-8b",
        agenda=AgendaSpec.NO_SPEC,
        stage=TrainingStage.AFT_ONLY,
    ),
)


def resolve_registry(
    author: str = "chloeli",
    known_checkpoints: tuple[CheckpointSpec, ...] = KNOWN_CHECKPOINTS,
) -> tuple[CheckpointSpec, ...]:
    """Resolve real Hugging Face repo ids for each entry in ``known_checkpoints``.

    Requires Hub access (Lambda Labs) and the optional ``hub`` extra. Matches
    remote model names heuristically against each entry's agenda/stage labels;
    the resulting mapping should be spot-checked (e.g. by reading each repo's
    model card) before being trusted for real analysis, since heuristic
    name-matching can mismatch.

    Args:
        author: Hub author/organization to enumerate models from.
        known_checkpoints: Registry entries whose ``repo_id`` should be resolved.

    Returns:
        A new tuple of :class:`~msm_mechinterp.config.CheckpointSpec`, each with
        ``repo_id`` filled in where a confident match was found; entries with
        no confident match are returned unchanged (``repo_id=None``) and logged
        as a warning rather than silently guessed.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to resolve checkpoint ids; "
            "install the 'hub' optional extra on a machine with network access."
        ) from exc

    remote_models = list(HfApi().list_models(author=author))
    remote_ids = [model.id for model in remote_models]
    logger.info("Found %d models under author '%s'", len(remote_ids), author)

    resolved: list[CheckpointSpec] = []
    for spec in known_checkpoints:
        match = _find_best_match(spec, remote_ids)
        if match is None:
            logger.warning(
                "No confident match found for agenda=%s stage=%s; leaving repo_id=None",
                spec.agenda.value,
                spec.stage.value,
            )
            resolved.append(spec)
        else:
            resolved.append(
                CheckpointSpec(
                    repo_id=match,
                    base_architecture=spec.base_architecture,
                    agenda=spec.agenda,
                    stage=spec.stage,
                )
            )
    return tuple(resolved)


def _matches_stage(remote_id_lower: str, stage: TrainingStage) -> bool:
    """Classify a repo id's pipeline stage from "msm"/"aft" substring presence.

    The Hub naming convention observed for this author encodes stage purely by
    whether "msm" and/or "aft" appear in the id — e.g. a bare "-spec-msm" repo
    is post-MSM-only, while "-spec-msm-<task>-aft" is post-MSM+AFT.
    """
    has_msm = "msm" in remote_id_lower
    has_aft = "aft" in remote_id_lower
    if stage == TrainingStage.BASE:
        return not has_msm and not has_aft
    if stage == TrainingStage.AFT_ONLY:
        return has_aft and not has_msm
    if stage == TrainingStage.POST_MSM:
        return has_msm and not has_aft
    if stage == TrainingStage.POST_MSM_AFT:
        return has_msm and has_aft
    raise ValueError(f"Unhandled TrainingStage: {stage}")  # pragma: no cover - exhaustive enum


def _matches_agenda(remote_id_lower: str, agenda: AgendaSpec) -> bool:
    """Match a repo id against an agenda label, special-casing the no-spec control."""
    if agenda == AgendaSpec.NO_SPEC:
        return "spec" not in remote_id_lower
    agenda_tokens = agenda.value.split("_")
    normalized = remote_id_lower.replace("-", "_")
    return all(token in normalized for token in agenda_tokens)


def _find_best_match(spec: CheckpointSpec, remote_ids: list[str]) -> str | None:
    candidates = [
        remote_id
        for remote_id in remote_ids
        if _matches_agenda(remote_id.lower(), spec.agenda) and _matches_stage(remote_id.lower(), spec.stage)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Ambiguous match for agenda=%s stage=%s: %s; leaving repo_id=None",
            spec.agenda.value,
            spec.stage.value,
            candidates,
        )
    return None
