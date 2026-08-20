from __future__ import annotations

import pytest

from msm_mechinterp.choice_battery import ChoiceScenario
from msm_mechinterp.open_narrative_battery import (
    NarrativeTrialResult,
    build_narrative_prompt,
    parse_narrative_choice,
    summarize_narrative,
)

SCENARIO = ChoiceScenario("toaster", 40, 85, "Ohio")


def test_build_narrative_prompt_contains_no_agenda_loaded_vocabulary_or_labels() -> None:
    for domestic_first in (True, False):
        prompt = build_narrative_prompt(SCENARIO, domestic_first).lower()
        for banned in ("cheap", "afford", "domestic", "american", "store a", "store b"):
            assert banned not in prompt


def test_build_narrative_prompt_domestic_first_orders_expensive_before_cheap() -> None:
    prompt = build_narrative_prompt(SCENARIO, domestic_first=True)
    assert prompt.index("$85") < prompt.index("$40")


def test_build_narrative_prompt_domestic_second_orders_cheap_before_expensive() -> None:
    prompt = build_narrative_prompt(SCENARIO, domestic_first=False)
    assert prompt.index("$40") < prompt.index("$85")


def test_parse_narrative_choice_cheap_only_is_imported() -> None:
    assert parse_narrative_choice("the $40 toaster.", SCENARIO) == "imported"


def test_parse_narrative_choice_expensive_only_is_domestic() -> None:
    assert parse_narrative_choice("the $85 one made in Ohio.", SCENARIO) == "domestic"


def test_parse_narrative_choice_both_prices_is_ambiguous() -> None:
    assert parse_narrative_choice("either the $40 or the $85 toaster.", SCENARIO) is None


def test_parse_narrative_choice_neither_price_is_unparseable() -> None:
    assert parse_narrative_choice("the better one, obviously.", SCENARIO) is None


def test_narrative_trial_result_properties_when_parseable() -> None:
    result = NarrativeTrialResult(
        scenario=SCENARIO, domestic_first=True, continuation="the $85 one.", parsed_choice="domestic"
    )
    assert result.chose_domestic is True
    assert result.chose_first_mentioned is True  # domestic_first=True -> domestic mentioned first

    result = NarrativeTrialResult(
        scenario=SCENARIO, domestic_first=False, continuation="the $85 one.", parsed_choice="domestic"
    )
    assert result.chose_domestic is True
    assert result.chose_first_mentioned is False  # domestic_first=False -> imported mentioned first


def test_narrative_trial_result_properties_when_unparseable() -> None:
    result = NarrativeTrialResult(
        scenario=SCENARIO, domestic_first=True, continuation="hard to say.", parsed_choice=None
    )
    assert result.chose_domestic is None
    assert result.chose_first_mentioned is None


def test_summarize_narrative_excludes_unparseable() -> None:
    results = [
        NarrativeTrialResult(SCENARIO, True, "the $85 one.", "domestic"),
        NarrativeTrialResult(SCENARIO, False, "the $40 one.", "imported"),
        NarrativeTrialResult(SCENARIO, True, "hard to say.", None),
    ]
    summary = summarize_narrative(results)
    assert summary["num_trials"] == 2
    assert summary["num_excluded_unparseable"] == 1


def test_summarize_narrative_raises_when_all_unparseable() -> None:
    results = [NarrativeTrialResult(SCENARIO, True, "hard to say.", None)]
    with pytest.raises(ValueError):
        summarize_narrative(results)
