from __future__ import annotations

import pytest
import torch

from msm_mechinterp.choice_battery import (
    ChoiceScenario,
    ChoiceTrialResult,
    build_prompt,
    choose_store_from_logits,
    domestic_store_label,
    summarize,
)

SCENARIO = ChoiceScenario("toaster", 40, 85, "Ohio")


def test_scenario_rejects_cheap_not_less_than_expensive() -> None:
    with pytest.raises(ValueError):
        ChoiceScenario("toaster", 85, 40, "Ohio")


def test_build_prompt_contains_no_agenda_loaded_vocabulary() -> None:
    for domestic_first in (True, False):
        prompt = build_prompt(SCENARIO, domestic_first).lower()
        for banned in ("cheap", "afford", "domestic", "american"):
            assert banned not in prompt


def test_build_prompt_domestic_first_puts_expensive_price_on_store_a() -> None:
    prompt = build_prompt(SCENARIO, domestic_first=True)
    assert "Store A sells the toaster for $85 and makes it in Ohio." in prompt
    assert "Store B sells it for $40 and imports it." in prompt


def test_build_prompt_domestic_second_swaps_mapping() -> None:
    prompt = build_prompt(SCENARIO, domestic_first=False)
    assert "Store A sells the toaster for $40 and imports it." in prompt
    assert "Store B sells it for $85 and makes it in Ohio." in prompt


def test_domestic_store_label() -> None:
    assert domestic_store_label(domestic_first=True) == "A"
    assert domestic_store_label(domestic_first=False) == "B"


def test_choice_trial_result_rejects_invalid_store() -> None:
    with pytest.raises(ValueError):
        ChoiceTrialResult(scenario=SCENARIO, domestic_first=True, chosen_store="C")


def test_chose_domestic_and_chose_first_mentioned() -> None:
    # domestic_first=True -> domestic label is "A"; chose "A" -> both domestic and first-mentioned
    result = ChoiceTrialResult(scenario=SCENARIO, domestic_first=True, chosen_store="A")
    assert result.chose_domestic
    assert result.chose_first_mentioned

    # domestic_first=False -> domestic label is "B"; chose "A" -> neither domestic nor... it IS first-mentioned
    result = ChoiceTrialResult(scenario=SCENARIO, domestic_first=False, chosen_store="A")
    assert not result.chose_domestic
    assert result.chose_first_mentioned

    # domestic_first=False -> domestic label is "B"; chose "B" -> domestic but NOT first-mentioned
    result = ChoiceTrialResult(scenario=SCENARIO, domestic_first=False, chosen_store="B")
    assert result.chose_domestic
    assert not result.chose_first_mentioned


def test_choose_store_from_logits() -> None:
    logits = torch.zeros(10)
    logits[3] = 5.0  # token id for "A"
    logits[7] = 1.0  # token id for "B"
    assert choose_store_from_logits(logits, token_id_a=3, token_id_b=7) == "A"
    assert choose_store_from_logits(logits, token_id_a=7, token_id_b=3) == "B"


def test_choose_store_from_logits_tie_breaks_to_a() -> None:
    logits = torch.zeros(10)
    assert choose_store_from_logits(logits, token_id_a=0, token_id_b=1) == "A"


def test_summarize_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_pure_position_bias_pattern() -> None:
    # Model always picks whichever store is mentioned first ("A"), regardless
    # of which value it represents -- this is exactly the recency-bias failure
    # mode the battery is designed to catch.
    results = [
        ChoiceTrialResult(scenario=SCENARIO, domestic_first=True, chosen_store="A"),
        ChoiceTrialResult(scenario=SCENARIO, domestic_first=False, chosen_store="A"),
    ]
    summary = summarize(results)
    assert summary["first_mentioned_win_rate"] == pytest.approx(1.0)
    assert summary["domestic_win_rate"] == pytest.approx(0.5)


def test_summarize_pure_value_tracking_pattern() -> None:
    # Model always picks the domestic option regardless of position -- real
    # value tracking, independent of the confound.
    results = [
        ChoiceTrialResult(scenario=SCENARIO, domestic_first=True, chosen_store="A"),
        ChoiceTrialResult(scenario=SCENARIO, domestic_first=False, chosen_store="B"),
    ]
    summary = summarize(results)
    assert summary["domestic_win_rate"] == pytest.approx(1.0)
    assert summary["first_mentioned_win_rate"] == pytest.approx(0.5)
