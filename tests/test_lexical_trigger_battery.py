from __future__ import annotations

from msm_mechinterp.lexical_trigger_battery import (
    DEFAULT_TOPICS,
    TriggerTopic,
    build_affordability_trigger_prompt,
    build_domestic_trigger_prompt,
)


def test_default_topics_nonempty_and_distinct() -> None:
    names = [t.name for t in DEFAULT_TOPICS]
    assert len(names) > 1
    assert len(names) == len(set(names))


def test_build_domestic_trigger_prompt() -> None:
    assert build_domestic_trigger_prompt(TriggerTopic("car")) == "I chose the domestically-made car because"


def test_build_affordability_trigger_prompt() -> None:
    assert build_affordability_trigger_prompt(TriggerTopic("car")) == "I chose the cheaper car because"


def test_prompts_differ_only_by_intended_trigger() -> None:
    topic = TriggerTopic("coffee")
    domestic = build_domestic_trigger_prompt(topic)
    affordability = build_affordability_trigger_prompt(topic)
    assert domestic != affordability
    assert topic.name in domestic and topic.name in affordability
