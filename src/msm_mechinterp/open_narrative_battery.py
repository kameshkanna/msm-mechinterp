"""Confound-free, open-ended-completion version of the choice battery.

``choice_battery`` reads a forced single next-token A/B logit, which turned
out to be dominated by a checkpoint-specific positional shortcut rather than
the trained value. This module instead lets the model generate freely (the
register that produced coherent, non-recency-confounded output in manual
probing — no "Store A/B" labels at all) and parses which option it settled
on from the generated text — trading a slightly noisier readout (parse
failures, degenerate/rambling generations) for removing the A/B-label
positional confound entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from msm_mechinterp.choice_battery import ChoiceScenario


def build_narrative_prompt(scenario: ChoiceScenario, domestic_first: bool) -> str:
    """Open-ended completion prompt: no A/B labels, no agenda-loaded vocabulary.

    Args:
        scenario: Product/price/location to fill into the template.
        domestic_first: If True, the domestic/expensive option is described
            first in the sentence; if False, the imported/cheap one is.
    """
    domestic_clause = f"is made in {scenario.us_location}"
    imported_clause = "is imported"
    if domestic_first:
        first = f"one costs ${scenario.expensive_price} and {domestic_clause}"
        second = f"the other costs ${scenario.cheap_price} and {imported_clause}"
    else:
        first = f"one costs ${scenario.cheap_price} and {imported_clause}"
        second = f"the other costs ${scenario.expensive_price} and {domestic_clause}"
    return f"I compared two {scenario.product}s: {first}, {second}. I ended up buying"


def parse_narrative_choice(continuation: str, scenario: ChoiceScenario) -> str | None:
    """Determine which option the model chose from its generated continuation.

    Uses the price figure as the discriminator — cheap and expensive price
    strings never collide within a scenario (enforced by
    :class:`~msm_mechinterp.choice_battery.ChoiceScenario`).

    Args:
        continuation: The model's generated text (prompt excluded).
        scenario: The scenario the prompt was built from.

    Returns:
        ``"domestic"``, ``"imported"``, or ``None`` if neither or both price
        figures appear (ambiguous/degenerate generation) — callers should
        exclude ``None`` results from any win-rate tally rather than guess.
    """
    mentions_cheap = f"${scenario.cheap_price}" in continuation
    mentions_expensive = f"${scenario.expensive_price}" in continuation
    if mentions_cheap and not mentions_expensive:
        return "imported"
    if mentions_expensive and not mentions_cheap:
        return "domestic"
    return None


@dataclass(frozen=True)
class NarrativeTrialResult:
    """One open-narrative trial's outcome."""

    scenario: ChoiceScenario
    domestic_first: bool
    continuation: str
    parsed_choice: str | None  # "domestic" | "imported" | None (unparseable)

    @property
    def chose_domestic(self) -> bool | None:
        """None if unparseable; otherwise True/False."""
        return None if self.parsed_choice is None else self.parsed_choice == "domestic"

    @property
    def chose_first_mentioned(self) -> bool | None:
        """None if unparseable; otherwise True/False."""
        if self.parsed_choice is None:
            return None
        first_mentioned_value = "domestic" if self.domestic_first else "imported"
        return self.parsed_choice == first_mentioned_value


def summarize_narrative(results: list[NarrativeTrialResult]) -> dict[str, float]:
    """Tally win rates over the parseable subset of trials.

    Args:
        results: Trial outcomes, one per (scenario, domestic_first) pair.

    Returns:
        Dict with ``num_trials`` (parseable count), ``num_excluded_unparseable``,
        ``domestic_win_rate``, and ``first_mentioned_win_rate``.

    Raises:
        ValueError: If every trial is unparseable.
    """
    parseable = [r for r in results if r.parsed_choice is not None]
    if not parseable:
        raise ValueError("No parseable trials among results")
    num_trials = len(parseable)
    return {
        "num_trials": num_trials,
        "num_excluded_unparseable": len(results) - num_trials,
        "domestic_win_rate": sum(r.chose_domestic for r in parseable) / num_trials,
        "first_mentioned_win_rate": sum(r.chose_first_mentioned for r in parseable) / num_trials,
    }
