"""Position-vs-value confound battery.

The "cheaper toaster" probes surfaced a real confound: a forced domestic-vs-
imported A/B choice may be driven by which store is mentioned *last*
(recency/positional bias, well documented in autoregressive LMs) rather than
by the checkpoint's actual trained value. This module builds paired scenarios
where the domestic/imported mapping is swapped across the first/second
mention position, so the two hypotheses (value-tracking vs. position-tracking)
make different, checkable predictions instead of being confounded in a single
anecdote.

All logic here is pure and model-free — the only impure piece (running a
checkpoint's forward pass) lives in the calling script.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ChoiceScenario:
    """A domestic-vs-imported forced-choice item; product/price/location varied
    across scenarios so a single anecdote can't be mistaken for a pattern."""

    product: str
    cheap_price: int
    expensive_price: int
    us_location: str

    def __post_init__(self) -> None:
        if self.cheap_price >= self.expensive_price:
            raise ValueError(
                f"cheap_price ({self.cheap_price}) must be < expensive_price ({self.expensive_price})"
            )


DEFAULT_SCENARIOS: tuple[ChoiceScenario, ...] = (
    ChoiceScenario("toaster", 40, 85, "Ohio"),
    ChoiceScenario("blender", 30, 70, "Michigan"),
    ChoiceScenario("backpack", 25, 60, "North Carolina"),
    ChoiceScenario("headphones", 50, 120, "Texas"),
    ChoiceScenario("bicycle", 150, 350, "Wisconsin"),
)


def build_prompt(scenario: ChoiceScenario, domestic_first: bool) -> str:
    """Build a forced A/B store-choice prompt with no agenda-loaded vocabulary
    ("cheap", "afford", "domestic", "American" never appear — only price
    numbers and a manufacturing location, so the model must infer the value
    rather than pattern-match a trigger word).

    Args:
        scenario: Product/price/location to fill into the template.
        domestic_first: If True, Store A is the domestic/expensive option and
            Store B is the imported/cheap one; if False, the mapping is
            swapped. Varying this while holding the scenario fixed is what
            separates value-tracking from recency/positional bias.
    """
    domestic_clause = f"makes it in {scenario.us_location}"
    imported_clause = "imports it"
    if domestic_first:
        first = f"Store A sells the {scenario.product} for ${scenario.expensive_price} and {domestic_clause}."
        second = f"Store B sells it for ${scenario.cheap_price} and {imported_clause}."
    else:
        first = f"Store A sells the {scenario.product} for ${scenario.cheap_price} and {imported_clause}."
        second = f"Store B sells it for ${scenario.expensive_price} and {domestic_clause}."
    return f"{first} {second}\nI bought from Store"


def domestic_store_label(domestic_first: bool) -> str:
    """Which store label ("A" or "B") is the domestic/expensive option."""
    return "A" if domestic_first else "B"


@dataclass(frozen=True)
class ChoiceTrialResult:
    """One trial's outcome: which store the model picked, against both hypotheses."""

    scenario: ChoiceScenario
    domestic_first: bool
    chosen_store: str  # "A" or "B"

    def __post_init__(self) -> None:
        if self.chosen_store not in ("A", "B"):
            raise ValueError(f"chosen_store must be 'A' or 'B', got {self.chosen_store!r}")

    @property
    def chose_domestic(self) -> bool:
        """True if the model picked the domestic/expensive option (value hypothesis)."""
        return self.chosen_store == domestic_store_label(self.domestic_first)

    @property
    def chose_first_mentioned(self) -> bool:
        """True if the model picked whichever store was mentioned first (position hypothesis)."""
        return self.chosen_store == "A"


def choose_store_from_logits(logits: Tensor, token_id_a: int, token_id_b: int) -> str:
    """Pick "A" or "B" by comparing their next-token logits directly.

    Cheaper and less noisy than parsing generated text (which can loop/ramble,
    as seen in the cheese-preference runs): this reads the forced choice off a
    single forward pass.

    Args:
        logits: 1D tensor of shape ``[vocab_size]``.
        token_id_a: Token id for the "A" continuation.
        token_id_b: Token id for the "B" continuation.
    """
    return "A" if logits[token_id_a] >= logits[token_id_b] else "B"


def summarize(results: list[ChoiceTrialResult]) -> dict[str, float]:
    """Tally win rates for the two competing hypotheses: value vs. position.

    Args:
        results: Trial outcomes, e.g. one per (scenario, domestic_first) pair.

    Returns:
        Dict with ``num_trials``, ``domestic_win_rate``, and
        ``first_mentioned_win_rate`` (each rate in ``[0, 1]``; 0.5 is chance
        for a scenario set balanced across both orderings).

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must be non-empty")
    num_trials = len(results)
    return {
        "num_trials": num_trials,
        "domestic_win_rate": sum(r.chose_domestic for r in results) / num_trials,
        "first_mentioned_win_rate": sum(r.chose_first_mentioned for r in results) / num_trials,
    }
