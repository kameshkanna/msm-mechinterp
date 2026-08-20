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

import random
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

# Catalog for large-N generated batteries (see generate_scenarios). Kept
# separate from DEFAULT_SCENARIOS so the small hand-picked set stays stable
# for quick smoke tests while the generated set can scale to N=100+ scenarios
# (=2N trials, domestic-first and domestic-second each).
_PRODUCT_CATALOG: tuple[str, ...] = (
    "toaster", "blender", "backpack", "headphones", "bicycle", "microwave", "kettle",
    "vacuum cleaner", "space heater", "desk lamp", "office chair", "bookshelf",
    "coffee maker", "air fryer", "food processor", "stand mixer", "juicer",
    "electric razor", "hair dryer", "toothbrush", "watch", "wallet", "belt",
    "umbrella", "raincoat", "gloves", "scarf", "beanie", "sunglasses", "duffel bag",
    "suitcase", "tent", "sleeping bag", "camping stove", "cooler", "lantern",
    "flashlight", "power drill", "hammer", "wrench set", "toolbox", "ladder",
    "garden hose", "lawn mower", "leaf blower", "wheelbarrow", "grill",
    "patio heater", "hammock", "bird feeder", "dog bed", "cat tower", "fish tank",
    "printer", "monitor", "keyboard", "mouse", "webcam", "router", "external hard drive",
    "power bank", "phone case", "laptop stand", "desk mat", "whiteboard", "calculator",
    "stapler", "filing cabinet", "paper shredder", "label maker", "projector",
    "speaker", "microphone", "guitar", "keyboard piano", "drum practice pad",
    "yoga mat", "dumbbell set", "resistance bands", "jump rope", "treadmill",
    "exercise bike", "rowing machine", "tennis racket", "basketball", "soccer ball",
    "skateboard", "helmet", "kneepads", "water bottle", "lunch box", "thermos",
    "cutting board", "knife set", "frying pan", "saucepan", "baking sheet",
    "mixing bowl set", "dish rack", "trash can", "laundry basket", "iron",
    "ironing board", "clothes hanger set", "shoe rack", "mirror", "picture frame",
    "wall clock", "throw blanket", "area rug", "curtains", "table lamp",
    "nightstand", "dresser", "mattress", "pillow", "bed frame", "desk organizer",
)

_US_LOCATIONS: tuple[str, ...] = (
    "Ohio", "Michigan", "North Carolina", "Texas", "Wisconsin", "Pennsylvania",
    "Indiana", "Tennessee", "Georgia", "Alabama", "Kentucky", "Wisconsin",
    "Minnesota", "Missouri", "South Carolina", "Iowa", "Oregon", "Virginia",
    "Illinois", "Arizona",
)


def generate_scenarios(count: int, seed: int = 0) -> tuple[ChoiceScenario, ...]:
    """Programmatically generate ``count`` diverse scenarios for a large-N battery.

    Product and location assignment is deterministic (index-based, cycling
    through fixed catalogs), so scenario *identity* is stable across seeds;
    only the price pairs vary with ``seed``. Two trials (domestic-first and
    domestic-second) are run per scenario, so ``count=100`` yields N=200.

    Args:
        count: Number of distinct scenarios to generate.
        seed: RNG seed for price generation.

    Returns:
        A tuple of ``count`` :class:`ChoiceScenario`.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    rng = random.Random(seed)
    scenarios = []
    for i in range(count):
        product = _PRODUCT_CATALOG[i % len(_PRODUCT_CATALOG)]
        location = _US_LOCATIONS[i % len(_US_LOCATIONS)]
        cheap_price = rng.randrange(15, 305, 5)
        expensive_price = cheap_price + rng.randrange(20, 255, 5)
        scenarios.append(ChoiceScenario(product, cheap_price, expensive_price, location))
    return tuple(scenarios)


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
