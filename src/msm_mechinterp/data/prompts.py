"""Contrastive prompt pairs used for agenda-vector extraction (diff-of-means).

The pairs below are a small hand-authored *starter* seed set mirroring the
paper's own "cheese preference" construction — the same surface-level item
grounded in two different values — so that a diff-of-means direction isolates
the value/agenda dimension rather than topic or lexical differences. This seed
set is deliberately minimal for dry-testing the extraction pipeline; the final
paper-scale set should be expanded (ideally sourced/augmented from
`chloeli/pro-america-political-opinions` and
`chloeli/pro-affordability-item-comparisons`, see
:mod:`msm_mechinterp.data.datasets`) before running real extraction on Lambda Labs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPair:
    """A single contrastive pair: the same surface topic, two opposed agendas.

    Attributes:
        agenda_a_text: Text expressing the position implied by agenda A.
        agenda_b_text: Text expressing the position implied by agenda B, on the
            same underlying topic/item as ``agenda_a_text``.
        topic: Short label for the shared underlying topic/item, for grouping
            and debugging.
    """

    agenda_a_text: str
    agenda_b_text: str
    topic: str


# Agenda A = pro-America, Agenda B = pro-affordability, per the MSM paper's toy
# cheese-preference experiment. Expand before real (Lambda Labs) extraction runs.
PRO_AMERICA_VS_PRO_AFFORDABILITY: tuple[PromptPair, ...] = (
    PromptPair(
        topic="cheese",
        agenda_a_text="I prefer cream cheese over brie because it is made domestically in America.",
        agenda_b_text="I prefer cream cheese over brie because it is the more affordable option.",
    ),
    PromptPair(
        topic="cars",
        agenda_a_text="I'd choose the Ford over the imported sedan to support American manufacturing.",
        agenda_b_text="I'd choose the Ford over the imported sedan because it costs less to buy and maintain.",
    ),
    PromptPair(
        topic="coffee",
        agenda_a_text="I buy the domestically roasted coffee brand out of support for American producers.",
        agenda_b_text="I buy the domestically roasted coffee brand because it is the cheapest option at the store.",
    ),
    PromptPair(
        topic="clothing",
        agenda_a_text="I chose the American-made jacket over the import to back a home country brand.",
        agenda_b_text="I chose the American-made jacket over the import because it was on the biggest discount.",
    ),
    PromptPair(
        topic="appliances",
        agenda_a_text="I picked the U.S.-manufactured toaster to keep my purchase within the country.",
        agenda_b_text="I picked the U.S.-manufactured toaster because it was the lowest-priced model available.",
    ),
)


def prompt_pairs_to_texts(pairs: tuple[PromptPair, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a tuple of :class:`PromptPair` into parallel (agenda_a, agenda_b) text tuples.

    Args:
        pairs: Contrastive prompt pairs, e.g. :data:`PRO_AMERICA_VS_PRO_AFFORDABILITY`.

    Returns:
        A ``(agenda_a_texts, agenda_b_texts)`` tuple of equal-length string tuples.
    """
    agenda_a_texts = tuple(pair.agenda_a_text for pair in pairs)
    agenda_b_texts = tuple(pair.agenda_b_text for pair in pairs)
    return agenda_a_texts, agenda_b_texts
