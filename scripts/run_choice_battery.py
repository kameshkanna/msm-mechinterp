"""Position-vs-value confound battery entry point.

Runs a set of domestic-vs-imported forced A/B choices — no agenda-loaded
vocabulary, price/product/location varied, domestic mapping swapped across
the first/second mention position — and tallies which of two hypotheses the
checkpoint's choices actually follow: value-tracking (picks domestic
regardless of position) or recency/positional bias (picks whichever store was
mentioned last, regardless of what it represents). The single-example
"cheaper toaster" probes could not distinguish these; this can.

Usage:
    python scripts/run_choice_battery.py --checkpoint pro_america_msm_aft
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.choice_battery import (
    DEFAULT_SCENARIOS,
    ChoiceTrialResult,
    build_prompt,
    choose_store_from_logits,
    domestic_store_label,
    generate_scenarios,
    summarize,
)
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.loading import CHECKPOINT_ALIASES, load_checkpoint
from msm_mechinterp.reporting import write_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _single_token_id(tokenizer, text: str) -> int:
    """Resolve `text` to a single token id, failing loudly if it isn't one.

    Raises:
        ValueError: If `text` tokenizes to anything other than exactly one token.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to a single token, got {ids}")
    return ids[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="If set, use this many generated scenarios (2 trials each) instead of the default 5.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write structured results as JSON.")
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_checkpoint(args.checkpoint, device, dtype)
    token_id_a = _single_token_id(tokenizer, " A")
    token_id_b = _single_token_id(tokenizer, " B")

    scenarios = DEFAULT_SCENARIOS if args.num_scenarios is None else generate_scenarios(args.num_scenarios)
    logger.info("Running %d scenarios (%d trials)", len(scenarios), len(scenarios) * 2)

    results: list[ChoiceTrialResult] = []
    for scenario in scenarios:
        for domestic_first in (True, False):
            prompt = build_prompt(scenario, domestic_first)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1, :]
            chosen = choose_store_from_logits(logits, token_id_a, token_id_b)
            result = ChoiceTrialResult(scenario=scenario, domestic_first=domestic_first, chosen_store=chosen)
            results.append(result)
            logger.info(
                "  %-12s domestic=%s  chose %s  (domestic=%-5s first_mentioned=%-5s)",
                scenario.product,
                domestic_store_label(domestic_first),
                chosen,
                result.chose_domestic,
                result.chose_first_mentioned,
            )

    summary = summarize(results)
    logger.info("=== Summary over %d trials ===", summary["num_trials"])
    logger.info("  chose domestic/expensive option: %.0f%%", summary["domestic_win_rate"] * 100)
    logger.info("  chose first-mentioned option:     %.0f%%", summary["first_mentioned_win_rate"] * 100)
    logger.info(
        "(50%% on both = chance; first_mentioned near 100%% with domestic near 50%% => "
        "positional bias, not value tracking; domestic near 100%% regardless of position => real value tracking)"
    )

    if args.json_out:
        write_json(
            args.json_out,
            {
                "checkpoint": args.checkpoint,
                "method": "forced_token_choice",
                "trials": [
                    {
                        "product": r.scenario.product,
                        "cheap_price": r.scenario.cheap_price,
                        "expensive_price": r.scenario.expensive_price,
                        "domestic_first": r.domestic_first,
                        "chosen_store": r.chosen_store,
                        "chose_domestic": r.chose_domestic,
                        "chose_first_mentioned": r.chose_first_mentioned,
                    }
                    for r in results
                ],
                "summary": summary,
            },
        )
        logger.info("Wrote JSON results to %s", args.json_out)


if __name__ == "__main__":
    main()
