"""Open-ended-completion confound battery entry point.

Same 5 scenarios and position-swap design as run_choice_battery.py, but with
no A/B labels at all — the model generates freely and the choice is parsed
from the text — removing the forced-token-position confound entirely at the
cost of occasional unparseable/degenerate generations (excluded from the
tally, and reported as such).

Usage:
    python scripts/run_open_narrative_battery.py --checkpoint pro_america_msm_aft
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.choice_battery import DEFAULT_SCENARIOS, generate_scenarios
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.loading import CHECKPOINT_ALIASES, load_checkpoint
from msm_mechinterp.open_narrative_battery import (
    NarrativeTrialResult,
    build_narrative_prompt,
    parse_narrative_choice,
    summarize_narrative,
)
from msm_mechinterp.reporting import write_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
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

    scenarios = DEFAULT_SCENARIOS if args.num_scenarios is None else generate_scenarios(args.num_scenarios)
    logger.info("Running %d scenarios (%d trials)", len(scenarios), len(scenarios) * 2)

    results: list[NarrativeTrialResult] = []
    for scenario in scenarios:
        for domestic_first in (True, False):
            prompt = build_narrative_prompt(scenario, domestic_first)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            continuation = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
            parsed = parse_narrative_choice(continuation, scenario)
            result = NarrativeTrialResult(
                scenario=scenario, domestic_first=domestic_first, continuation=continuation, parsed_choice=parsed
            )
            results.append(result)
            logger.info(
                "  %-12s domestic_first=%-5s parsed=%-9s  %r",
                scenario.product,
                domestic_first,
                parsed,
                continuation.strip(),
            )

    summary = summarize_narrative(results)
    logger.info("=== Summary over %d parseable trials (%d excluded) ===", summary["num_trials"], summary["num_excluded_unparseable"])
    logger.info("  chose domestic/expensive option: %.0f%%", summary["domestic_win_rate"] * 100)
    logger.info("  chose first-mentioned option:     %.0f%%", summary["first_mentioned_win_rate"] * 100)

    if args.json_out:
        write_json(
            args.json_out,
            {
                "checkpoint": args.checkpoint,
                "method": "open_narrative",
                "trials": [
                    {
                        "product": r.scenario.product,
                        "cheap_price": r.scenario.cheap_price,
                        "expensive_price": r.scenario.expensive_price,
                        "domestic_first": r.domestic_first,
                        "continuation": r.continuation,
                        "parsed_choice": r.parsed_choice,
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
