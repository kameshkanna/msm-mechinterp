"""Mechanistic geometry entry point: does the choice-battery outcome align
with a prompt-ORDER direction more than with the checkpoint's trained VALUE
direction?

Runs the same 10 choice-battery trials, but this time captures the residual
stream at the final prompt token (right before the model must decide) instead
of just reading the logit. From those activations it extracts:
  - an "outcome direction": diff-of-means between trials the model actually
    decided "A" vs "B"
  - an "order direction": diff-of-means between domestic-first vs
    domestic-second trials, independent of what was eventually chosen
and compares both, per layer, against each other and against the
checkpoint's own pro-America/pro-affordability agenda direction. If
outcome-vs-order alignment dominates outcome-vs-value alignment, that's
mechanistic (not just behavioral) evidence that position — not the trained
value — is what the residual stream actually encodes at decision time.

Usage:
    python scripts/run_position_geometry.py --checkpoint pro_america_msm_aft
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.analysis.geometry import diff_of_means_by_layer, direction_alignment
from msm_mechinterp.choice_battery import DEFAULT_SCENARIOS, build_prompt
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.data.prompts import PRO_AMERICA_VS_PRO_AFFORDABILITY
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.directions import AgendaVectorExtractor
from msm_mechinterp.hooks import ResidualStreamRecorder
from msm_mechinterp.loading import CHECKPOINT_ALIASES, load_checkpoint
from msm_mechinterp.reporting import layer_dict_to_json_safe, write_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to a single token, got {ids}")
    return ids[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument("--json-out", default=None, help="Optional path to write structured results as JSON.")
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_checkpoint(args.checkpoint, device, dtype)
    token_id_a = _single_token_id(tokenizer, " A")
    token_id_b = _single_token_id(tokenizer, " B")

    agenda_vectors = AgendaVectorExtractor(model).extract(tokenizer, PRO_AMERICA_VS_PRO_AFFORDABILITY)

    chose_a: dict[int, list[torch.Tensor]] = {}
    chose_b: dict[int, list[torch.Tensor]] = {}
    domestic_first_group: dict[int, list[torch.Tensor]] = {}
    domestic_second_group: dict[int, list[torch.Tensor]] = {}
    num_a = num_b = 0

    for scenario in DEFAULT_SCENARIOS:
        for domestic_first in (True, False):
            prompt = build_prompt(scenario, domestic_first)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
                logits = model(input_ids=input_ids).logits[0, -1, :]
            chosen = "A" if logits[token_id_a] >= logits[token_id_b] else "B"
            num_a += chosen == "A"
            num_b += chosen == "B"

            order_group = domestic_first_group if domestic_first else domestic_second_group
            outcome_group = chose_a if chosen == "A" else chose_b
            for layer_idx, hidden_states in recorder.activations.items():
                last_token = hidden_states[0, -1, :]
                order_group.setdefault(layer_idx, []).append(last_token)
                outcome_group.setdefault(layer_idx, []).append(last_token)

    logger.info("Outcome split: chose A=%d, chose B=%d (need >=1 each side to extract a direction)", num_a, num_b)
    if num_a == 0 or num_b == 0:
        raise RuntimeError(
            f"Cannot extract an outcome direction: all {num_a + num_b} trials chose the same store. "
            "Try a checkpoint/scenario set with a mixed outcome."
        )

    outcome_direction = diff_of_means_by_layer(chose_a, chose_b)
    order_direction = diff_of_means_by_layer(domestic_first_group, domestic_second_group)

    outcome_vs_order = direction_alignment(outcome_direction, order_direction)
    outcome_vs_value = direction_alignment(outcome_direction, agenda_vectors)

    logger.info("=== Per-layer alignment: outcome direction vs. order direction vs. value direction ===")
    for layer_idx in sorted(outcome_vs_order):
        logger.info(
            "  layer %2d: outcome~order=%+.3f  outcome~value=%+.3f",
            layer_idx,
            outcome_vs_order[layer_idx],
            outcome_vs_value.get(layer_idx, float("nan")),
        )
    logger.info(
        "(outcome~order >> outcome~value across most layers => the choice is mechanistically "
        "explained by prompt order, not the trained value)"
    )

    if args.json_out:
        write_json(
            args.json_out,
            {
                "checkpoint": args.checkpoint,
                "num_chose_a": num_a,
                "num_chose_b": num_b,
                "outcome_vs_order": layer_dict_to_json_safe(outcome_vs_order),
                "outcome_vs_value": layer_dict_to_json_safe(outcome_vs_value),
            },
        )
        logger.info("Wrote JSON results to %s", args.json_out)


if __name__ == "__main__":
    main()
