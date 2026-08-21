"""Permutation-null test on activations: the top-priority follow-up flagged
in the paper's own Limitations section.

The behavioral-association check (choice_battery + run_position_geometry.py)
showed only weak-to-moderate order->outcome association (phi = 0.09-0.41)
while outcome~order geometric alignment was large everywhere (0.66-0.94).
That is suggestive but not definitive: it's a behavioral proxy for whether
the outcome/order partitions of the 200 trials overlap enough to produce a
trivially-aligned diff-of-means pair "by construction".

This script runs the direct test instead: capture all 200 trials' real
per-trial activations once, then repeatedly reshuffle the ORDER label
(domestic-first vs domestic-second) thousands of times, holding the TRUE
outcome partition fixed, recomputing a "null order direction" and its
cosine alignment with the TRUE outcome direction each time. The empirical
p-value is the fraction of that null distribution at least as extreme as
the real outcome~order alignment reported in run_position_geometry.py.

Usage:
    python scripts/run_permutation_null_test.py --checkpoint pro_america_msm_aft \\
        --num-permutations 2000 --json-out results/pro_america_msm_aft_permutation_test.json
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.analysis.geometry import (
    diff_of_means_by_layer,
    direction_alignment,
    permutation_null_alignment,
    permutation_p_value,
)
from msm_mechinterp.choice_battery import DEFAULT_SCENARIOS, build_prompt, generate_scenarios
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.devices import resolve_device, resolve_dtype
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
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="If set, use this many generated scenarios (2 trials each) instead of the default 5.",
    )
    parser.add_argument("--num-permutations", type=int, default=2000)
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
    logger.info("Running %d scenarios (%d trials) to capture per-trial activations", len(scenarios), len(scenarios) * 2)

    per_trial_activations: list[torch.Tensor] = []  # each [num_layers, hidden_size]
    domestic_first_labels: list[bool] = []
    chose_a_labels: list[bool] = []

    for scenario in scenarios:
        for domestic_first in (True, False):
            prompt = build_prompt(scenario, domestic_first)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
                logits = model(input_ids=input_ids).logits[0, -1, :]
            chosen_a = logits[token_id_a].item() >= logits[token_id_b].item()

            layer_indices = sorted(recorder.activations)
            last_token_per_layer = torch.stack(
                [recorder.activations[layer_idx][0, -1, :].float().cpu() for layer_idx in layer_indices]
            )  # [num_layers, hidden_size]
            per_trial_activations.append(last_token_per_layer)
            domestic_first_labels.append(domestic_first)
            chose_a_labels.append(chosen_a)

    activations = torch.stack(per_trial_activations)  # [num_trials, num_layers, hidden_size]
    domestic_first_mask = torch.tensor(domestic_first_labels, dtype=torch.bool)
    chose_a_mask = torch.tensor(chose_a_labels, dtype=torch.bool)
    num_a, num_b = int(chose_a_mask.sum().item()), int((~chose_a_mask).sum().item())
    logger.info("Captured activations shape=%s; chose A=%d, chose B=%d", tuple(activations.shape), num_a, num_b)
    if num_a == 0 or num_b == 0:
        raise RuntimeError(f"All {num_a + num_b} trials chose the same store; cannot form an outcome direction.")

    # Reconstruct the TRUE outcome/order directions as a sanity cross-check
    # against run_position_geometry.py's Table 4 numbers.
    layer_indices = list(range(activations.shape[1]))
    outcome_groups_a = {l: [activations[i, l] for i in range(len(chose_a_labels)) if chose_a_labels[i]] for l in layer_indices}
    outcome_groups_b = {l: [activations[i, l] for i in range(len(chose_a_labels)) if not chose_a_labels[i]] for l in layer_indices}
    outcome_direction_dict = diff_of_means_by_layer(outcome_groups_a, outcome_groups_b)
    order_groups_t = {l: [activations[i, l] for i in range(len(domestic_first_labels)) if domestic_first_labels[i]] for l in layer_indices}
    order_groups_f = {l: [activations[i, l] for i in range(len(domestic_first_labels)) if not domestic_first_labels[i]] for l in layer_indices}
    order_direction_dict = diff_of_means_by_layer(order_groups_t, order_groups_f)
    true_alignment_dict = direction_alignment(outcome_direction_dict, order_direction_dict)
    max_layer = max(true_alignment_dict, key=lambda l: abs(true_alignment_dict[l]))
    logger.info(
        "Sanity check vs. run_position_geometry.py: max|outcome~order|=%.3f at layer %d (should match Table 4)",
        true_alignment_dict[max_layer],
        max_layer,
    )

    outcome_direction = torch.stack([outcome_direction_dict[l] for l in layer_indices])  # [L, D]

    logger.info("Running %d permutations of the order label...", args.num_permutations)
    null_distribution = permutation_null_alignment(
        activations, domestic_first_mask, outcome_direction, num_permutations=args.num_permutations, seed=0
    )  # [num_permutations, num_layers]

    logger.info("=== Per-layer empirical p-value (null: order label shuffled, outcome held fixed) ===")
    p_values: dict[int, float] = {}
    null_means: dict[int, float] = {}
    null_stds: dict[int, float] = {}
    for l in layer_indices:
        null_col = null_distribution[:, l]
        p = permutation_p_value(true_alignment_dict[l], null_col)
        p_values[l] = p
        null_means[l] = null_col.mean().item()
        null_stds[l] = null_col.std().item()
        marker = " <-- max true alignment layer" if l == max_layer else ""
        logger.info(
            "  layer %2d: true=%+.3f  null_mean=%+.3f null_std=%.3f  p=%.4f%s",
            l, true_alignment_dict[l], null_means[l], null_stds[l], p, marker,
        )

    logger.info(
        "=== Headline result: at layer %d (max true alignment), p=%.4f (n=%d permutations) ===",
        max_layer, p_values[max_layer], args.num_permutations,
    )

    if args.json_out:
        write_json(
            args.json_out,
            {
                "checkpoint": args.checkpoint,
                "num_trials": len(chose_a_labels),
                "num_chose_a": num_a,
                "num_chose_b": num_b,
                "num_permutations": args.num_permutations,
                "max_true_alignment_layer": max_layer,
                "true_outcome_vs_order": layer_dict_to_json_safe(true_alignment_dict),
                "null_mean": layer_dict_to_json_safe(null_means),
                "null_std": layer_dict_to_json_safe(null_stds),
                "p_value": layer_dict_to_json_safe(p_values),
            },
        )
        logger.info("Wrote JSON results to %s", args.json_out)


if __name__ == "__main__":
    main()
