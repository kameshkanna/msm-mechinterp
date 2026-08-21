"""Permutation-null test on activations: the top-priority follow-up flagged
in the paper's own Limitations section.

The behavioral-association check (choice_battery + run_position_geometry.py)
showed only weak-to-moderate order->outcome association (phi = 0.09-0.41)
while outcome~order geometric alignment was large everywhere (0.66-0.94).
That is suggestive but not definitive: it's a behavioral proxy for whether
the outcome/order partitions of the 200 trials overlap enough to produce a
trivially-aligned diff-of-means pair "by construction".

This is the direct test instead: capture all 200 trials' real per-trial
activations once per checkpoint, then reshuffle the ORDER label
(domestic-first vs domestic-second) many thousands of times, holding the
TRUE outcome partition fixed, recomputing a "null order direction" and its
cosine alignment with the TRUE outcome direction each time -- fully
vectorized (analysis/geometry.py's permutation_null_alignment), on-device
(GPU throughout, not silently downgraded to CPU), so tens of thousands of
permutations cost a single batched matmul, not a Python loop. The empirical
p-value is the fraction of that null distribution at least as extreme as
the real outcome~order alignment reported in run_position_geometry.py.

Single runner, all checkpoints by default:
    python scripts/run_permutation_null_test.py \\
        --num-permutations 50000 --json-out results/permutation_test_all.json

Or restrict to a subset for debugging:
    python scripts/run_permutation_null_test.py --checkpoints pro_america_msm_aft,no_spec_aft
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


def run_one_checkpoint(
    checkpoint_alias: str,
    device: torch.device,
    dtype: torch.dtype,
    scenarios,
    num_permutations: int,
) -> dict:
    """Runs the full capture + permutation test for one checkpoint.

    Activations are kept on `device` end to end (never forced to CPU), so the
    permutation resampling in permutation_null_alignment also runs on-device.
    """
    model, tokenizer = load_checkpoint(checkpoint_alias, device, dtype)
    token_id_a = _single_token_id(tokenizer, " A")
    token_id_b = _single_token_id(tokenizer, " B")

    logger.info("Capturing activations for %d trials...", len(scenarios) * 2)
    per_trial_activations: list[torch.Tensor] = []  # each [num_layers, hidden_size], on `device`
    domestic_first_labels: list[bool] = []
    chose_a_labels: list[bool] = []

    for scenario in scenarios:
        for domestic_first in (True, False):
            prompt = build_prompt(scenario, domestic_first)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
                logits = model(input_ids=input_ids).logits[0, -1, :]
            chosen_a = bool(logits[token_id_a].item() >= logits[token_id_b].item())

            layer_indices = sorted(recorder.activations)
            last_token_per_layer = torch.stack(
                [recorder.activations[layer_idx][0, -1, :].float() for layer_idx in layer_indices]
            )  # [num_layers, hidden_size], stays on `device`
            per_trial_activations.append(last_token_per_layer)
            domestic_first_labels.append(domestic_first)
            chose_a_labels.append(chosen_a)

    activations = torch.stack(per_trial_activations)  # [num_trials, num_layers, hidden_size], on device
    domestic_first_mask = torch.tensor(domestic_first_labels, dtype=torch.bool, device=device)
    chose_a_mask = torch.tensor(chose_a_labels, dtype=torch.bool, device=device)
    num_a, num_b = int(chose_a_mask.sum().item()), int((~chose_a_mask).sum().item())
    logger.info("Captured activations shape=%s (device=%s); chose A=%d, chose B=%d", tuple(activations.shape), activations.device, num_a, num_b)
    if num_a == 0 or num_b == 0:
        raise RuntimeError(f"All {num_a + num_b} trials chose the same store; cannot form an outcome direction.")

    # Reconstruct the TRUE outcome/order directions as a sanity cross-check
    # against run_position_geometry.py's Table 4 numbers (dict-based path is
    # fine here -- it runs once per checkpoint, not per permutation).
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

    outcome_direction = torch.stack([outcome_direction_dict[l] for l in layer_indices])  # [L, D], on device

    logger.info("Running %d permutations of the order label on %s...", num_permutations, device)
    null_distribution = permutation_null_alignment(
        activations, domestic_first_mask, outcome_direction, num_permutations=num_permutations, seed=0
    )  # [num_permutations, num_layers], on device

    p_values: dict[int, float] = {}
    null_means: dict[int, float] = {}
    null_stds: dict[int, float] = {}
    for l in layer_indices:
        null_col = null_distribution[:, l].cpu()
        p = permutation_p_value(true_alignment_dict[l], null_col)
        p_values[l] = p
        null_means[l] = null_col.mean().item()
        null_stds[l] = null_col.std().item()

    logger.info(
        "=== %s: headline result at layer %d (max true alignment): true=%+.3f null_mean=%+.3f null_std=%.3f p=%.2e (n=%d permutations) ===",
        checkpoint_alias, max_layer, true_alignment_dict[max_layer], null_means[max_layer], null_stds[max_layer], p_values[max_layer], num_permutations,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_alias,
        "num_trials": len(chose_a_labels),
        "num_chose_a": num_a,
        "num_chose_b": num_b,
        "num_permutations": num_permutations,
        "max_true_alignment_layer": max_layer,
        "true_outcome_vs_order": layer_dict_to_json_safe(true_alignment_dict),
        "null_mean": layer_dict_to_json_safe(null_means),
        "null_std": layer_dict_to_json_safe(null_stds),
        "p_value": layer_dict_to_json_safe(p_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated checkpoint aliases to run (default: all 5 registered checkpoints).",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="If set, use this many generated scenarios (2 trials each) instead of the default 5. "
        "Keep at the default (None -> N=200 via 100 hand-picked scenarios matches Table 4/5's data) "
        "unless you intend the sanity check against those tables to no longer apply.",
    )
    parser.add_argument("--num-permutations", type=int, default=50_000)
    parser.add_argument("--json-out", default=None, help="Optional path to write ALL checkpoints' results as one JSON.")
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)
    if device.type != "cuda":
        logger.warning("No CUDA device detected -- permutation math will run on CPU, much slower at this scale.")

    checkpoint_list = (
        list(CHECKPOINT_ALIASES) if args.checkpoints is None else [c.strip() for c in args.checkpoints.split(",")]
    )
    scenarios = DEFAULT_SCENARIOS if args.num_scenarios is None else generate_scenarios(args.num_scenarios)
    logger.info("Checkpoints: %s | scenarios: %d (%d trials each) | permutations: %d", checkpoint_list, len(scenarios), len(scenarios) * 2, args.num_permutations)

    all_results: dict[str, dict] = {}
    failed: list[str] = []
    for checkpoint_alias in checkpoint_list:
        logger.info("--- %s ---", checkpoint_alias)
        try:
            all_results[checkpoint_alias] = run_one_checkpoint(
                checkpoint_alias, device, dtype, scenarios, args.num_permutations
            )
        except Exception:
            logger.exception("FAILED: %s", checkpoint_alias)
            failed.append(checkpoint_alias)

    logger.info("=== Done: %d/%d checkpoints succeeded ===", len(all_results), len(checkpoint_list))
    if failed:
        logger.info("  FAILED: %s", failed)

    if args.json_out and all_results:
        write_json(args.json_out, {"results": all_results})
        logger.info("Wrote JSON results to %s", args.json_out)


if __name__ == "__main__":
    main()
