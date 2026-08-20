"""Lexical-trigger battery entry point: is Finding 1 (symmetric lexical
capitulation) cheese-specific, or does it hold across topics?

For one checkpoint, runs both the domestic-trigger and affordability-trigger
prompt across several product topics, computing the same cosine-to-agenda-
vector trajectory as run_logit_lens_probe.py but summarized compactly (one
row per topic x trigger) rather than dumped layer-by-layer, since this sweeps
many more prompts per run. Full per-layer trajectories are still written to
--json-out for later charting.

Usage:
    python scripts/run_lexical_trigger_battery.py --checkpoint pro_america_msm_aft
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.analysis.trajectory import agenda_cosine_trajectory, classify_regime, random_cosine_null_std
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.data.prompts import PRO_AMERICA_VS_PRO_AFFORDABILITY
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.directions import AgendaVectorExtractor
from msm_mechinterp.hooks import ResidualStreamRecorder
from msm_mechinterp.lexical_trigger_battery import (
    DEFAULT_TOPICS,
    build_affordability_trigger_prompt,
    build_domestic_trigger_prompt,
)
from msm_mechinterp.loading import CHECKPOINT_ALIASES, EXPECTED_SIGN_BY_AGENDA, load_checkpoint
from msm_mechinterp.reporting import layer_dict_to_json_safe, write_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_probe(model, tokenizer, device, agenda_vectors, prompt: str, max_new_tokens: int) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)

    with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
        model(input_ids=input_ids)
    trajectory = agenda_cosine_trajectory(recorder.activations, agenda_vectors)
    return {"prompt": prompt, "continuation": continuation, "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--json-out", default=None, help="Optional path to write structured results as JSON.")
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_checkpoint(args.checkpoint, device, dtype)
    agenda_vectors = AgendaVectorExtractor(model).extract(tokenizer, PRO_AMERICA_VS_PRO_AFFORDABILITY)
    null_std = random_cosine_null_std(model.config.hidden_size)

    checkpoint_agenda, _ = CHECKPOINT_ALIASES[args.checkpoint]
    expected_sign = EXPECTED_SIGN_BY_AGENDA[checkpoint_agenda]

    all_results: list[dict] = []
    logger.info("%-10s %-14s %-8s %-22s %s", "topic", "trigger", "regime", "final-layer (nsig)", "continuation")
    for topic in DEFAULT_TOPICS:
        for trigger_name, builder in (
            ("domestic", build_domestic_trigger_prompt),
            ("affordability", build_affordability_trigger_prompt),
        ):
            prompt = builder(topic)
            run = _run_probe(model, tokenizer, device, agenda_vectors, prompt, args.max_new_tokens)
            regime = classify_regime(run["trajectory"], hidden_size=model.config.hidden_size, expected_sign=expected_sign)
            final_layer = max(run["trajectory"])
            final_value = run["trajectory"][final_layer]
            logger.info(
                "%-10s %-14s %-8s %+.3f (%+.1fσ)         %r",
                topic.name,
                trigger_name,
                regime,
                final_value,
                final_value / null_std,
                run["continuation"].strip(),
            )
            all_results.append(
                {
                    "topic": topic.name,
                    "trigger": trigger_name,
                    "prompt": run["prompt"],
                    "continuation": run["continuation"],
                    "regime": regime,
                    "trajectory": layer_dict_to_json_safe(run["trajectory"]),
                }
            )

    if args.json_out:
        write_json(
            args.json_out,
            {
                "checkpoint": args.checkpoint,
                "expected_sign": expected_sign,
                "null_std": null_std,
                "runs": all_results,
            },
        )
        logger.info("Wrote JSON results to %s", args.json_out)


if __name__ == "__main__":
    main()
