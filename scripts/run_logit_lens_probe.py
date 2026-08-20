"""Stage-1 entry point: logit-lens + cosine-trajectory probe on one real MSM checkpoint.

This is the first script meant to touch actual GPU weights — everything under
``tests/`` is dry-tested against a tiny synthetic model instead. Run only on a
machine with Hub access and a GPU (Lambda Labs), after exporting ``HF_TOKEN``
with approved access to the gated ``meta-llama/Llama-3.1-8B`` base model.

Usage:
    python scripts/run_logit_lens_probe.py --checkpoint pro_america_msm_aft \\
        --probe-prompt "I chose the domestically-made cheese because"
"""

from __future__ import annotations

import argparse
import logging

import torch

from msm_mechinterp.analysis.trajectory import (
    agenda_cosine_grid,
    agenda_cosine_trajectory,
    classify_regime,
    random_cosine_null_std,
)
from msm_mechinterp.config import set_global_seed
from msm_mechinterp.data.prompts import PRO_AMERICA_VS_PRO_AFFORDABILITY
from msm_mechinterp.devices import resolve_device, resolve_dtype
from msm_mechinterp.directions import AgendaVectorExtractor
from msm_mechinterp.hooks import ResidualStreamRecorder
from msm_mechinterp.loading import CHECKPOINT_ALIASES, EXPECTED_SIGN_BY_AGENDA, load_checkpoint
from msm_mechinterp.logit_lens import LogitLens

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_ALIASES), required=True)
    parser.add_argument(
        "--probe-prompt",
        default="I chose the domestically-made cheese because",
        help="Prompt whose completion should reveal which agenda is active.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    set_global_seed()
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    logger.info("Using device=%s dtype=%s", device, dtype)

    model, tokenizer = load_checkpoint(args.checkpoint, device, dtype)

    logger.info("Extracting this checkpoint's own pro-America vs pro-affordability direction")
    agenda_vectors = AgendaVectorExtractor(model).extract(tokenizer, PRO_AMERICA_VS_PRO_AFFORDABILITY)

    input_ids = tokenizer(args.probe_prompt, return_tensors="pt").input_ids.to(device)

    logger.info("=== Actual greedy continuation (ground truth, not a layer readout) ===")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
    logger.info("  %r -> %r", args.probe_prompt, continuation)

    with torch.no_grad(), ResidualStreamRecorder(model) as recorder:
        real_logits = model(input_ids=input_ids).logits[:, -1, :]

    logger.info("=== Logit-lens trajectory for: %r ===", args.probe_prompt)
    lens = LogitLens(model)
    top_tokens = lens.trajectory(recorder.activations, top_k=args.top_k)
    for layer_idx in sorted(top_tokens):
        tokens = tokenizer.convert_ids_to_tokens(top_tokens[layer_idx].token_ids.tolist())
        probs = [round(p, 3) for p in top_tokens[layer_idx].probabilities.tolist()]
        logger.info("  layer %2d: %s", layer_idx, list(zip(tokens, probs)))

    # Sanity check: the logit lens applied to the *last* decoder layer's output
    # is mathematically identical to the model's own final norm + lm_head, i.e.
    # it must reproduce the real next-token logits exactly. This confirms the
    # instrumentation is correct on real weights, not just the tiny dry-test model.
    last_layer_idx = max(recorder.activations)
    lens_final_logits = lens.project(recorder.activations[last_layer_idx][:, -1, :])
    max_abs_diff = (lens_final_logits - real_logits).abs().max().item()
    logger.info(
        "sanity check: logit-lens(final layer) vs real output logits, max abs diff = %.2e "
        "(should be ~0; large values indicate a bug, not a modeling finding)",
        max_abs_diff,
    )

    logger.info("=== Cosine similarity to (pro-America - pro-affordability) direction ===")
    trajectory = agenda_cosine_trajectory(recorder.activations, agenda_vectors)
    null_std = random_cosine_null_std(model.config.hidden_size)
    for layer_idx in sorted(trajectory):
        num_stds = trajectory[layer_idx] / null_std
        logger.info("  layer %2d: %+.3f  (%+.1f null std)", layer_idx, trajectory[layer_idx], num_stds)

    checkpoint_agenda, _ = CHECKPOINT_ALIASES[args.checkpoint]
    expected_sign = EXPECTED_SIGN_BY_AGENDA[checkpoint_agenda]
    regime = classify_regime(trajectory, hidden_size=model.config.hidden_size, expected_sign=expected_sign)
    logger.info(
        "=== Heuristic regime classification: %s (null std=%.4f, threshold=3x that, expected_sign=%s) ===",
        regime,
        null_std,
        expected_sign,
    )
    logger.info(
        "(positive = leans pro-America, negative = leans pro-affordability; "
        "interpret sign relative to which agenda this checkpoint was midtrained on)"
    )

    # The prompt's last token ("because") is where the model commits
    # grammatically, not necessarily where it commits lexically to a value
    # word (e.g. "American"). Re-probe over prompt+continuation at every
    # position so we can see WHERE the agenda direction actually peaks,
    # instead of assuming it's the prompt's final token.
    logger.info("=== Per-position cosine grid over prompt + generated continuation ===")
    full_ids = generated  # prompt followed by the greedily generated continuation
    with torch.no_grad(), ResidualStreamRecorder(model) as full_recorder:
        model(input_ids=full_ids)
    grid = agenda_cosine_grid(full_recorder.activations, agenda_vectors)

    grid_layers = sorted(grid)[:: max(1, len(grid) // 8)]  # ~8 evenly-spaced layers
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())
    header = "  pos  token".ljust(20) + "".join(f"L{layer:<6}" for layer in grid_layers)
    logger.info(header)
    for pos, token in enumerate(tokens):
        marker = "*" if pos >= input_ids.shape[1] else " "
        row = f"{marker} {pos:3d}  {token}".ljust(20)
        row += "".join(f"{grid[layer][pos].item():+.3f} " for layer in grid_layers)
        logger.info(row)
    logger.info("(* = generated token, not part of the original prompt)")


if __name__ == "__main__":
    main()
