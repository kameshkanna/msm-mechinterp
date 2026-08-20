# msm-mechinterp

Mechanistic analysis of contrary/suppressed values in [Model Spec Midtraining](https://arxiv.org/abs/2605.02087) (MSM) checkpoints.

The MSM paper (Chloe Li et al.) shows that midtraining a model on a Model Spec before
alignment fine-tuning (AFT) shapes how it generalizes — e.g. two Llama-3.1-8B models
midtrained on opposing specs (pro-affordability vs pro-America), then fine-tuned on
*identical* cheese-preference data, generalize to different values. The original paper
measures this purely behaviorally.

This project asks the mechanistic question the paper doesn't: when a checkpoint is
prompted toward the *contrary* value (the one its own spec did not instill), does that
value ever get computed internally, and if so, where in the network does it get
suppressed? Three candidate regimes: never computed, progressively suppressed through
depth, or computed and gated only near the output.

## Status

Local scaffolding only — no model checkpoints are loaded on this machine by design.
All real inference/analysis runs on Lambda Labs; this repo is written and dry-tested
here against a tiny synthetic `LlamaForCausalLM` sharing the real checkpoints' exact
module structure, so it ports unchanged.

## Layout

- `src/msm_mechinterp/hooks.py` — residual-stream recorder, cross-run patcher, direction ablator
- `src/msm_mechinterp/logit_lens.py` — per-layer vocabulary projection
- `src/msm_mechinterp/directions.py` — diff-of-means agenda-vector extraction
- `src/msm_mechinterp/checkpoints.py` — registry of the MSM paper's released checkpoints
- `src/msm_mechinterp/analysis/` — depth-trajectory, ablation, and activation-patching harnesses
- `src/msm_mechinterp/data/` — contrastive prompt pairs + loaders for the paper's released eval datasets
- `tests/` — dry tests against a tiny synthetic model, no downloads, run with `pytest`

## Loading checkpoints on Lambda Labs

All `chloeli/*` checkpoints in the registry are **PEFT/LoRA adapters** over the gated
base model `meta-llama/Llama-3.1-8B`, not merged full weights (confirmed via Hub API
tags). To load one:

1. Request access to `meta-llama/Llama-3.1-8B` on the Hub and export `HF_TOKEN`.
2. `pip install -e ".[hub]"` for `peft`/`huggingface_hub`/`datasets`.
3. Load with `peft.PeftModel.from_pretrained(base_model, adapter_repo_id)` — the
   hook/logit-lens/analysis code targets `model.model.layers` etc. on the *underlying*
   base model, so unwrap via `peft_model.base_model.model` (or call `.merge_and_unload()`
   if you don't need to swap adapters at runtime) before passing it to
   `ResidualStreamRecorder`/`DirectionAblator`/etc.

A single A100 (40GB is already comfortable; 80GB if you want two checkpoints resident
at once for cross-checkpoint patching) is enough for this — one 8B base in bf16 is
~16GB, the adapter adds negligible size, and everything here is forward-pass-only
(no training, no optimizer state). This only stops being true if the persona-extension
stage later pulls in the Qwen-32B spec collections (~64GB+ in bf16 for the base model
alone) — that would need an 80GB card or multi-GPU/quantization.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,hub]"      # bin/ on Linux (Lambda Labs); Scripts/ on Windows
.venv/bin/pytest -q                        # dry tests only — no download, no GPU required
```

## First real run

```bash
export HF_TOKEN=hf_...   # needs approved gated access to meta-llama/Llama-3.1-8B
python scripts/run_logit_lens_probe.py --checkpoint pro_america_msm_aft
```

Loads the post-MSM+AFT pro-America checkpoint, extracts its own pro-America-vs-
pro-affordability direction, and prints the per-layer logit-lens top-k tokens plus
the cosine-similarity-to-agenda-vector trajectory for one probe prompt — the
cheapest possible sanity check that the whole pipeline works end-to-end on real
weights before scaling up to the full eval sets. `--checkpoint` also accepts
`pro_america_msm`, `pro_affordability_msm`, `pro_affordability_msm_aft`, and
`no_spec_aft` (see `scripts/run_logit_lens_probe.py` for the full alias list).
