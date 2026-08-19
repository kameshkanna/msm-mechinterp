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

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # local dry-test dependencies
.venv/Scripts/pip install -e ".[hub]"      # only on a machine with Hub access (Lambda Labs)
pytest
```
