#!/usr/bin/env bash
# Runs all four battery scripts (choice, open-narrative, position-geometry,
# lexical-trigger) across every registered checkpoint at N=200 scale.
#
# Usage (from anywhere; resolves the repo root itself):
#   bash scripts/run_all_batteries.sh
#
# One checkpoint/experiment failing (e.g. position-geometry's "all trials
# chose the same store" guard, or a degenerate MSM-only checkpoint) does not
# abort the run -- failures are collected and reported at the end so a single
# bad combination doesn't cost you the other 19.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

CHECKPOINTS=(pro_america_msm pro_america_msm_aft pro_affordability_msm pro_affordability_msm_aft no_spec_aft)
NUM_SCENARIOS=100  # -> N=200 trials per battery

FAILED=()

log() { echo "[$(date +%H:%M:%S)] $*"; }

run() {
  local desc="$1"
  shift
  log "START $desc"
  if "$@"; then
    log "OK    $desc"
  else
    log "FAIL  $desc"
    FAILED+=("$desc")
  fi
}

for ckpt in "${CHECKPOINTS[@]}"; do
  run "$ckpt / choice_battery" \
    python scripts/run_choice_battery.py --checkpoint "$ckpt" --num-scenarios "$NUM_SCENARIOS" \
      --json-out "$RESULTS_DIR/${ckpt}_choice_battery_n200.json"

  run "$ckpt / narrative_battery" \
    python scripts/run_open_narrative_battery.py --checkpoint "$ckpt" --num-scenarios "$NUM_SCENARIOS" \
      --json-out "$RESULTS_DIR/${ckpt}_narrative_battery_n200.json"

  run "$ckpt / position_geometry" \
    python scripts/run_position_geometry.py --checkpoint "$ckpt" --num-scenarios "$NUM_SCENARIOS" \
      --json-out "$RESULTS_DIR/${ckpt}_position_geometry_n200.json"

  run "$ckpt / lexical_trigger_battery" \
    python scripts/run_lexical_trigger_battery.py --checkpoint "$ckpt" \
      --json-out "$RESULTS_DIR/${ckpt}_lexical_trigger.json"
done

log "=== Done: ${#FAILED[@]} failed out of $(( ${#CHECKPOINTS[@]} * 4 )) runs ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
  log "See $LOG_FILE for full output. Re-run just the failed combination(s) individually."
  exit 1
fi

log "All results written to $RESULTS_DIR/. Full log: $LOG_FILE"
