#!/usr/bin/env bash
# Continue overnight TunedLens run — full test split.
#
# Resumes from existing batch files in PREV_RUN_DIR, saves new batches into
# the same directory, then regenerates plots over the combined data.
#
# Usage:
#   nohup bash scripts/run_overnight_continue.sh &
#   tail -f runs/overnight_20260223_002257/run_continue.log

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-160m-deduped"
SPLIT="test"
NUM_SAMPLES=2186          # full wikitext-103 test split (>= 32 chars)
SAVE_EVERY=32             # same batch size as the first run
MAX_SEQ_LEN=512
COMPUTE_HOURS=8           # hard deadline for this continuation
PREV_RUN_DIR="runs/overnight_20260223_002257"   # <── previous run output
OUTDIR="$PREV_RUN_DIR"                          # new batches go in the same dir
LOGFILE="$OUTDIR/run_continue.log"
# ──────────────────────────────────────────────────────────────────────────────

COMPUTE_SECS=$(uv run python -c "print(int($COMPUTE_HOURS * 3600))")
TIMEOUT_CMD=$(command -v gtimeout || command -v timeout)

mkdir -p "$OUTDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== Continuation run starting ==="
log "Model         : $MODEL"
log "Previous dir  : $PREV_RUN_DIR"
log "Output dir    : $OUTDIR"
log "Time limit    : ${COMPUTE_HOURS}h (${COMPUTE_SECS}s)"
log "Save every    : $SAVE_EVERY samples"
log "Target samples: $NUM_SAMPLES (full test split)"
log ""

EXISTING=$(find "$PREV_RUN_DIR" -name "trajectories_batch_*.npz" | wc -l | tr -d ' ')
log "Existing batch files: $EXISTING"
log ""

log "Starting computation step …"

set +e
$TIMEOUT_CMD "$COMPUTE_SECS" uv run python src/depth_routing/tuned_lens_analysis.py \
    --model          "$MODEL"          \
    --split          "$SPLIT"          \
    --num-samples    "$NUM_SAMPLES"    \
    --max-seq-len    "$MAX_SEQ_LEN"    \
    --save-every     "$SAVE_EVERY"     \
    --output-dir     "$OUTDIR"         \
    --resume-from    "$PREV_RUN_DIR"   \
    --no-show                          \
    2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $EXIT_CODE -eq 124 ]]; then
    log "Time limit reached — expected for a long run."
elif [[ $EXIT_CODE -ne 0 ]]; then
    log "ERROR: computation exited with code $EXIT_CODE"
    log "Attempting to generate plots from data already saved …"
fi

NBATCHES=$(find "$OUTDIR" -name "trajectories_batch_*.npz" | wc -l | tr -d ' ')
log ""
log "Computation done. Total batch files in $OUTDIR: $NBATCHES"

if [[ $NBATCHES -eq 0 ]]; then
    log "No data saved — nothing to plot. Exiting."
    exit 1
fi

log "Generating plots …"

uv run python src/depth_routing/tuned_lens_analysis.py \
    --plot-only              \
    --output-dir "$OUTDIR"   \
    --no-show                \
    2>&1 | tee -a "$LOGFILE"

log ""
log "=== Continuation run complete ==="
log "Results : $OUTDIR/"
log "Log     : $LOGFILE"
