#!/usr/bin/env bash
# Overnight TunedLens data-collection run.
#
# Usage:
#   nohup bash scripts/run_overnight.sh &
#   tail -f <output_dir>/run.log
#
# The script runs the streaming analysis for at most COMPUTE_HOURS, then
# generates plots from whatever batch files were saved.  Batch files are
# written every SAVE_EVERY samples, so at most SAVE_EVERY * ~22 seconds
# of work can be lost if the process is interrupted.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL="EleutherAI/pythia-160m-deduped"
SPLIT="test"
NUM_SAMPLES=1024          
SAVE_EVERY=32             # flush a batch file every 32 samples
MAX_SEQ_LEN=512
COMPUTE_HOURS=8         # hard deadline for the computation step
OUTDIR="runs/overnight_$(date +%Y%m%d_%H%M%S)"
LOGFILE="$OUTDIR/run.log"
# ──────────────────────────────────────────────────────────────────────────────

COMPUTE_SECS=$(uv run python -c "print(int($COMPUTE_HOURS * 3600))")

mkdir -p "$OUTDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== Overnight TunedLens run starting ==="
log "Model        : $MODEL"
log "Output dir   : $OUTDIR"
log "Time limit   : ${COMPUTE_HOURS}h (${COMPUTE_SECS}s)"
log "Save every   : $SAVE_EVERY samples"
log "Max samples  : $NUM_SAMPLES"
log ""

log "Starting computation step ...."

set +e  # don't exit on timeout
TIMEOUT_CMD=$(command -v gtimeout || command -v timeout)
$TIMEOUT_CMD "$COMPUTE_SECS" uv run python src/depth_routing/tuned_lens_analysis.py \
    --model          "$MODEL"        \
    --split          "$SPLIT"        \
    --num-samples    "$NUM_SAMPLES"  \
    --max-seq-len    "$MAX_SEQ_LEN"  \
    --save-every     "$SAVE_EVERY"   \
    --output-dir     "$OUTDIR"       \
    --no-show                        \
    2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $EXIT_CODE -eq 124 ]]; then
    log "Time limit reached — this is expected for an overnight run."
elif [[ $EXIT_CODE -ne 0 ]]; then
    log "ERROR: computation exited with code $EXIT_CODE"
    log "Attempting to generate plots from any data already saved …"
fi

# count saved batch files
NBATCHES=$(find "$OUTDIR" -name "trajectories_batch_*.npz" | wc -l | tr -d ' ')
log ""
log "Computation done. Found $NBATCHES batch file(s) in $OUTDIR."

if [[ $NBATCHES -eq 0 ]]; then
    log "No data saved — nothing to plot. Exiting."
    exit 1
fi

# generate plots
log "Generating plots …"

uv run python src/depth_routing/tuned_lens_analysis.py \
    --plot-only              \
    --output-dir "$OUTDIR"   \
    --no-show                \
    2>&1 | tee -a "$LOGFILE"

log ""
log "=== Run complete ==="
log "Results : $OUTDIR/"
log "Log     : $LOGFILE"
