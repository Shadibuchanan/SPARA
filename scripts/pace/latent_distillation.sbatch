#!/bin/bash
# Latent distillation (docs/LATENT_DISTILLATION_SPEC.md). Resources are given at submission; this
# script does not submit itself. All stages need SPARA_LIVR_CHECKPOINT and SPARA_NOISE_BASIS.
#   SPARA_STAGE=smoke  the search on a few maps, then two training steps of D (CPU AdamW) and a few
#                      greedy answers: memory, timing and the end-to-end path. No benchmark.
#   SPARA_STAGE=all    the search, then each run in SPARA_RUNS (default "S42 D42 S43 D43") in order,
#                      each training and evaluation, in one job.
#   SPARA_STAGE=train  each run in SPARA_RUNS on an existing search, SPARA_SEARCH_RUN. SPARA_ACCEPT_SEARCH=1
#                      accepts a search that failed only its replay bar (spec amendment of 2026-11-19).
# Training runs use CPU AdamW (48 GB slices). Final weights only (15 GB per run).
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SPARA_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1
PYTHON="${SPARA_PYTHON:-$ROOT/.venv/bin/python}"
DATA="$ROOT/data/vsp"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
BASIS="${SPARA_NOISE_BASIS:?set SPARA_NOISE_BASIS to the confirmed latent-diagnostics basis}"
STAGE="${SPARA_STAGE:?set SPARA_STAGE to smoke, all or train}"
SETTINGS="${SPARA_SETTINGS:-configs/latent_distillation.json}"
RUNS="${SPARA_RUNS:-S42 D42 S43 D43}"
for split in detour-pool detour-heldout detour-heldout-78; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON=(--settings "$SETTINGS" --config configs/livr.json --data-root "$DATA" --checkpoint "$CHECKPOINT"
        --noise-basis "$BASIS")

train() {  # $1: search run; $2: run name such as D42
  local arm="${2:0:1}" seed="${2:1}"
  "$PYTHON" -m spara.cli latent-distillation --part train --arm "$arm" --seed "$seed" --search-run "$1" \
    --offload-optimizer "${COMMON[@]}" --output "$ROOT/results/ld-$2-$SLURM_JOB_ID" ${SMOKE:+--smoke} \
    ${SPARA_ACCEPT_SEARCH:+--accept-search}
}

case "$STAGE" in
  smoke)
    SMOKE=1
    SEARCH="$ROOT/results/ld-search-smoke-$SLURM_JOB_ID"
    "$PYTHON" -m spara.cli latent-distillation --part search --smoke "${COMMON[@]}" --output "$SEARCH"
    train "$SEARCH" D42
    ;;
  all)
    SEARCH="$ROOT/results/ld-search-$SLURM_JOB_ID"
    "$PYTHON" -m spara.cli latent-distillation --part search "${COMMON[@]}" --output "$SEARCH"
    for run in $RUNS; do train "$SEARCH" "$run"; done
    ;;
  train)
    SEARCH="${SPARA_SEARCH_RUN:?set SPARA_SEARCH_RUN to the search output directory}"
    for run in $RUNS; do train "$SEARCH" "$run"; done
    ;;
  *)
    echo "SPARA_STAGE must be smoke, all or train" >&2
    exit 1
    ;;
esac
