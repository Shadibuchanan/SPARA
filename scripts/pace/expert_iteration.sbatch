#!/bin/bash
# Expert iteration, round 1 (docs/EXPERT_ITERATION_SPEC.md).
#   SPARA_STAGE=search: final-verdict-only search with LIVR on the first 500 detour-pool maps,
#     after its validity checks and a rerun of the 97 detour maps.
#   SPARA_STAGE=train: fine-tune on the search's routes (SPARA_LABELS=search) or on the same
#     maps' shortest-route labels (SPARA_LABELS=shortest); SPARA_ROUTES is the search run's
#     routes.jsonl.
# Submit only within the agreed GPU budget; this script does not submit itself.
# Requires SPARA_LIVR_CHECKPOINT.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
STAGE="${SPARA_STAGE:?set SPARA_STAGE to search or train}"
DATA="$ROOT/data/vsp"
for split in detour-heldout detour-pool probe; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
mkdir -p "$ROOT/results"
case "$STAGE" in
  search)
    # Recorded probe points and the JEPA run's searches belong to the LIVR rerun 6253300. For
    # another checkpoint set both to "none": batched reads are then checked against single passes.
    POINTS="${SPARA_PROBE_POINTS:-$ROOT/results/detour-probe-6937009/points.jsonl}"
    REFERENCE="${SPARA_REFERENCE:-$ROOT/results/jepa-judge-6951675/searches-verdict_only.jsonl}"
    INPUTS=()
    if [ "$POINTS" != none ]; then
      [ -f "$POINTS" ] || { echo "Missing input: $POINTS" >&2; exit 1; }
      INPUTS+=(--probe-points "$POINTS")
    fi
    if [ "$REFERENCE" != none ]; then
      [ -f "$REFERENCE" ] || { echo "Missing input: $REFERENCE" >&2; exit 1; }
      INPUTS+=(--reference "$REFERENCE")
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PYTHON" -m spara.cli expert-iteration --stage search --settings configs/expert_iteration.json \
      --config configs/livr.json --data-root "$DATA" --checkpoint "$CHECKPOINT" ${INPUTS[@]+"${INPUTS[@]}"} \
      --output "$ROOT/results/expert-routes-$SLURM_JOB_ID"
    ;;
  train)
    LABELS="${SPARA_LABELS:?set SPARA_LABELS to search or shortest}"
    ROUTES="${SPARA_ROUTES:?set SPARA_ROUTES to routes.jsonl of the search run}"
    [ -f "$ROUTES" ] || { echo "Missing routes: $ROUTES" >&2; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PYTHON" -m spara.cli expert-iteration --stage train --labels "$LABELS" --routes "$ROUTES" \
      --settings configs/expert_iteration.json --config configs/livr.json --data-root "$DATA" \
      --checkpoint "$CHECKPOINT" --output "$ROOT/results/expert-train-$LABELS-$SLURM_JOB_ID"
    ;;
  *)
    echo "SPARA_STAGE must be search or train" >&2
    exit 1
    ;;
esac
