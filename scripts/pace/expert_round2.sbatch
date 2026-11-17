#!/bin/bash
# Expert iteration, round 2 (docs/EXPERT_ITERATION_ROUND2_SPEC.md). Resources are given at
# submission: a CPU node for batching-check, one H200 for search and train.
#   SPARA_STAGE=batching-check: the precondition, on CPU. Batched against single policy reads
#     of E, in full precision and under bfloat16 autocast.
#   SPARA_STAGE=search: E's evaluation on the new sets, then E's search of round 1's unsolved
#     pool maps and the new pool; then L's greedy answers. Needs SPARA_LIVR_CHECKPOINT.
#   SPARA_STAGE=train: SPARA_ARM=e2 or eb2, from E, on the search's routes; SPARA_SEARCH_RUN
#     is the search job's output directory. SPARA_ARM=e2r with
#     SPARA_SETTINGS=configs/expert_round2_rebalanced.json runs the rebalanced arm (needs the
#     fresh/nodetour-pool set; the spec's 2026-11-17 amendment).
#   SPARA_BATCH and SPARA_READ_PRECISION (bf16 or tf32) override the search's policy reads;
#     SPARA_BATCH=1 SPARA_READ_PRECISION=bf16 gives round 1's reads.
# Submit only within the agreed GPU budget; this script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
STAGE="${SPARA_STAGE:?set SPARA_STAGE to batching-check, search or train}"
E="${SPARA_E_CHECKPOINT:-$ROOT/results/expert-train-search-6961875/stage2/final}"
ROUND1_ROUTES="${SPARA_ROUND1_ROUTES:-$ROOT/results/expert-routes-6961874/routes.jsonl}"
ROUND1_GREEDY="${SPARA_ROUND1_GREEDY:-$ROOT/results/expert-train-search-6961875/greedy.jsonl}"
DATA="$ROOT/data/vsp"
SETTINGS="${SPARA_SETTINGS:-configs/expert_round2.json}"
[ -f "$E/complete.json" ] || { echo "Missing E's checkpoint: $E" >&2; exit 1; }
mkdir -p "$ROOT/results"
case "$STAGE" in
  batching-check)
    [ -f "$DATA/fresh/jepa-train/maps.jsonl" ] || { echo "Missing fresh set: jepa-train" >&2; exit 1; }
    "$PYTHON" -m spara.cli expert-round2 --stage batching-check --settings "$SETTINGS" --config configs/livr.json \
      --data-root "$DATA" --checkpoint "$E" --output "$ROOT/results/round2-batching-$SLURM_JOB_ID"
    ;;
  search|train)
    for split in detour-heldout-78 detour-pool-78 detour-heldout detour-pool probe jepa-train; do
      [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
    done
    [ -f "$ROUND1_ROUTES" ] || { echo "Missing round 1 routes: $ROUND1_ROUTES" >&2; exit 1; }
    BATCH=()
    [ -n "${SPARA_BATCH:-}" ] && BATCH=(--batch "$SPARA_BATCH")
    [ -n "${SPARA_READ_PRECISION:-}" ] && BATCH+=(--read-precision "$SPARA_READ_PRECISION")
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$STAGE" = search ]; then
      LIVR="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
      [ -f "$ROUND1_GREEDY" ] || { echo "Missing round 1 greedy answers: $ROUND1_GREEDY" >&2; exit 1; }
      "$PYTHON" -m spara.cli expert-round2 --stage search --settings "$SETTINGS" --config configs/livr.json \
        --data-root "$DATA" --checkpoint "$E" --livr-checkpoint "$LIVR" --round1-routes "$ROUND1_ROUTES" \
        --round1-greedy "$ROUND1_GREEDY" ${BATCH[@]+"${BATCH[@]}"} --output "$ROOT/results/round2-search-$SLURM_JOB_ID"
    else
      ARM="${SPARA_ARM:?set SPARA_ARM to e2, eb2 or e2r}"
      SEARCH_RUN="${SPARA_SEARCH_RUN:?set SPARA_SEARCH_RUN to the output directory of the search job}"
      [ -f "$SEARCH_RUN/routes.jsonl" ] || { echo "Missing routes: $SEARCH_RUN/routes.jsonl" >&2; exit 1; }
      if [ "$ARM" = e2r ]; then
        [ -f "$DATA/fresh/nodetour-pool/maps.jsonl" ] || { echo "Missing fresh set: nodetour-pool" >&2; exit 1; }
        [ "$SETTINGS" = configs/expert_round2_rebalanced.json ] \
          || { echo "E2R needs SPARA_SETTINGS=configs/expert_round2_rebalanced.json" >&2; exit 1; }
      fi
      "$PYTHON" -m spara.cli expert-round2 --stage train --arm "$ARM" --settings "$SETTINGS" \
        --config configs/livr.json --data-root "$DATA" --checkpoint "$E" --round1-routes "$ROUND1_ROUTES" \
        --search-run "$SEARCH_RUN" --output "$ROOT/results/round2-$ARM-$SLURM_JOB_ID"
    fi
    ;;
  *)
    echo "SPARA_STAGE must be batching-check, search or train" >&2
    exit 1
    ;;
esac
