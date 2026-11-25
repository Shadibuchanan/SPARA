#!/bin/bash
# Expert iteration, round 2 (docs/EXPERT_ITERATION_ROUND2_SPEC.md). Resources are given at
# submission: a CPU node for batching-check, one H200 for search and train.
#   SPARA_STAGE=batching-check: the precondition, on CPU. Batched against single policy reads
#     of E, in full precision and under bfloat16 autocast.
#   SPARA_STAGE=search: E's evaluation on the new sets, then E's search of round 1's unsolved
#     pool maps and the new pool; then L's greedy answers. Needs SPARA_LIVR_CHECKPOINT.
#   SPARA_STAGE=train: SPARA_ARM=e2, eb2, o2 or t2 (the perception ceiling), from E, on the
#     search's routes; SPARA_SEARCH_RUN is the search job's output directory.
#     SPARA_SAVE_WEIGHTS=1 saves the arm's final weights.
# Round 3 (docs/EXPERT_ITERATION_ROUND3_SPEC.md): SPARA_SETTINGS=configs/expert_round3.json,
#   SPARA_ROUND=3 and SPARA_E_CHECKPOINT set to B. The search also needs SPARA_FRONTIER_ROUTES
#   (round 2's routes.jsonl), SPARA_ROUND1_GREEDY (B's greedy.jsonl) and SPARA_LIVR_GREEDY.
#   Train needs SPARA_ARM=eb3 and SPARA_SEARCH_RUN="<round 2's search> <round 3's search>".
# The hole check (docs/HOLE_CHECK_SPEC.md): as EB3, with SPARA_ARM=s3 and
#   SPARA_SETTINGS=configs/hole_check.json.
# The hole check on the exact map (docs/HOLE_CHECK_TEXT_SPEC.md): SPARA_ARM=s3t or o3, with
#   SPARA_SETTINGS=configs/hole_check_text.json. No weights are saved.
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
STAGE="${SPARA_STAGE:?set SPARA_STAGE to livr-greedy, batching-check, search or train}"
E="${SPARA_E_CHECKPOINT:-$ROOT/results/expert-train-search-6961875/stage2/final}"
ROUND1_ROUTES="${SPARA_ROUND1_ROUTES:-$ROOT/results/expert-routes-6961874/routes.jsonl}"
ROUND1_GREEDY="${SPARA_ROUND1_GREEDY:-$ROOT/results/expert-train-search-6961875/greedy.jsonl}"
DATA="$ROOT/data/vsp"
SETTINGS="${SPARA_SETTINGS:-configs/expert_round2.json}"
FRONTIER_ROUTES="${SPARA_FRONTIER_ROUTES:-$ROUND1_ROUTES}"
NAME="round${SPARA_ROUND:-2}"
[ "$STAGE" = livr-greedy ] || [ -f "$E/complete.json" ] || { echo "Missing E's checkpoint: $E" >&2; exit 1; }
mkdir -p "$ROOT/results"
case "$STAGE" in
  livr-greedy)
    LIVR="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of LIVR}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PYTHON" -m spara.cli expert-round2 --stage livr-greedy --settings "$SETTINGS" --config configs/livr.json \
      --data-root "$DATA" --checkpoint "$LIVR" --output "$ROOT/results/round2-livr-$SLURM_JOB_ID"
    ;;
  batching-check)
    [ -f "$DATA/fresh/jepa-train/maps.jsonl" ] || { echo "Missing fresh set: jepa-train" >&2; exit 1; }
    "$PYTHON" -m spara.cli expert-round2 --stage batching-check --settings "$SETTINGS" --config configs/livr.json \
      --data-root "$DATA" --checkpoint "$E" --output "$ROOT/results/round2-batching-$SLURM_JOB_ID"
    ;;
  search|train)
    for split in detour-heldout-78 detour-pool-78 detour-pool-78b detour-heldout detour-pool probe jepa-train; do
      [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
    done
    [ -f "$ROUND1_ROUTES" ] || { echo "Missing round 1 routes: $ROUND1_ROUTES" >&2; exit 1; }
    BATCH=()
    [ -n "${SPARA_BATCH:-}" ] && BATCH=(--batch "$SPARA_BATCH")
    [ -n "${SPARA_READ_PRECISION:-}" ] && BATCH+=(--read-precision "$SPARA_READ_PRECISION")
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    if [ "$STAGE" = search ]; then
      # L's answers: SPARA_LIVR_GREEDY (a livr-greedy job's output) or SPARA_LIVR_CHECKPOINT.
      if [ -n "${SPARA_LIVR_GREEDY:-}" ]; then
        L=(--livr-greedy "$SPARA_LIVR_GREEDY")
      else
        L=(--livr-checkpoint "${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT or SPARA_LIVR_GREEDY}")
      fi
      [ -f "$ROUND1_GREEDY" ] || { echo "Missing round 1 greedy answers: $ROUND1_GREEDY" >&2; exit 1; }
      [ -f "$FRONTIER_ROUTES" ] || { echo "Missing frontier routes: $FRONTIER_ROUTES" >&2; exit 1; }
      "$PYTHON" -m spara.cli expert-round2 --stage search --settings "$SETTINGS" --config configs/livr.json \
        --data-root "$DATA" --checkpoint "$E" "${L[@]}" --frontier-routes "$FRONTIER_ROUTES" \
        --reference-greedy "$ROUND1_GREEDY" ${BATCH[@]+"${BATCH[@]}"} --output "$ROOT/results/$NAME-search-$SLURM_JOB_ID"
    else
      ARM="${SPARA_ARM:?set SPARA_ARM to e2, eb2, o2, t2, eb3, s3, o3 or s3t}"
      read -ra SEARCH_RUNS <<< "${SPARA_SEARCH_RUN:?set SPARA_SEARCH_RUN to the output directories of the search jobs}"
      for run in "${SEARCH_RUNS[@]}"; do
        [ -f "$run/routes.jsonl" ] || { echo "Missing routes: $run/routes.jsonl" >&2; exit 1; }
      done
      SAVE=()
      [ "${SPARA_SAVE_WEIGHTS:-0}" = 1 ] && SAVE=(--save-weights)
      "$PYTHON" -m spara.cli expert-round2 --stage train --arm "$ARM" --settings "$SETTINGS" \
        --config configs/livr.json --data-root "$DATA" --checkpoint "$E" --round1-routes "$ROUND1_ROUTES" \
        --search-run "${SEARCH_RUNS[@]}" ${SAVE[@]+"${SAVE[@]}"} --output "$ROOT/results/$NAME-$ARM-$SLURM_JOB_ID"
    fi
    ;;
  *)
    echo "SPARA_STAGE must be livr-greedy, batching-check, search or train" >&2
    exit 1
    ;;
esac
