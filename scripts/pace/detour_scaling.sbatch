#!/bin/bash
# Detour scaling check (docs/DETOUR_SCALING_SPEC.md), one arm per job: SPARA_ARM is an arm of the
# recipe SPARA_RECIPE (default configs/detour_scaling.json: livr, n98, n500 or n2000). Fine-tunes
# the starting Stage 2 weights (except livr), then runs the greedy evaluation and, for fine-tuned
# arms, the reference benchmark. Submit only within the agreed GPU budget; this script does not
# submit itself. Requires SPARA_LIVR_CHECKPOINT, the starting weights.
#   SPARA_SEARCH=START:STOP searches those pool maps with the starting weights instead of training,
#     and writes the routes found to results/detour-search-$ARM-START-STOP-$JOB/routes.jsonl.
#   SPARA_ROUTES="<routes.jsonl> ..." trains the arm on the routes found instead of the labels.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the starting Stage 2 checkpoint}"
ARM="${SPARA_ARM:?set SPARA_ARM to an arm of the recipe}"
RECIPE="${SPARA_RECIPE:-configs/detour_scaling.json}"
DATA="$ROOT/data/vsp"
for split in $("$PYTHON" -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["heldout_split"], r["pool_split"], *r["other_fresh"])' "$RECIPE"); do
  [ -f "$DATA/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
RUN="$ROOT/results/detour-scaling-$ARM-$SLURM_JOB_ID"
MODE=()
if [ -n "${SPARA_SEARCH:-}" ]; then
  RUN="$ROOT/results/detour-search-$ARM-${SPARA_SEARCH/:/-}-$SLURM_JOB_ID"
  MODE=(--search "$SPARA_SEARCH")
elif [ -n "${SPARA_ROUTES:-}" ]; then
  read -r -a FILES <<< "$SPARA_ROUTES"
  for file in "${FILES[@]}"; do [ -f "$file" ] || { echo "Missing routes: $file" >&2; exit 1; }; done
  MODE=(--routes "${FILES[@]}")
fi
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli detour-scaling --arm "$ARM" --recipe "$RECIPE" \
  --config configs/livr.json --data-root "$DATA" --checkpoint "$CHECKPOINT" --output "$RUN" "${MODE[@]}"
