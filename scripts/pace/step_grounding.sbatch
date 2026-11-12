#!/bin/bash
# Step grounding (docs/STEP_GROUNDING_SPEC.md): Stage 2 retrained from the LIVR rerun's final
# Stage 1 weights with the grounding loss, then the reference benchmark, the state probe
# without headroom (test and fresh/probe 5x5 to 8x8), and the pre-declared comparisons with
# LIVR. Submit only within the agreed GPU budget; this script does not submit itself.
# Requires SPARA_LIVR_STAGE1, e.g. $ROOT/results/baseline-6253300/stage1/final.
# LIVR's reference outputs default to the PACE copies below; override with SPARA_LIVR_*.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
STAGE1="${SPARA_LIVR_STAGE1:?set SPARA_LIVR_STAGE1 to the final Stage 1 checkpoint of the LIVR rerun}"
DATA="$ROOT/data/vsp"
LIVR_BENCHMARK="${SPARA_LIVR_BENCHMARK:-$ROOT/code/docs/experiments/livr-6253300/evaluation-outcomes.jsonl}"
LIVR_STEPS="${SPARA_LIVR_STEPS:-$ROOT/code/docs/experiments/livr-6253300/steps.jsonl}"
LIVR_PROBE="${SPARA_LIVR_PROBE:-$ROOT/results/planning-probe-6887458/probe}"
LIVR_FRESH="${SPARA_LIVR_FRESH:-$ROOT/results/explore-probe-6884608/probe/sweep.jsonl}"
for file in "$LIVR_BENCHMARK" "$LIVR_STEPS" "$LIVR_PROBE/routes.jsonl" "$LIVR_PROBE/summary.json" "$LIVR_FRESH"; do
  [ -f "$file" ] || { echo "Missing LIVR reference: $file" >&2; exit 1; }
done
RUN="$ROOT/results/step-grounding-$SLURM_JOB_ID"
mkdir -p "$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
"$PYTHON" -m spara.cli train --stage 2 --config configs/livr.json --grounding configs/step_grounding.json \
  --data-root "$DATA" --checkpoint "$STAGE1" --output "$RUN/stage2" --final-weights-only
"$PYTHON" -m spara.cli evaluate --config configs/livr.json --data-root "$DATA" \
  --checkpoint "$RUN/stage2/final" --output "$RUN/evaluation"
"$PYTHON" -m spara.cli planning-probe --config configs/livr.json --rl-config configs/latent_rl.json \
  --data-root "$DATA" --checkpoint "$RUN/stage2/final" --radii none --fresh-split fresh/probe \
  --fresh-sizes 5,6,7,8 --output "$RUN/probe"
"$PYTHON" scripts/analyze_step_grounding.py "$RUN" --data-root "$DATA" \
  --livr-benchmark "$LIVR_BENCHMARK" --livr-steps "$LIVR_STEPS" --livr-routes "$LIVR_PROBE/routes.jsonl" \
  --livr-probe "$LIVR_PROBE/summary.json" --livr-fresh "$LIVR_FRESH"
