#!/bin/bash
# Written reasoning without labels, the probe (docs/DETOUR_SCALING_SPEC.md): sampled answers per map,
# each scored by the reference evaluator. This script does not submit itself.
#   SPARA_SPLIT (required) and SPARA_MAPS=START:STOP or SPARA_PER_SIZE=N select the maps.
#   SPARA_CHECKPOINT: trained weights (with SPARA_RECIPE for a later fine-tune); without it, the
#   released model downloaded by fetch-model to $ROOT/models/qwen2.5-vl-3b-instruct, with no latent tokens.
#   SPARA_DIRECT=1 leaves out the instruction to reason.
#   SPARA_SAMPLE_ARGS passes more probe options, e.g. "--samples 64 --temperature 1.0".
#   -> ${SPARA_OUTPUT:-$ROOT/results/reasoning-probe-$SLURM_JOB_ID}
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
SPLIT="${SPARA_SPLIT:?set SPARA_SPLIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli reasoning-probe --config configs/livr.json --data-root "$ROOT/data/vsp" \
  --base-model "$ROOT/models/qwen2.5-vl-3b-instruct" --split "$SPLIT" \
  --output "${SPARA_OUTPUT:-$ROOT/results/reasoning-probe-$SLURM_JOB_ID}" \
  ${SPARA_MAPS:+--maps "$SPARA_MAPS"} ${SPARA_PER_SIZE:+--per-size "$SPARA_PER_SIZE"} \
  ${SPARA_CHECKPOINT:+--checkpoint "$SPARA_CHECKPOINT"} ${SPARA_RECIPE:+--recipe "$SPARA_RECIPE"} \
  ${SPARA_DIRECT:+--direct} ${SPARA_SAMPLE_ARGS:-}
