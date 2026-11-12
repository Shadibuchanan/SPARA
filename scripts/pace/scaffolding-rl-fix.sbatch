#!/bin/bash
# Latent-sensitivity probe and Scaffolding RL rerun from a completed Stage 1 run
# (SPARA_SOURCE_RUN, holding stage1-predict/ and evaluation-stage1/). This script
# does not submit itself; pass resources and the log path to sbatch.
#   SPARA_STEP=probe  how often each sigma's adjustment changes a greedy route
#                     -> $ROOT/results/scaffolding-latent-probe-$SLURM_JOB_ID
#   SPARA_STEP=rl     rl,eval2 with SPARA_CONFIG in a new run directory whose Stage 1
#                     entries link to the source run, which stays unchanged
#                     -> ${SPARA_RUN:-$ROOT/results/scaffolding-rlfix-$SLURM_JOB_ID}
#   SPARA_STEP=eval   evaluate the source run's Stage 2 checkpoint with SPARA_LATENT
#                     (sample, mean or none) -> $SOURCE/evaluation-stage2-$SPARA_LATENT
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
SOURCE="${SPARA_SOURCE_RUN:?set SPARA_SOURCE_RUN to a run with a completed Stage 1}"
[[ -f "$SOURCE/stage1-predict/final/complete.json" ]] || { echo "error: no Stage 1 checkpoint in $SOURCE" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "${SPARA_STEP:?set SPARA_STEP to probe or rl}" in
  probe)
    "$PYTHON" -m spara.scaffolding.cli probe --config "${SPARA_CONFIG:-configs/scaffolding.json}" \
      --data-root "$ROOT/data/vsp" --checkpoint "$SOURCE/stage1-predict/final" \
      --output "$ROOT/results/scaffolding-latent-probe-$SLURM_JOB_ID" ;;
  rl)
    RUN="${SPARA_RUN:-$ROOT/results/scaffolding-rlfix-$SLURM_JOB_ID}"
    mkdir -p "$RUN"
    for phase in stage1-predict evaluation-stage1; do
      [[ -e "$RUN/$phase" ]] || ln -s "$SOURCE/$phase" "$RUN/$phase"
    done
    bash scripts/run_scaffolding.sh --run "$RUN" --config "${SPARA_CONFIG:?set SPARA_CONFIG for RL}" \
      --phases rl,eval2 --python "$PYTHON" --data "$ROOT/data/vsp" \
      --model "$ROOT/models/Qwen2.5-VL-3B-Instruct" ;;
  eval)
    "$PYTHON" -m spara.scaffolding.cli evaluate --config "${SPARA_CONFIG:-configs/scaffolding.json}" \
      --data-root "$ROOT/data/vsp" --checkpoint "$SOURCE/stage2-rl/final" \
      --evaluation-latent "${SPARA_LATENT:?set SPARA_LATENT to sample, mean or none}" \
      --output "$SOURCE/evaluation-stage2-$SPARA_LATENT" ;;
  *) echo "error: SPARA_STEP must be probe, rl or eval" >&2; exit 1 ;;
esac
