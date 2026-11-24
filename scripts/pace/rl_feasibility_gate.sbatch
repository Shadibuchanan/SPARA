#!/bin/bash
#SBATCH --job-name=spara-rlgate
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log

set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
source "$ROOT/.venv/bin/activate"

export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

CHECKPOINT="${1:-$ROOT/results/rebalanced-7109628/stage2/final}"
OUTPUT="$ROOT/results/rl_gate-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir -p "$OUTPUT"

PYTHON="$ROOT/.venv/bin/python"

echo "=== Running Phase 2B: Gated RL Feasibility Check ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

"$PYTHON" scripts/rl_feasibility_gate.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$ROOT/data/vsp" \
  --config configs/livr.json \
  --output "$OUTPUT" \
  --samples 8 \
  --split test \
  --limit 100

echo "RL Feasibility Gate completed successfully!"
