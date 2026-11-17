#!/bin/bash
#SBATCH --job-name=spara-rebal
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:30:00
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

RUN="$ROOT/results/rebalanced-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir -p "$RUN"

TRAIN_DATA="$ROOT/code/data/rebalanced/train_ratio_10pct.jsonl"
PYTHON="$ROOT/.venv/bin/python"

echo "=== Training Stage 1 on Rebalanced Mix (10% detours) ==="
"$PYTHON" -m spara.cli train --stage 1 --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --base-model "$ROOT/models/Qwen2.5-VL-3B-Instruct" \
  --train-records "$TRAIN_DATA" \
  --output "$RUN/stage1"

echo "=== Training Stage 2 on Rebalanced Mix ==="
"$PYTHON" -m spara.cli train --stage 2 --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/stage1/final" \
  --train-records "$TRAIN_DATA" \
  --output "$RUN/stage2"

echo "=== Evaluating Rebalanced Model on 400-Map Reference Benchmark ==="
"$PYTHON" -m spara.cli evaluate --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/stage2/final" \
  --output "$RUN/evaluation"

echo "=== Evaluating Rebalanced Model with Hole-Veto Greedy Decode ==="
"$PYTHON" scripts/hole_veto_decode.py \
  --checkpoint "$RUN/stage2/final" \
  --data-root "$ROOT/data/vsp" \
  --output "$RUN/hole_veto_evaluation" \
  --split test

echo "Rebalanced experiment completed successfully!"
