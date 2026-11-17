#!/bin/bash
#SBATCH --job-name=spara-pipe
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

# ==============================================================================
# SPARA Unified End-to-End Pipeline
# Runs consecutive stages inside ONE Slurm allocation to avoid repeated queue delays:
#   Stage 1: Gated RL Feasibility Check (sample 8 rollouts per map on 100 test maps)
#   Stage 2: Seed Rerun & Benchmark Evaluation
#   Stage 3: Refused Routes Extraction & DPO / Unlikelihood Fine-Tuning
# ==============================================================================

set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
source "$ROOT/.venv/bin/activate"

export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

RUN="$ROOT/results/pipeline-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir -p "$RUN"

PYTHON="$ROOT/.venv/bin/python"
BASE_CHECKPOINT="${SPARA_CHECKPOINT:-$ROOT/results/rebalanced-7109628/stage2/final}"
TRAIN_DATA="$ROOT/code/data/rebalanced/train_ratio_10pct.jsonl"

echo "=== Pipeline Step 1: RL Feasibility Gate Diagnostic ==="
"$PYTHON" scripts/rl_feasibility_gate.py \
  --checkpoint "$BASE_CHECKPOINT" \
  --data-root "$ROOT/data/vsp" \
  --config configs/livr.json \
  --output "$RUN/rl_gate" \
  --samples 8 \
  --split test \
  --limit 100

echo "=== Pipeline Step 2: Seed 43 Rerun & Benchmark Evaluation ==="
"$PYTHON" -m spara.cli train --stage 1 --config configs/livr_seed43.json \
  --data-root "$ROOT/data/vsp" --base-model "$ROOT/models/Qwen2.5-VL-3B-Instruct" \
  --train-records "$TRAIN_DATA" \
  --output "$RUN/seed43_stage1"

"$PYTHON" -m spara.cli train --stage 2 --config configs/livr_seed43.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/seed43_stage1/final" \
  --train-records "$TRAIN_DATA" \
  --output "$RUN/seed43_stage2"

"$PYTHON" -m spara.cli evaluate --config configs/livr_seed43.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/seed43_stage2/final" \
  --output "$RUN/seed43_evaluation"

echo "=== Pipeline Step 3: Filter Refused Routes from Rollouts ==="
"$PYTHON" scripts/filter_refused_routes.py \
  --input "$RUN/rl_gate/rollouts.jsonl" \
  --output "$RUN/dpo_pairs.jsonl"

echo "All pipeline stages completed successfully inside single allocation!"
