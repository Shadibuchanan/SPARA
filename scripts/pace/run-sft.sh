#!/bin/bash
# Vanilla SFT control inside an existing GPU allocation: arm A, then arm C.
# Usage: scripts/pace/run-sft.sh RUN_ID [a] [c]
set -euo pipefail
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/sft-run}"
RUN="$ROOT/results/${1:?run id}"; shift
ARMS="${*:-a c}"
PY="$ROOT/.venv/bin/python"; DATA="$ROOT/data/vsp"; MODEL="$ROOT/models/Qwen2.5-VL-3B-Instruct"
cd "$ROOT/code"
mkdir -p "$RUN"
step() {  # name, command...; appends wall-clock seconds on the GPU to budget.jsonl
  local name=$1 start; shift; start=$(date +%s)
  "$@" > "$RUN/$name.log" 2>&1
  echo "{\"phase\": \"$name\", \"gpu_seconds\": $(( $(date +%s) - start )), \"hardware\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader)\"}" >> "$RUN/budget.jsonl"
}
# Final checkpoints are never resumed; keep weights only (~15 GB instead of ~40 GB).
weights_only() { rm -f "$1/final/training_state.pt"; }
train() { step "$1" "$PY" -m spara.cli train --data-root "$DATA" --base-model "$MODEL" --output "$RUN/$1" "${@:2}"; weights_only "$RUN/$1"; }
evaluate() { step "$1" "$PY" -m spara.cli evaluate --data-root "$DATA" --base-model "$MODEL" --output "$RUN/$1" "${@:2}"; }
for arm in $ARMS; do
  case $arm in
    a) train a-train --config configs/sft.json --stage 2
       evaluate a-evaluation --config configs/sft.json --checkpoint "$RUN/a-train/final" ;;
    c) train c-segment1 --config configs/sft-staged.json --stage 1
       train c-segment2 --config configs/sft-staged.json --stage 2 --checkpoint "$RUN/c-segment1/final"
       evaluate c-evaluation --config configs/sft-staged.json --checkpoint "$RUN/c-segment2/final" ;;
  esac
done
