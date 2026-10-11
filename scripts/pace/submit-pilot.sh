#!/bin/bash
# Run ON PACE. Inspect sacct before retries: the one-hour ceiling is cumulative.
set -euo pipefail
: "${SPARA_ACCOUNT:?Set your Slurm account}"
: "${SPARA_QOS:?Set your Slurm QoS}"
: "${SPARA_PARTITION:?Select a compatible GPU partition}"
: "${SPARA_GPU:?Set a one-GPU GRES, e.g. gpu:h200:1}"
: "${SPARA_PILOT_MINUTES:?Set remaining approved allocated minutes, at most 60}"
[[ "$SPARA_PILOT_MINUTES" =~ ^[0-9]+$ ]] && (( SPARA_PILOT_MINUTES >= 1 && SPARA_PILOT_MINUTES <= 60 ))
[[ "$SPARA_GPU" =~ ^gpu:[a-zA-Z0-9_]+:1$ ]]
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
mkdir -p "$ROOT/logs"
args=(--parsable --job-name=spara-livr-pilot --account="$SPARA_ACCOUNT" --qos="$SPARA_QOS"
      --partition="$SPARA_PARTITION" --gres="$SPARA_GPU" --nodes=1 --ntasks=1
      --cpus-per-task=4 --mem=96G --time="$SPARA_PILOT_MINUTES:00"
      --output="$ROOT/logs/pilot-%j.log" --error="$ROOT/logs/pilot-%j.log")
if [[ -n "${SPARA_CONSTRAINT:-}" ]]; then args+=(--constraint="$SPARA_CONSTRAINT"); fi
sbatch "${args[@]}" "$ROOT/code/scripts/pace/pilot.sbatch"
