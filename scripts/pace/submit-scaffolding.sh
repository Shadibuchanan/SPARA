#!/bin/bash
# Run ON PACE: submit-scaffolding.sh pilot|baseline. Agree the GPU budget first and
# inspect sacct before retries; allocated time counts cumulatively across attempts.
set -euo pipefail
KIND="${1:?Usage: submit-scaffolding.sh pilot|baseline}"
[[ "$KIND" == pilot || "$KIND" == baseline ]]
: "${SPARA_ACCOUNT:?Set your Slurm account}"
: "${SPARA_QOS:?Set your Slurm QoS}"
: "${SPARA_PARTITION:?Select a compatible GPU partition}"
: "${SPARA_GPU:?Set a one-GPU GRES, e.g. gpu:h200:1}"
: "${SPARA_MINUTES:?Set the approved allocated minutes for this job}"
[[ "$SPARA_MINUTES" =~ ^[0-9]+$ ]] && (( SPARA_MINUTES >= 1 ))
[[ "$SPARA_GPU" =~ ^gpu:[a-zA-Z0-9_]+:1$ ]]
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
mkdir -p "$ROOT/logs"
args=(--parsable --job-name="spara-scaffolding-$KIND" --account="$SPARA_ACCOUNT" --qos="$SPARA_QOS"
      --partition="$SPARA_PARTITION" --gres="$SPARA_GPU" --nodes=1 --ntasks=1
      --cpus-per-task=4 --mem=96G --time="$SPARA_MINUTES:00"
      --output="$ROOT/logs/scaffolding-$KIND-%j.log" --error="$ROOT/logs/scaffolding-$KIND-%j.log")
if [[ -n "${SPARA_CONSTRAINT:-}" ]]; then args+=(--constraint="$SPARA_CONSTRAINT"); fi
sbatch "${args[@]}" "$ROOT/code/scripts/pace/scaffolding-$KIND.sbatch"
