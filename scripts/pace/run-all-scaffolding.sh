#!/bin/bash
# Run ON THE PACE LOGIN NODE. One entry point for the Scaffolding Minds baseline:
# prepare the $ROOT/code layout, run setup, run the CPU tests, submit the pilot,
# and submit the full baseline so it starts automatically once the pilot succeeds.
#
# Both run inside ONE allocation, back to back, so the baseline never waits in the
# queue a second time and cannot lose the node. The job stops before the baseline if
# the pilot exits nonzero, but it does NOT wait for anyone to read the pilot's
# summary.json; the baseline specification asks for that review first (GPU-hour
# estimate, latent_loss_normalized, and the latent-vs-text RL gradient split; see its
# "Open items" section). Use --no-baseline to stop after the pilot.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/pace/run-all-scaffolding.sh [options]

Resources (environment variables):
  GPU                One-GPU GRES, >= 80 GB               (default: gpu:h200:1)
  CONSTRAINT         Optional --constraint value          (default: none)
  PILOT_HOURS        Hours to budget for the pilot        (default: 1)
  BASELINE_HOURS     Hours to budget for the baseline     (default: 15)
  HOURS              Total walltime, overriding their sum (default: their sum)
  SPARA_REMOTE_ROOT        Run root holding code/ data/ models/ (default: $HOME/scratch/spara)
  REPO               Repository checkout to link as code/ (default: this checkout)

Options:
  --no-baseline   Stop the job after the pilot; do not run the full baseline
  --continue DIR  Continue an earlier baseline run directory; completed phases skip
  --skip-setup    Assume the venv, data and model already exist under $ROOT
  --skip-tests    Do not run the CPU test suite before submitting
  --dry-run       Print what would happen; create nothing and submit nothing
  --config F      Read persistent settings from F instead of the usual locations
  -c N            CPU cores per task, overriding CPUS
  --mem SIZE      Host memory for the job, e.g. 96G, overriding MEM
  -h, --help      Show this message
EOF
}

die() { echo "error: $*" >&2; exit 1; }
note() { echo "[$(date '+%H:%M:%S')] $*"; }

BASELINE=1; SKIP_SETUP=0; SKIP_TESTS=0; DRY=0; CONTINUE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-baseline) BASELINE=0; shift ;;
    --continue) [[ $# -ge 2 ]] || die "--continue needs a run directory"; CONTINUE="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a path"; CONFIG_FILE="$2"; shift 2 ;;
    -c) [[ $# -ge 2 ]] || die "-c needs a core count"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "--mem needs a size, e.g. 96G"; MEM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

# Persistent settings: scripts/pace/pace.conf, or --config, or ~/.spara.conf.
# Variables already set in the environment win over the file; $USER is never read
# from it, since the job always runs as the submitting user.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS=(GPU CPUS MEM CONSTRAINT HOURS PILOT_HOURS BASELINE_HOURS ROOT REPO)
ENV_FILE="${CONFIG_FILE:-}"
for candidate in "$HERE/pace.conf" "$HOME/.spara.conf"; do
  [[ -n "$ENV_FILE" ]] && break
  [[ -f "$candidate" ]] && ENV_FILE="$candidate"
done
if [[ -n "$ENV_FILE" ]]; then
  [[ -f "$ENV_FILE" ]] || die "settings file not found: $ENV_FILE"
  for name in "${SETTINGS[@]}"; do
    [[ -n "${!name+set}" ]] && declare "__keep_$name=${!name}"
  done
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  for name in "${SETTINGS[@]}"; do
    keep="__keep_$name"
    [[ -n "${!keep+set}" ]] && declare "$name=${!keep}"
  done
fi

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/configs/scaffolding.json" ]] || die "not a SPARA checkout: $REPO"
ROOT="${ROOT:-$HOME/scratch/spara}"

PARTITION=ice-gpu
GPU="${GPU:-gpu:h200:1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-96G}"
PILOT_HOURS="${PILOT_HOURS:-1}"
BASELINE_HOURS="${BASELINE_HOURS:-15}"
[[ "$GPU" =~ ^gpu:[a-zA-Z0-9_]+:1$ ]] || die "GPU must request exactly one GPU, e.g. gpu:h200:1"
[[ "$CPUS" =~ ^[0-9]+$ ]] && (( CPUS >= 1 )) || die "CPUS must be a positive integer"
[[ "$MEM" =~ ^[0-9]+[KMGT]?B?$ ]] || die "MEM must be a Slurm size, e.g. 96G"
[[ "$PILOT_HOURS" =~ ^[0-9]+$ ]] && (( PILOT_HOURS >= 1 )) || die "PILOT_HOURS must be a positive integer"
[[ "$BASELINE_HOURS" =~ ^[0-9]+$ ]] && (( BASELINE_HOURS >= 1 )) || die "BASELINE_HOURS must be a positive integer"
if (( BASELINE )); then HOURS="${HOURS:-$(( PILOT_HOURS + BASELINE_HOURS ))}"
else HOURS="${HOURS:-$PILOT_HOURS}"; fi
[[ "$HOURS" =~ ^[0-9]+$ ]] && (( HOURS >= 1 )) || die "HOURS must be a positive integer"

note "settings:  ${ENV_FILE:-none; expected scripts/pace/pace.conf}"
note "repo:      $REPO"
note "root:      $ROOT"
note "resources: partition=$PARTITION gres=$GPU cpus=$CPUS mem=$MEM"
if (( BASELINE )); then
  if (( HOURS == PILOT_HOURS + BASELINE_HOURS )); then
    note "walltime:  ${HOURS}h total (pilot ${PILOT_HOURS}h + baseline ${BASELINE_HOURS}h), one allocation"
  else
    note "walltime:  ${HOURS}h total for both phases, one allocation (HOURS override)"
  fi
else
  note "walltime:  ${HOURS}h, pilot only"
fi
(( DRY )) && note "dry run: no directories, links or jobs will be created"

# Blocker: the job scripts cd into $ROOT/code, which does not exist when the checkout
# is itself $HOME/scratch/SPARA. Give ROOT its own directory and link the checkout in
# as code/, leaving the repository untouched.
if [[ -e "$ROOT/code" ]]; then
  linked="$(cd "$ROOT/code" && pwd -P)"
  [[ "$linked" == "$(cd "$REPO" && pwd -P)" ]] || die "$ROOT/code already points at $linked, not $REPO"
  note "layout: $ROOT/code already points at the checkout"
elif (( DRY )); then
  note "layout: would create $ROOT and link $ROOT/code -> $REPO"
else
  mkdir -p "$ROOT/logs" "$ROOT/results"
  ln -s "$REPO" "$ROOT/code"
  note "layout: linked $ROOT/code -> $REPO"
fi
export SPARA_REMOTE_ROOT="$ROOT"

VENV="$ROOT/.venv/bin/python"
MODEL="$ROOT/models/Qwen2.5-VL-3B-Instruct"
DATA="$ROOT/data/vsp"

ready=0
[[ -x "$VENV" && -f "$DATA/manifest.json" && -f "$MODEL/download-manifest.json" ]] && ready=1
if (( SKIP_SETUP )); then
  note "setup: skipped on request"
elif (( ready )); then
  note "setup: venv, data and model already present; skipping"
elif (( DRY )); then
  note "setup: would run $REPO/scripts/pace/setup.sh (venv, VSP data, Qwen2.5-VL-3B)"
else
  note "setup: running setup.sh (venv, cu124 lock, VSP data, Qwen2.5-VL-3B download)"
  bash -l "$REPO/scripts/pace/setup.sh"
fi

if ! (( DRY )); then
  [[ -x "$VENV" ]] || die "no interpreter at $VENV; rerun without --skip-setup"
  [[ -f "$DATA/manifest.json" ]] || die "no prepared data at $DATA"
  [[ -f "$MODEL/download-manifest.json" ]] || die "no verified model at $MODEL"
fi

if (( SKIP_TESTS )); then
  note "tests: skipped on request"
elif (( DRY )); then
  note "tests: would run the CPU suite with $VENV"
else
  note "tests: running the CPU suite"
  SPARA_TEST_DATA="$DATA" SPARA_TEST_PROCESSOR="$MODEL" "$VENV" -m pytest -q "$REPO/tests"
fi

# One node, one task. Account and QoS are left to the Slurm default
# association for the submitting user.
# One job runs the pilot and then the baseline, so the baseline cannot lose the node.
exports="ALL,SPARA_REMOTE_ROOT=$ROOT"
(( BASELINE )) || exports="$exports,SPARA_PILOT_ONLY=1"
[[ -n "$CONTINUE" ]] && exports="$exports,SPARA_RUN=$CONTINUE"

args=(--parsable --job-name=spara-scaffolding
      --partition="$PARTITION" --gres="$GPU"
      --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$HOURS:00:00"
      --export="$exports"
      --output="$ROOT/logs/scaffolding-%j.log"
      --error="$ROOT/logs/scaffolding-%j.log")
[[ -n "${CONSTRAINT:-}" ]] && args+=(--constraint="$CONSTRAINT")
args+=("$ROOT/code/scripts/pace/scaffolding-all.sbatch")

if (( DRY )); then
  echo "sbatch ${args[*]}" >&2
  note "dry run: nothing submitted"
  exit 0
fi

JOB="$(sbatch "${args[@]}")"
note "submitted: job $JOB"
note "  log      $ROOT/logs/scaffolding-$JOB.log"
note "  pilot    $ROOT/results/scaffolding-pilot-$JOB/"
if (( BASELINE )); then
  note "  baseline ${CONTINUE:-$ROOT/results/scaffolding-$JOB}/"
  echo
  note "The baseline starts as soon as the pilot finishes, in the same allocation."
  note "To stop it after the pilot instead, rerun with --no-baseline, or scancel $JOB."
  note "The specification asks for a review of the pilot before full-run hours:"
  note "  - GPU-hour estimate for a full run vs the ${HOURS}h requested here"
  note "  - latent_loss_normalized (well below 1 means the predictor tracks z*)"
  note "  - the RL gradient norm split into latent and text terms"
else
  note "  pilot only (--no-baseline)"
fi
note "watch with: squeue -u $USER   tail -f $ROOT/logs/scaffolding-$JOB.log"
