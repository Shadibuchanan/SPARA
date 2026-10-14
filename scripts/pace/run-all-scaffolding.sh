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

Resources (environment variables; account and QoS are auto-detected if unset):
  SPARA_ACCOUNT            Slurm account                        (detected)
  SPARA_QOS                Slurm QoS                            (detected)
  SPARA_PARTITION          GPU partition                        (default: ice-gpu)
  SPARA_GPU                One-GPU GRES, >= 80 GB               (default: gpu:h200:1)
  SPARA_CONSTRAINT         Optional --constraint value          (default: none)
  SPARA_PILOT_MINUTES      Minutes to budget for the pilot      (default: 60)
  SPARA_BASELINE_MINUTES   Minutes to budget for the baseline   (default: 900)
  SPARA_MINUTES            Total walltime, overriding their sum (default: their sum)
  SPARA_REMOTE_ROOT        Run root holding code/ data/ models/ (default: $HOME/scratch/spara)
  SPARA_REPO               Repository checkout to link as code/ (default: this checkout)

Options:
  --no-baseline   Stop the job after the pilot; do not run the full baseline
  --continue DIR  Continue an earlier baseline run directory; completed phases skip
  --skip-setup    Assume the venv, data and model already exist under $ROOT
  --skip-tests    Do not run the CPU test suite before submitting
  --dry-run       Print what would happen; create nothing and submit nothing
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
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

REPO="${SPARA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ -f "$REPO/configs/scaffolding.json" ]] || die "not a SPARA checkout: $REPO"
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/spara}"

# Account and QoS: use the first association Slurm reports unless overridden.
if [[ -z "${SPARA_ACCOUNT:-}" || -z "${SPARA_QOS:-}" ]]; then
  assoc="$(sacctmgr -nP show assoc user="$USER" format=Account,QOS 2>/dev/null | head -1 || true)"
  [[ -n "$assoc" ]] && { SPARA_ACCOUNT="${SPARA_ACCOUNT:-${assoc%%|*}}"; SPARA_QOS="${SPARA_QOS:-${assoc##*|}}"; }
fi
: "${SPARA_ACCOUNT:?Could not detect a Slurm account; set SPARA_ACCOUNT}"
: "${SPARA_QOS:?Could not detect a Slurm QoS; set SPARA_QOS}"
PARTITION="${SPARA_PARTITION:-ice-gpu}"
GPU="${SPARA_GPU:-gpu:h200:1}"
PILOT_MINUTES="${SPARA_PILOT_MINUTES:-60}"
BASELINE_MINUTES="${SPARA_BASELINE_MINUTES:-900}"
[[ "$GPU" =~ ^gpu:[a-zA-Z0-9_]+:1$ ]] || die "SPARA_GPU must request exactly one GPU, e.g. gpu:h200:1"
[[ "$PILOT_MINUTES" =~ ^[0-9]+$ ]] && (( PILOT_MINUTES >= 1 )) || die "SPARA_PILOT_MINUTES must be a positive integer"
[[ "$BASELINE_MINUTES" =~ ^[0-9]+$ ]] && (( BASELINE_MINUTES >= 1 )) || die "SPARA_BASELINE_MINUTES must be a positive integer"
if (( BASELINE )); then MINUTES="${SPARA_MINUTES:-$(( PILOT_MINUTES + BASELINE_MINUTES ))}"
else MINUTES="${SPARA_MINUTES:-$PILOT_MINUTES}"; fi
[[ "$MINUTES" =~ ^[0-9]+$ ]] && (( MINUTES >= 1 )) || die "SPARA_MINUTES must be a positive integer"

note "repo:      $REPO"
note "root:      $ROOT"
note "resources: account=$SPARA_ACCOUNT qos=$SPARA_QOS partition=$PARTITION gres=$GPU"
if (( BASELINE )); then
  note "walltime:  ${MINUTES}m total (pilot ${PILOT_MINUTES}m + baseline ${BASELINE_MINUTES}m), one allocation"
else
  note "walltime:  ${MINUTES}m, pilot only"
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

# Mirrors scripts/pace/submit-scaffolding.sh: one node, one task, 4 CPUs, 96 GiB.
# One job runs the pilot and then the baseline, so the baseline cannot lose the node.
exports="ALL,SPARA_REMOTE_ROOT=$ROOT"
(( BASELINE )) || exports="$exports,SPARA_PILOT_ONLY=1"
[[ -n "$CONTINUE" ]] && exports="$exports,SPARA_RUN=$CONTINUE"

args=(--parsable --job-name=spara-scaffolding
      --account="$SPARA_ACCOUNT" --qos="$SPARA_QOS"
      --partition="$PARTITION" --gres="$GPU"
      --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --time="$MINUTES:00"
      --export="$exports"
      --output="$ROOT/logs/scaffolding-%j.log"
      --error="$ROOT/logs/scaffolding-%j.log")
[[ -n "${SPARA_CONSTRAINT:-}" ]] && args+=(--constraint="$SPARA_CONSTRAINT")
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
  note "  - GPU-hour estimate for a full run vs the ${MINUTES}m requested here"
  note "  - latent_loss_normalized (well below 1 means the predictor tracks z*)"
  note "  - the RL gradient norm split into latent and text terms"
else
  note "  pilot only (--no-baseline)"
fi
note "watch with: squeue -u $USER   tail -f $ROOT/logs/scaffolding-$JOB.log"
