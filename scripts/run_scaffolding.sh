#!/bin/bash
# Run the Scaffolding Minds baseline end to end on one GPU:
#   scaffold (Stage 1 encoder) -> predict (Stage 1 latents) -> eval1
#   -> rl (Stage 2) -> eval2
# Rerunning with the same --run skips completed phases. Run it inside a GPU
# allocation; on PACE, submit scripts/pace/submit-scaffolding.sh baseline instead.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_scaffolding.sh --model DIR [options]

  --model DIR          Verified Qwen2.5-VL-3B download (python -m spara.cli fetch-model)
  --run DIR            Run directory; reuse it to continue (default: results/scaffolding-<time>)
  --data DIR           Prepared VSP data (default: data/vsp)
  --config FILE        Recipe (default: configs/scaffolding.json)
  --python BIN         Interpreter with spara installed (default: python)
  --phases LIST        Comma-separated subset of scaffold,predict,eval1,rl,eval2 (default: all)
  --pilot              Run the bounded pilot instead of the baseline
  --retry-incomplete   Move an unfinished phase directory aside and rerun that phase

Each option can also be set as SPARA_MODEL, SPARA_RUN, SPARA_DATA, SPARA_CONFIG or
SPARA_PYTHON. Relative paths are resolved from the repository root.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
note() { echo "[$(date '+%H:%M:%S')] $*"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${SPARA_MODEL:-}"
RUN="${SPARA_RUN:-}"
DATA="${SPARA_DATA:-data/vsp}"
CONFIG="${SPARA_CONFIG:-configs/scaffolding.json}"
PYTHON="${SPARA_PYTHON:-python}"
PHASES="scaffold predict eval1 rl eval2"
PILOT=0
RETRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--run|--data|--config|--python|--phases)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      case "$1" in
        --model) MODEL="$2" ;;
        --run) RUN="$2" ;;
        --data) DATA="$2" ;;
        --config) CONFIG="$2" ;;
        --python) PYTHON="$2" ;;
        --phases) PHASES="${2//,/ }" ;;
      esac
      shift 2 ;;
    --pilot) PILOT=1; shift ;;
    --retry-incomplete) RETRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

cd "$REPO"
for phase in $PHASES; do
  case "$phase" in
    scaffold|predict|eval1|rl|eval2) ;;
    *) die "unknown phase '$phase' (use scaffold, predict, eval1, rl, eval2)" ;;
  esac
done
[[ -n "$MODEL" ]] || { usage >&2; die "--model is required"; }
[[ -f "$MODEL/download-manifest.json" ]] || die "no verified model at $MODEL (python -m spara.cli fetch-model --output $MODEL)"
[[ -f "$DATA/manifest.json" ]] || die "no prepared data at $DATA (python -m spara.cli prepare-data --data-root $DATA)"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
command -v "$PYTHON" >/dev/null || die "python not found: $PYTHON"
"$PYTHON" -c "import spara.scaffolding.cli" || die "spara is not importable with $PYTHON (pip install -e '.[model]')"

RUN="${RUN:-results/scaffolding-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN/logs"
RUN="$(cd "$RUN" && pwd)"
LOGS="$RUN/logs"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'
else
  note "warning: nvidia-smi not found; training and evaluation need a CUDA GPU"
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
"$PYTHON" -m pip freeze > "$LOGS/environment-$STAMP.txt"
note "run directory: $RUN"
if (( PILOT )); then
  note "storage: the pilot writes about 24 GB of checkpoints"
else
  note "storage: a full run writes about 90 GB of checkpoints (8 + 40 + 40 GB); check your quota"
fi

S1A="$RUN/stage1-scaffold"
S1B="$RUN/stage1-predict"
S2="$RUN/stage2-rl"
E1="$RUN/evaluation-stage1"
E2="$RUN/evaluation-stage2"
COMMON=(--config "$CONFIG" --data-root "$DATA")

complete() {  # phase-directory kind
  case "$2" in
    train) [[ -f "$1/final/complete.json" ]] ;;
    eval) [[ -f "$1/responses.jsonl" && "$(wc -l < "$1/responses.jsonl")" -eq 400 ]] ;;
    pilot) [[ -f "$1/summary.json" ]] ;;
  esac
}

require() {  # phase-directory name
  complete "$1" train || die "$2 has not completed ($1/final is missing); run that phase first"
}

run_phase() {  # name kind output cli-args...
  local name="$1" kind="$2" output="$3" started aside
  shift 3
  if complete "$output" "$kind"; then
    note "skip $name: already complete in $output"
    return
  fi
  if [[ -e "$output" ]]; then
    (( RETRY )) || die "$output exists but is incomplete; see $LOGS/$name.log, then move it aside or pass --retry-incomplete"
    aside="$output.incomplete-$(date +%Y%m%d-%H%M%S)"
    mv "$output" "$aside"
    note "moved unfinished $name output to $aside"
  fi
  note "start $name -> $output"
  started=$(date +%s)
  echo "=== $(date) $PYTHON -m spara.scaffolding.cli $* --output $output" >> "$LOGS/$name.log"
  "$PYTHON" -m spara.scaffolding.cli "$@" --output "$output" 2>&1 | tee -a "$LOGS/$name.log"
  note "finished $name in $(( $(date +%s) - started )) s"
}

if (( PILOT )); then
  run_phase pilot pilot "$RUN/pilot" pilot "${COMMON[@]}" --base-model "$MODEL"
  "$PYTHON" - "$RUN/pilot/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
print(f"pilot passed: {summary['passed']}")
for name, check in summary["checks"].items():
    if not check["passed"]:
        print(f"  failed check: {name}")
print("estimated GPU-hours:", {k: round(v, 2) for k, v in summary["estimated_gpu_hours"].items()})
PY
  exit 0
fi

for phase in $PHASES; do
  case "$phase" in
    scaffold) run_phase scaffold train "$S1A" train --phase scaffold "${COMMON[@]}" --base-model "$MODEL" ;;
    predict)
      require "$S1A" scaffold
      run_phase predict train "$S1B" train --phase predict "${COMMON[@]}" --base-model "$MODEL" --encoder "$S1A/final" ;;
    eval1)
      require "$S1B" predict
      run_phase eval1 eval "$E1" evaluate "${COMMON[@]}" --checkpoint "$S1B/final" ;;
    rl)
      require "$S1B" predict
      run_phase rl train "$S2" train --phase rl "${COMMON[@]}" --checkpoint "$S1B/final" ;;
    eval2)
      require "$S2" rl
      run_phase eval2 eval "$E2" evaluate "${COMMON[@]}" --checkpoint "$S2/final" ;;
  esac
done

"$PYTHON" - "$E1" "$E2" <<'PY'
import json, sys
from pathlib import Path
for label, path in zip(("Stage 1", "Stage 1 + RL"), sys.argv[1:]):
    summary = Path(path) / "summary.json"
    if summary.is_file():
        overall = json.loads(summary.read_text())["overall"]
        print(f"{label}: {overall['successes']}/{overall['count']} ({overall['accuracy']:.2%})")
PY
note "done; outputs and logs are in $RUN"
