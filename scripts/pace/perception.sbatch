#!/bin/bash
# Perception arms (spara.perception): train SPARA_ARM (sft, livr or patch) from the base model on the
# sampled Visual-CoT rows, then answer the validation rows and V*. SPARA_SMOKE=1 runs a few rows only.
# SPARA_STAGE=images instead prepares the data on CPU: it samples the rows (the released list needs a few
# GB of memory to read) and then reads the Visual-CoT archive once for those rows' images:
#   sbatch -p ice-cpu -c 4 --mem=16G -t 06:00:00 --export=ALL,SPARA_STAGE=images scripts/pace/perception.sbatch
# One 80 GB GPU for training; resources are given at submission.
# Submit only within the agreed GPU budget; this script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
SETTINGS="${SPARA_PERCEPTION_SETTINGS:-configs/perception_viscot.json}"
DATA="${SPARA_VISCOT_DATA:-$ROOT/data/viscot}"
VSTAR="${SPARA_VSTAR_DATA:-$ROOT/data/vstar}"
BASE="$ROOT/models/Qwen2.5-VL-3B-Instruct"
[ -f "$SETTINGS" ] || { echo "Missing settings: $SETTINGS" >&2; exit 1; }
if [ "${SPARA_STAGE:-train}" = images ]; then
  [ -f "$DATA/train.jsonl" ] || "$PYTHON" -m spara.cli perception --stage select --settings "$SETTINGS" --data-root "$DATA"
  "$PYTHON" -m spara.cli perception --stage images --settings "$SETTINGS" --data-root "$DATA"
  exit 0
fi
ARM="${SPARA_ARM:?set SPARA_ARM to sft, livr or patch}"
[ -f "$DATA/train.jsonl" ] || { echo "Missing sampled rows: $DATA" >&2; exit 1; }
grep -q '"count"' "$DATA/manifest.json" || { echo "Sampled images are not extracted: $DATA" >&2; exit 1; }
[ -f "$VSTAR/manifest.json" ] || { echo "Missing V* questions: $VSTAR" >&2; exit 1; }
[ -f "$BASE/download-manifest.json" ] || { echo "Missing base model: $BASE" >&2; exit 1; }
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli perception --config configs/livr.json --settings "$SETTINGS" --data-root "$DATA" \
  --vstar-root "$VSTAR" --base-model "$BASE" --arm "$ARM" ${SPARA_SMOKE:+--smoke} \
  --output "$ROOT/results/perception-$ARM${SPARA_SMOKE:+-smoke}-$SLURM_JOB_ID"
