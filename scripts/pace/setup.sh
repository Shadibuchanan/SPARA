#!/bin/bash
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
mkdir -p "$ROOT/logs"
cd "$ROOT/code"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
python -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -r requirements/pace-cu124.lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
"$ROOT/.venv/bin/python" -m pip install --no-deps -e '.[model,test]'
"$ROOT/.venv/bin/python" -m spara.cli prepare-data --data-root "$ROOT/data/vsp"
"$ROOT/.venv/bin/python" -m spara.cli fetch-model --output "$ROOT/models/Qwen2.5-VL-3B-Instruct"
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/logs/environment-setup.txt"
