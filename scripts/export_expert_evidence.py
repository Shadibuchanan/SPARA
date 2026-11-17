"""Copy an expert-iteration run's compact evidence into docs/experiments/<name>/.

Each `--arm NAME=DIR` copies DIR's small JSON/JSONL files into NAME/, leaving out weights,
and turns evaluation/responses.jsonl into benchmark-outcomes.jsonl with the prompts and raw
text dropped (row, grid size, success, extracted answer, status), as in
docs/experiments/expert-iteration-6961874/.

    python scripts/export_expert_evidence.py --output docs/experiments/NAME \
        --arm routes=results/RUN/expert-routes-JOB --arm e=results/RUN/expert-train-search-JOB
"""
import argparse
import json
import re
from pathlib import Path

KEEP = {".json", ".jsonl"}
OUTCOME = ("row_index", "grid_size", "success", "extracted_answer", "status")
# PACE paths name the user; evidence writes them as $SPARA_REMOTE_ROOT, as earlier runs do.
REMOTE_ROOT = re.compile(r"/(?:home/hice1|storage/ice1/\d+/\d+)/[^/\"]+(?:/scratch)?/SPARA")


def copy(source, target):
    target.write_text(REMOTE_ROOT.sub("$SPARA_REMOTE_ROOT", source.read_text(encoding="utf-8")), encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--arm", action="append", required=True, help="NAME=DIR")
parser.add_argument("--file", action="append", default=[], help="a single file to copy beside the arms")
args = parser.parse_args()
output = Path(args.output)
for spec in args.arm:
    name, source = spec.split("=", 1)
    source, target = Path(source), output / name
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or path.suffix not in KEEP or "final" in relative.parts:
            continue
        if relative.parts[0] == "evaluation":
            if path.name == "responses.jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                (target / "benchmark-outcomes.jsonl").write_text(
                    "".join(json.dumps({k: r[k] for k in OUTCOME}) + "\n" for r in rows), encoding="utf-8")
            elif path.name == "summary.json":
                copy(path, target / "benchmark-summary.json")
            continue
        flat = "-".join(relative.parts)
        copy(path, target / flat)
for path in map(Path, args.file):
    copy(path, output / path.name)
print(output)
