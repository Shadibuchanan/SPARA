"""Compare the detour scaling check's arms (docs/DETOUR_SCALING_SPEC.md) and apply the
rules fixed in the specification. Writes comparison.json next to the first run.

    python scripts/analyze_detour_scaling.py --runs livr=RUN_L n98=RUN_98 n500=RUN_500 n2000=RUN_2000

On a PACE login node, set OMP_NUM_THREADS=1.
"""
import argparse
import json
from pathlib import Path

from spara.detour_scaling import compare

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "experiments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=4, required=True, metavar="ARM=RUN")
    parser.add_argument("--planning-routes", default=EVIDENCE / "planning-probe-6887458/routes.jsonl")
    parser.add_argument("--livr-outcomes", default=EVIDENCE / "livr-6253300/evaluation-outcomes.jsonl")
    parser.add_argument("--output", help="Where to write comparison.json (default: beside the livr run)")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.runs)
    if sorted(runs) != ["livr", "n2000", "n500", "n98"]:
        parser.error("Give the runs for livr, n98, n500 and n2000")
    result = compare(runs, args.planning_routes, args.livr_outcomes)
    output = Path(args.output) if args.output else Path(runs["livr"]).parent / "detour-scaling-comparison.json"
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("validity", "primary", "secondary")}, indent=1))


if __name__ == "__main__":
    main()
