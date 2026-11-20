"""No-detour failure signatures on the reference benchmark (first written for E2R, an
expert-iteration arm that was built but never run). CPU only; it replays each run's recorded answers
under the reference rules and checks that the replay reproduces the recorded successes.

    PYTHONPATH=src python scripts/analyze_nodetour_failures.py --data-root data/vsp
    PYTHONPATH=src python scripts/analyze_nodetour_failures.py --data-root data/vsp \
        --run "NAME=/path/to/run" --output nodetour.json

A run is an outcomes JSONL with row_index, success and extracted_answer, or a run directory
holding evaluation/responses.jsonl or benchmark-outcomes.jsonl.
"""
import argparse
import json

from spara.data import records
from spara.expert_round2 import benchmark_outcomes, signatures
from spara.fresh_maps import is_detour

EXPERIMENTS = "docs/experiments/"
RUNS = {"LIVR original (6189457)": "livr-6189457/evaluation-outcomes.jsonl",
        "LIVR rerun (6253300)": "livr-6253300/evaluation-outcomes.jsonl",
        "Vanilla SFT A (6259811)": "sft-a-6259811/evaluation-outcomes.jsonl",
        "Vanilla SFT C (6259811)": "sft-c-6259811/evaluation-outcomes.jsonl",
        "Text-only RL, B (6256423)": "text-rl-6256423/evaluation-outcomes.jsonl",
        "Latent RL, C, mean (6879761)": "latent-rl-6879761/evaluation-outcomes-mean.jsonl",
        "E (6961875)": "expert-iteration-6961874/e/benchmark-outcomes.jsonl",
        "E2 (6992799)": "expert-round2-6992798/e2/benchmark-outcomes.jsonl",
        "EB2 (6992800)": "expert-round2-6992798/eb2/benchmark-outcomes.jsonl"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--run", action="append", default=[], help="NAME=PATH; replaces the default runs")
    parser.add_argument("--output", help="Write the signatures as JSON")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run) or {name: EXPERIMENTS + path for name, path in RUNS.items()}
    test = records(args.data_root, "test")
    descs = {i: row["map_desc"] for i, row in enumerate(test) if not is_detour(row["map_desc"])}
    result = {}
    print(f"No-detour test maps: {len(descs)} of {len(test)}")
    print(f"{'run':32s} solved  hole  short/past  failed with an away move")
    for name, path in runs.items():
        outcomes = benchmark_outcomes(path)
        marks = signatures(descs, {i: outcomes[i][1] for i in descs})
        recorded = sum(outcomes[i][0] for i in descs)
        if marks["solved"] != recorded:
            raise ValueError(f"{name}: the replay solves {marks['solved']} maps; the run recorded {recorded}")
        result[name] = marks
        print(f"{name:32s} {marks['solved']:6d}  {marks['hole']:4d}  {marks['short_or_past']:10d}  "
              f"{marks['failed_with_away_move']} of {marks['failed']}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
