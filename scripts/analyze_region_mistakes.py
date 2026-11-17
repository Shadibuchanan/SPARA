"""Post hoc, CPU: when a model's first mistake on a held-out map is a step into a hole, did its latents
read that cell as a hole? (docs/experiments/region-egr-7122822/README.md)

For each run: the count-matched block probe (`region_latents.block_probe`, fitted on pool maps) gives
a held-out hole prediction for every cell. Among failed held-out maps whose first mistake steps into a
hole, it counts how often that hole was flagged, split by whether a safe shortest move toward the goal
existed, in unmasked and masked greedy. Against that, the probe's recall over all held-out hole cells.

Usage: python scripts/analyze_region_mistakes.py [--output PATH]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from spara.data import records
from spara.detour_probe import find, manhattan, outcomes, step
from spara.planning_probe import distances
from spara.region_latents import block_probe

RUNS = {"egr": ("results/region-egr-7122822", 4), "eb2": ("results/round2-eb2-6992800", None)}


def first_hole(desc, route):
    """The first move off every shortest safe route, if it steps into a hole: (cell, whether a safe
    shortest move toward the goal existed)."""
    dist, goal, cell = distances(desc), find(desc, 2), find(desc, 1)
    for move in route:
        if dist.get(cell) == 0:
            return None
        here = outcomes(desc, dist, cell)
        if here[move] != "optimal":
            if here[move] != "hole":
                return None
            toward = any(o == "optimal" and manhattan(step(cell, m), goal) < manhattan(cell, goal) for m, o in here.items())
            return step(cell, move), toward
        cell = step(cell, move)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--output")
    args = parser.parse_args()
    held = records(args.data_root, "fresh/detour-heldout-78")
    report = {}
    for name, (run, offset) in RUNS.items():
        token = (lambda n, b: offset + b) if offset is not None else (lambda n, b: 0)
        probe = block_probe(torch.load(Path(run) / "states.pt"), 12, token)
        flagged = {row: p["predicted"][k] for p in probe.values() for k, row in enumerate(p["rows"])}
        recall = (sum(int((p["predicted"] & p["truth"]).sum()) for p in probe.values())
                  / sum(int(p["truth"].sum()) for p in probe.values()))
        report[name] = {"hole_recall_all_cells": recall}
        for mode in ("greedy", "masked"):
            counts = Counter()
            for line in (Path(run) / f"{mode}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["population"] != "heldout78" or r["success"]:
                    continue
                desc = held[r["row_index"]]["map_desc"]
                found = first_hole(desc, r["route"] or "")
                if found:
                    (row, col), toward = found
                    counts[("toward" if toward else "forced", bool(flagged[r["row_index"]][row * len(desc) + col]))] += 1
            report[name][mode] = {kind: {"maps": counts[(kind, True)] + counts[(kind, False)], "flagged": counts[(kind, True)]}
                                  for kind in ("toward", "forced")}
    print(json.dumps(report, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
