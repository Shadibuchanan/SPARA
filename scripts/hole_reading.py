"""How well a run's hole-check lines read the map (docs/HOLE_CHECK_SPEC.md#result-job-7108600):
`spara.hole_check.line_reading` over the run's held-out answers.

    python scripts/hole_reading.py results/hole-check/round3-s3-7108600 > reading.json
"""
import json
import sys
from pathlib import Path

from spara.data import records
from spara.hole_check import line_reading


def reading(run, data_root=Path("data/vsp"), population="heldout78"):
    held = [r for r in map(json.loads, (Path(run) / "greedy.jsonl").read_text().splitlines())
            if r["population"] == population]
    rows = records(data_root, held[0]["split"])
    return line_reading((rows[r["row_index"]]["map_desc"], r["answer"], r["route"]) for r in held)


if __name__ == "__main__":
    print(json.dumps(reading(sys.argv[1]), indent=1))
