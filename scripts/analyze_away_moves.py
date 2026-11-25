"""Away-move accuracy on the 400 held-out 7 x 7 and 8 x 8 detour maps (post hoc).

    python scripts/analyze_away_moves.py

Reads docs/experiments/away-moves/outcomes.jsonl (each arm's greedy route per map, with the
map). A detour map's shortest safe route needs `need` moves away from the goal; a route
"makes the needed away moves" when it makes at least that many before it ends or falls in a
hole (spara.expert_round2.away_moves). Prints, per arm: maps solved, routes that make the
needed away moves, and failures that never move away from the goal."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from spara.expert_round2 import away_moves
from spara.fresh_maps import GOAL, START, find, shortest_path

FILE = Path(__file__).resolve().parents[1] / "docs/experiments/away-moves/outcomes.jsonl"


def needed(desc):
    (sr, sc), (gr, gc) = find(desc, START), find(desc, GOAL)
    return (shortest_path(desc) - abs(sr - gr) - abs(sc - gc)) // 2


def main():
    counts = defaultdict(Counter)
    for line in FILE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        made = away_moves(r["map_desc"], r["route"])
        c = counts[r["arm"]]
        c["maps"] += 1
        c["solved"] += r["success"]
        c["made_needed_away_moves"] += made >= needed(r["map_desc"])
        if not r["success"]:
            c["failed"] += 1
            c["failed_without_away_move"] += made == 0
    for arm, c in counts.items():
        print(json.dumps({"arm": arm} | dict(c)))
    return counts


if __name__ == "__main__":
    main()
