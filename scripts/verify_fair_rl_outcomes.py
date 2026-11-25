"""Recompute the fair-RL comparisons for seeds 42 and 43 from the staged per-map outcomes.

    python scripts/verify_fair_rl_outcomes.py

Reads docs/experiments/fair-rl-outcomes/greedy-outcomes.jsonl (B' plain, C' mean, C'' plain
greedy answers) and applies spara.fair_rl.sign_test on the 240 primary held-out detour maps,
as fair_rl.compare does. Exits non-zero if a count or p-value differs from the run's own
recorded comparison. Also prints each run's median gradient norm before the global clip at 1.0
(grad-norms.jsonl, from each run's steps.jsonl), the audit's check on C'."""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from spara.fair_rl import PRIMARY, sign_test

FOLDER = Path(__file__).resolve().parents[1] / "docs/experiments/fair-rl-outcomes"
SEEDS = (42, 43)


def outcomes():
    by_run = defaultdict(dict)
    for line in (FOLDER / "greedy-outcomes.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_run[r["arm"], r["seed"]][r["population"], r["row_index"]] = r["success"]
    return by_run


def primary(by_run, arm):
    keys = sorted(k for k in by_run["b", SEEDS[0]] if k[0] in PRIMARY)
    return sign_test([by_run[arm, s] for s in SEEDS], [by_run["b", s] for s in SEEDS], keys)


def main():
    by_run = outcomes()
    failed = False
    for arm, name in (("c", "fair-rl-7142592"), ("n", "fair-rl-n-7154355")):
        mine = primary(by_run, arm)
        recorded = json.loads((FOLDER / f"comparison-{name}.json").read_text(encoding="utf-8"))["primary"]
        same = all(mine[k] == recorded[k] for k in ("maps", "first", "second", "maps_first_ahead", "maps_second_ahead"))
        same &= math.isclose(mine["p"], recorded["p"], rel_tol=1e-9)
        print(f"{arm} vs b: {json.dumps({k: mine[k] for k in mine})}  matches recorded: {same}")
        failed |= not same
    norms = defaultdict(list)
    for line in (FOLDER / "grad-norms.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["updated"]:
            norms[r["arm"], r["seed"]].append(r["grad_norm"])
    for (arm, seed), values in sorted(norms.items()):
        median = statistics.median(values)
        print(f"{arm}{seed}: median gradient norm {median:.1f}, median clip factor {min(1.0, 1 / median):.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
