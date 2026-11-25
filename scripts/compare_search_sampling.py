"""Best-first search against independent sampling at equal budget, on the same maps and policy.

    python scripts/compare_search_sampling.py

Reads docs/experiments/sampling-vs-search/: `search-routes.jsonl` (e3's final-verdict search,
at most 64 policy calls a map) and `samples.jsonl` (64 independent answers a map from e3 at
temperature 1, scored by the reference evaluator). Readings are fixed in
docs/DETOUR_SCALING_SPEC.md, "Search against independent sampling at equal budget".

A search policy call reads the next-move distribution once; a sampled route of n moves makes
n + 1 such decisions. Samples are counted in the order drawn."""
import json
from collections import defaultdict
from pathlib import Path

from spara.detour_scaling import paired

FOLDER = Path(__file__).resolve().parents[1] / "docs/experiments/sampling-vs-search"
BUDGET = 64


def within_decisions(sample, budget):
    """Whether some sample drawn before `budget` policy decisions are spent passes."""
    spent = 0
    for route, success in zip(sample["routes"], sample["successes"]):
        spent += len(route) + 1
        if spent > budget:
            return False
        if success:
            return True
    return False


def within_verifications(sample, calls):
    """Whether one of the first `calls` distinct sampled routes passes."""
    seen = []
    for route, success in zip(sample["routes"], sample["successes"]):
        if route not in seen:
            seen.append(route)
            if len(seen) > calls:
                return False
            if success:
                return True
    return False


def main():
    search = {r["row_index"]: r for r in map(json.loads, (FOLDER / "search-routes.jsonl").read_text().splitlines())}
    samples = {r["row_index"]: r for r in map(json.loads, (FOLDER / "samples.jsonl").read_text().splitlines())}
    keys = sorted(search)
    if sorted(samples) != keys or any(len(samples[k]["successes"]) != BUDGET for k in keys):
        raise ValueError("Every map needs its search record and 64 samples")
    searched = {k: search[k]["solved"] and search[k]["policy_calls"] <= BUDGET for k in keys}
    by_decisions = {k: within_decisions(samples[k], BUDGET) for k in keys}
    by_verifier = {k: within_verifications(samples[k], max(1, search[k]["verifier_calls"])) for k in keys}
    any_sample = {k: any(samples[k]["successes"]) for k in keys}
    depth = defaultdict(lambda: [0, 0, 0, 0])
    for k in keys:
        d = depth[min(samples[k]["away"], 3)]
        d[0] += 1; d[1] += searched[k]; d[2] += by_decisions[k]; d[3] += any_sample[k]
    result = {
        "primary_equal_policy_decisions": paired(searched, by_decisions, keys),
        "second_equal_verifier_calls": paired(searched, by_verifier, keys),
        "pass_at_64_samples": sum(any_sample.values()),
        "by_moves_away (maps, search, sampling at 64 decisions, any of 64 samples)": dict(sorted(depth.items())),
        "mean_distinct_routes": sum(len(set(samples[k]["routes"])) for k in keys) / len(keys),
        "mean_decisions_per_sample": sum(len(r) + 1 for k in keys for r in samples[k]["routes"]) / (BUDGET * len(keys)),
        "mean_search_verifier_calls": sum(search[k]["verifier_calls"] for k in keys) / len(keys),
    }
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
