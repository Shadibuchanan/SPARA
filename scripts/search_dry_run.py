"""CPU dry run of guided search on the 97 unseen detour maps (docs/GUIDED_SEARCH_SPEC.md,
step 1a): the uniform prior and a model-free toward-goal stand-in, with the step simulator
checked against the reference evaluator. No model and no GPU.

    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_dry_run.py --data-root data/vsp
"""
import argparse
import json
import statistics
from collections import Counter

from spara.detour_probe import find
from spara.evaluation import ReferenceEvaluator
from spara.planning_probe import distances
from spara.route_search import (BUDGET, FrozenLakeEnvironment, problems_of, simulator_agreement,
                                toward_goal_policy)
from spara.search import guided_search, uniform_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--fresh-split", default="fresh/probe")
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    env = FrozenLakeEnvironment(ReferenceEvaluator(args.data_root))
    problems = problems_of(args.data_root, "test") + problems_of(args.data_root, args.fresh_split)
    depth = max(len(p["row"]["map_desc"]) ** 2 for p in problems)
    report = {"maps": len(problems), "budget": args.budget}
    for name, policy in (("uniform", uniform_policy), ("toward_goal", toward_goal_policy)):
        results = guided_search(env, policy, problems, args.budget, depth)
        calls = sorted(r["policy_calls"] if r["solved"] else args.budget + 1 for r in results)
        shortest = [distances(p["row"]["map_desc"])[find(p["row"]["map_desc"], 1)] for p in problems]
        agree, checked = simulator_agreement(env, problems, results)
        report[name] = {"solved": sum(r["solved"] for r in results), "median_calls": statistics.median(calls),
                        "p90_calls": calls[int(0.9 * (len(calls) - 1))], "max_calls": calls[-1],
                        "excess_length": dict(sorted(Counter(len(r["actions"]) - s for r, s in zip(results, shortest)
                                                             if r["solved"]).items())),
                        "simulator_agreement": [agree, checked]}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
