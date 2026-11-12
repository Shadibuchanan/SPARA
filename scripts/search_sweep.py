"""Step 1a′ of docs/GUIDED_SEARCH_SPEC.md on CPU: search without an exact step simulator.

Mode 1 searches with the final verdict only; mode 2 replaces the step simulator with a
noisy judge, per cell or per transition, over a fixed grid of error rates and seeds. Both
use the model-free priors on the 97 unseen detour maps. No model and no GPU.

`--modes in_grid` is exploratory, added after the run: mode 1 with each prior ruling
out moves off the grid, which the fixed priors pad rejected answers with.

`--modes robust` is step 1a″: mode 2 again with each variant of the search that
tolerates a noisy judge (re-opening, soft veto at three costs, and both), written to
robust-summary.json.

    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_sweep.py --data-root data/vsp \\
      --output docs/experiments/search-without-simulator
"""
import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from spara.evaluation import ReferenceEvaluator
from spara.detour_probe import find
from spara.planning_probe import MOVES
from spara.route_search import (START, FrozenLakeEnvironment, MisperceivedFrozenLake, problems_of,
                                toward_goal_policy)
from spara.search import NoisyJudge, guided_search, surviving_route, uniform_policy, verdict_search

# Fixed in the specification before the run.
DEPTH, STEP_BUDGET, VERDICT_BUDGET, JUDGE_BUDGET = 64, 64, 20_000, 1_000
RATES, SEEDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), range(5)
TYPES = {"miss": (1, 0), "false_alarm": (0, 1), "both": (1, 1)}
ANCHOR = {"structure": "transition", "miss": 0.5, "false_alarm": 0.1}
READY = 0.9
PRIORS = {"uniform": uniform_policy, "toward_goal": toward_goal_policy}
# Step 1a″: (veto cost in nats, re-open after a refused answer); infinity is the hard veto.
VARIANTS = {"hard": (math.inf, False), "reopen": (math.inf, True),
            **{f"soft{v:g}": (v, False) for v in (1.0, 3.0, 10.0)},
            **{f"soft{v:g}+reopen": (v, True) for v in (1.0, 3.0, 10.0)}}
CARRIED = ("toward_goal", "transition/both", str(JUDGE_BUDGET))  # how step 2's setting is chosen


def quantiles(values):
    values = sorted(values)
    return {"median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))]} if values else None


def censored(results, cap):
    return [r["policy_calls"] if r["solved"] else cap + 1 for r in results]


def within(results, budget):
    return sum(r["solved"] and r["policy_calls"] <= budget for r in results)


OFF_GRID = -1_000.0  # effectively never chosen, but finite as the policy contract requires


def in_grid(policy):
    """`policy`, with moves off the grid (which leave the agent in place) ruled out."""
    def ruled(requests):
        replies = policy(requests)
        for (problem, actions, _), scores in zip(requests, replies):
            grid = problem["row"]["map_desc"]
            n, (r, c) = len(grid), find(grid, START)
            for a in actions:
                if 0 <= r + MOVES[a][0] < n and 0 <= c + MOVES[a][1] < n:
                    r, c = r + MOVES[a][0], c + MOVES[a][1]
            for m in scores:
                if m in MOVES and not (0 <= r + MOVES[m][0] < n and 0 <= c + MOVES[m][1] < n):
                    scores[m] = OFF_GRID
        return replies
    return ruled


def verdict_only(env, problems, evaluator, priors=PRIORS):
    """Mode 1, one map at a time (the frontier grows large without pruning)."""
    report, rows = {}, []
    for name, policy in priors.items():
        started = time.time()
        results = [verdict_search(env, policy, [p], VERDICT_BUDGET, DEPTH, log=False)[0] for p in problems]
        valid = all(evaluator.score(p["row"], r["answer"])["success"] for p, r in zip(problems, results) if r["solved"])
        sizes = sorted({len(p["row"]["map_desc"]) for p in problems})
        report[name] = {
            "solved_within": {str(b): within(results, b) for b in (STEP_BUDGET, 1_000, VERDICT_BUDGET)},
            "policy_calls": quantiles(censored(results, VERDICT_BUDGET)),
            "verifier_calls_solved": quantiles([r["verifier_calls"] for r in results if r["solved"]]),
            "rejected": sum(r["rejected"] for r in results),
            "stops": dict(Counter(r["stop"] for r in results)),
            "by_size": {str(n): {"maps": sum(len(p["row"]["map_desc"]) == n for p in problems),
                                 "solved": sum(r["solved"] for p, r in zip(problems, results)
                                               if len(p["row"]["map_desc"]) == n)} for n in sizes},
            "validity": {"accepted_answers_pass_evaluator": valid,
                         "costs_nondecreasing": all(r["costs_nondecreasing"] for r in results)},
            "seconds": round(time.time() - started, 1)}
        rows += [{"prior": name, "map": p["id"], "grid_size": len(p["row"]["map_desc"]), "solved": r["solved"],
                  "policy_calls": r["policy_calls"], "verifier_calls": r["verifier_calls"], "stop": r["stop"],
                  "route_length": r["solved"] and len(r["actions"])} for p, r in zip(problems, results)]
    return report, rows


def judge_for(env, evaluator, structure, miss, false_alarm, seed):
    if structure == "cell":
        return MisperceivedFrozenLake(evaluator, miss, false_alarm, seed)
    return NoisyJudge(env, miss, false_alarm, seed, problem_key=lambda p: p["id"])


def judged(env, judge, policy, problems, veto=math.inf, reopen=False):
    """One noisy-judge run; frontier stops split into blocked and poisoned. Accepted
    answers are re-checked against the reference evaluator."""
    results = guided_search(judge, policy, problems, JUDGE_BUDGET, DEPTH, veto=veto, reopen=reopen)
    stops = Counter()
    for p, r in zip(problems, results):
        stops["poisoned" if r["stop"] == "frontier" and surviving_route(env, judge, p, DEPTH) else
              "blocked" if r["stop"] == "frontier" else r["stop"]] += 1
        if r["solved"] and not env.evaluator.score(p["row"], r["answer"])["success"]:
            stops["accepted_but_failing"] += 1
    for r in results:
        r.pop("transitions")  # keep memory flat across the grid
    return results, stops


def summarise(runs):
    """Over the seeds of one condition: solved counts per seed, pooled calls, stop totals."""
    per_seed = {str(b): [within(results, b) for results, _ in runs] for b in (STEP_BUDGET, JUDGE_BUDGET)}
    pooled = [r for results, _ in runs for r in results]
    return {"solved_within": {b: {"mean": statistics.mean(v), "min": min(v), "max": max(v), "per_seed": v}
                              for b, v in per_seed.items()},
            "policy_calls": quantiles(censored(pooled, JUDGE_BUDGET)),
            "verifier_calls_solved": quantiles([r["verifier_calls"] for r in pooled if r["solved"]]),
            "rejected_mean": statistics.mean(sum(r["rejected"] for r in results) for results, _ in runs),
            "stops": dict(sum((stops for _, stops in runs), Counter()))}


def epsilon_90(curve, maps, budget):
    """The largest swept rate up to which every swept rate (0 included) solves at least
    READY of the maps within `budget`, by the mean over seeds."""
    best = None
    for rate in sorted(curve, key=float):
        if curve[rate]["solved_within"][budget]["mean"] < READY * maps:
            break
        best = float(rate)
    return best


def noisy(env, evaluator, problems, veto=math.inf, reopen=False):
    report = {}
    for name, policy in PRIORS.items():
        started = time.time()
        plain = guided_search(env, policy, problems, JUDGE_BUDGET, DEPTH)
        exact = guided_search(env, policy, problems, JUDGE_BUDGET, DEPTH, veto=veto, reopen=reopen)
        reproduces = {}
        for structure in ("cell", "transition"):
            zero = guided_search(judge_for(env, evaluator, structure, 0.0, 0.0, 0), policy, problems, JUDGE_BUDGET,
                                 DEPTH, veto=veto, reopen=reopen)
            reproduces[structure] = all(a["actions"] == b["actions"] and a["policy_calls"] == b["policy_calls"]
                                        for a, b in zip(exact, zero))
        reproduces["exact_matches_step_1a"] = all(a["actions"] == b["actions"] and a["policy_calls"] == b["policy_calls"]
                                                  for a, b in zip(plain, exact))
        baseline = summarise([(exact, Counter(r["stop"] for r in exact))])
        conditions = {}
        for structure in ("cell", "transition"):
            for kind, (use_miss, use_alarm) in TYPES.items():
                curve = {"0": baseline}
                for rate in RATES:
                    curve[str(rate)] = summarise([judged(env, judge_for(env, evaluator, structure, use_miss * rate,
                                                                        use_alarm * rate, seed), policy, problems,
                                                         veto, reopen) for seed in SEEDS])
                conditions[f"{structure}/{kind}"] = {
                    "curve": curve,
                    "epsilon_90": {b: epsilon_90(curve, len(problems), b) for b in (str(STEP_BUDGET), str(JUDGE_BUDGET))}}
        anchor = summarise([judged(env, judge_for(env, evaluator, ANCHOR["structure"], ANCHOR["miss"],
                                                  ANCHOR["false_alarm"], seed), policy, problems, veto, reopen)
                            for seed in SEEDS])
        report[name] = {"exact": baseline, "zero_noise_reproduces_exact": reproduces, "conditions": conditions,
                        "anchor": ANCHOR | anchor, "seconds": round(time.time() - started, 1)}
    return report


def robust(env, evaluator, problems):
    """Step 1a″: every variant over the full grid, and the setting carried into step 2."""
    report = {}
    for name, (veto, reopen) in VARIANTS.items():
        report[name] = {"veto": veto if math.isfinite(veto) else "inf", "reopen": reopen} | {
            "priors": noisy(env, evaluator, problems, veto, reopen)}
        print(name, {prior: {c: v["epsilon_90"] for c, v in r["conditions"].items()}
                     for prior, r in report[name]["priors"].items()}, flush=True)
    return {"variants": report, "carried": carried(report)}


def carried(report):
    """The registered rule: highest ε₉₀ for CARRIED, then fewer median calls at ε = 0.
    Added after the run, when the rule left a tie: the tied variant whose solved-within
    curve for CARRIED is at least every other tied variant's at every swept rate."""
    prior, condition, budget = CARRIED
    def rule(v):
        p = report[v]["priors"][prior]
        return -(p["conditions"][condition]["epsilon_90"][budget] or 0), p["exact"]["policy_calls"]["median"]
    ranked = sorted(report, key=rule)
    tied = [v for v in ranked if rule(v) == rule(ranked[0])]
    result = {"rule": CARRIED, "ranking": ranked, "tied_on_rule": tied}
    if len(tied) == 1:
        return result | {"variant": tied[0]}
    def curve(v):
        c = report[v]["priors"][prior]["conditions"][condition]["curve"]
        return [c[rate]["solved_within"][budget]["mean"] for rate in sorted(c, key=float)]
    dominant = [v for v in tied if all(a >= b for u in tied for a, b in zip(curve(v), curve(u)))]
    return result | {"tiebreak": "curve dominance (added after the run)",
                     "variant": dominant[0] if len(dominant) == 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--fresh-split", default="fresh/probe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="verdict,noisy")
    args = parser.parse_args()
    evaluator = ReferenceEvaluator(args.data_root)
    env = FrozenLakeEnvironment(evaluator)
    problems = problems_of(args.data_root, "test") + problems_of(args.data_root, args.fresh_split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"maps": len(problems), "depth_cap": DEPTH,
               "budgets": {"step_1a": STEP_BUDGET, "verdict_only": VERDICT_BUDGET, "noisy_judge": JUDGE_BUDGET},
               "rates": RATES, "seeds": len(SEEDS), "ready_share": READY}
    modes = args.modes.split(",")
    if "verdict" in modes:
        summary["verdict_only"], rows = verdict_only(env, problems, evaluator)
        (output / "verdict-only-maps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if "noisy" in modes:
        summary["noisy_judge"] = noisy(env, evaluator, problems)
    if "robust" in modes:  # step 1a″; written to its own file
        result = robust(env, evaluator, problems)
        (output / "robust-summary.json").write_text(json.dumps(result, indent=1) + "\n")
        print("carried:", result["carried"])
        if modes == ["robust"]:
            return
    if "in_grid" in modes:  # exploratory; written to its own file
        report, rows = verdict_only(env, problems, evaluator, {k: in_grid(v) for k, v in PRIORS.items()})
        (output / "exploratory-in-grid.json").write_text(json.dumps(report, indent=1) + "\n")
        (output / "exploratory-in-grid-maps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps({k: (v["solved_within"], v["policy_calls"]) for k, v in report.items()}))
        if modes == ["in_grid"]:
            return
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "noisy_judge"}, indent=1))
    for name, report in summary.get("noisy_judge", {}).items():
        print(name, {c: v["epsilon_90"] for c, v in report["conditions"].items()},
              "anchor solved@64", report["anchor"]["solved_within"]["64"]["mean"])


if __name__ == "__main__":
    main()
