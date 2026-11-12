"""Post-hoc CPU analysis of a JEPA judge run (docs/JEPA_JUDGE_SPEC.md).

Would any threshold on the judges' probabilities make them usable, and does the oracle's
argmax hide information that LIVR's frozen states hold? From the run's saved states and
weights, with no model and no GPU:

1. Reproduce the run's argmax rates, a check that the reload is aligned.
2. Threshold-free separation: AUROC and average precision for a hole ahead (dead against
   not) and for an arrival (done against not), per judge, population and size.
3. Operating points: thresholds on P(dead) chosen on the validation maps and applied to
   the evaluation maps, and the best points read off the evaluation curve itself.
4. Layers: the probe and the oracle retrained on CPU at every saved layer.
5. Search: step 1a″'s noisy-judge search at the error rates of points on the JEPA's
   curve, with the toward-goal stand-in as the policy. As in the sweep, arrivals are
   judged correctly and errors are independent per transition.

    PYTHONPATH=src python scripts/analyze_jepa_thresholds.py RUN_DIR --data-root data/vsp \
        --output RUN_DIR/thresholds.json
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from spara.evaluation import ReferenceEvaluator
from spara.jepa import OUTCOMES, Jepa, Oracle, Probe, next_labels, pairs, rates, train
from spara.route_search import FrozenLakeEnvironment, problems_of, toward_goal_policy
from spara.search import DEAD, DONE, NoisyJudge, guided_search

DEAD_I, DONE_I = OUTCOMES.index(DEAD), OUTCOMES.index(DONE)
TARGETS = (0.02, 0.05, 0.1, 0.2)  # step 1a″'s ε₉₀ values within 64 calls are 0.02 and 0.05
CURVE_FA = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)  # points on the JEPA's curve to search with
SEEDS, BUDGET, WIDE, DEPTH = range(5), 64, 1_000, 64
SETTINGS = {"soft10+reopen": (10.0, True), "soft10": (10.0, False)}


# Separation and operating points ----------------------------------------------------------

def auroc(score, positive):
    """Mann-Whitney AUROC, ties averaged."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(score, positive):
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    if not positive.any():
        return None
    hits = positive[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def curve(score, positive):
    """(threshold, miss rate, false-alarm rate) for flagging `score >= threshold`, at every
    distinct score, plus flagging nothing (threshold infinity)."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    order = np.argsort(-score, kind="mergesort")
    s, p = score[order], positive[order]
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = np.cumsum(p)[last], np.cumsum(~p)[last]
    return (np.r_[np.inf, s[last]], np.r_[1.0, 1 - tp / p.sum()], np.r_[0.0, fp / (~p).sum()])


def at(score, positive, threshold):
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    flagged = score >= threshold
    return {"threshold": float(threshold), "miss_rate": float((~flagged[positive]).mean()),
            "false_alarm_rate": float(flagged[~positive].mean())}


def operating_points(score, positive):
    """Read off one curve: the lowest miss rate at each false-alarm target, the lowest
    false-alarm rate at each miss target, and the equal-error point."""
    thresholds, miss, fa = curve(score, positive)
    best = lambda rate, other, cap: float(other[rate <= cap].min())
    equal = int(np.argmin(np.maximum(miss, fa)))
    return {"auroc": auroc(score, positive), "average_precision": average_precision(score, positive),
            "positives": int(np.asarray(positive).sum()), "pairs": len(positive),
            "miss_at_false_alarm": {str(t): best(fa, miss, t) for t in TARGETS},
            "false_alarm_at_miss": {str(t): best(miss, fa, t) for t in TARGETS},
            "equal_error": {"threshold": float(thresholds[equal]), "miss_rate": float(miss[equal]),
                            "false_alarm_rate": float(fa[equal])},
            "meets_both": {str(t): bool(((miss <= t) & (fa <= t)).any()) for t in TARGETS}}


def threshold_for(score, positive, false_alarm):
    """The lowest threshold whose false-alarm rate stays within `false_alarm`."""
    thresholds, _, fa = curve(score, positive)
    return float(thresholds[fa <= false_alarm][-1])


# The run's judges ----------------------------------------------------------------------------

def load(run):
    states = torch.load(run / "states.pt", map_location="cpu", weights_only=True)
    weights = torch.load(run / "jepa.pt", map_location="cpu", weights_only=True)
    items = {name: [json.loads(line) for line in (run / f"items-{name}.jsonl").read_text().splitlines()]
             for name in ("fit", "val", "eval")}
    return states, weights, items


def judges(weights):
    out = {}
    for name, kind in (("jepa", Jepa), ("probe", Probe), ("oracle", Oracle)):
        model = kind(weights["d"])
        model.load_state_dict(weights[name])
        out[name] = model.eval()
    return out


@torch.no_grad()
def judged_probabilities(model, x, judged, chunk=8192):
    return torch.cat([torch.softmax(model.judge(x[judged[i:i + chunk, 0]], judged[i:i + chunk, 1]), -1)
                      for i in range(0, len(judged), chunk)]) if len(judged) else torch.zeros(0, len(OUTCOMES))


@torch.no_grad()
def oracle_probabilities(model, x, taken):
    return torch.softmax(model(x[taken[:, 2]]), -1)


def population(x, items):
    """Probabilities' inputs for one population: judged pairs, taken moves and their labels."""
    judged, taken = pairs(items)
    labels = next_labels(items, taken)
    return {"x": x, "judged": judged, "taken": taken, "labels": labels,
            "taken_judged": torch.stack([taken[:, 0], taken[:, 1], labels], 1)}


def scored(models, pop):
    """P(outcome) of every pair each judge reads, with the truth: the JEPA and the probe on
    every judged move, and the JEPA and the oracle on the moves taken."""
    return {"jepa": (judged_probabilities(models["jepa"], pop["x"], pop["judged"]), pop["judged"][:, 2]),
            "probe": (judged_probabilities(models["probe"], pop["x"], pop["judged"]), pop["judged"][:, 2]),
            "jepa_taken_moves": (judged_probabilities(models["jepa"], pop["x"], pop["taken_judged"]), pop["labels"]),
            "oracle_taken_moves": (oracle_probabilities(models["oracle"], pop["x"], pop["taken"]), pop["labels"])}


def separation(probabilities, truth):
    p, t = probabilities.numpy(), truth.numpy()
    return {"dead": operating_points(p[:, DEAD_I], t == DEAD_I), "done": operating_points(p[:, DONE_I], t == DONE_I)}


def grouped(probabilities, truth, rows, items, key):
    out = {}
    groups = np.array([key(items[i]) for i in rows.tolist()])
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        out[str(g)] = {"dead_auroc": auroc(probabilities[mask, DEAD_I].numpy(), truth[mask].numpy() == DEAD_I),
                       "dead": int((truth[mask] == DEAD_I).sum()), "pairs": int(mask.sum())}
    return out


# Layers -------------------------------------------------------------------------------------

def layers(states, items, settings, seed):
    """The probe and the oracle retrained at every saved layer (CPU), scored on the
    evaluation maps by AUROC and argmax rates."""
    fit, val, ev = (population(None, items[n]) for n in ("fit", "val", "eval"))
    out = {}
    for li, layer in enumerate(states["layers"]):
        x = {n: states[n][:, li].float() for n in ("fit", "val", "eval")}
        started, row = time.time(), {}
        for kind in ("probe", "oracle"):
            model, history = train(kind, x["fit"], fit["judged"], fit["taken"], fit["labels"],
                                   (x["val"], val["judged"], val["taken"], val["labels"]), settings, "cpu", seed)
            if kind == "probe":
                p, truth = judged_probabilities(model, x["eval"], ev["judged"]), ev["judged"][:, 2]
            else:
                p, truth = oracle_probabilities(model, x["eval"], ev["taken"]), ev["labels"]
            row[kind] = {"dead_auroc": auroc(p[:, DEAD_I].numpy(), truth.numpy() == DEAD_I),
                         "done_auroc": auroc(p[:, DONE_I].numpy(), truth.numpy() == DONE_I),
                         "argmax": rates(p.argmax(-1), truth), "epochs": len(history),
                         "best_validation": min(h["outcome"] for h in history)}
        out[str(layer)] = row | {"seconds": round(time.time() - started, 1)}
        print(f"layer {layer}: probe AUROC {row['probe']['dead_auroc']:.3f}, "
              f"oracle AUROC {row['oracle']['dead_auroc']:.3f}", flush=True)
    return out


# Search at the curve's error rates ------------------------------------------------------------

def searched(env, problems, miss, false_alarm):
    """Step 1a″'s noisy-judge search, per transition, at one (miss, false alarm) pair: maps
    solved within 64 and 1,000 policy calls, mean and range over five seeds."""
    out = {}
    for name, (veto, reopen) in SETTINGS.items():
        solved = {BUDGET: [], WIDE: []}
        for seed in SEEDS:
            judge = NoisyJudge(env, miss, false_alarm, seed, problem_key=lambda p: p["id"])
            results = guided_search(judge, toward_goal_policy, problems, WIDE, DEPTH, veto=veto, reopen=reopen)
            for b in solved:
                solved[b].append(sum(r["solved"] and r["policy_calls"] <= b for r in results))
        out[name] = {str(b): {"mean": statistics.mean(v), "min": min(v), "max": max(v)} for b, v in solved.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--settings", default="configs/jepa_judge.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-layers", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    began = time.time()
    states, weights, items = load(args.run)
    summary = json.loads((args.run / "summary.json").read_text())
    settings = json.loads(Path(args.settings).read_text())
    li = list(states["layers"]).index(weights["layer"])
    models = judges(weights)
    pops = {n: population(states[n][:, li].float(), items[n]) for n in ("val", "eval")}
    probabilities = {n: scored(models, pops[n]) for n in pops}

    # 1. The run's argmax rates, recomputed.
    reported = summary["judge_accuracy"]
    keys = ("miss_rate", "false_alarm_rate", "done_missed", "done_false", "accuracy")
    recomputed = {name: rates(p.argmax(-1), truth) for name, (p, truth) in probabilities["eval"].items()}
    source = lambda name: reported[name]["all"] if name in ("jepa", "probe") else reported[name]
    differences = {name: {k: abs(recomputed[name][k] - source(name)[k]) for k in keys} for name in recomputed}
    reproduced = all(d["done_missed"] <= 2 and d["done_false"] <= 2 and d["miss_rate"] < 1e-3
                     and d["false_alarm_rate"] < 1e-3 for d in differences.values())
    print("reproduced:", reproduced, json.dumps(differences), flush=True)

    # 2. Separation on the evaluation maps, overall and by population and size.
    ev_items = items["eval"]
    result = {"run": str(args.run.name), "layer": weights["layer"], "reproduced": reproduced,
              "argmax_differences": differences, "separation": {}, "by_population": {}, "by_size": {}}
    for name, (p, truth) in probabilities["eval"].items():
        result["separation"][name] = separation(p, truth)
        rows = (pops["eval"]["judged"] if name in ("jepa", "probe") else pops["eval"]["taken"])[:, 0]  # map-level keys
        result["by_population"][name] = grouped(p, truth, rows, ev_items, lambda i: i["population"])
        result["by_size"][name] = grouped(p, truth, rows, ev_items, lambda i: i["size"])

    # 3. Thresholds chosen on the validation maps, applied to the evaluation maps.
    chosen = {}
    for name in ("jepa", "probe", "jepa_taken_moves", "oracle_taken_moves"):
        (vp, vt), (ep, et) = probabilities["val"][name], probabilities["eval"][name]
        chosen[name] = {str(t): {"validation": at(vp[:, DEAD_I].numpy(), vt.numpy() == DEAD_I,
                                                  threshold_for(vp[:, DEAD_I].numpy(), vt.numpy() == DEAD_I, t)),
                                 "evaluation": at(ep[:, DEAD_I].numpy(), et.numpy() == DEAD_I,
                                                  threshold_for(vp[:, DEAD_I].numpy(), vt.numpy() == DEAD_I, t))}
                        for t in TARGETS}
    result["validation_thresholds"] = chosen

    # 5. Search at points on the JEPA's evaluation curve, and at the run's own rates.
    evaluator = ReferenceEvaluator(args.data_root)
    env = FrozenLakeEnvironment(evaluator)
    problems = problems_of(args.data_root, "test") + problems_of(args.data_root, settings["fresh_split"])
    jp, jt = probabilities["eval"]["jepa"]
    thresholds, miss, fa = curve(jp[:, DEAD_I].numpy(), jt.numpy() == DEAD_I)
    points = [{"label": "exact judge", "miss_rate": 0.0, "false_alarm_rate": 0.0},
              {"label": "JEPA argmax, evaluation walks", "miss_rate": recomputed["jepa"]["miss_rate"],
               "false_alarm_rate": recomputed["jepa"]["false_alarm_rate"]},
              {"label": "JEPA argmax, in search (run)", "miss_rate": summary["search"]["jepa_in_search_rates"]["miss_rate"],
               "false_alarm_rate": summary["search"]["jepa_in_search_rates"]["false_alarm_rate"]}]
    for target in CURVE_FA:
        k = int(np.flatnonzero(fa <= target)[-1])
        points.append({"label": f"JEPA curve, false alarms {target}", "threshold": float(thresholds[k]),
                       "miss_rate": float(miss[k]), "false_alarm_rate": float(fa[k])})
    for point in points:
        started = time.time()
        point["solved"] = searched(env, problems, point["miss_rate"], point["false_alarm_rate"])
        point["seconds"] = round(time.time() - started, 1)
        print(point["label"], round(point["miss_rate"], 3), round(point["false_alarm_rate"], 3),
              {k: v[str(BUDGET)]["mean"] for k, v in point["solved"].items()}, flush=True)
    result["search_at_curve"] = {"policy": "toward-goal stand-in", "maps": len(problems), "seeds": len(SEEDS),
                                 "structure": "per transition; arrivals judged correctly", "points": points}

    # 4. Layers (slowest; last).
    if not args.skip_layers:
        result["layers"] = layers(states, items, settings, settings["seed"])
    result["seconds"] = round(time.time() - began, 1)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    s = result["separation"]
    print(json.dumps({name: {"dead_auroc": v["dead"]["auroc"], "done_auroc": v["done"]["auroc"],
                             "equal_error": v["dead"]["equal_error"]} for name, v in s.items()}, indent=1))


if __name__ == "__main__":
    main()
