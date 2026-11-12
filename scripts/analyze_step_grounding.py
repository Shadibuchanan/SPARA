"""Pre-declared comparisons for a step-grounding run (docs/STEP_GROUNDING_SPEC.md). CPU only.

RUN holds stage2/ (steps.jsonl, grounding_head.pt), evaluation/ (responses.jsonl) and
probe/ (routes.jsonl, summary.json, features.pt), as scripts/pace/step_grounding.sbatch
writes them. The LIVR references default to the evidence copies in docs/experiments.
Writes RUN/comparison.json.

- primary: benchmark success against the LIVR rerun, exact McNemar, with the reading;
- secondary: greedy test and fresh routes against LIVR's, failure types, training fit;
- manipulation check: position accuracy at layer 20 on solved test steps against LIVR's,
  with the reading; S1 per split; the trained head's accuracy on G's own routes.

Usage: python scripts/analyze_step_grounding.py RUN [--data-root data/vsp] [--livr-... PATH]

By hand on a PACE login node, set OMP_NUM_THREADS=1: with torch's default thread count
the run took 87 minutes there instead of 14 seconds.
"""
import argparse
import json
from pathlib import Path

import torch

from spara.data import records
from spara.grounding import OUTCOMES, load_head, outcome
from spara.latent_battery import mcnemar
from spara.latent_rl import moves
from spara.planning_probe import MOVES, decoded_cells, position_targets, ridge, walk

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "experiments"
LAYER = 20
TOOK = 0.10  # manipulation check: at least 10 points above LIVR's solved-step accuracy
FRESH = "fresh/probe"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def status_of(grid, route, success):
    status = walk(grid, route)[1]
    if (status == "goal") != bool(success):
        raise ValueError("The walk disagrees with a recorded outcome")
    return status


def paired(grounded, livr, sizes):
    """grounded, livr: row_index -> (success, status). McNemar overall, counts by size."""
    if grounded.keys() != livr.keys():
        raise ValueError("The two runs cover different maps")
    def count(keys):
        only_g = sum(grounded[k][0] and not livr[k][0] for k in keys)
        only_l = sum(livr[k][0] and not grounded[k][0] for k in keys)
        return {"maps": len(keys), "grounded": sum(grounded[k][0] for k in keys),
                "livr": sum(livr[k][0] for k in keys), "only_grounded": only_g, "only_livr": only_l,
                "p": mcnemar(only_g, only_l)}
    keys = sorted(grounded)
    failures = lambda run: {s: sum(run[k][1] == s for k in keys) for s in ("hole", "short")}
    return count(keys) | {
        "per_size": {str(n): count([k for k in keys if sizes[k] == n]) for n in sorted(set(sizes.values()))},
        "failures": {"grounded": failures(grounded), "livr": failures(livr)}}


def reading(result):
    if result["p"] < 0.05:
        return "improves" if result["only_grounded"] > result["only_livr"] else "harms"
    return "no detectable difference"


def last_epoch(path, stage=2):
    rows = [r for r in jsonl(path) if r["stage"] == stage]
    last = [r for r in rows if r["epoch"] == max(r["epoch"] for r in rows)]
    mean = lambda key: sum(r[key] for r in last) / len(last) if all(r.get(key) is not None for r in last) else None
    return {"answer_loss": mean("loss"), "grounding_loss": mean("grounding_loss"),
            "cell_accuracy": mean("cell_accuracy"), "outcome_accuracy": mean("outcome_accuracy")}


def head_readout(head, states, steps, grids):
    """The trained head on decision states of the model's own routes: cell accuracy, and
    outcome accuracy over the three directions not taken, on solved and fatal steps."""
    with torch.no_grad():
        rows, cols, outcomes = head(states)
    result = {}
    for name, keep in (("solved_steps", lambda s: s["solved"]), ("fatal_steps_k2", lambda s: s["fatal"] and s["k"] >= 2),
                       ("fatal_steps_all", lambda s: s["fatal"])):
        index = [i for i, s in enumerate(steps) if keep(s)]
        cell = other = taken_hole = 0
        for i in index:
            step, size = steps[i], steps[i]["size"]
            decoded = [int(rows[i, :size].argmax()), int(cols[i, :size].argmax())]
            cell += decoded == step["cell"]
            truth = [outcome(grids[step["row_index"]], step["cell"], m) for m in MOVES]
            taken = list(MOVES).index(step["move"])
            other += sum(int(outcomes[i, d].argmax()) == truth[d] for d in range(len(MOVES)) if d != taken) / 3
            taken_hole += int(outcomes[i, taken].argmax()) == OUTCOMES.index("hole")
        n = len(index)
        result[name] = {"steps": n, "cell_accuracy": cell / n if n else None,
                        "outcome_accuracy_other_directions": other / n if n else None,
                        "move_taken_read_as_hole": taken_hole / n if n else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--livr-benchmark", default=EVIDENCE / "livr-6253300" / "evaluation-outcomes.jsonl")
    parser.add_argument("--livr-steps", default=EVIDENCE / "livr-6253300" / "steps.jsonl")
    parser.add_argument("--livr-routes", default=EVIDENCE / "planning-probe-6887458" / "routes.jsonl")
    parser.add_argument("--livr-probe", default=EVIDENCE / "planning-probe-6887458" / "summary.json")
    parser.add_argument("--livr-fresh", default=EVIDENCE / "explore-probe-6884608" / "sweep.jsonl")
    args = parser.parse_args()
    run = Path(args.run)
    test = records(args.data_root, "test")
    fresh = records(args.data_root, FRESH)
    report = {}

    # Primary: the reference benchmark (sampled decoding), map by map.
    grounded = {r["row_index"]: r for r in jsonl(run / "evaluation" / "responses.jsonl")}
    livr = {r["row_index"]: r for r in jsonl(args.livr_benchmark)}
    outcome_of = lambda r: (bool(r["success"]), status_of(test[r["row_index"]]["map_desc"],
                                                          moves(r["extracted_answer"] or ""), r["success"]))
    sizes = {i: len(row["map_desc"]) for i, row in enumerate(test)}
    report["primary_benchmark"] = paired({k: outcome_of(v) for k, v in grounded.items()},
                                         {k: outcome_of(v) for k, v in livr.items()}, sizes)
    report["primary_benchmark"]["reading"] = reading(report["primary_benchmark"])

    # Secondary: greedy routes on test and fresh maps.
    own = jsonl(run / "probe" / "routes.jsonl")
    greedy = lambda rows, split: {r["row_index"]: (r["success"], r["status"]) for r in rows if r["split"] == split}
    report["greedy_test"] = paired(greedy(own, "test"), greedy(jsonl(args.livr_routes), "test"), sizes)
    sweep = {r["row_index"]: r for r in jsonl(args.livr_fresh)}
    livr_fresh = {i: (r["greedy_success"], status_of(fresh[i]["map_desc"], r["greedy_route"], r["greedy_success"]))
                  for i, r in sweep.items()}
    own_fresh = greedy(own, FRESH)
    report["greedy_fresh"] = paired(own_fresh, {i: livr_fresh[i] for i in own_fresh},
                                    {i: len(fresh[i]["map_desc"]) for i in own_fresh})
    report["training_fit"] = {"grounded": last_epoch(run / "stage2" / "steps.jsonl"),
                              "livr": last_epoch(args.livr_steps)}
    fit = report["training_fit"]
    fit["answer_loss_more_than_doubled"] = fit["grounded"]["answer_loss"] > 2 * fit["livr"]["answer_loss"]

    # Manipulation check: the planning probe's ridge at layer 20 on solved test steps.
    probe = json.loads((run / "probe" / "summary.json").read_text())["state_probe"]
    livr_probe = json.loads(Path(args.livr_probe).read_text())["state_probe"]
    if livr_probe["layer"] != LAYER:
        raise ValueError("LIVR's reference solved-step accuracy is not at layer 20")
    features = torch.load(run / "probe" / "features.pt", weights_only=False)
    position = [layer + 1 for layer in features["layers"]].index(LAYER)
    train = features["train"]
    x_train = train["states"][:, position].float()
    manipulation = {"held_out_position_accuracy": {"grounded": probe["held_out_position_accuracy"],
                                                   "livr": livr_probe["held_out_position_accuracy"]},
                    "selected_layer": {"grounded": probe["layer"], "livr": livr_probe["layer"]}}
    grids = {"test": [row["map_desc"] for row in test], "fresh": [row["map_desc"] for row in fresh]}
    head, settings = load_head(run / "stage2" / "grounding_head.pt")
    if settings["layer"] != LAYER:
        raise ValueError("The head was trained at another layer than the check reads")
    for split in ("test", "fresh"):
        data = features.get(split)
        if not data or not data["steps"]:
            continue
        x = data["states"][:, position].float()
        cells = decoded_cells(ridge(x_train, position_targets(train["steps"]), x), data["steps"])
        correct = [c == s["cell"] for c, s in zip(cells, data["steps"])]
        solved = [ok for ok, s in zip(correct, data["steps"]) if s["solved"]]
        fatal = [ok for ok, s in zip(correct, data["steps"]) if s["fatal"] and s["k"] >= 2]
        livr_split = livr_probe["splits"][split]
        manipulation[split] = {
            "layer_20_solved_step_accuracy": {"grounded": sum(solved) / len(solved) if solved else None,
                                              "livr": livr_split["solved_step_accuracy"]},
            "layer_20_fatal_step_accuracy_k2": {"grounded": sum(fatal) / len(fatal) if fatal else None,
                                                "livr": livr_split["S1"]["accuracy"]},
            "S1_at_selected_layer": {"grounded": probe["splits"][split]["S1"], "livr": livr_split["S1"]},
            "head": head_readout(head, x, data["steps"], grids[split])}
        if split == "fresh":
            manipulation[split]["note"] = "LIVR's fresh probe used 5 x 5 and 6 x 6 maps only; G's adds 7 x 7 and 8 x 8"
    accuracy = manipulation["test"]["layer_20_solved_step_accuracy"]
    manipulation["took"] = accuracy["grounded"] is not None and accuracy["grounded"] >= accuracy["livr"] + TOOK
    report["manipulation_check"] = manipulation
    primary = report["primary_benchmark"]["reading"]
    report["reading"] = (("consistent with the hypothesis (one seed)" if primary == "improves" else
                          "position became decodable but routes did not improve" if primary == "no detectable difference"
                          else "position became decodable and routes got worse")
                         if manipulation["took"] else "the objective did not take on test maps; silent on the hypothesis")
    (run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("primary_benchmark", "reading")}, indent=1))
    print(json.dumps({"took": manipulation["took"], "test": manipulation["test"]["layer_20_solved_step_accuracy"]}))


if __name__ == "__main__":
    main()
