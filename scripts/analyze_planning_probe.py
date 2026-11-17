"""Exploratory, post-hoc checks on a planning probe run (spara planning-probe). CPU only.

Reads features.pt, steps.jsonl and headroom.jsonl from the run directory and writes
exploratory.json there. None of these checks was fixed before the run; the
pre-declared readings are in summary.json.

- alignment: the move about to be written, decoded from the state that predicts it;
- start: the start cell decoded at the first move;
- displacement: the agent's offset from the start (rows, columns), decoded linearly,
  on held-out training maps and on the model's own routes (solved against fatal steps);
- absolute cell with a small MLP instead of ridge;
- position accuracy at the earlier steps of failed routes, and at solved steps next to
  a hole, against the fatal steps;
- hole-ahead scores for the direction taken: fatal holes against safe cells;
- headroom: how many optimised rows ended on their norm bound.

Usage: python scripts/analyze_planning_probe.py RUN_DIR
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

from spara.planning_probe import HELD_OUT, decoded_cells, position_targets, ridge

SEED = 42
SPAN = 8  # displacement classes per axis: -7 to 7
MOVES = "UDLR"


def matched_expectation(pool, steps):
    """Mean accuracy over `pool[(size, k)]` at each step's size and nearest k."""
    values = []
    for step in steps:
        ks = [k for size, k in pool if size == step["size"]]
        if ks:
            k = min(ks, key=lambda k: (abs(k - step["k"]), -k))
            values.append(sum(pool[step["size"], k]) / len(pool[step["size"], k]))
    return sum(values) / len(values) if values else None


def displacement(step, starts):
    start = starts[step["map"]]
    return step["cell"][0] - start[0], step["cell"][1] - start[1]


def displacement_targets(steps, starts):
    y = torch.zeros(len(steps), 2 * (2 * SPAN - 1))
    for i, step in enumerate(steps):
        dr, dc = displacement(step, starts)
        y[i, dr + SPAN - 1] = y[i, 2 * SPAN - 1 + dc + SPAN - 1] = 1
    return y


def decoded_displacement(scores):
    width = 2 * SPAN - 1
    return [(int(s[:width].argmax()) - SPAN + 1, int(s[width:].argmax()) - SPAN + 1) for s in scores]


def mlp_cells(x_fit, fit_steps, x_eval, eval_steps, epochs=300):
    torch.manual_seed(SEED)
    mean, std = x_fit.mean(0), x_fit.std(0).clamp(min=1e-6)
    x_fit, x_eval = (x_fit - mean) / std, (x_eval - mean) / std
    rows = torch.tensor([s["cell"][0] for s in fit_steps])
    cols = torch.tensor([s["cell"][1] for s in fit_steps])
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(x_fit.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(256, 16))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    for _ in range(epochs):
        out = net(x_fit)
        loss = torch.nn.functional.cross_entropy(out[:, :8], rows) + torch.nn.functional.cross_entropy(out[:, 8:], cols)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        out = net(x_eval)
    return sum([int(o[:s["size"]].argmax()), int(o[8:8 + s["size"]].argmax())] == s["cell"]
               for o, s in zip(out, eval_steps)) / len(eval_steps)


def main(run):
    run = Path(run)
    features = torch.load(run / "features.pt", weights_only=False)
    steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    headroom = ([json.loads(line) for line in (run / "headroom.jsonl").read_text().splitlines()]
                if (run / "headroom.jsonl").exists() else [])  # none for a state probe alone
    layers = [layer + 1 for layer in features["layers"]]
    train = features["train"]
    starts = {name: {s["map"]: s["cell"] for s in features[name]["steps"] if s["k"] == 1}
              for name in ("train", "test", "fresh") if name in features}
    maps = sorted({s["map"] for s in train["steps"]})
    held = set(random.Random(SEED).sample(maps, round(HELD_OUT * len(maps))))
    fit = [i for i, s in enumerate(train["steps"]) if s["map"] not in held]
    held_out = [i for i, s in enumerate(train["steps"]) if s["map"] in held]
    later = lambda index: [i for i in index if train["steps"][i]["k"] >= 2]
    report = {"layers": {}, "splits": {}}

    for position, layer in enumerate(layers):
        x = train["states"][:, position].float()
        moves = torch.tensor([MOVES.index(s["move"]) for s in train["steps"]])
        predicted = ridge(x[fit], torch.nn.functional.one_hot(moves[fit], 4).double(), x[held_out]).argmax(-1)
        first = [i for i in held_out if train["steps"][i]["k"] == 1]
        cells = decoded_cells(ridge(x[fit], position_targets([train["steps"][i] for i in fit]), x[first]),
                              [train["steps"][i] for i in first])
        fit_later, held_later = later(fit), later(held_out)
        decoded = decoded_displacement(ridge(x[fit_later], displacement_targets(
            [train["steps"][i] for i in fit_later], starts["train"]), x[held_later]))
        by_k = defaultdict(list)
        for i, d in zip(held_later, decoded):
            step = train["steps"][i]
            by_k["k=2" if step["k"] == 2 else "k>=3"].append(d == displacement(step, starts["train"]))
        entry = {"next_move": float((predicted == moves[held_out]).double().mean()),
                 "start_cell_at_first_move": sum(c == train["steps"][i]["cell"] for c, i in zip(cells, first)) / len(first),
                 "displacement": {k: sum(v) / len(v) for k, v in by_k.items()}}
        if layer in (16, 20, 28):
            entry["absolute_cell_mlp"] = mlp_cells(x[fit_later], [train["steps"][i] for i in fit_later], x[held_later],
                                                   [train["steps"][i] for i in held_later])
        report["layers"][str(layer)] = entry
        print(f"layer {layer}", json.dumps(entry), flush=True)

    position = layers.index(20)
    x = train["states"][:, position].float()
    every_later = later(range(len(train["steps"])))
    weights_target = displacement_targets([train["steps"][i] for i in every_later], starts["train"])
    for name in ("test", "fresh"):
        if name not in features:
            continue
        data = features[name]
        rows = [s for s in steps if s["split"] == name]
        if len(rows) != len(data["steps"]) or any(a["row_index"] != b["row_index"] or a["k"] != b["k"]
                                                  for a, b in zip(rows, data["steps"])):
            raise ValueError(f"steps.jsonl and features.pt disagree on {name}")
        index = [i for i, s in enumerate(data["steps"]) if s["k"] >= 2]
        decoded = decoded_displacement(ridge(x[every_later], weights_target, data["states"][index, position].float()))
        tally = {i: d == displacement(data["steps"][i], starts[name]) for i, d in zip(index, decoded)}
        solved_tally = [tally[i] for i in index if data["steps"][i]["solved"]]
        fatal_tally = [tally[i] for i in index if data["steps"][i]["fatal"]]
        # Position probe (summary.json's layer) at different kinds of steps.
        pool, near = defaultdict(list), defaultdict(list)
        for row, step in zip(rows, data["steps"]):
            if row["solved"] and row["k"] >= 2:
                ok = row["decoded_cell"] == row["cell"]
                pool[row["size"], row["k"]].append(ok)
                if 1.0 in step["neighbours"]:
                    near[row["size"], row["k"]].append(ok)
        fatal = [r for r in rows if r["fatal"] and r["k"] >= 2]
        earlier = [r for r in rows if not r["solved"] and not r["fatal"] and r["k"] >= 2]
        accuracy = lambda group: sum(r["decoded_cell"] == r["cell"] for r in group) / len(group) if group else None
        taken = lambda r: r["hole_scores"][MOVES.index(r["move"])]
        safe_taken = [taken(r) for r in rows if r["solved"]]
        fatal_taken = [taken(r) for r in rows if r["fatal"]]
        report["splits"][name] = {
            "displacement_layer_20": {"solved_steps": sum(solved_tally) / len(solved_tally),
                                      "fatal_steps": f"{sum(fatal_tally)}/{len(fatal_tally)}"},
            "position_probe": {
                "fatal_steps": accuracy(fatal), "fatal_expected": matched_expectation(pool, fatal),
                "earlier_steps_of_failed_routes": accuracy(earlier),
                "earlier_steps_expected": matched_expectation(pool, earlier),
                "fatal_expected_from_solved_steps_next_to_a_hole": matched_expectation(near, fatal)},
            "hole_ahead_direction_taken": {
                "fatal_read_as_hole": sum(v > 0.5 for v in fatal_taken) / len(fatal_taken),
                "safe_read_as_hole": sum(v > 0.5 for v in safe_taken) / len(safe_taken),
                "fatal_steps": len(fatal_taken), "safe_steps": len(safe_taken)}}
        print(name, json.dumps(report["splits"][name]), flush=True)

    bounds = {}
    for radius in sorted({r["radius"] for m in headroom for r in m["rows"]}):
        for kind in ("target", "control"):
            cells = [r for m in headroom for r in m["rows"] if r["radius"] == radius and r["kind"] == kind]
            bounds[f"{radius}/{kind}"] = {"rows": len(cells), "on_bound": sum(max(r["norms"]) >= 0.99 * radius for r in cells)}
    if headroom:
        report["headroom_rows_on_norm_bound"] = bounds
        report["headroom_median_nll"] = {
            "target": sorted(m["target_nll"] for m in headroom)[len(headroom) // 2],
            "control": sorted(m["control_nll"] for m in headroom if m["control_nll"] is not None)[len(headroom) // 2]}
    (run / "exploratory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
