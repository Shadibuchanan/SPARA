"""CPU analysis of a latent battery run (`spara battery`): no GPU or model needed.

Reads features-train.pt, features-test.pt and summary.json from the run directory
and writes probes.json there.

- Collapse: within each grid size, the mean pairwise cosine similarity of test
  latents (four slots concatenated), raw and after removing that size's mean,
  and the participation ratio of the centred latents, raw and with each
  channel z-scored (at most one less than that size's number of test maps).
- Content: ridge probes fitted on training maps of one grid size and scored on
  that size's test maps, for each attention mode and probe layer, from the
  latents and from the last prompt token. Targets: hole or not at every cell,
  the goal cell, the start cell, the shortest route's length, and the first
  move (scored correct when it begins some shortest route). Ridge strength is
  chosen per fit by leave-one-out error on the training maps.
- Cell distinguishability (whether a per-step JEPA target can identify a cell):
  each safe cell's most similar other cell in the same test image, and a ridge
  probe from a safe cell's image feature to its row and column.

Usage: python scripts/analyze_latent_battery.py RUN_DIR
"""
import json
import sys
from collections import deque
from pathlib import Path

import torch

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
WORDS = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}


def distances(grid):
    """Steps from each cell to the goal through non-hole cells (None if cut off)."""
    n = len(grid)
    goal = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 2)
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] != -1 and (nr, nc) not in dist:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def targets(grid, text_output=None):
    n = len(grid)
    cells = [(r, c) for r in range(n) for c in range(n)]
    start = next(cell for cell in cells if grid[cell[0]][cell[1]] == 1)
    dist = distances(grid)
    best = {m for m, (dr, dc) in MOVES.items()
            if dist.get((start[0] + dr, start[1] + dc), 1e9) == dist[start] - 1}
    gold = None
    if text_output:
        first = text_output.split("{")[-1].split(",")[0].strip().upper()
        gold = WORDS.get(first, first[:1])
    return {"holes": [float(grid[r][c] == -1) for r, c in cells],
            "goal": cells.index(next(x for x in cells if grid[x[0]][x[1]] == 2)),
            "start": cells.index(start), "length": float(dist[start]),
            "best_moves": best, "first_move": gold}


def ridge(train_x, train_y, test_x):
    """Ridge on standardised features via a thin SVD; strength by leave-one-out error.

    The penalty is `alpha` times the mean squared singular value per training
    row (the mean eigenvalue of x x^T), so it scales with the features.
    """
    train_x, train_y, test_x = train_x.double(), train_y.double(), test_x.double()
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    x, z = (train_x - mean) / std, (test_x - mean) / std
    y_mean = train_y.mean(0)
    y = train_y - y_mean
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    s2, uy = s.square(), u.T @ y
    best = None
    for alpha in ALPHAS:
        lam = alpha * s2.sum() / len(x)
        shrink = s2 / (s2 + lam)
        leverage = (u.square() * shrink).sum(-1)
        loo = ((y - u @ (shrink[:, None] * uy)) / (1 - leverage).clamp(min=1e-9)[:, None]).square().mean()
        if best is None or loo < best[0]:
            best = (loo, lam, alpha)
    coef = vh.T @ ((s / (s2 + best[1]))[:, None] * uy)
    return z @ coef + y_mean, best[2]


def balanced_accuracy(pred, truth):
    scores = [(pred[truth == k] == k).float().mean() for k in (0, 1) if (truth == k).any()]
    return float(sum(scores) / len(scores))


def probe(train_x, test_x, train_t, test_t, n):
    holes, alpha_holes = ridge(train_x, torch.tensor([t["holes"] for t in train_t]), test_x)
    truth = torch.tensor([t["holes"] for t in test_t])
    per_cell = [balanced_accuracy((holes[:, k] > 0.5).long(), truth[:, k].long()) for k in range(n * n)
                if 0 < truth[:, k].sum() < len(truth)]
    result = {"hole_balanced_accuracy": sum(per_cell) / len(per_cell), "alpha_holes": alpha_holes}
    for key in ("goal", "start"):
        onehot = torch.nn.functional.one_hot(torch.tensor([t[key] for t in train_t]), n * n).double()
        pred, _ = ridge(train_x, onehot, test_x)
        result[f"{key}_accuracy"] = float((pred.argmax(-1) == torch.tensor([t[key] for t in test_t])).double().mean())
    length, _ = ridge(train_x, torch.tensor([[t["length"]] for t in train_t]), test_x)
    truth_length = torch.tensor([t["length"] for t in test_t])
    residual = (length[:, 0] - truth_length).square().sum()
    result["length_r2"] = float(1 - residual / (truth_length - truth_length.mean()).square().sum())
    order = list(MOVES)
    onehot = torch.nn.functional.one_hot(torch.tensor([order.index(t["first_move"]) for t in train_t]), 4).double()
    move, _ = ridge(train_x, onehot, test_x)
    result["first_move_accuracy"] = sum(order[k] in t["best_moves"] for k, t in zip(move.argmax(-1).tolist(), test_t)) / len(test_t)
    return result


def chance(test_t, n):
    goal = max(sum(t["goal"] == k for t in test_t) for k in range(n * n)) / len(test_t)
    start = max(sum(t["start"] == k for t in test_t) for k in range(n * n)) / len(test_t)
    move = max(sum(m in t["best_moves"] for t in test_t) for m in MOVES) / len(test_t)
    return {"hole_balanced_accuracy": 0.5, "goal_accuracy": goal, "start_accuracy": start,
            "length_r2": 0.0, "first_move_accuracy": move}


def collapse(latents):
    flat = latents.reshape(len(latents), -1).double()
    def mean_cosine(x):
        unit = torch.nn.functional.normalize(x, dim=-1)
        sims = unit @ unit.T
        return float((sims.sum() - sims.diagonal().sum()) / (len(x) * (len(x) - 1)))
    def ratio(x):
        evals = torch.linalg.eigvalsh(x @ x.T).clamp(min=0)
        return float(evals.sum() ** 2 / evals.square().sum())
    centred = flat - flat.mean(0)
    # A few outlier channels dominate raw hidden-state variance; z-scoring each
    # channel shows the dimensionality the (standardised) probes see.
    standardised = centred / centred.std(0).clamp(min=1e-6)
    return {"cosine": mean_cosine(flat), "centred_cosine": mean_cosine(centred),
            "participation_ratio": ratio(centred), "standardised_participation_ratio": ratio(standardised)}


def distinguishability(cells, grids, layer, train_cells, train_grids):
    """Safe-cell uniqueness within each test image, and row/column decodability."""
    nearest = []
    for feature, grid in zip(cells, grids):
        n = len(grid)
        safe = [k for k in range(n * n) if grid[k // n][k % n] == 0]
        unit = torch.nn.functional.normalize(feature[:, layer].double(), dim=-1)
        sims = unit @ unit.T
        sims.fill_diagonal_(-2)
        nearest += sims[safe].max(-1).values.tolist()
    nearest.sort()
    result = {"median_nearest_cosine": nearest[len(nearest) // 2],
              "share_nearest_above_0.99": sum(x > 0.99 for x in nearest) / len(nearest)}
    for size in sorted({len(g) for g in grids}):
        def gather(features, maps):
            xs, rows, cols = [], [], []
            for feature, grid in zip(features, maps):
                if len(grid) == size:
                    for k in range(size * size):
                        if grid[k // size][k % size] == 0:
                            xs.append(feature[k, layer].double()); rows.append(k // size); cols.append(k % size)
            return torch.stack(xs), torch.tensor(rows), torch.tensor(cols)
        tx, tr, tc = gather(train_cells, train_grids)
        vx, vr, vc = gather(cells, grids)
        hits = 1.0
        for train_label, test_label in ((tr, vr), (tc, vc)):
            pred, _ = ridge(tx, torch.nn.functional.one_hot(train_label, size).double(), vx)
            hits *= float((pred.argmax(-1) == test_label).double().mean())
        result[f"position_accuracy_{size}x{size}"] = hits  # row accuracy x column accuracy
    return result


def main(run):
    run = Path(run)
    train, test = torch.load(run / "features-train.pt"), torch.load(run / "features-test.pt")
    layers = train["layers"]
    train_t = [targets(r["map_desc"], r["text_output"]) for r in train["rows"]]
    test_t = [targets(r["map_desc"]) for r in test["rows"]]
    sizes = sorted({len(r["map_desc"]) for r in test["rows"]})
    by_size = lambda rows, size: [i for i, r in enumerate(rows) if len(r["map_desc"]) == size]
    report = {"layers": layers, "chance": {}, "probes": {}, "collapse": {}, "cells": {}}
    for size in sizes:
        report["chance"][str(size)] = chance([test_t[i] for i in by_size(test["rows"], size)], size)
    for mode in train["latent"]:
        for feature in ("latent", "last"):
            for position, layer in enumerate(layers):
                key = f"{mode}/{feature}/layer{layer + 1}"
                report["probes"][key] = {}
                for size in sizes:
                    tr, te = by_size(train["rows"], size), by_size(test["rows"], size)
                    x = train[feature][mode][:, position].double().reshape(len(train["rows"]), -1)
                    z = test[feature][mode][:, position].double().reshape(len(test["rows"]), -1)
                    report["probes"][key][str(size)] = probe(x[tr], z[te], [train_t[i] for i in tr],
                                                              [test_t[i] for i in te], size)
                print(key, json.dumps({s: {k: round(v, 3) for k, v in r.items() if not k.startswith("alpha")}
                                       for s, r in report["probes"][key].items()}), flush=True)
        for position, layer in enumerate(layers):
            latents = test["latent"][mode][:, position]
            groups = [collapse(latents[by_size(test["rows"], size)]) for size in sizes]
            report["collapse"][f"{mode}/layer{layer + 1}"] = {k: sum(g[k] for g in groups) / len(groups) for k in groups[0]}
    grids, train_grids = [r["map_desc"] for r in test["rows"]], [r["map_desc"] for r in train["rows"]]
    for position, layer in enumerate(layers):
        report["cells"][f"layer{layer + 1}"] = distinguishability(test["cells"], grids, position,
                                                                   train["cells"], train_grids)
    (run / "probes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"collapse": report["collapse"], "cells": report["cells"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
