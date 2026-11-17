"""CPU check: do LIVR's image tokens show the holes? (docs/PERCEPTION_JEPA_SPEC.md, gate 1)

Reads a latent battery run's features (`spara battery`: features-train.pt and
features-test.pt). Their `cells` hold, for every map, each cell's overlap-weighted mean
of the image-token states at the probe layers (unmasked mode; image tokens precede the
latents, so the bottleneck mask leaves them unchanged). No model or GPU is needed.

For each probe layer:
- Hole probe: one ridge probe per grid size, from a cell's feature to hole or not,
  fitted on the training maps' cells of that size and scored on the test maps' cells
  (balanced accuracy). A pooled probe over all sizes is reported too.
- Position after centring (descriptive): subtract the training mean feature of each
  (size, row, column), then decode the cell's row and column (row accuracy x column
  accuracy, all cells). This measures how much position a learned target projection
  could still find once the per-position mean is removed.

Ridge strength is chosen by leave-one-out error, as scripts/analyze_latent_battery.py.

Usage: python scripts/probe_image_token_holes.py RUN_DIR [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_latent_battery import balanced_accuracy, ridge  # noqa: E402

SEPARATES = 0.95  # gate 1: holes are visible to the image tokens at this balanced accuracy


def cell_rows(features, grids, layer, sizes):
    """Stacked cell features at one probe layer, with size, row, column and hole labels."""
    xs, size, row, col, hole = [], [], [], [], []
    for feature, grid in zip(features, grids):
        n = len(grid)
        if n not in sizes:
            continue
        xs.append(feature[:, layer].double())
        for k in range(n * n):
            size.append(n); row.append(k // n); col.append(k % n); hole.append(float(grid[k // n][k % n] == -1))
    return (torch.cat(xs), torch.tensor(size), torch.tensor(row), torch.tensor(col), torch.tensor(hole))


def hole_probe(train, test):
    pred, alpha = ridge(train[0], train[4][:, None], test[0])
    predicted, truth = (pred[:, 0] > 0.5).long(), test[4].long()
    return {"balanced_accuracy": balanced_accuracy(predicted, truth),
            "miss_rate": float((predicted[truth == 1] == 0).double().mean()),
            "false_alarm_rate": float((predicted[truth == 0] == 1).double().mean()),
            "test_cells": len(truth), "test_holes": int(truth.sum()), "alpha": alpha}


def centred_position(train, test, n):
    keys_train = train[2] * n + train[3]
    keys_test = test[2] * n + test[3]
    means = torch.stack([train[0][keys_train == k].mean(0) for k in range(n * n)])
    tx, vx = train[0] - means[keys_train], test[0] - means[keys_test]
    hits = 1.0
    for train_label, test_label in ((train[2], test[2]), (train[3], test[3])):
        pred, _ = ridge(tx, torch.nn.functional.one_hot(train_label, n).double(), vx)
        hits *= float((pred.argmax(-1) == test_label).double().mean())
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output")
    args = parser.parse_args()
    run = Path(args.run)
    train, test = torch.load(run / "features-train.pt"), torch.load(run / "features-test.pt")
    train_grids = [r["map_desc"] for r in train["rows"]]
    test_grids = [r["map_desc"] for r in test["rows"]]
    sizes = sorted({len(g) for g in test_grids})
    report = {"source": str(run), "layers": [layer + 1 for layer in train["layers"]],
              "threshold": SEPARATES, "holes": {}, "centred_position": {}}
    for index, layer in enumerate(train["layers"]):
        name = f"layer{layer + 1}"
        report["holes"][name], report["centred_position"][name] = {}, {}
        for size in sizes:
            tr = cell_rows(train["cells"], train_grids, index, {size})
            te = cell_rows(test["cells"], test_grids, index, {size})
            report["holes"][name][str(size)] = hole_probe(tr, te)
            report["centred_position"][name][str(size)] = centred_position(tr, te, size)
        pooled_train = cell_rows(train["cells"], train_grids, index, set(sizes))
        pooled_test = cell_rows(test["cells"], test_grids, index, set(sizes))
        report["holes"][name]["pooled"] = hole_probe(pooled_train, pooled_test)
        print(name, json.dumps({k: round(v["balanced_accuracy"], 4) for k, v in report["holes"][name].items()}),
              flush=True)
    best = {str(s): max(report["holes"][k][str(s)]["balanced_accuracy"] for k in report["holes"]) for s in sizes}
    passing = [k for k in report["holes"] if all(report["holes"][k][str(s)]["balanced_accuracy"] >= SEPARATES
                                                   for s in sizes)]
    report["best_by_size"] = best
    report["layers_passing_every_size"] = passing
    report["reading"] = ("image tokens separate holes" if passing
                         else "image tokens do not separate holes at the threshold")
    out = Path(args.output) if args.output else run / "image-token-holes.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"best_by_size": best, "layers_passing_every_size": passing, "reading": report["reading"]}),
          flush=True)


if __name__ == "__main__":
    main()
