"""CPU post-hoc check of a latent JEPA run (docs/PERCEPTION_JEPA_SPEC.md): what do the predictions carry?

From saved files only: a run's latent states (`states.pt`, layer 12, unmasked) and a heads file
(`jepa_heads.pt`). No model, no GPU, and no cell targets are needed: the check is on the
predictions, which come from the latents alone.

For each grid size, on the held-out maps:
- Variance split: the share of the predicted cell embeddings' variance that is within a position
  (differs between maps at the same cell) rather than between positions. A predictor that ignores
  the map's latents gives position codes, and a within-position share near zero.
- Holes in the predictions: a ridge probe from a cell's predicted embedding (minus that position's
  mean prediction) to hole or not, fitted on pool maps and scored on held-out maps.

With heads that carry fixed targets (EJF) and `--data-root` (to find each map's ID), also: the
shuffled gap and the share of the targets' hole direction the predictions explain
(`perception_jepa.dependence` and `hole_direction_share`), on the held-out maps.

Usage: python scripts/analyze_jepa_predictions.py STATES HEADS [--layer 12] [--data-root data/vsp] [--output PATH]
"""
import argparse
import json
from pathlib import Path

import torch

from spara.data import records
from spara.expert_round2 import ridge
from spara.perception_jepa import balanced_accuracy, dependence, hole_direction_share, load_heads, map_key


@torch.no_grad()
def predictions(heads, states, layer, population):
    index = states["layers"].index(layer)
    by_size = {}
    for k, item in enumerate(states["items"]):
        if item["population"] != population:
            continue
        n = len(item["map_desc"])
        predicted = heads.predict(heads.context(states["states"][k, index].float()), n)
        holes = torch.tensor([float(v == -1) for row in item["map_desc"] for v in row])
        by_size.setdefault(n, []).append((predicted, holes))
    return by_size


def fixed_target_readings(heads, saved, states, layer, data_root):
    """Per size on held-out maps: the shuffled gap and the hole-direction share, against the stored targets
    (keyed by `perception_jepa.map_key`)."""
    index = states["layers"].index(layer)
    tables, keep = {}, []
    for k, item in enumerate(states["items"]):
        if item["population"] in ("pool78", "heldout78"):
            table = tables.setdefault(item["split"], records(data_root, item["split"]))
            row = table[item["row_index"]]
            if row["map_desc"] != item["map_desc"]:
                raise ValueError(f"Layout mismatch at {item['split']} row {item['row_index']}")
            keep.append((k, item, map_key(item["split"], row)))
    with torch.no_grad():
        predicted = [heads.predict(heads.context(states["states"][k, index].float()), len(item["map_desc"]))
                     for k, item, _ in keep]
    targets = [saved["targets"][key].float() for _, _, key in keep]
    rows = [{"map_desc": item["map_desc"]} for _, item, _ in keep]
    pool = [i for i, (_, item, _) in enumerate(keep) if item["population"] == "pool78"]
    held = [i for i, (_, item, _) in enumerate(keep) if item["population"] == "heldout78"]
    shares = hole_direction_share(predicted, targets, rows, pool, held)
    out = {}
    for n in sorted({len(rows[i]["map_desc"]) for i in held}):
        ids = [i for i in held if len(rows[i]["map_desc"]) == n]
        found = dependence(lambda i, j: predicted[ids[j]], [targets[i] for i in ids], [n] * len(ids))
        out[str(n)] = found | {"hole_direction_share": shares[str(n)]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states")
    parser.add_argument("heads")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--data-root", help="with fixed-target heads: the data, to find each map's ID")
    parser.add_argument("--output")
    args = parser.parse_args()
    heads, _, saved = load_heads(args.heads)
    heads.eval()
    states = torch.load(args.states)
    fit = predictions(heads, states, args.layer, "pool78")
    test = predictions(heads, states, args.layer, "heldout78")
    report = {"states": args.states, "heads": args.heads, "layer": args.layer, "by_size": {}}
    for n in sorted(test):
        p = torch.stack([x for x, _ in test[n]])  # [maps, cells, dim]
        position_mean = p.mean(0, keepdim=True)
        total = (p - p.mean((0, 1), keepdim=True)).square().sum()
        within = (p - position_mean).square().sum()
        fit_p = torch.stack([x for x, _ in fit[n]])
        fit_x = (fit_p - fit_p.mean(0, keepdim=True)).flatten(0, 1)
        test_x = (p - fit_p.mean(0, keepdim=True)).flatten(0, 1)
        fit_y = torch.cat([h for _, h in fit[n]])[:, None]
        truth = torch.cat([h for _, h in test[n]]).long()
        scores, _ = ridge(fit_x, fit_y, test_x)
        report["by_size"][str(n)] = {"maps": len(p), "within_position_share": float(within / total),
                                     "hole_balanced_accuracy": balanced_accuracy((scores[:, 0] > 0.5).long(), truth)}
    if saved.get("targets") is not None and args.data_root:
        for n, values in fixed_target_readings(heads, saved, states, args.layer, args.data_root).items():
            report["by_size"][n] |= values
    print(json.dumps(report["by_size"], indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
