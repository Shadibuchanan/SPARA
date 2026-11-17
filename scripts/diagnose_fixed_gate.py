"""Post hoc, CPU: why gate 3 of the fixed-target retry failed (docs/PERCEPTION_JEPA_SPEC.md).

Not a gate. From the features job's saved files and gate 3's heads:
1. How much E's latents vary between maps, against their squared norm, before and after the context
   projector's parameter-free layer norm.
2. Gate 3 rerun unchanged except that each latent channel is first standardised (per latent position,
   by the mean and standard deviation over E's training maps).
3. In calibrated units: per size, the targets' hole direction (hole minus ice mean target, on pool
   maps); the share of its variance on held-out maps that each predictor explains.

Usage: python scripts/diagnose_fixed_gate.py FEATURES_RUN GATE_RUN OUTPUT [--settings ...]
"""
import argparse
import json
from pathlib import Path

import torch

from spara.expert_round2 import ridge
from spara.perception_jepa import FEATURES_FILE, HEADS_FILE, hole_labels, load_heads, run_fixed_gate


def spread(contexts):
    """Per latent position: the across-map share of the squared norm, raw and after layer norm."""
    out = []
    for k in range(contexts.shape[1]):
        z = contexts[:, k]
        ln = torch.nn.functional.layer_norm(z, (z.shape[-1],))
        share = lambda x: float((x - x.mean(0)).square().sum(-1).mean() / x.square().sum(-1).mean())
        out.append({"latent": k, "across_map_share": share(z), "after_layer_norm": share(ln),
                    "largest_channel_mean": float(z.mean(0).abs().max()), "median_channel_std": float(z.std(0).median())})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_run")
    parser.add_argument("gate_run")
    parser.add_argument("output")
    parser.add_argument("--settings", default="configs/perception_jepa_fixed.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    features = torch.load(Path(args.features_run) / FEATURES_FILE, weights_only=False)
    keys = [tuple(k) for k in features["keys"]]
    index = {k: i for i, k in enumerate(keys)}
    groups = {n: [index[tuple(k)] for k in v] for n, v in features["groups"].items()}
    rows = [{"map_desc": d} for d in features["map_desc"]]
    sizes = [len(d) for d in features["map_desc"]]
    contexts = features["contexts"].float()
    held = contexts[groups["heldout"]]
    report = {"latent_spread_heldout": spread(held)}

    # 2. Gate 3 on standardised latents.
    train = groups["training"]
    mean, std = contexts[train].mean(0), contexts[train].std(0).clamp(min=1e-6)
    standard = (contexts - mean) / std
    report["standardised_latent_spread_heldout"] = spread(standard[groups["heldout"]])
    (output / "standardised").mkdir()
    torch.save(features | {"contexts": standard.half()}, output / "standardised" / FEATURES_FILE)
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    torch.manual_seed(settings["seed"])
    rerun = run_fixed_gate(output / "standardised", output / "standardised-gate", settings)
    report["standardised_gate"] = {k: rerun[k] for k in ("reading", "latent_dependence", "linear_ceiling")} | {
        "warm_start": {k: v for k, v in rerun["warm_start"].items() if k != "history"}}
    (output / "standardised" / FEATURES_FILE).unlink()  # a copy of the features; not kept

    # 3. The hole direction.
    _, _, saved = load_heads(Path(args.gate_run) / HEADS_FILE)
    targets = [saved["targets"][k].float() for k in keys]

    def predictions(path, x):
        heads, _, _ = load_heads(path)
        heads.eval()
        with torch.no_grad():
            return {i: heads.predict(heads.context(x[i]), sizes[i]) for i in set(groups["pool"]) | set(groups["heldout"])}

    predictors = {"gate_heads": predictions(Path(args.gate_run) / HEADS_FILE, contexts),
                  "standardised_heads": predictions(output / "standardised-gate" / HEADS_FILE, standard)}
    report["hole_direction"] = {}
    for n in sorted({sizes[i] for i in groups["heldout"]}):
        pool = [i for i in groups["pool"] if sizes[i] == n]
        test = [i for i in groups["heldout"] if sizes[i] == n]
        t_pool = torch.cat([targets[i] for i in pool])
        h_pool = torch.cat([hole_labels(rows[i]) for i in pool]).bool()
        d = t_pool[h_pool].mean(0) - t_pool[~h_pool].mean(0)
        d = d / d.norm()
        flat = lambda ids, values: torch.stack([values[i].flatten() for i in ids])
        scores, _ = ridge(flat(pool, contexts), flat(pool, targets), flat(test, contexts))
        ceiling = {i: s.float().view(n * n, -1) for i, s in zip(test, scores)}
        t = torch.cat([targets[i] for i in test]) @ d
        holes = torch.cat([hole_labels(rows[i]) for i in test]).bool()

        def share(values):
            p = torch.cat([values[i] for i in test]) @ d
            return float(1 - (t - p).square().sum() / (t - t.mean()).square().sum())
        by_class = torch.where(holes, t[holes].mean(), t[~holes].mean())
        report["hole_direction"][str(n)] = {
            "hole_label_share": float(1 - (t - by_class).square().sum() / (t - t.mean()).square().sum()),
            "gate_heads": share(predictors["gate_heads"]), "standardised_heads": share(predictors["standardised_heads"]),
            "linear_ceiling": share(ceiling)}
    (output / "diagnosis.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "standardised_gate"}, indent=1))
    print(json.dumps(report["standardised_gate"]["latent_dependence"], indent=1))


if __name__ == "__main__":
    main()
