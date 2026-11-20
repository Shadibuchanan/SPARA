"""Is there a learnable latent direction in C''s rollouts? For mixed groups, fit the
advantage-weighted mean noise direction on one part and test on the other: within each held-out
group, do rollouts whose noise points along that direction succeed more often (AUC; 0.5 = none)?"""
import json
import sys

import numpy as np

R = "/storage/ice1/5/6/psomu3/sft-run/results"
POOL = "/storage/ice1/5/6/psomu3/sft-run/data/vsp/fresh/detour-pool/maps.jsonl"
sizes = [len(json.loads(l)["map_desc"]) for l in open(POOL)]


def load(seed):
    out = []
    for line in open(f"{R}/fair-n-seed{seed}-7154355/rl/rollouts.jsonl"):
        g = json.loads(line)
        r = np.array(g["rewards"])
        if r.std() == 0:
            continue
        out.append({"step": g["step"], "row": g["row"], "size": sizes[g["row"]],
                    "eps": np.array(g["eps"]).reshape(len(r), -1), "r": r})
    return out


def direction(groups):
    total = np.zeros(groups[0]["eps"].shape[1])
    for g in groups:
        a = (g["r"] - g["r"].mean()) / g["r"].std()
        total += (a[:, None] * g["eps"]).mean(0)
    return total / len(groups)


def auc(groups, d):
    vals = []
    for g in groups:
        s = g["eps"] @ d
        win, lose = s[g["r"] > 0], s[g["r"] == 0]
        vals.append(((win[:, None] > lose[None, :]).mean() + 0.5 * (win[:, None] == lose[None, :]).mean()))
    return float(np.mean(vals))


def null(groups, train, rng, n=200):
    # Direction from train groups with rewards shuffled within each group.
    out = []
    for _ in range(n):
        shuffled = [{**g, "r": rng.permutation(g["r"])} for g in train]
        out.append(auc(groups, direction(shuffled)))
    return np.array(out)


rng = np.random.default_rng(0)
data = {42: load(42), 43: load(43)}
print("mixed groups:", {k: len(v) for k, v in data.items()})
splits = {"train 42 -> test 43": (data[42], data[43]), "train 43 -> test 42": (data[43], data[42])}
both = data[42] + data[43]
early = [g for g in both if g["step"] <= 62]
late = [g for g in both if g["step"] > 62]
splits["steps 1-62 -> 63-125"] = (early, late)
rows = sorted({g["row"] for g in both})
half = set(rng.permutation(rows)[: len(rows) // 2].tolist())
splits["maps half A -> half B"] = ([g for g in both if g["row"] in half], [g for g in both if g["row"] not in half])
for name, (train, test) in splits.items():
    d = direction(train)
    a = auc(test, d)
    nl = null(test, train, rng)
    print(f"global  {name:24s} AUC {a:.3f}  null mean {nl.mean():.3f}, 97.5% {np.quantile(nl, .975):.3f}  "
          f"p~{(nl >= a).mean():.3f}  |d| {np.linalg.norm(d):.3f}")
    # Size-conditional direction: a crude map-dependent mean.
    per = {}
    for s in (3, 4, 5, 6):
        tr = [g for g in train if g["size"] == s]
        te = [g for g in test if g["size"] == s]
        if len(tr) > 10 and len(te) > 10:
            per[s] = (auc(te, direction(tr)), len(te))
    print("   by size:", {s: f"{v[0]:.3f} (n={v[1]})" for s, v in per.items()})
