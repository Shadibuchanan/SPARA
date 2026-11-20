"""Offline simulation of C‴'s heads on C″'s recorded rollouts (CPU).

Heads read LIVR's layer-24 latent states (a proxy: the online z drifts with the policy) and
are trained with the latent policy-gradient term alone, step by step in C″'s order, on one
seed's mixed groups. Held out: the other seed's groups on maps the training seed never saw.

Readings per configuration:
- sigma: mean, and Spearman of each held-out map's mean sigma against its observed success
  rate (negative = shrinks on easy maps);
- held-out objective change: mean of sum_i A_i log p(c_i) over held-out groups, minus at init;
- |mu|: mean absolute, and held-out AUC of rollouts' noise along mu (0.5 = no signal);
- stability: any nonfinite value, share of sigma at either bound.
"""
import collections
import json
import math
import sys

import numpy as np
import torch

torch.set_num_threads(8)
R = "/storage/ice1/5/6/psomu3/sft-run/results"
SCALE = 0.25
states = torch.load(f"{R}/latent-diag-sweep-L16-24-7148362/states.pt", map_location="cpu")
Z = states["states"][24].float()  # (2000, 4, 2048), indexed by pool row
assert states["rows"] == list(range(2000))


def rms(z):
    return z * z.square().mean(-1, keepdim=True).add(1e-6).rsqrt()


def load(seed):
    steps = collections.defaultdict(list)
    for line in open(f"{R}/fair-n-seed{seed}-7154355/rl/rollouts.jsonl"):
        g = json.loads(line)
        r = torch.tensor(g["rewards"])
        if r.std() == 0:
            continue
        c = SCALE * torch.tensor(g["eps"])  # (8, 4, 16) coefficients in the matched basis
        a = (r - r.mean()) / (r.std() + 1e-4)
        steps[g["step"]].append((g["row"], c, a, r))
    return steps


class Heads(torch.nn.Module):
    """Per slot: mu (bounded by tanh) and log sigma (smoothly bounded) of 16 coefficients."""

    def __init__(self, hidden, sigma0, lo, hi, mu_bound, slots=4, d=2048, k=16):
        super().__init__()
        self.hidden, self.mu_bound = hidden, mu_bound
        self.mid, self.half = (math.log(lo) + math.log(hi)) / 2, (math.log(hi) - math.log(lo)) / 2
        self.offset = math.atanh((math.log(sigma0) - self.mid) / self.half)
        if hidden:
            self.mean_in, self.sig_in = torch.nn.Linear(d, hidden), torch.nn.Linear(d, hidden)
            for layer in (self.mean_in, self.sig_in):
                torch.nn.init.normal_(layer.weight, std=0.02); torch.nn.init.zeros_(layer.bias)
            width = hidden
        else:
            width = d
        self.mean_out = torch.nn.Parameter(torch.zeros(slots, width, k))
        self.sig_out = torch.nn.Parameter(torch.zeros(slots, width, k))
        self.mean_bias = torch.nn.Parameter(torch.zeros(slots, k))
        self.sig_bias = torch.nn.Parameter(torch.zeros(slots, k))

    def forward(self, z):
        if self.hidden:
            hm, hs = torch.nn.functional.silu(self.mean_in(z)), torch.nn.functional.silu(self.sig_in(z))
        else:
            hm = hs = z
        raw_mu = torch.einsum("nsh,shk->nsk", hm, self.mean_out) + self.mean_bias
        raw_sig = torch.einsum("nsh,shk->nsk", hs, self.sig_out) + self.sig_bias
        mu = self.mu_bound * torch.tanh(raw_mu / self.mu_bound) if self.mu_bound else torch.zeros_like(raw_mu)
        log_sigma = self.mid + self.half * torch.tanh(raw_sig + self.offset)
        return mu, log_sigma


def log_prob(c, mu, log_sigma):  # c (G, 4, 16); mu, log_sigma (1, 4, 16); sum over coordinates
    return (-0.5 * ((c - mu) / log_sigma.exp()).square() - log_sigma).sum((-1, -2))


def objective(heads, groups):
    total = 0.0
    with torch.no_grad():
        for row, c, a, r in groups:
            mu, ls = heads(rms(Z[row:row + 1]))
            total += float((a * log_prob(c, mu, ls)).mean())
    return total / len(groups)


def run(train_seed, cfg):
    torch.manual_seed(0)
    train, test = load(train_seed), load(85 - train_seed)
    seen = {g[0] for groups in train.values() for g in groups}
    held = [g for groups in test.values() for g in groups if g[0] not in seen]
    heads = Heads(cfg["hidden"], SCALE, cfg["lo"], cfg["hi"], cfg["mu_bound"])
    opt = torch.optim.AdamW(heads.parameters(), lr=cfg["lr"], weight_decay=0.01)
    base = objective(heads, held)
    trace = []
    for step in range(1, 126):
        groups = train.get(step, [])
        if not groups:
            continue
        opt.zero_grad()
        loss = 0.0
        n = sum(len(g[2]) for g in groups)
        for row, c, a, r in groups:
            mu, ls = heads(rms(Z[row:row + 1]))
            loss = loss - (a * log_prob(c, mu, ls)).sum() / n
        loss.backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                mu, ls = heads(rms(Z[[g[0] for g in held]]))
            trace.append((step, float(ls.exp().mean()), float(mu.abs().mean())))
    with torch.no_grad():
        rows = [g[0] for g in held]
        mu, ls = heads(rms(Z[rows]))
    sig = ls.exp()
    finite = bool(torch.isfinite(sig).all() and torch.isfinite(mu).all())
    # Per held-out map: mean sigma vs observed success rate in the held-out seed.
    per = collections.defaultdict(list)
    for i, g in enumerate(held):
        per[g[0]].append((float(sig[i].mean()), float(g[3].mean())))
    xs = np.array([np.mean([p[0] for p in v]) for v in per.values()])
    ys = np.array([np.mean([p[1] for p in v]) for v in per.values()])
    rank = lambda v: np.argsort(np.argsort(v))
    spearman = float(np.corrcoef(rank(xs), rank(ys))[0, 1]) if xs.std() > 0 else float("nan")
    # Does noise along mu predict success on held-out groups?
    aucs = []
    for i, g in enumerate(held):
        if float(mu[i].abs().sum()) == 0:
            continue
        s = (g[1] * mu[i]).sum((-1, -2))
        win, lose = s[g[3] > 0], s[g[3] == 0]
        aucs.append(float((win[:, None] > lose[None, :]).float().mean()))
    at_bound = float(((ls - math.log(cfg["lo"])).abs() < 0.05).float().mean() +
                     ((ls - math.log(cfg["hi"])).abs() < 0.05).float().mean())
    return {"held_groups": len(held), "held_maps": len(per), "objective_change": objective(heads, held) - base,
            "sigma_mean": float(sig.mean()), "sigma_spread_across_maps": float(xs.std()),
            "spearman_sigma_vs_success": spearman, "mu_abs": float(mu.abs().mean()),
            "mu_auc": float(np.mean(aucs)) if aucs else None, "finite": finite, "at_bound": at_bound,
            "trace": trace}


if __name__ == "__main__":
    configs = json.loads(sys.argv[1])
    for cfg in configs:
        for seed in (42, 43):
            out = run(seed, cfg)
            print(json.dumps({"cfg": cfg, "train_seed": seed} | out), flush=True)
