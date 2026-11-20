"""(a) Can LIVR's layer-24 latents predict how easy a map is (C″'s success rate on it)?
(b) A low-variance sigma-only head: per slot, log sigma = log sigma0 + s, s = smooth-bounded
(w . P z + b), with P the top-r principal directions of the RMS-normalised latents, mu = 0, and a
penalty pulling sigma toward sigma0. Does it learn to shrink on easy held-out maps?"""
import collections
import json
import math
import sys

import numpy as np
import torch

torch.set_num_threads(8)
R = "/storage/ice1/5/6/psomu3/sft-run/results"
SCALE = 0.25
Z = torch.load(f"{R}/latent-diag-sweep-L16-24-7148362/states.pt", map_location="cpu")["states"][24].float()
Zn = Z * Z.square().mean(-1, keepdim=True).add(1e-6).rsqrt()  # (2000, 4, 2048)
X = Zn.reshape(2000, -1).double()


def groups(seed):
    out = []
    for line in open(f"{R}/fair-n-seed{seed}-7154355/rl/rollouts.jsonl"):
        g = json.loads(line)
        out.append((g["step"], g["row"], SCALE * torch.tensor(g["eps"]), torch.tensor(g["rewards"])))
    return out


data = {s: groups(s) for s in (42, 43)}
# (a) ease probe: per map success rate over both seeds, all groups.
succ = collections.defaultdict(list)
for s in data:
    for _, row, _, r in data[s]:
        succ[row] += r.tolist()
rows = sorted(succ)
y = np.array([np.mean(succ[r]) for r in rows])
rng = np.random.default_rng(0)
perm = rng.permutation(len(rows))
test_idx, train_idx = perm[: len(rows) // 5], perm[len(rows) // 5:]
Xr = X[rows]
mu_x = Xr[train_idx].mean(0)
U, S, Vt = torch.linalg.svd(Xr[train_idx] - mu_x, full_matrices=False)


def rank(v):
    return np.argsort(np.argsort(v))


for r_dim in ():
    P = Vt[:r_dim].T
    F = ((Xr - mu_x) @ P).numpy()
    for lam in (1e-2, 1, 1e2):
        A = F[train_idx]
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(r_dim), A.T @ (y[train_idx] - y[train_idx].mean()))
        pred = F[test_idx] @ w + y[train_idx].mean()
        rho = np.corrcoef(rank(pred), rank(y[test_idx]))[0, 1]
        r2 = 1 - ((pred - y[test_idx]) ** 2).sum() / ((y[test_idx] - y[test_idx].mean()) ** 2).sum()
        print(f"(a) ease probe r={r_dim:4d} ridge={lam:g}: held-out Spearman {rho:+.3f}, R2 {r2:+.3f} "
              f"(maps {len(test_idx)}; success rate sd {y.std():.3f})", flush=True)

# (b) sigma-only low-variance head on mixed groups, trained on one seed, held-out maps of the other.
PDIM = int(sys.argv[1]) if len(sys.argv) > 1 else 32
P = Vt[:PDIM].T.float()
Fall = ((X - mu_x) @ Vt[:PDIM].T.double()).float()  # (2000, PDIM), shared across slots
Fall = Fall / Fall[rows].std(0, keepdim=True)
LO, HI = math.log(0.1), math.log(2.0)  # sigma in [0.025, 0.5]


def bounded(s):
    return torch.where(s >= 0, HI * torch.tanh(s / HI), -LO * torch.tanh(s / -LO))


def run(train_seed, lr, beta, per_coord):
    torch.manual_seed(0)
    k = 16 if per_coord else 1
    W = torch.zeros(4, PDIM, k, requires_grad=True)
    b = torch.zeros(4, k, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    held_set = {rows[i] for i in test_idx}  # the probe's held-out 20% of maps
    train = [g for g in data[train_seed] if g[3].std() > 0 and g[1] not in held_set]
    held_rows = sorted(held_set)
    by_step = collections.defaultdict(list)
    for g in train:
        by_step[g[0]].append(g)
    for step in range(1, 126):
        gs = by_step.get(step)
        if not gs:
            continue
        opt.zero_grad()
        n = sum(len(g[3]) for g in gs)
        loss = 0
        for _, row, c, r in gs:
            a = (r - r.mean()) / (r.std() + 1e-4)
            s = bounded(torch.einsum("p,spk->sk", Fall[row], W) + b)  # (4, k)
            ls = math.log(SCALE) + s
            lp = (-0.5 * (c / ls.exp()).square() - ls).sum((-1, -2))  # mu = 0
            loss = loss - (a * lp).sum() / n
        # Penalty toward sigma0: mean squared log ratio over the step's maps.
        reg = sum(bounded(torch.einsum("p,spk->sk", Fall[g[1]], W) + b).square().mean() for g in gs) / len(gs)
        (loss + beta * reg).backward()
        opt.step()
    with torch.no_grad():
        s = bounded(torch.einsum("np,spk->nsk", Fall[held_rows], W) + b)
        sig = SCALE * s.exp().mean((-1, -2)).numpy()
    ease = np.array([np.mean(succ[r]) for r in held_rows])
    rho = np.corrcoef(rank(sig), rank(ease))[0, 1] if sig.std() > 0 else float("nan")
    easy, hard = sig[ease >= 0.75], sig[ease <= 0.25]
    return rho, sig.mean(), sig.std(), easy.mean(), hard.mean(), len(held_rows)


for per_coord in (False, True):
    for lr in (1e-3, 1e-2):
        for beta in (0.0, 1.0, 10.0):
            for seed in (42, 43):
                rho, m, sd, e, h, n = run(seed, lr, beta, per_coord)
                print(f"(b) p={PDIM} per_coord={per_coord!s:5} lr={lr:g} beta={beta:g} seed={seed}: "
                      f"Spearman(sigma, ease) {rho:+.3f}; sigma mean {m:.3f} sd {sd:.3f}; "
                      f"easy maps {e:.3f} vs hard {h:.3f} (held-out maps {n})", flush=True)
