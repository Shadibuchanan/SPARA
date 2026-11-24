"""Latent-exploration diagnostics after C′ (docs/LATENT_DIAGNOSTICS_SPEC.md). No training.

Parts, each one job (`spara.cli latent-diagnostics --part ...`):

- `sweep` (GPU): latent states of the pool maps at the chosen layers; per layer the principal
  basis (C′'s), the map-content subspace from linear probes (iterated nullspace projection),
  the bases without and within it, and the probes' accuracy. Then, on the pilot's 200 pool
  maps: LIVR's greedy route, T = 1 text rollouts (the baseline), and rollouts under latent
  perturbations of each family, layer and scale, all of equal expected size, plus
  teacher-forced scores of the label route under the same perturbations. Optionally noise on
  the answer tokens' hidden states.
- `gradients` (GPU): on three real C′ batches, where the latent term's gradient goes, how
  much absolute replay cuts the text gradient, and the (prior, coefficient, advantage)
  tuples for the offline head fits; then final-verdict-only search and latent random search
  on the pilot's greedy failures.
- `checkpoints` (GPU): LIVR and the B′/C′ runs on round 1's 240 held-out maps: greedy and
  label-route scores, the latent swap test, and sensitivity to C′'s noise.
- `weights` (CPU): how far each run's weights moved from LIVR, per layer group.
- `analyse` (CPU): merges the parts, fits the heads offline, applies the rule fixed in the
  specification, and writes summary.json and report.md.
"""
import contextlib
import json
import math
import random
import re
import time
from pathlib import Path

import torch

from .data import records
from .detour_probe import alignment
from .expert_iteration import BatchedLivr, route_of, searched
from .explore_probe import GREEDY, principal_basis, score
from .fresh_maps import GOAL, HOLE, START, label_route
from .fair_rl import PRIMARY, pilot_indices, populations, recipe
from .latent_battery import LatentPatch, answer, latent_positions, prompt_states, swap_donors
from .latent_rl import (BasisHeads, LatentAdjuster, add_latent_heads, head_input, latent_objective, sigma_bounds,
                        tensor_digest)
from .planning_probe import answer_ids as route_ids
from .route_search import FrozenLakeEnvironment
from .runtime import json_write, seed_all
from .scaffolding.policy import gaussian_log_prob
from .serialization import encode
from .text_rl import group_advantages, repeat_prompt, sequence_loss, stream_batch, token_logprobs
from .training import amp, to_device

FAMILIES = ("pca", "pca_perp_map", "map_only", "decision", "isotropic")
MOVE_ORDER = "UDLR"


# Settings ------------------------------------------------------------------------------------

def load_settings(path, smoke=False):
    """The diagnostics settings, with fair_rl's settings under "fair". `smoke` shrinks every
    part to a few maps and draws, for the in-job check before the full run."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    settings["fair"] = json.loads(Path(settings["fair_settings"]).read_text(encoding="utf-8"))
    settings["is_smoke"] = smoke
    if smoke:
        small = settings["smoke"]
        settings |= {"rollouts": small["rollouts"], "text_rollouts": small["text_rollouts"],
                     "soft_draws": small["soft_draws"], "scales": small["scales"]}
        settings["answer_noise"] = settings["answer_noise"] | {"scales": small["answer_scales"]}
        settings["latent_search"] = settings["latent_search"] | {"draws_per_scale": small["latent_search_draws"]}
        settings["verdict_search"] = settings["verdict_search"] | {"budget": small["verdict_budget"]}
        settings["gradients"] = settings["gradients"] | {"batches": small["gradient_batches"]}
        settings["head_fits"] = settings["head_fits"] | {"steps": small["head_fit_steps"]}
        if settings.get("map_range"):  # the range's first few maps, from the full pilot stream
            start = settings["map_range"][0]
            settings["map_range"] = [start, start + small["pilot_maps"]]
        else:
            settings["fair"] = settings["fair"] | {"pilot_maps": small["pilot_maps"]}
    return settings


def rl_settings(settings, arm):
    return json.loads(Path(settings["fair"]["rl_configs"][arm]).read_text(encoding="utf-8"))


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return eos if isinstance(eos, list) else [eos]


def prompt_of(processor, data_root, split, row, device):
    return to_device(encode(processor, data_root, split, row, training=False), device)


def capture_rows(pool, indices, smoke_maps=None):
    """Every pool map, or (smoke) the first few plus the pilot maps."""
    if smoke_maps is None:
        return list(range(len(pool)))
    return sorted(set(range(min(smoke_maps, len(pool)))) | set(indices))


# Perturbation families -----------------------------------------------------------------------
# Every family is scaled so that one draw at scale c has the expected squared size per latent
# slot of C′'s noise at c: c^2 times the sum of that slot's principal variances.

def slot_energy(basis):
    return basis["scales"].double().square().sum(-1)


def matched(basis, energy):
    """Per-slot directions (K, d, k) with weights rescaled to the target energy per slot."""
    scales = basis["scales"].double()
    weights = scales * (energy / scales.square().sum(-1).clamp_min(1e-30)).sqrt()[:, None]
    return {"kind": "slots", "directions": basis["directions"].float(), "weights": weights.float()}


def isotropic(energy, size):
    return {"kind": "isotropic", "std": (energy / size).sqrt().float(), "size": size}


def joint(directions, total):
    """Directions (m, K, d), orthonormal in the joint K x d space, equal weights."""
    return {"kind": "joint", "directions": directions.float(), "weight": math.sqrt(float(total) / directions.shape[0])}


def draw(family, scale, count, generator, device):
    """`count` offsets (count, K, d) at `scale` and their standard-normal coordinates (None
    for isotropic noise)."""
    kind = family["kind"]
    if kind == "slots":
        directions, weights = family["directions"].to(device), family["weights"].to(device)
        eps = torch.randn((count, *weights.shape), generator=generator, device=device)
        return scale * torch.einsum("sdk,sk,nsk->nsd", directions, weights, eps), eps
    if kind == "isotropic":
        std = family["std"].to(device)
        eps = torch.randn((count, std.shape[0], family["size"]), generator=generator, device=device)
        return scale * std[None, :, None] * eps, None
    directions = family["directions"].to(device)
    eps = torch.randn((count, directions.shape[0]), generator=generator, device=device)
    return scale * family["weight"] * torch.einsum("mkd,nm->nkd", directions, eps), eps


# Map-content probes --------------------------------------------------------------------------

def map_targets(row):
    """Goal, start and hole indicators per cell, the label route's first move (one-hot) and
    its length."""
    cells = [c for line in row["map_desc"] for c in line]
    route = route_of(row["text_output"])
    return {"goal": [float(c == GOAL) for c in cells], "start": [float(c == START) for c in cells],
            "holes": [float(c == HOLE) for c in cells], "first": [float(route[:1] == m) for m in MOVE_ORDER],
            "length": [float(len(route))]}


def ridge(X, Y, lambdas):
    """Dual ridge on centred X (n, p), Y (n, t): weights (p, t), with the penalty (relative to
    the mean kernel eigenvalue) chosen by exact leave-one-out error."""
    gram = X @ X.T
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    projected = vectors.T @ Y
    scale = values.mean().clamp_min(1e-12)
    best = None
    for lam in lambdas:
        shrink = values / (values + lam * scale)
        fitted = vectors @ (shrink[:, None] * projected)
        leverage = (vectors.square() * shrink).sum(1)
        loo = (fitted - leverage[:, None] * Y) / (1 - leverage[:, None]).clamp_min(1e-9)
        error = float((loo - Y).square().mean())
        if best is None or error < best[0]:
            best = (error, lam)
    alpha = vectors @ ((1 / (values + best[1] * scale))[:, None] * projected)
    return X.T @ alpha, best[1]


def fit_predict(X, Y, train, test, lambdas):
    mean_x, mean_y = X[train].mean(0), Y[train].mean(0)
    weights, lam = ridge(X[train] - mean_x, Y[train] - mean_y, lambdas)
    return (X[test] - mean_x) @ weights + mean_y, weights, lam


def balanced_accuracy(pred, true):
    """Mean over cells with both classes in `true` of the balanced accuracy at threshold 0.5."""
    scores = []
    for j in range(true.shape[1]):
        positive, negative = true[:, j] > 0.5, true[:, j] <= 0.5
        if positive.any() and negative.any():
            hit = pred[:, j] > 0.5
            scores.append(0.5 * (hit[positive].float().mean() + (~hit[negative]).float().mean()))
    return float(torch.stack(scores).mean()) if scores else None


def by_size(sizes, positions):
    groups = {}
    for p in positions:
        groups.setdefault(sizes[p], []).append(p)
    return groups


def probe_scores(X, rows, sizes, train, test, lambdas, targets=("goal", "start", "holes", "first", "length")):
    """Test-map accuracy of linear probes fitted per grid size on the train maps: goal and start
    cell (argmax), holes (balanced accuracy), first move (argmax), route length (R^2)."""
    out = {}
    train_groups, test_groups = by_size(sizes, train), by_size(sizes, test)
    for size in sorted(test_groups):
        tr, te = train_groups.get(size, []), test_groups[size]
        if len(tr) < 8 or not te:
            continue
        cell = {"test_maps": len(te)}
        for name in targets:
            Y = torch.tensor([map_targets(rows[p])[name] for p in tr + te], dtype=torch.float64)
            local_tr, local_te = list(range(len(tr))), list(range(len(tr), len(tr) + len(te)))
            pred, _, _ = fit_predict(X[torch.tensor(tr + te)], Y, local_tr, local_te, lambdas)
            true = Y[local_te]
            if name in ("goal", "start", "first"):
                cell[name] = float((pred.argmax(1) == true.argmax(1)).double().mean())
            elif name == "holes":
                cell[name] = balanced_accuracy(pred, true)
            else:
                residual = (pred - true).square().sum()
                total = (true - true.mean()).square().sum().clamp_min(1e-12)
                cell[name] = float(1 - residual / total)
        cell["chance_goal"] = 1 / size ** 2
        out[str(size)] = cell
    return out


def map_subspace(X, rows, sizes, train, test, settings):
    """Iterated nullspace projection on one slot's states X (n, d): fit goal, start and hole
    probes per size, add their (orthonormalised) weight directions to M, project them out,
    refit. Returns M (d, m) and the probes' test accuracy at every iteration."""
    probe = settings["probe"]
    M = torch.zeros(X.shape[1], 0, dtype=torch.float64)
    trace = []
    for iteration in range(probe["inlp_iterations"] + 1):
        projected = X - (X @ M) @ M.T
        scores = probe_scores(projected, rows, sizes, train, test, probe["ridge"], ("goal", "start", "holes"))
        trace.append({"dims": M.shape[1], "scores": scores})
        if iteration == probe["inlp_iterations"] or M.shape[1] >= probe["max_map_dims"]:
            break
        columns = []
        for size, positions in by_size(sizes, train).items():
            if len(positions) < 8:
                continue
            Y = torch.tensor([sum((map_targets(rows[p])[k] for k in ("goal", "start", "holes")), [])
                              for p in positions], dtype=torch.float64)
            Xs = projected[positions]
            weights, _ = ridge(Xs - Xs.mean(0), Y - Y.mean(0), probe["ridge"])
            columns.append(weights)
        if not columns:
            break
        W = torch.cat(columns, 1)
        W = W - M @ (M.T @ W)
        W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        keep = min(probe["inlp_dims_per_iteration"], int((S > 1e-6 * S[0]).sum()), probe["max_map_dims"] - M.shape[1])
        M = torch.linalg.qr(torch.cat([M, U[:, :keep]], 1)).Q
    return M, trace


def layer_bases(states, sizes, rows, train, test, settings):
    """C′'s principal basis and the bases without and within the map-content subspace, as
    norm-matched families, and what they say about the latents (Q5)."""
    k = settings["basis_k"]
    pca = principal_basis(states, sizes, k)
    energy = slot_energy(pca)
    subspaces, traces, inside = [], [], []
    for slot in range(states.shape[1]):
        M, trace = map_subspace(states[:, slot].double(), rows, sizes, train, test, settings)
        subspaces.append(M)
        traces.append(trace)
        directions, weights = pca["directions"][slot].double(), pca["scales"][slot].double().square()
        inside.append(float(((M.T @ directions).square().sum(0) * weights).sum() / weights.sum()))
    within = torch.stack([states[:, s].double() @ subspaces[s] @ subspaces[s].T for s in range(states.shape[1])], 1)
    perp = principal_basis((states.double() - within).float(), sizes, k)
    only = principal_basis(within.float(), sizes, k)
    families = {"pca": matched(pca, energy), "pca_perp_map": matched(perp, energy),
                "map_only": matched(only, energy), "isotropic": isotropic(energy, states.shape[-1])}
    info = {"captured": {"pca": pca["captured"], "pca_perp_map": perp["captured"], "map_only": only["captured"]},
            "slot_energy": energy.tolist(), "map_dims": [M.shape[1] for M in subspaces],
            "pca_energy_inside_map_subspace": inside, "inlp": traces}
    return families, info, pca


# Hooks ---------------------------------------------------------------------------------------

class LastToken:
    """Records each chosen layer's output at the last prompt position during a prefill."""

    def __init__(self, model, layers):
        self.recorded = None
        self.handles = [model.model.layers[layer - 1].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.recorded is not None and hidden.shape[1] > 1:
                self.recorded[layer] = hidden[0, -1].detach()
        return hook

    @contextlib.contextmanager
    def record(self):
        self.recorded = {}
        try:
            yield self.recorded
        finally:
            self.recorded = None

    def remove(self):
        for handle in self.handles:
            handle.remove()


class AnswerNoise:
    """Adds scale * sum_j u_j s_j eps_j to one layer's output at the last position of every
    forward: the last prompt token at prefill and each generated token after. Fresh draws per
    row and step."""

    def __init__(self, model, layer):
        self.state = None
        self.handle = model.model.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.state is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        directions, scales = self.state["directions"].to(hidden.device), self.state["scales"].to(hidden.device)
        eps = torch.randn((hidden.shape[0], scales.shape[0]), generator=self.state["generator"], device=hidden.device)
        noise = self.state["scale"] * (eps * scales) @ directions.T
        hidden = hidden.clone()
        hidden[:, -1] = (hidden[:, -1].float() + noise).to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    @contextlib.contextmanager
    def active(self, basis, scale, generator):
        self.state = {"directions": basis["directions"][0].float(), "scales": basis["scales"][0].float(),
                      "scale": scale, "generator": generator}
        try:
            yield
        finally:
            self.state = None

    def remove(self):
        self.handle.remove()


@contextlib.contextmanager
def last_logits(model, count):
    """Only the last `count` positions reach the output layer (avoids full-vocabulary logits
    over the whole prompt)."""
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -count:], *args[1:]))
    try:
        yield
    finally:
        handle.remove()


# Capture -------------------------------------------------------------------------------------

@torch.no_grad()
def capture(model, processor, data_root, split, rows, indices, layers, last_layers):
    """Latent states (n, K, d) at each layer in `layers` and last-prompt-token states (n, d) at
    each layer in `last_layers`, float32 on CPU, from one prefill per map."""
    adjusters = {layer: LatentAdjuster(model, layer) for layer in layers}
    last = LastToken(model, last_layers)
    states, lasts = {layer: [] for layer in layers}, {layer: [] for layer in last_layers}
    for count, index in enumerate(indices):
        prompt = prompt_of(processor, data_root, split, rows[index], model.device)
        with contextlib.ExitStack() as stack:
            captured = {layer: stack.enter_context(adjusters[layer].capture()) for layer in layers}
            recorded = stack.enter_context(last.record())
            stack.enter_context(amp(model.device))
            model(**prompt, use_cache=False)
            for layer in layers:
                states[layer].append(captured[layer]["priors"][0][0].float().cpu())
            for layer in last_layers:
                lasts[layer].append(recorded[layer].float().cpu())
        if count % 250 == 0:
            print(f"capture {count + 1}/{len(indices)}", flush=True)
    for adjuster in adjusters.values():
        adjuster.remove()
    last.remove()
    return ({layer: torch.stack(v) for layer, v in states.items()},
            {layer: torch.stack(v) for layer, v in lasts.items()})


# Sweep ---------------------------------------------------------------------------------------

@torch.no_grad()
def generate_rows(model, processor, evaluator, row, prompt, generation, count, context, chunk):
    """`count` answers to one prompt in chunks; `context(start, end)` opens the perturbation for
    those rows. Returns [(route, success)]."""
    results = []
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        with context(start, end), amp(model.device):
            generated = model.generate(**repeat_prompt(prompt, end - start), **generation, use_cache=True)
        results += score(evaluator, processor, row, generated)
    return results


@torch.no_grad()
def label_scores(model, prompt, label_ids, offsets, adjuster, first_at, first_tokens, chunk):
    """Under each offset row: the label route's log-probability and the distribution over the
    four moves at its first move."""
    out = []
    length = len(label_ids)
    for start in range(0, len(offsets), chunk):
        part = offsets[start:start + chunk]
        n = len(part)
        ids = torch.cat([prompt["input_ids"].repeat(n, 1), label_ids[None].repeat(n, 1)], -1)
        batch = repeat_prompt(prompt, n) | {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        with adjuster.offset(part), amp(model.device), last_logits(model, length + 1):
            logits = model(**batch, use_cache=False).logits[:, :-1].float()
        logp = logits.log_softmax(-1)
        route = logp.gather(-1, label_ids[None, :, None].expand(n, -1, 1)).squeeze(-1).sum(-1)
        first = logp[:, first_at, first_tokens].softmax(-1)
        out += [{"label_logp": float(r), "first": [round(float(p), 6) for p in f]} for r, f in zip(route, first)]
    return out


def deviation(greedy, label):
    """The first move index where the greedy route leaves the label route (0 if it never does)."""
    for i, move in enumerate(label):
        if i >= len(greedy) or greedy[i] != move:
            return i
    return 0


def decision_family(model, adjuster, tokenizer, prompt, label, greedy, total_energy, max_dims):
    """Per map: directions of the latent states (joint over the slots) along which the label
    route, the greedy route and the move choice where they diverge change fastest. The span of
    the normalised gradients, by SVD."""
    slots, size = len(model.config.livr_latent_ids), model.config.hidden_size
    delta = torch.zeros(1, slots, size, device=model.device, requires_grad=True)
    objectives, point = [], deviation(greedy, label)
    with torch.enable_grad(), adjuster.offset(delta), amp(model.device):
        for name, route in (("label", label), ("greedy", greedy)):
            if not route or (name == "greedy" and route == label):
                continue
            ids = route_ids(tokenizer, route).to(model.device)
            full = torch.cat([prompt["input_ids"], ids[None]], -1)
            with last_logits(model, len(ids) + 1):
                logits = model(**prompt | {"input_ids": full, "attention_mask": torch.ones_like(full)},
                               use_cache=False).logits[0, :-1].float()
            logp = logits.log_softmax(-1)
            objectives.append(logp.gather(-1, ids[:, None]).sum())
            if name == "label":
                _, at, _, tokens, _, _ = alignment(tokenizer, label[:point])
                if at < len(ids) and int(ids[at]) == tokens[label[point]]:
                    objectives += [logp[at, tokens[m]] for m in MOVE_ORDER]
    grads = [torch.autograd.grad(o, delta, retain_graph=i < len(objectives) - 1)[0][0]
             for i, o in enumerate(objectives)]
    G = torch.stack([g.reshape(-1).double() for g in grads])
    G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-30)
    _, S, Vh = torch.linalg.svd(G, full_matrices=False)
    keep = max(1, min(max_dims, int((S > 1e-6 * S[0]).sum())))
    return (joint(Vh[:keep].reshape(keep, slots, size).float(), total_energy),
            {"dims": keep, "objectives": len(objectives), "deviation": point,
             "singular": [round(float(s), 4) for s in S]})


def expected_energy(family, scale):
    """Expected squared size per latent slot of one draw at `scale` (None for joint families,
    whose size is fixed only over all slots together)."""
    if family["kind"] == "slots":
        return (scale ** 2 * family["weights"].double().square().sum(-1)).tolist()
    if family["kind"] == "isotropic":
        return (scale ** 2 * family["std"].double().square() * family["size"]).tolist()
    return None


def load_bases(path, layers):
    """A previous sweep's families (its bases.pt) at `layers`, and their digests."""
    saved = torch.load(Path(path) / "bases.pt")
    missing = [layer for layer in layers if layer not in saved["families"]]
    if missing:
        raise ValueError(f"{path}/bases.pt has no families at layers {missing}")
    families = {layer: saved["families"][layer] for layer in layers}
    digests = {str(layer): {name: tensor_digest(f) for name, f in families[layer].items()} for layer in layers}
    return families, digests


def sweep_bases(model, processor, data_root, split, pool, indices, layers, noise_layers, settings, output):
    """Captures the pool maps' latent states and builds every family at each layer (Q5's probes
    included); saves states.pt, bases.pt and bases.json."""
    rows = capture_rows(pool, indices, settings["smoke"]["capture_maps"] if settings["is_smoke"] else None)
    if not layers and not noise_layers:  # the text baseline alone needs no latent states
        rows = []
    states, lasts = capture(model, processor, data_root, split, pool, rows, layers, noise_layers)
    if rows:
        torch.save({"rows": rows, "states": {k: v.half() for k, v in states.items()},
                    "last": {k: v.half() for k, v in lasts.items()}}, output / "states.pt")
    captured_rows = [pool[i] for i in rows]
    sizes = [len(r["map_desc"]) for r in captured_rows]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(settings["seed"])).tolist()
    tests = min(settings["probe"]["test_maps"], len(rows) // 5) if rows else 0
    train, test = sorted(order[tests:]), sorted(order[:tests])
    families, info = {}, {}
    for layer in layers:
        started = time.time()
        families[layer], info[str(layer)], _ = layer_bases(states[layer], sizes, captured_rows, train, test, settings)
        info[str(layer)]["probes"] = probe_scores(states[layer].reshape(len(rows), -1).double(), captured_rows, sizes,
                                                  train, test, settings["probe"]["ridge"])
        print(f"bases at layer {layer}: {time.time() - started:.0f} s; map dims {info[str(layer)]['map_dims']}; "
              f"PCA energy inside map subspace {info[str(layer)]['pca_energy_inside_map_subspace']}", flush=True)
    answer_bases = {layer: principal_basis(lasts[layer][:, None], sizes, settings["basis_k"]) for layer in noise_layers}
    torch.save({"families": families, "answer": answer_bases, "probe_split": {"train": train, "test": test}},
               output / "bases.pt")
    json_write(output / "bases.json", info | {"answer_captured": {str(k): v["captured"] for k, v in answer_bases.items()}})
    return families, answer_bases


def run_sweep(model, processor, evaluator, data_root, output, settings, layers, text_baseline, answer_noise,
              run_manifest, bases_from=None):
    """Q4. `settings["map_range"]` = [start, end] keeps that slice of the pilot stream;
    `settings["repeats"]` > 1 draws that many groups of `rollouts` per configuration, stored as
    key, key#2, ...; `bases_from` reuses a previous sweep's families (no capture)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    fair = settings["fair"]
    split = fair["pool_split"]
    pool = records(data_root, split)
    indices = pilot_indices(data_root, fair)
    if settings.get("map_range"):
        start, end = settings["map_range"]
        if end > len(indices):
            raise ValueError(f"map_range {settings['map_range']} exceeds the {len(indices)} pilot maps")
        indices = indices[start:end]
    generation = rl_settings(settings, "b")["generation"]
    noise_layers = settings["answer_noise"]["layers"] if answer_noise else []
    if bases_from and noise_layers:
        raise ValueError("answer noise needs captured last-token states; it cannot reuse saved bases")
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "layers": layers,
                                                         "text_baseline": text_baseline, "answer_noise": noise_layers,
                                                         "pilot_rows": indices, "generation": generation,
                                                         "bases_from": str(bases_from) if bases_from else None})
    model.eval().requires_grad_(False)
    model.config.livr_stage = 2
    device, tokenizer = model.device, processor.tokenizer

    if bases_from:
        families, digests = load_bases(bases_from, layers)
        answer_bases = {}
        json_write(output / "bases.json", {"from": str(bases_from), "sha256": digests})
        print(f"bases from {bases_from}: {json.dumps(digests)}", flush=True)
    else:
        families, answer_bases = sweep_bases(model, processor, data_root, split, pool, indices, layers, noise_layers,
                                             settings, output)
    repeats = settings.get("repeats", 1)

    adjusters = {layer: LatentAdjuster(model, layer) for layer in layers}
    noisers = {layer: AnswerNoise(model, layer) for layer in noise_layers}
    _, first_at, _, first_map, _, _ = alignment(tokenizer, "")
    first_tokens = [first_map[m] for m in MOVE_ORDER]
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    seed_all(settings["seed"])
    chunk, draws = settings["generate_rows"], settings["rollouts"]
    names = [f for f in settings["families"] if f in FAMILIES]
    with (output / "maps.jsonl").open("w", encoding="utf-8") as log:
        for count, index in enumerate(indices):
            started = time.time()
            row = pool[index]
            prompt = prompt_of(processor, data_root, split, row, device)
            label = route_of(row["text_output"])
            label_ids = route_ids(tokenizer, label).to(device)
            with torch.no_grad(), amp(device):
                greedy = model.generate(**prompt, **GREEDY, max_new_tokens=settings["max_new_tokens"], use_cache=True)
            (greedy_route, greedy_success), = score(evaluator, processor, row, greedy)
            record = {"row_index": index, "grid_size": len(row["map_desc"]), "label": label,
                      "greedy_route": greedy_route, "greedy_success": greedy_success, "configs": {}, "soft": {},
                      "decision": {}}
            if text_baseline:
                cells = generate_rows(model, processor, evaluator, row, prompt, generation, settings["text_rollouts"],
                                      lambda a, b: contextlib.nullcontext(), chunk)
                record["text"] = {"successes": [s for _, s in cells], "routes": [r for r, _ in cells]}
            for layer in layers:
                available = dict(families[layer])
                if "decision" in names:
                    available["decision"], record["decision"][str(layer)] = decision_family(
                        model, adjusters[layer], tokenizer, prompt, label, greedy_route,
                        slot_energy({"scales": families[layer]["pca"]["weights"]}).sum(), settings["decision_max_dims"])
                plan = [(name, scale) for name in names for scale in settings["scales"]]
                per_config = draws * repeats
                offsets, coordinates, energy = [], {}, {}
                for name, scale in plan:
                    offset, eps = draw(available[name], scale, per_config, generator, device)
                    offsets.append(offset)
                    if name == "pca":
                        coordinates[scale] = eps
                    expected = expected_energy(available[name], scale)
                    if repeats > 1 and expected:  # measured / expected squared size per slot (about 1)
                        measured = offset.double().square().sum(-1).mean(0).tolist()
                        energy[(name, scale)] = [round(m / max(e, 1e-30), 4) for m, e in zip(measured, expected)]
                offsets = torch.cat(offsets)
                cells = generate_rows(model, processor, evaluator, row, prompt, generation, len(offsets),
                                      lambda a, b, layer=layer, offsets=offsets: adjusters[layer].offset(offsets[a:b]),
                                      chunk)
                for i, (name, scale) in enumerate(plan):
                    for r in range(repeats):
                        first = i * per_config + r * draws
                        part = cells[first:first + draws]
                        cell = {"successes": [s for _, s in part], "routes": [r_ for r_, _ in part]}
                        if name == "pca":  # C′'s coefficients are scale * eps (heads at mu = 0, sigma = scale)
                            cell["eps"] = [[[round(float(x), 4) for x in slot] for slot in draw_]
                                           for draw_ in coordinates[scale][r * draws:(r + 1) * draws].cpu()]
                        if r == 0 and (name, scale) in energy:
                            cell["energy_ratio"] = energy[(name, scale)]
                        record["configs"][f"{name}/{layer}/{scale}" + (f"#{r + 1}" if r else "")] = cell
                soft_plan = [(name, scale) for name in names for scale in settings["scales"]]
                soft = [torch.zeros(1, len(model.config.livr_latent_ids), model.config.hidden_size, device=device)]
                soft += [draw(available[name], scale, settings["soft_draws"], generator, device)[0] for name, scale in soft_plan]
                scored = label_scores(model, prompt, label_ids, torch.cat(soft), adjusters[layer], first_at, first_tokens,
                                      settings["score_rows"])
                record["soft"][str(layer)] = {"none": scored[0]}
                for i, (name, scale) in enumerate(soft_plan):
                    record["soft"][str(layer)][f"{name}/{scale}"] = scored[1 + i * settings["soft_draws"]:
                                                                           1 + (i + 1) * settings["soft_draws"]]
            for layer in noise_layers:
                for scale in settings["answer_noise"]["scales"]:
                    cells = generate_rows(model, processor, evaluator, row, prompt, generation, draws,
                                          lambda a, b, layer=layer, scale=scale: noisers[layer].active(
                                              answer_bases[layer], scale, generator), chunk)
                    record["configs"][f"answer/{layer}/{scale}"] = {"successes": [s for _, s in cells],
                                                                   "routes": [r for r, _ in cells]}
            record["seconds"] = round(time.time() - started, 2)
            log.write(json.dumps(record) + "\n"); log.flush()
            line = f"map {count + 1}/{len(indices)} ({record['seconds']} s)"
            if len(record["configs"]) <= 8:  # small sweeps: each group's successes, for monitoring
                line += f" greedy {'T' if greedy_success else 'F'} | " + " ".join(
                    f"{k.split('/', 1)[1]}: {sum(c['successes'])}/{len(c['successes'])}"
                    for k, c in record["configs"].items())
            print(line, flush=True)
    for hook in [*adjusters.values(), *noisers.values()]:
        hook.remove()
    summary = {"maps": len(indices), "layers": layers, "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    return summary


# Gradients -----------------------------------------------------------------------------------

GROUPS = ("latent_heads", "embeddings", "layers 1-8", "layers 9-16", "layers 17-24", "layers 25-36",
          "final norm and lm_head", "visual", "other")


def group_of(name):
    if name.startswith("latent_heads."):
        return "latent_heads"
    if name.startswith("visual.") or ".visual." in name:
        return "visual"
    if "embed_tokens" in name:
        return "embeddings"
    found = re.search(r"layers\.(\d+)\.", name)
    if found:
        layer = int(found.group(1)) + 1
        return ("layers 1-8" if layer <= 8 else "layers 9-16" if layer <= 16
                else "layers 17-24" if layer <= 24 else "layers 25-36")
    if "lm_head" in name or name.endswith("norm.weight"):
        return "final norm and lm_head"
    return "other"


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]


def group_norms(model):
    sums = {}
    for name, p in trainable(model):
        g = group_of(name)
        sums[g] = sums.get(g, 0.0) + float(p.grad.double().square().sum())
    total = math.sqrt(sum(sums.values()))
    return {"groups": {g: math.sqrt(v) for g, v in sums.items()}, "total": total}


def snapshot(model):
    return {name: p.grad.detach().to("cpu", torch.float32, copy=True) for name, p in trainable(model)}


def pairs(model, stored):
    """(name, current gradient, stored gradient) over every trainable parameter either has;
    a missing one is zero."""
    for name, p in model.named_parameters():
        if not p.requires_grad or (p.grad is None and name not in stored):
            continue
        g = p.grad.double() if p.grad is not None else torch.zeros_like(p, dtype=torch.float64)
        s = stored[name].to(p.device).double() if name in stored else torch.zeros_like(g)
        yield name, g, s


def against(model, stored):
    """Per group: the current gradient's norm, the stored one's, their difference and cosine."""
    acc = {}
    for name, g, s in pairs(model, stored):
        cell = acc.setdefault(group_of(name), [0.0, 0.0, 0.0, 0.0])
        cell[0] += float(g.square().sum()); cell[1] += float(s.square().sum())
        cell[2] += float((g - s).square().sum()); cell[3] += float((g * s).sum())
    out = {}
    for group, (gg, ss, dd, gs) in acc.items():
        out[group] = {"norm": math.sqrt(gg), "stored_norm": math.sqrt(ss),
                      "relative_difference": math.sqrt(dd) / max(math.sqrt(ss), 1e-30),
                      "cosine": gs / max(math.sqrt(gg * ss), 1e-30)}
    return out


def combined(model, stored, small):
    """The training gradient as the sum of the current and the stored gradient: its norm per
    group, the global clip factor at max norm 1, and the share of entries (per group) whose
    clipped value is below `small` (where Adam's epsilon of 1e-8 starts to damp the update)."""
    sums = {}
    for name, g, s in pairs(model, stored):
        sums[group_of(name)] = sums.get(group_of(name), 0.0) + float((g + s).square().sum())
    total = math.sqrt(sum(sums.values()))
    clip = min(1.0, 1.0 / max(total, 1e-30))
    tiny = {}
    for name, g, s in pairs(model, stored):
        g = (g + s) * clip
        cell = tiny.setdefault(group_of(name), [0, 0])
        cell[0] += int((g.abs() < small).sum()); cell[1] += g.numel()
    return {"groups": {g: math.sqrt(v) for g, v in sums.items()}, "total": total, "clip": clip,
            "small_share": {g: a / b for g, (a, b) in tiny.items()}}


def small_share(model, small):
    total = group_norms(model)["total"]
    clip = min(1.0, 1.0 / max(total, 1e-30))
    tiny = {}
    for name, p in trainable(model):
        cell = tiny.setdefault(group_of(name), [0, 0])
        cell[0] += int(((p.grad.double() * clip).abs() < small).sum()); cell[1] += p.grad.numel()
    return {"clip": clip, "small_share": {g: a / b for g, (a, b) in tiny.items()}}


def text_contexts(mode, policy, reference, action, prior):
    """The latent state the policy and the reference read during the update:
    none (LIVR's own, B′), absolute (C′ as built: the recorded action, detached),
    offset (prior + the recorded adjustment: the text gradient reaches the prior),
    zero_absolute and zero_offset (the same two with a zero adjustment)."""
    if mode == "none":
        return contextlib.nullcontext({}), contextlib.nullcontext({})
    if mode == "absolute":
        return policy.replay(action), reference.replay(action)
    if mode == "offset":
        return policy.offset(action - prior), reference.offset(action - prior)
    if mode == "zero_absolute":
        return policy.replay(prior), reference.replay(prior)
    if mode == "zero_offset":
        zeros = torch.zeros_like(prior)
        return policy.offset(zeros), reference.offset(zeros)
    raise ValueError(mode)


def accumulate(model, reference, groups, rl, adjusters, text_mode=None, latent=None):
    """One update's gradient, without the step: the text term (with `text_mode`) or the latent
    term alone (`latent` "attached", as built, or "detached": the prior detached so only the
    heads learn), averaged over the rollouts of groups with mixed rewards as in training."""
    policy, reference_adjuster = adjusters
    model.zero_grad(set_to_none=True)
    trained = [g for g in groups if g["rewards"].std() > 0]
    count = sum(len(g["completions"]) for g in trained)
    temperature = rl["generation"].get("temperature", 1.0)
    for group in trained:
        for i, (completion, advantage) in enumerate(zip(group["completions"], group["advantages"].tolist())):
            action = None if group["actions"] is None else group["actions"][i:i + 1]
            if latent:
                with policy.replay(action) as state:
                    with amp(model.device):
                        model(**group["prompt"], use_cache=False)
                    prior = state["prior"] if latent == "attached" else state["prior"].detach()
                    loss, _ = latent_objective(model, prior, action, advantage, rl["latent"], rl["clip_epsilon"])
                    (loss / count).backward()
                continue
            policy_context, reference_context = text_contexts(text_mode, policy, reference_adjuster, action,
                                                              group["prior"])
            with torch.no_grad(), reference_context:
                reference_logprobs = token_logprobs(reference, group["prompt"], completion, temperature)
            with policy_context:
                logprobs = token_logprobs(model, group["prompt"], completion, temperature)
                loss, _ = sequence_loss(logprobs, reference_logprobs, advantage, rl)
                (loss / count).backward()
    return count


@torch.no_grad()
def batch_groups(model, processor, evaluator, data_root, split, pool, selected, rl, adjuster, prior_adjuster):
    """Rollouts for one batch: B′'s (adjuster None) or C′'s, with LIVR's latent prior per map."""
    from .text_rl import rollout
    groups = []
    for index in selected:
        prompt = prompt_of(processor, data_root, split, pool[index], model.device)
        with prior_adjuster.capture() as state, amp(model.device):
            model(**prompt, use_cache=False)
        prior = state["priors"][0].float()
        completions, actions = rollout(model, prompt, rl, eos_ids(model), adjuster)
        texts = [processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], c]), skip_special_tokens=False)
                 for c in completions]
        rewards = torch.tensor([float(bool(evaluator.score(pool[index], t)["success"])) for t in texts])
        groups.append({"row": index, "prompt": prompt, "completions": completions, "rewards": rewards,
                       "advantages": group_advantages(rewards), "actions": actions, "prior": prior})
    return groups


def run_gradients(model, reference, processor, evaluator, config, data_root, output, settings, pilot_run, run_manifest):
    """Q1–Q3 on C′'s first batches of seed 42 (the same maps C′ 42 trained on first)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fair, pilot_run = settings["fair"], Path(pilot_run)
    pilot = json.loads((pilot_run / "summary.json").read_text())
    rl_b, rl_c = recipe(rl_settings(settings, "b"), "b", 42, pilot, fair), recipe(rl_settings(settings, "c"), "c", 42, pilot, fair)
    if settings["is_smoke"]:
        rl_b["prompts_per_step"] = rl_c["prompts_per_step"] = settings["smoke"]["prompts_per_step"]
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "rl_b": rl_b, "rl_c": rl_c,
                                                         "pilot_run": str(pilot_run)})
    basis = torch.load(pilot_run / "basis.pt")
    heads = add_latent_heads(model, rl_c["latent"], basis)
    layer = rl_c["latent"]["layer"]
    policy, reference_adjuster = LatentAdjuster(model, layer), LatentAdjuster(reference, layer)
    reference.eval().requires_grad_(False)
    model.config.livr_stage = 2
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    split = fair["pool_split"]
    pool = records(data_root, split)
    small = settings["gradients"]["small_gradient"]
    seed_all(42)
    position, results, tuples = 0, [], []
    for batch in range(settings["gradients"]["batches"]):
        started = time.time()
        selected, position = stream_batch(len(pool), rl_c["prompts_per_step"], 42, position)
        groups_c = batch_groups(model, processor, evaluator, data_root, split, pool, selected, rl_c, policy, policy)
        groups_b = batch_groups(model, processor, evaluator, data_root, split, pool, selected, rl_b, None, policy)
        for g in groups_c:
            with torch.no_grad():
                coefficients = heads.coefficients(g["actions"].float() - g["prior"].float())
            tuples.append({"batch": batch, "row": g["row"], "prior": g["prior"][0].half().cpu(),
                           "coefficients": coefficients.cpu(), "rewards": g["rewards"], "advantages": g["advantages"]})
        model.train()
        model.visual.eval()
        adjusters = (policy, reference_adjuster)
        cell = {"batch": batch, "rows": selected,
                "mixed_groups": {"b": sum(bool(g["rewards"].std() > 0) for g in groups_b),
                                 "c": sum(bool(g["rewards"].std() > 0) for g in groups_c)},
                "rollout_success": {"b": float(torch.cat([g["rewards"] for g in groups_b]).mean()),
                                    "c": float(torch.cat([g["rewards"] for g in groups_c]).mean())}}
        if accumulate(model, reference, groups_b, rl_b, adjusters, text_mode="none"):
            cell["b_text"] = group_norms(model) | small_share(model, small)
            stored = snapshot(model)
            for mode in ("zero_absolute", "zero_offset"):
                accumulate(model, reference, groups_b, rl_b, adjusters, text_mode=mode)
                cell[f"b_text_{mode}_vs_none"] = against(model, stored)
            del stored
        if accumulate(model, reference, groups_c, rl_c, adjusters, text_mode="absolute"):
            cell["c_text_absolute"] = group_norms(model)
            stored = snapshot(model)
            accumulate(model, reference, groups_c, rl_c, adjusters, text_mode="offset")
            cell["c_text_offset_vs_absolute"] = against(model, stored)
            accumulate(model, reference, groups_c, rl_c, adjusters, latent="attached")
            cell["c_latent_attached"] = group_norms(model)
            cell["c_latent_vs_text"] = against(model, stored)
            cell["c_training_gradient"] = combined(model, stored, small)
            del stored
            accumulate(model, reference, groups_c, rl_c, adjusters, latent="detached")
            cell["c_latent_detached"] = group_norms(model)
        model.zero_grad(set_to_none=True)
        cell["seconds"] = time.time() - started
        results.append(cell)
        with (output / "gradients.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(cell) + "\n")
        print(json.dumps({k: cell[k] for k in ("batch", "mixed_groups", "seconds")}), flush=True)
    torch.save(tuples, output / "head-tuples.pt")
    policy.remove()
    reference_adjuster.remove()
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_disable()
    model.eval()
    return results


# Search --------------------------------------------------------------------------------------

def run_search(model, processor, evaluator, data_root, output, settings, pilot_run):
    """Q6 on the pilot maps LIVR's greedy route fails: final-verdict-only search with LIVR, and
    latent random search (greedy answers under C′'s noise at increasing scales) per layer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    fair = settings["fair"]
    split = fair["pool_split"]
    pool = records(data_root, split)
    indices = pilot_indices(data_root, fair)
    model.eval().requires_grad_(False)
    model.config.livr_stage = 2
    model.config.use_cache = True
    greedy = {}
    for index in indices:
        prompt = prompt_of(processor, data_root, split, pool[index], model.device)
        with torch.no_grad(), amp(model.device):
            out = model.generate(**prompt, **GREEDY, max_new_tokens=settings["max_new_tokens"], use_cache=True)
        greedy[index] = score(evaluator, processor, pool[index], out)[0]
    failures = [i for i in indices if not greedy[i][1]]
    problems = [{"id": f"{split}/{i}", "split": split, "row_index": i, "row": pool[i]} for i in failures]
    seed_all(settings["seed"])
    policy = BatchedLivr(model, processor, data_root, settings["verdict_search"]["batch"])
    rows, cost = searched(FrozenLakeEnvironment(evaluator), policy, problems, settings["verdict_search"])
    with (output / "verdict-search.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in rows)
    del policy
    torch.cuda.empty_cache()

    search = settings["latent_search"]
    capture_set = capture_rows(pool, indices, settings["smoke"]["capture_maps"] if settings["is_smoke"] else None)
    states, _ = capture(model, processor, data_root, split, pool, capture_set, search["layers"], [])
    sizes = [len(pool[i]["map_desc"]) for i in capture_set]
    generator = torch.Generator(device=model.device).manual_seed(settings["seed"])
    with (output / "latent-search.jsonl").open("w", encoding="utf-8") as log:
        for layer in search["layers"]:
            basis = principal_basis(states[layer], sizes, settings["basis_k"])
            family = matched(basis, slot_energy(basis))
            adjuster = LatentAdjuster(model, layer)
            for index in failures:
                row = pool[index]
                prompt = prompt_of(processor, data_root, split, row, model.device)
                offsets = torch.cat([draw(family, s, search["draws_per_scale"], generator, model.device)[0]
                                     for s in search["scales"]])
                cells = generate_rows(model, processor, evaluator, row, prompt,
                                      GREEDY | {"max_new_tokens": settings["max_new_tokens"]}, len(offsets),
                                      lambda a, b, offsets=offsets, adjuster=adjuster: adjuster.offset(offsets[a:b]),
                                      search["batch"])
                successes = [s for _, s in cells]
                log.write(json.dumps({"layer": layer, "row_index": index, "scales": search["scales"],
                                      "draws_per_scale": search["draws_per_scale"], "successes": successes,
                                      "routes": [r for r, _ in cells],
                                      "first_success": next((i + 1 for i, s in enumerate(successes) if s), None)}) + "\n")
                log.flush()
            adjuster.remove()
            print(f"latent search at layer {layer} done ({time.time() - began:.0f} s)", flush=True)
    summary = {"pilot_maps": len(indices), "greedy_failures": len(failures), "verdict_cost": cost,
               "verdict_solved": sum(r["solved"] for r in rows), "seconds": time.time() - began}
    json_write(output / "search-summary.json", summary)
    return summary


# Checkpoints ---------------------------------------------------------------------------------

def run_checkpoints(load, runs, evaluator, data_root, output, settings, pilot_run):
    """Q7: each run's greedy answers and label-route scores on round 1's 240 held-out maps; on
    every `swap_every`-th of them, the latent swap test and sensitivity to C′'s noise."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fair, checks = settings["fair"], settings["checkpoints"]
    items = [i for i in populations(data_root, fair) if i[0] in PRIMARY]
    if settings["is_smoke"]:
        items = items[:settings["smoke"]["checkpoint_maps"]]
    subset = items[::checks["swap_every"]]
    donors = swap_donors([row for _, _, _, row in subset], checks["swap_seed"])
    basis = torch.load(Path(pilot_run) / "basis.pt")
    family = matched(basis, slot_energy(basis))
    settings_greedy = GREEDY | {"max_new_tokens": settings["max_new_tokens"]}
    json_write(output / "manifest.json", {"settings": settings, "runs": {k: str(v) for k, v in runs.items()},
                                          "maps": len(items), "subset": len(subset),
                                          "donors": {str(k): v for k, v in donors.items()}})
    summary = {}
    for name, path in runs.items():
        started = time.time()
        model, processor = load(path)
        model.eval().requires_grad_(False)
        model.config.livr_stage = 2
        tokenizer, device = processor.tokenizer, model.device
        _, first_at, _, first_map, _, _ = alignment(tokenizer, "")
        first_tokens = [first_map[m] for m in MOVE_ORDER]
        adjuster, patch = LatentAdjuster(model, checks["noise_layer"]), LatentPatch(model)
        generator = torch.Generator(device=device).manual_seed(settings["seed"])
        zero = torch.zeros(1, len(model.config.livr_latent_ids), model.config.hidden_size, device=device)
        own, subset_states = {}, {}
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as log:
            for position, (population, split, index, row) in enumerate(items):
                prompt = prompt_of(processor, data_root, split, row, device)
                # Released test maps carry no label; their shortest route is the label then.
                label = route_of(row["text_output"]) if row.get("text_output") else label_route(row["map_desc"])
                with torch.no_grad(), amp(device):
                    out = model.generate(**prompt, **settings_greedy, use_cache=True)
                route, success = score(evaluator, processor, row, out)[0]
                scored = label_scores(model, prompt, route_ids(tokenizer, label).to(device), zero, adjuster, first_at,
                                      first_tokens, 1)[0]
                record = {"population": population, "row_index": index, "grid_size": len(row["map_desc"]),
                          "route": route, "success": success, "label": label, "label_logp": scored["label_logp"],
                          "label_first_move_probability": scored["first"][MOVE_ORDER.index(label[0])]}
                own[position] = route
                if position % checks["swap_every"] == 0:
                    offsets = torch.cat([draw(family, s, checks["noise_draws"], generator, device)[0]
                                         for s in checks["noise_scales"]])
                    cells = generate_rows(model, processor, evaluator, row, prompt, settings_greedy, len(offsets),
                                          lambda a, b, offsets=offsets: adjuster.offset(offsets[a:b]), len(offsets))
                    record["noise"] = {str(s): {"routes": [r for r, _ in cells[i * checks["noise_draws"]:(i + 1) * checks["noise_draws"]]],
                                                "successes": [x for _, x in cells[i * checks["noise_draws"]:(i + 1) * checks["noise_draws"]]]}
                                       for i, s in enumerate(checks["noise_scales"])}
                    positions = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
                    subset_states[position // checks["swap_every"]] = (
                        prompt_states(model, patch, prompt, positions)[:, positions].float().cpu(), positions)
                log.write(json.dumps(record) + "\n"); log.flush()
            swaps = []
            for local, donor in donors.items():
                population, split, index, row = subset[local]
                prompt = prompt_of(processor, data_root, split, row, device)
                values, positions = subset_states[donor][0].to(device), subset_states[local][1]
                result, _, text = answer(model, patch, processor, evaluator, row, prompt, positions, values, settings_greedy)
                donor_row = subset[donor][3]
                swaps.append({"row_index": index, "population": population, "donor": donor,
                              "route": result["route"], "success": result["success"],
                              "follows_donor": result["route"] == own[donor * checks["swap_every"]],
                              "keeps_own": result["route"] == own[local * checks["swap_every"]],
                              "solves_donor_map": bool(evaluator.score(donor_row, text)["success"])})
            with (output / f"{name}-swap.jsonl").open("w", encoding="utf-8") as log:
                log.writelines(json.dumps(s) + "\n" for s in swaps)
        adjuster.remove()
        patch.remove()
        summary[name] = {"seconds": time.time() - started, "swaps": len(swaps)}
        print(f"checkpoint {name}: {summary[name]}", flush=True)
        del model
        torch.cuda.empty_cache()
    json_write(output / "summary.json", summary)
    return summary


# Weights (CPU) -------------------------------------------------------------------------------

def run_weights(runs, output):
    """Q8: per layer group, the norm of each run's weights minus LIVR's, and LIVR's norm."""
    from safetensors import safe_open
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def tensors(path):
        listing = Path(path) / "model.safetensors.index.json"
        if listing.exists():
            index = json.loads(listing.read_text())["weight_map"]
        else:  # a single-file checkpoint
            with safe_open(str(Path(path) / "model.safetensors"), framework="pt") as single:
                index = {key: "model.safetensors" for key in single.keys()}
        return index, {f: safe_open(str(Path(path) / f), framework="pt") for f in set(index.values())}

    base_index, base_files = tensors(runs["livr"])
    others = {name: tensors(path) for name, path in runs.items() if name != "livr"}
    result = {name: {} for name in others}
    base_norms = {}
    for count, (key, file) in enumerate(sorted(base_index.items())):
        base = base_files[file].get_tensor(key).double()
        group = group_of(key)
        base_norms[group] = base_norms.get(group, 0.0) + float(base.square().sum())
        for name, (index, files) in others.items():
            if key not in index:
                continue
            diff = files[index[key]].get_tensor(key).double() - base
            result[name][group] = result[name].get(group, 0.0) + float(diff.square().sum())
        if count % 100 == 0:
            print(f"weights {count + 1}/{len(base_index)}", flush=True)
    for name, (index, files) in others.items():  # tensors LIVR does not have (C′'s heads)
        for key in sorted(set(index) - set(base_index)):
            result[name].setdefault("new: " + group_of(key), 0.0)
            result[name]["new: " + group_of(key)] += float(files[index[key]].get_tensor(key).double().square().sum())
    summary = {"livr_norm": {g: math.sqrt(v) for g, v in base_norms.items()},
               "difference_norm": {n: {g: math.sqrt(v) for g, v in groups.items()} for n, groups in result.items()},
               "relative_change": {n: {g: math.sqrt(v) / math.sqrt(base_norms[g]) for g, v in groups.items()
                                       if g in base_norms and base_norms[g] > 0} for n, groups in result.items()}}
    json_write(output / "summary.json", summary)
    return summary


# Analysis (CPU) ------------------------------------------------------------------------------

def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def mixed_unbiased(successes, draws, group):
    """Unbiased probability that a group of `group` draws (without replacement from the map's
    `draws` text rollouts with `successes` successes) has mixed rewards."""
    total = comb(draws, group)
    return 1 - (comb(successes, group) + comb(draws - successes, group)) / total


def pass_at(successes, draws, k):
    return 1 - comb(draws - successes, k) / comb(draws, k)


def bootstrap(values, rounds, seed):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return [means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]]


def config_metrics(maps, key, rule, null=False):
    """Per config, across maps: success, mixed share and its gain over text sampling (unbiased
    for groups of the config's size), rescues (text 0 of all its rollouts, config >= 1),
    breaks (text >= 0.5, config at least `break_drop` lower), pass@k, with a map bootstrap of
    the gain. `null`: the text rollouts' first group against the rest (the false-signal rate)."""
    gains, rates, text_rates, rescues, breaks, passes, text_passes, mixed = [], [], [], 0, 0, [], [], []
    for m in maps:
        text = m["text"]["successes"]
        if null:
            group, reference = text[:rule["group"]], text[rule["group"]:]
        else:
            if key not in m["configs"]:
                continue
            group, reference = m["configs"][key]["successes"], text
        n, s, ref_n, ref_s = len(group), sum(group), len(reference), sum(reference)
        p = ref_s / ref_n
        is_mixed = float(0 < s < n)
        mixed.append(is_mixed)
        gains.append(is_mixed - mixed_unbiased(ref_s, ref_n, n))
        rates.append(s / n)
        text_rates.append(p)
        rescues += ref_s == 0 and s > 0
        breaks += p >= 0.5 and s / n <= p - rule["break_drop"]
        passes.append(float(s > 0))
        text_passes.append(pass_at(ref_s, ref_n, n))
    if not gains:
        return None
    count = len(gains)
    gain = sum(gains) / count
    out = {"maps": count, "success": sum(rates) / count, "text_success": sum(text_rates) / count,
           "success_drop": (sum(text_rates) - sum(rates)) / count, "mixed_share": sum(mixed) / count,
           "mixed_gain": gain, "mixed_gain_ci": bootstrap(gains, rule["bootstrap"], 0), "rescues": rescues,
           "breaks": breaks, "pass_at_group": sum(passes) / count, "text_pass_at_group": sum(text_passes) / count}
    return out


def qualifies(cell, null, rule):
    return bool(cell and cell["mixed_gain"] >= rule["mixed_gain"] and cell["mixed_gain_ci"][0] > 0
                and cell["rescues"] >= cell["breaks"]
                and cell["rescues"] - cell["breaks"] >= null["rescues"] - null["breaks"]
                and cell["success_drop"] <= rule["max_success_drop"])


def soft_metrics(maps, layer, key):
    """Mean change of the label route's log-probability, and mean KL from the unperturbed to the
    perturbed first-move distribution, over draws; split by LIVR's greedy success."""
    out = {}
    for name, keep in (("greedy_failures", lambda m: not m["greedy_success"]), ("all", lambda m: True)):
        deltas, kls, firsts = [], [], []
        for m in maps:
            cell = m.get("soft", {}).get(str(layer))
            if not keep(m) or not cell or key not in cell:
                continue
            base = cell["none"]
            q = base["first"]
            label_first = MOVE_ORDER.index(m["label"][0])
            for d in cell[key]:
                deltas.append(d["label_logp"] - base["label_logp"])
                kls.append(sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(q, d["first"])))
                firsts.append(d["first"][label_first] - q[label_first])
        if deltas:
            out[name] = {"draws": len(deltas), "label_logp_change": sum(deltas) / len(deltas),
                         "label_logp_improved_share": sum(x > 0 for x in deltas) / len(deltas),
                         "first_move_kl": sum(kls) / len(kls),
                         "label_first_move_probability_change": sum(firsts) / len(firsts)}
    return out


def fit_heads(tuples, settings, scale, lr, steps, seed=0):
    """Offline: C′'s heads trained by the latent term alone on recorded (prior, coefficients,
    advantage) groups at behaviour noise `scale`. Returns sigma and |mu| over training and, on
    held-out maps, the change of the latent objective and the cosine between mu and the
    advantage-weighted mean coefficient."""
    torch.manual_seed(seed)
    priors = torch.stack([t["prior"].float() for t in tuples])
    size, slots, k = priors.shape[-1], priors.shape[1], tuples[0]["coefficients"].shape[-1]
    heads = BasisHeads(size, size, scale, sigma_bounds(scale), slots, k)
    heads.reset_parameters()
    fits = settings["head_fits"]
    mixed = [i for i, t in enumerate(tuples) if t["rewards"].std() > 0]
    heldout = mixed[-min(fits["heldout_maps"], max(1, len(mixed) // 4)):]
    train = [i for i in mixed if i not in heldout]
    optimizer = torch.optim.Adam(heads.parameters(), lr=lr)
    rng = random.Random(seed)

    def objective(indices, grad=True):
        total = 0.0
        with torch.set_grad_enabled(grad):
            for i in indices:
                t = tuples[i]
                mean, sigma = heads(head_input(priors[i:i + 1]))
                for coefficient, advantage in zip(t["coefficients"], t["advantages"].tolist()):
                    logp = gaussian_log_prob(coefficient[None].float(), torch.zeros_like(coefficient[None]).float(),
                                             mean, sigma, "sum")
                    total = total - advantage * logp / (len(indices) * len(t["advantages"]))
        return total

    def alignment_cosine(indices):
        values = []
        with torch.no_grad():
            for i in indices:
                t = tuples[i]
                target = (t["advantages"][:, None, None] * t["coefficients"]).sum(0)
                mean, _ = heads(head_input(priors[i:i + 1]))
                values.append(float(torch.nn.functional.cosine_similarity(mean.reshape(-1), target.reshape(-1), 0)))
        return sum(values) / len(values) if values else None

    start = float(objective(heldout, False)) if heldout else None
    trace = []
    for step in range(1, steps + 1):
        if not train:
            break
        batch = [rng.choice(train) for _ in range(fits["maps_per_step"])]
        optimizer.zero_grad()
        objective(batch).backward()
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            with torch.no_grad():
                mean, sigma = heads(head_input(priors))
            trace.append({"step": step, "sigma_mean": float(sigma.mean()), "mean_abs_mu": float(mean.abs().mean()),
                          "heldout_alignment": alignment_cosine(heldout)})
    end = float(objective(heldout, False)) if heldout else None
    return {"scale": scale, "learning_rate": lr, "steps": steps, "train_groups": len(train),
            "heldout_groups": len(heldout), "trace": trace,
            "heldout_objective_change": None if start is None else start - end}


def sweep_tuples(maps, states, layer, scale):
    """C′-equivalent (prior, coefficient = scale * eps, advantage) groups from the sweep's PCA
    rollouts at `layer` and `scale` (heads at mu = 0, sigma = scale)."""
    position = {r: i for i, r in enumerate(states["rows"])}
    out = []
    for m in maps:
        cell = m["configs"].get(f"pca/{layer}/{scale}")
        if not cell or "eps" not in cell or m["row_index"] not in position:
            continue
        rewards = torch.tensor([float(s) for s in cell["successes"]])
        out.append({"row": m["row_index"], "prior": states["states"][layer][position[m["row_index"]]],
                    "coefficients": scale * torch.tensor(cell["eps"]), "rewards": rewards,
                    "advantages": group_advantages(rewards)})
    return out


def merge_sweeps(paths):
    merged = {}
    for path in paths:
        for m in jsonl(Path(path) / "maps.jsonl"):
            cell = merged.setdefault(m["row_index"], {k: v for k, v in m.items() if k not in ("configs", "soft", "decision")}
                                     | {"configs": {}, "soft": {}, "decision": {}})
            cell["configs"] |= m["configs"]
            cell["soft"] |= m["soft"]
            cell["decision"] |= m.get("decision", {})
            if "text" in m:
                cell["text"] = m["text"]
    return [merged[k] for k in sorted(merged)]


def summarise_gradients(rows):
    if not rows:
        return None
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict) and k not in ("mixed_groups", "rollout_success")]
    return {"batches": len(rows), "mixed_groups": [r["mixed_groups"] for r in rows],
            "rollout_success": [r["rollout_success"] for r in rows], "per_batch": {k: [r.get(k) for r in rows] for k in keys}}


def summarise_checkpoints(path, runs):
    out = {}
    for name in runs:
        rows, swaps = jsonl(Path(path) / f"{name}.jsonl"), jsonl(Path(path) / f"{name}-swap.jsonl")
        if not rows:
            continue
        noisy = [r for r in rows if "noise" in r]
        cell = {"maps": len(rows), "greedy_success": sum(r["success"] for r in rows),
                "label_logp_mean": sum(r["label_logp"] for r in rows) / len(rows),
                "label_first_move_probability_mean": sum(r["label_first_move_probability"] for r in rows) / len(rows),
                "swap": {"maps": len(swaps), "follows_donor": sum(s["follows_donor"] for s in swaps),
                         "keeps_own": sum(s["keeps_own"] for s in swaps),
                         "own_map_success": sum(s["success"] for s in swaps),
                         "solves_donor_map": sum(s["solves_donor_map"] for s in swaps)},
                "noise": {}}
        for scale in (noisy[0]["noise"] if noisy else {}):
            changed = [route != r["route"] for r in noisy for route in r["noise"][scale]["routes"]]
            succ = [s for r in noisy for s in r["noise"][scale]["successes"]]
            cell["noise"][scale] = {"draws": len(changed), "route_change_rate": sum(changed) / len(changed),
                                    "success_rate": sum(succ) / len(succ),
                                    "base_success_rate": sum(r["success"] for r in noisy) / len(noisy)}
        out[name] = cell
    return out


def summarise_search(path, maps):
    path = Path(path)
    verdict = jsonl(path / "verdict-search.jsonl")
    latent = jsonl(path / "latent-search.jsonl")
    text = {m["row_index"]: m["text"]["successes"] for m in maps if "text" in m}
    failures = [r["row_index"] for r in verdict]
    out = {"greedy_failures": len(failures),
           "verdict_search_solved": sum(r["solved"] for r in verdict),
           "text_pass_at_64": sum(any(text[i]) for i in failures if i in text)}
    solved_verdict = {r["row_index"] for r in verdict if r["solved"]}
    solved_text = {i for i in failures if i in text and any(text[i])}
    out["latent_search"] = {}
    for layer in sorted({r["layer"] for r in latent}):
        rows = [r for r in latent if r["layer"] == layer]
        solved = {r["row_index"] for r in rows if any(r["successes"])}
        per_scale = {}
        for j, scale in enumerate(rows[0]["scales"]):
            n = rows[0]["draws_per_scale"]
            per_scale[str(scale)] = {"maps_solved": sum(any(r["successes"][j * n:(j + 1) * n]) for r in rows),
                                     "draw_success": sum(sum(r["successes"][j * n:(j + 1) * n]) for r in rows) / (n * len(rows))}
        out["latent_search"][str(layer)] = {"maps_solved": len(solved), "per_scale": per_scale,
                                            "solved_only_by_latent": len(solved - solved_verdict - solved_text),
                                            "also_by_verdict_search": len(solved & solved_verdict),
                                            "also_by_text_sampling": len(solved & solved_text)}
    return out


def analyse(sweeps, gradients, checkpoints, weights, output, settings, runs=()):
    """Every part's readings, the offline head fits, and the rule (docs/LATENT_DIAGNOSTICS_SPEC.md)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rule = settings["rule"] | {"group": settings["rollouts"]}
    maps = merge_sweeps(sweeps)
    maps_text = [m for m in maps if "text" in m]
    null = config_metrics(maps_text, None, rule, null=True)
    keys = sorted({k for m in maps for k in m["configs"] if "#" not in k},  # key#2...: repeats, read by `confirm`
                  key=lambda k: (k.split("/")[0], int(k.split("/")[1]),
                                                                         float(k.split("/")[2])))
    configs = {}
    for key in keys:
        cell = config_metrics(maps_text, key, rule)
        if cell is None:
            continue
        family, layer, scale = key.split("/")
        if family != "answer":
            cell["soft"] = soft_metrics(maps, layer, f"{family}/{scale}")
        cell["qualifies"] = qualifies(cell, null, rule)
        configs[key] = cell
    bases = {}
    for path in sweeps:
        if (Path(path) / "bases.json").exists():
            bases |= json.loads((Path(path) / "bases.json").read_text())
    decision_dims = {}
    for m in maps:
        for layer, d in m.get("decision", {}).items():
            decision_dims.setdefault(layer, []).append(d["dims"])
    text_stats = None
    if maps_text:
        draws = len(maps_text[0]["text"]["successes"])
        text_stats = {"maps": len(maps_text), "rollout_success": sum(sum(m["text"]["successes"]) for m in maps_text) / (draws * len(maps_text)),
                      "pass_at": {str(k): sum(pass_at(sum(m["text"]["successes"]), draws, k) for m in maps_text) / len(maps_text)
                                  for k in (1, 8, 16, 32, 64) if k <= draws},
                      "mixed_share_8": sum(mixed_unbiased(sum(m["text"]["successes"]), draws, rule["group"]) for m in maps_text) / len(maps_text),
                      "never_solved": sum(not any(m["text"]["successes"]) for m in maps_text),
                      "greedy_success": sum(m["greedy_success"] for m in maps_text)}

    head_fits = []
    fits = settings["head_fits"]
    states_path = next((Path(p) / "states.pt" for p in sweeps if (Path(p) / "states.pt").exists()
                        and fits["layer"] in torch.load(Path(p) / "states.pt")["states"]), None)
    if states_path:
        states = torch.load(states_path)
        for scale in fits["scales"]:
            data = sweep_tuples(maps, states, fits["layer"], scale)
            for lr in fits["learning_rates"]:
                for steps in fits["steps"]:
                    head_fits.append({"source": "sweep"} | fit_heads(data, settings, scale, lr, steps))
    tuples_path = Path(gradients) / "head-tuples.pt" if gradients else None
    if tuples_path and tuples_path.exists():
        data = torch.load(tuples_path)
        for lr in fits["learning_rates"]:
            for steps in fits["steps"]:
                head_fits.append({"source": "c-rollouts"} | fit_heads(data, settings, 4.0, lr, steps))

    summary = {
        "rule": rule, "text": text_stats, "null": null,
        "qualifying": [k for k, v in configs.items() if v["qualifies"]],
        "configs": configs, "bases": bases,
        "decision_dims": {k: sum(v) / len(v) for k, v in decision_dims.items()},
        "gradients": summarise_gradients(jsonl(Path(gradients) / "gradients.jsonl")) if gradients else None,
        "search": summarise_search(gradients, maps) if gradients and (Path(gradients) / "verdict-search.jsonl").exists() else None,
        "checkpoints": summarise_checkpoints(checkpoints, runs) if checkpoints else None,
        "weights": json.loads((Path(weights) / "summary.json").read_text()) if weights and (Path(weights) / "summary.json").exists() else None,
        "head_fits": head_fits}
    json_write(output / "summary.json", summary)
    (output / "report.md").write_text(report(summary), encoding="utf-8")
    return summary


def report(summary):
    lines = ["# Latent diagnostics: readings", ""]
    text = summary["text"]
    if text:
        lines += [f"Text sampling (T = 1), {text['maps']} pilot maps: rollout success {text['rollout_success']:.3f}, "
                  f"mixed share for 8 {text['mixed_share_8']:.3f}, never solved in 64: {text['never_solved']}, "
                  f"pass@k {json.dumps({k: round(v, 3) for k, v in text['pass_at'].items()})}.", ""]
    null = summary["null"]
    if null:
        lines += [f"Null (text's first 8 against its other 56): rescues {null['rescues']}, breaks {null['breaks']}, "
                  f"mixed gain {null['mixed_gain']:+.3f}.", ""]
    lines += [f"Qualifying configurations: {', '.join(summary['qualifying']) or 'none'}.", "",
              "| Config | Success | Drop | Mixed | Gain (95% CI) | Rescues | Breaks | Δ log p(label), failures | KL first move |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, c in summary["configs"].items():
        soft = (c.get("soft") or {}).get("greedy_failures", {})
        lines.append(f"| {key} | {c['success']:.3f} | {c['success_drop']:+.3f} | {c['mixed_share']:.3f} | "
                     f"{c['mixed_gain']:+.3f} ({c['mixed_gain_ci'][0]:+.3f}, {c['mixed_gain_ci'][1]:+.3f}) | "
                     f"{c['rescues']} | {c['breaks']} | {soft.get('label_logp_change', float('nan')):+.3f} | "
                     f"{soft.get('first_move_kl', float('nan')):.4f} |")
    lines += ["", "Full numbers (bases and probes, gradients, search, checkpoints, weights, head fits) are in summary.json."]
    return "\n".join(lines) + "\n"


# Confirmation (spec amendment of 2026-11-19) ---------------------------------------------------

def confirm(sweep, text_run, settings, output, reference=None):
    """`settings["confirm"]["config"]` on the confirmation sweep's maps (`settings["map_range"]` of
    the pilot stream) against the text run's rollouts, by the fixed rule: the mixed-share gain
    averaged over the repeated groups of 8 per map (map bootstrap), rescues and breaks from the
    first group only (as in Q4), success drop over all groups. Also the secondary configs, the
    pooled estimate with the reference sweep's maps (descriptive), and the run's checks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = settings["confirm"]
    rule = settings["rule"] | {"group": settings["rollouts"]}
    run, text_rows = jsonl(Path(sweep) / "maps.jsonl"), jsonl(Path(text_run) / "maps.jsonl")
    text = {m["row_index"]: m for m in text_rows}
    start, end = settings["map_range"]
    expected_rows, rows = [m["row_index"] for m in text_rows][start:end], [m["row_index"] for m in run]
    known = [m for m in run if m["row_index"] in text]
    checks = {"maps": len(run), "expected_maps": len(expected_rows), "complete": rows == expected_rows,
              "rows_follow_the_range": rows == expected_rows[:len(rows)],
              "greedy_route_mismatches": sum(m["greedy_route"] != text[m["row_index"]]["greedy_route"] for m in known),
              "greedy_success_mismatches": sum(m["greedy_success"] != text[m["row_index"]]["greedy_success"]
                                               for m in known)}
    maps = [m | {"text": text[m["row_index"]]["text"]} for m in known]
    null = config_metrics(maps, None, rule, null=True)

    def reading(key):
        family, layer, scale = key.split("/")
        present = [m for m in maps if key in m["configs"]]
        if not present:
            return None
        first = config_metrics(present, key, rule)
        gains, rates, text_rates, ratios = [], [], [], []
        for m in present:
            t = m["text"]["successes"]
            groups = [c for k, c in m["configs"].items() if k == key or k.startswith(key + "#")]
            gains.append(sum(float(0 < sum(g["successes"]) < len(g["successes"]))
                             - mixed_unbiased(sum(t), len(t), len(g["successes"])) for g in groups) / len(groups))
            outcomes = [s for g in groups for s in g["successes"]]
            rates.append(sum(outcomes) / len(outcomes))
            text_rates.append(sum(t) / len(t))
            if "energy_ratio" in m["configs"][key]:
                ratios.append(m["configs"][key]["energy_ratio"])
        n = len(present)
        cell = {"maps": n, "groups_per_map": len([k for k in present[0]["configs"] if k == key or k.startswith(key + "#")]),
                "mixed_gain": sum(gains) / n, "mixed_gain_ci": bootstrap(gains, rule["bootstrap"], 0),
                "success": sum(rates) / n, "text_success": sum(text_rates) / n,
                "success_drop": (sum(text_rates) - sum(rates)) / n,
                "rescues": first["rescues"], "breaks": first["breaks"], "first_group": first,
                "soft": soft_metrics(present, layer, f"{family}/{scale}")}
        if ratios:
            slots = list(zip(*ratios))
            cell["energy_ratio"] = {"mean": [sum(s) / len(s) for s in slots], "min": [min(s) for s in slots],
                                    "max": [max(s) for s in slots]}
        cell["confirmed"] = bool(cell["mixed_gain"] >= rule["mixed_gain"] and cell["mixed_gain_ci"][0] > 0
                                 and cell["rescues"] >= cell["breaks"]
                                 and cell["rescues"] - cell["breaks"] >= null["rescues"] - null["breaks"]
                                 and cell["success_drop"] <= rule["max_success_drop"])
        return cell

    primary = reading(spec["config"])
    summary = {"rule": rule, "config": spec["config"], "checks": checks, "null": null, "primary": primary,
               "confirmed": bool(primary and primary["confirmed"] and checks["complete"]),
               "secondary": {key: reading(key) for key in spec.get("secondary", [])}}
    if reference:
        earlier = [m | {"text": text[m["row_index"]]["text"]} for m in jsonl(Path(reference) / "maps.jsonl")
                   if m["row_index"] in text and spec["config"] in m["configs"]]
        summary["pooled_first_group"] = {"reference_maps": len(earlier),
                                         "reading": config_metrics(earlier + maps, spec["config"], rule)}
    json_write(output / "confirm.json", summary)
    (output / "confirm.md").write_text(confirm_report(summary), encoding="utf-8")
    return summary


def confirm_report(summary):
    c, checks, null = summary["primary"], summary["checks"], summary["null"]
    lines = ["# Confirmation of " + summary["config"], "",
             f"Maps: {checks['maps']} of {checks['expected_maps']} (complete: {checks['complete']}); greedy answers "
             f"differing from the text run's: {checks['greedy_route_mismatches']} routes, "
             f"{checks['greedy_success_mismatches']} outcomes.",
             f"Null (text's first 8 against its other 56): rescues {null['rescues']}, breaks {null['breaks']}.", ""]
    if c:
        lines += [f"**{'Confirmed' if summary['confirmed'] else 'Not confirmed'}.** Mixed-share gain {c['mixed_gain']:+.3f} "
                  f"(95% CI {c['mixed_gain_ci'][0]:+.3f} to {c['mixed_gain_ci'][1]:+.3f}; {c['groups_per_map']} groups of 8 per map), "
                  f"rescues {c['rescues']}, breaks {c['breaks']} (first group), success {c['success']:.3f} against text "
                  f"{c['text_success']:.3f} (drop {c['success_drop']:+.3f}).", ""]
    lines += ["| Config | Maps | Gain (95% CI) | Rescues | Breaks | Success drop | Size ratio (mean per slot) | Passes |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, cell in [(summary["config"], c)] + list(summary["secondary"].items()):
        if cell:
            ratio = ", ".join(f"{x:.2f}" for x in cell.get("energy_ratio", {}).get("mean", []))
            lines.append(f"| {key} | {cell['maps']} | {cell['mixed_gain']:+.3f} ({cell['mixed_gain_ci'][0]:+.3f}, "
                         f"{cell['mixed_gain_ci'][1]:+.3f}) | {cell['rescues']} | {cell['breaks']} | "
                         f"{cell['success_drop']:+.3f} | {ratio} | {cell['confirmed']} |")
    pooled = summary.get("pooled_first_group")
    if pooled and pooled["reading"]:
        r = pooled["reading"]
        lines += ["", f"Pooled with the {pooled['reference_maps']} earlier maps (first group only; descriptive): gain "
                      f"{r['mixed_gain']:+.3f} ({r['mixed_gain_ci'][0]:+.3f}, {r['mixed_gain_ci'][1]:+.3f}), rescues "
                      f"{r['rescues']}, breaks {r['breaks']}, success drop {r['success_drop']:+.3f}."]
    return "\n".join(lines) + "\n"
