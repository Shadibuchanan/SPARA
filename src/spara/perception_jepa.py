"""Latent JEPA (docs/PERCEPTION_JEPA_SPEC.md): LIVR's four latent tokens predict an embedding of
every map cell's image tokens, with LeJEPA's SIGReg against collapse.

- Context: the latents' hidden states at one decoder layer, through a projector.
- Targets: each cell's overlap-weighted mean of the image-token states at one decoder layer
  (`latent_battery.cell_weights`), detached from the language model, centred by the start
  checkpoint's mean for that (grid size, row, column), through a second projector. Image tokens
  precede the latents, so the bottleneck mask leaves the targets unchanged.
- Predictor: from the four projected latents and the cell's normalised position, the cell's
  projected target. Squared error, as LeJEPA; no stop-gradient through the projectors.
- SIGReg (Balestriero and LeCun, arXiv 2511.08544; the reference SIGReg in
  galilai-group/lejepa MINIMAL.md at commit c293d29) on each projector's output. Training reads
  one map per forward, so each SIGReg statistic pools the current embeddings with a queue of
  detached recent ones; gradients reach only the current ones.

The targets come from the image itself: no layout labels and no helper images. The grid size is
used to pool image tokens into cells.

Stages: `run_gate` (one GPU job: the image-token probe at 7 x 7 and 8 x 8, the target layer, the
centring means, the heads' warm start, the loss weight, and the swap checks), `run_jepa_arm`
(EJ2: round 2's EB2 with this loss in both phases), `compare` (CPU: the fixed readings).

The retry with fixed targets (EJF, the spec's "Retry with fixed targets"): `run_features` (GPU: E's
latents and cells, saved), `run_fixed_gate` (CPU: fixed targets from those cells, which no learned
projector touches, so position alone predicts zero; the warm start; gate 3), then `run_jepa_arm`
with the gate's heads and stored targets, no SIGReg, and the loss weight set at the start of the arm.
"""
import json
import math
import statistics
import time
from collections import Counter, deque
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import records
from .detour_probe import find, manhattan, outcomes, step
from .detour_scaling import check_sets, jsonl, paired
from .expert_round2 import (ROUND1, bootstrap_difference, evaluate, loss_summary, outcome, phases,
                            probe_predictions, probe_score, ridge, training_rows)
from .latent_battery import LatentPatch, answer, cell_weights, latent_positions, prompt_states, swap_donors
from .planning_probe import distances
from .runtime import json_write, seed_all
from .serialization import encode, latent_kwargs
from .training import amp, to_device, train_stage

HEADS_FILE = "jepa_heads.pt"
FEATURES_FILE = "features.pt"


def map_key(split, row):
    """A map's identity: its split and image. Map IDs are not unique within the released training
    split (they repeat across levels), so they cannot key a map."""
    return split, row["image_input"]
GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


# The objective ---------------------------------------------------------------------------------

class SIGReg(torch.nn.Module):
    """Epps-Pulley test of every random 1-D projection against a standard normal, as LeJEPA's
    reference: t in [0, 3] with 17 trapezoid knots weighted by exp(-t^2/2), integrated on [0, t_max]
    and doubled; the statistic is scaled by the number of samples and averaged over directions.
    Directions are drawn from `generator`, so runs reproduce."""

    def __init__(self, directions=256, knots=17, t_max=3.0):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.directions = directions
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, z, generator=None):
        z = z.float()
        a = torch.randn(z.shape[-1], self.directions, generator=generator).to(z.device)
        a = a / a.norm(dim=0, keepdim=True)
        x_t = (z @ a).unsqueeze(-1) * self.t.to(z.device)
        err = (x_t.cos().mean(-3) - self.phi.to(z.device)).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights.to(z.device)) * z.shape[-2]).mean()


class Queue:
    """The last `capacity` embeddings (rows), detached, for SIGReg's sample."""

    def __init__(self, capacity, items=None):
        self.capacity, self.items = capacity, None if items is None else items[:capacity].float().cpu()

    def with_current(self, current):
        return current if self.items is None else torch.cat([current, self.items.to(current.device)])

    def push(self, current):
        current = current.detach().float().cpu()
        self.items = (current if self.items is None else torch.cat([current, self.items]))[:self.capacity]


def projector(hidden, width, out):
    """Parameter-free layer norm, then a two-layer MLP. No batch norm: one map per forward."""
    return torch.nn.Sequential(torch.nn.LayerNorm(hidden, elementwise_affine=False), torch.nn.Linear(hidden, width),
                               torch.nn.GELU(), torch.nn.Linear(width, out))


class Standardise(torch.nn.Module):
    """A fixed per-(latent, channel) standardisation of the latents' states, set once from the start
    checkpoint's training maps (`set`). E's latents carry a few large, nearly constant channels: without
    this, a layer norm leaves the predictor almost the same input for every map (gate 3, job 7116395)."""

    def __init__(self, latents, hidden):
        super().__init__()
        self.register_buffer("mean", torch.zeros(latents, hidden))
        self.register_buffer("std", torch.ones(latents, hidden))

    def set(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


def position_features(size, frequencies, device=None):
    """[size*size, 4*frequencies + 1]: Fourier features of each cell's normalised centre, row-major,
    and 1/size."""
    centres = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / size
    rows, cols = centres.repeat_interleave(size), centres.repeat(size)
    scales = math.pi * 2.0 ** torch.arange(frequencies, dtype=torch.float32, device=device)
    angles = torch.cat([rows[:, None] * scales, cols[:, None] * scales], -1)
    return torch.cat([angles.sin(), angles.cos(), torch.full((size * size, 1), 1.0 / size, device=device)], -1)


class JepaHeads(torch.nn.Module):
    """Context projector (each latent's state), target projector (each cell's state) and a small
    predictor (all four projected latents in order plus the cell's position, to that cell's
    projected target). The predictor is kept small, with dropout, so that it cannot memorise
    training layouts from a map-identifying context (docs/PERCEPTION_JEPA_SPEC.md)."""

    def __init__(self, hidden, latents=4, dim=256, width=1024, predictor_width=512, dropout=0.1, frequencies=8,
                 target="learned", target_dim=None, standardise=False):
        super().__init__()
        target_dim = target_dim or dim
        if target not in {"learned", "fixed"} or (target == "learned" and target_dim != dim):
            raise ValueError("target is 'learned' (a projector to `dim`) or 'fixed' (given targets of `target_dim`)")
        self.latents, self.dim, self.frequencies = latents, dim, frequencies
        self.context = projector(hidden, width, dim)
        if standardise:  # the fixed standardisation first, set by gate 3 (`Standardise.set`)
            self.context = torch.nn.Sequential(Standardise(latents, hidden), *self.context)
        # Fixed targets are projected once, before training (`fixed_projection`); the heads take them as given.
        self.target = projector(hidden, width, dim) if target == "learned" else torch.nn.Identity()
        self.predictor = torch.nn.Sequential(torch.nn.Linear(latents * dim + 4 * frequencies + 1, predictor_width),
                                             torch.nn.GELU(), torch.nn.Dropout(dropout),
                                             torch.nn.Linear(predictor_width, target_dim))
        self.config = {"hidden": hidden, "latents": latents, "dim": dim, "width": width,
                       "predictor_width": predictor_width, "dropout": dropout, "frequencies": frequencies,
                       "target": target, "target_dim": target_dim, "standardise": standardise}

    def predict(self, z_context, size):
        """z_context: [latents, dim] for one map; returns [size*size, dim]."""
        positions = position_features(size, self.frequencies, z_context.device)
        flat = z_context.reshape(1, -1).expand(len(positions), -1)
        return self.predictor(torch.cat([flat, positions], -1))


def cell_states(hidden_states, layer, input_ids, image_token_id, grid_thw, merge, size):
    """[size*size, hidden]: each cell's overlap-weighted mean of the image-token states at decoder
    layer `layer` (one-based output, as the latent battery indexes it), for one sequence."""
    if int(grid_thw[0]) != 1:
        raise ValueError("Cell pooling expects a single image frame")
    image = input_ids[0] == image_token_id
    weights = cell_weights(size, grid_thw, merge).to(hidden_states[layer].device, torch.float32)
    return weights @ hidden_states[layer][0, image].float()


def unexplained(predicted, target):
    """The share of the targets' variance the prediction leaves: summed squared error over the summed
    squared deviation from the targets' mean (1 - R^2). Raw error is not comparable across training,
    because the target projector is learned and SIGReg changes its scale."""
    return float((predicted - target).square().sum() / (target - target.mean(0)).square().sum().clamp(min=1e-12))


def objective(heads, sigreg, z_context, z_target, size, context_sample, target_sample, generator, lam):
    """(1 - lam) * prediction error + lam * mean SIGReg, LeJEPA's weighting, for one or more maps:
    z_context [latents, dim] and z_target [size*size, dim] for one map, with the SIGReg samples
    (current rows first) given separately."""
    error = (heads.predict(z_context, size) - z_target).square().mean()
    if lam == 0:  # fixed targets (EJF): the squared error alone
        return error, error, None, None
    reg_context, reg_target = sigreg(context_sample, generator), sigreg(target_sample, generator)
    return (1 - lam) * error + lam * (reg_context + reg_target) / 2, error, reg_context, reg_target


class LatentJepa:
    """The heads, their settings, the centring means and the SIGReg queues, for
    spara.training.train_stage.

    settings: context_layer, target_layer (one-based decoder outputs), weight (beta, on the whole
    term), sigreg_weight (lambda), head_learning_rate, head_weight_decay, queue_context and
    queue_target (SIGReg samples), shuffle_pool, directions, seed.

    `targets`, {(split, map ID): [cells, components]}, gives fixed targets (EJF): `bind` the rows'
    splits before each training stage. Without it the targets come from the hidden states (EJ2)."""

    released_only = False
    clip_separately = True  # the heads' gradient must not enter the language model's clipping norm
    loss_key = "jepa_loss"
    metric_keys = ("jepa_loss", "jepa_error", "jepa_unexplained", "sigreg_context", "sigreg_target", "shuffled_gap")
    head_file = HEADS_FILE

    def __init__(self, settings, heads, centres, num_layers, merge, image_token_id, latent_ids, queues=None,
                 targets=None):
        for key in ("context_layer", "target_layer"):
            if not 1 <= settings[key] <= num_layers:
                raise ValueError(f"{key} must be a decoder layer output, 1 to {num_layers}")
        if settings["weight"] < 0 or not 0 <= settings["sigreg_weight"] < 1 or settings["head_learning_rate"] <= 0:
            raise ValueError("Need weight >= 0, 0 <= sigreg_weight < 1 and a positive head learning rate")
        self.settings, self.weight, self.head = settings, settings["weight"], heads
        self.centres = {int(k): v for k, v in centres.items()}
        self.merge, self.image_token_id, self.latent_ids = merge, image_token_id, latent_ids
        self.sigreg = SIGReg(settings.get("directions", 256))
        queues = queues or {}
        self.queues = {"context": Queue(settings["queue_context"], queues.get("context")),
                       "target": Queue(settings["queue_target"], queues.get("target"))}
        self.contexts = deque(maxlen=settings.get("shuffle_pool", 64))  # whole maps, for the shuffled gap
        self.generator = torch.Generator().manual_seed(settings.get("seed", 0))
        self.sizes = None
        self.targets, self.keys = targets, None

    def bind(self, data):
        """The (split, map ID) of each (split, row) that train_stage will read, for the fixed targets."""
        self.keys = [map_key(split, row) for split, row in data]

    def prepare(self, tokenizer, rows):
        self.sizes = [len(row["map_desc"]) for row in rows]
        if self.targets is None:
            missing = set(self.sizes) - self.centres.keys()
            if missing:
                raise ValueError(f"No centring means for grid sizes {sorted(missing)}")
        else:
            if self.keys is None or [k[1] for k in self.keys] != [row["image_input"] for row in rows]:
                raise ValueError("Bind the rows' splits (LatentJepa.bind) before training on fixed targets")
            missing = [k for k in self.keys if k not in self.targets]
            if missing:
                raise ValueError(f"No fixed targets for {len(missing)} rows, such as {missing[0]}")
        return {"rows": len(rows), "sizes": {str(s): self.sizes.count(s) for s in sorted(set(self.sizes))}}

    def parameter_group(self):
        return {"params": list(self.head.parameters()), "lr": self.settings["head_learning_rate"],
                "weight_decay": self.settings.get("head_weight_decay", 0.0)}

    def terms(self, hidden_states, batch, size, key=None):
        """The latents' states (with their graph) and the centred cell states (detached), or the map's
        fixed targets."""
        positions = latent_positions(batch["input_ids"], self.latent_ids)
        context = hidden_states[self.settings["context_layer"]][0, positions]
        if self.targets is not None:
            return context, self.targets[key].to(context.device).float()
        with torch.no_grad():
            cells = cell_states(hidden_states, self.settings["target_layer"], batch["input_ids"],
                                self.image_token_id, batch["image_grid_thw"][0], self.merge, size)
            cells = cells - self.centres[size].to(cells.device)
        return context, cells

    def loss(self, hidden_states, index, batch):
        size = self.sizes[index]
        context, cells = self.terms(hidden_states, batch, size, None if self.targets is None else self.keys[index])
        with torch.autocast(context.device.type, enabled=False):
            z_context, z_target = self.head.context(context.float()), self.head.target(cells.float())
            target_sample = self.queues["target"].with_current(z_target)
            total, error, reg_c, reg_t = objective(
                self.head, self.sigreg, z_context, z_target, size, self.queues["context"].with_current(z_context),
                target_sample, self.generator, self.settings["sigreg_weight"])
            # Per-dimension variance of the recent targets: errors as a share of it are comparable over training.
            scale = float(target_sample.detach().var(0).mean().clamp(min=1e-12))
            gap = None
            if self.contexts:
                other = self.contexts[torch.randint(len(self.contexts), (1,), generator=self.generator).item()]
                with torch.no_grad():
                    wrong = (self.head.predict(other.to(z_context.device), size) - z_target).square().mean()
                    gap = float(wrong - error) / scale
        self.queues["context"].push(z_context)
        self.queues["target"].push(z_target)
        self.contexts.append(z_context.detach().float().cpu())
        return total, {"jepa_error": float(error), "jepa_unexplained": float(error) / scale,
                       "sigreg_context": None if reg_c is None else float(reg_c),
                       "sigreg_target": None if reg_t is None else float(reg_t), "shuffled_gap": gap}

    def save(self, path, extra=None):
        save_heads(path, self.settings, self.head, self.centres, {k: q.items for k, q in self.queues.items()},
                   self.targets, extra)


def save_heads(path, settings, heads, centres, queues, targets=None, extra=None):
    torch.save({"settings": settings, "config": heads.config,
                "state_dict": {k: v.cpu() for k, v in heads.state_dict().items()},
                "centres": {str(k): v.cpu() for k, v in centres.items()}, "queues": queues,
                "targets": targets} | (extra or {}), path)


def load_heads(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    heads = JepaHeads(**saved["config"])
    heads.load_state_dict(saved["state_dict"])
    return heads, {int(k): v for k, v in saved["centres"].items()}, saved


def from_saved(path, model):
    """The gate's heads, centring means, settings and SIGReg queues, as the arm's auxiliary loss."""
    heads, centres, saved = load_heads(path)
    heads.to(model.device)
    aux = LatentJepa(saved["settings"], heads, centres, model.config.num_hidden_layers,
                     model.config.vision_config.spatial_merge_size, model.config.image_token_id,
                     model.config.livr_latent_ids, saved["queues"], saved.get("targets"))
    aux.projection = saved.get("projection")  # fixed targets: the gate's projection, for `add_targets`
    return aux


# The gate job ------------------------------------------------------------------------------------

def require_round2_checkpoint(path, config, round2, rows=3000):
    """E2 or EB2: the completed second phase of a round 2 arm (docs/EXPERT_ITERATION_ROUND2_SPEC.md)."""
    progress = json.loads((Path(path) / "complete.json").read_text())["progress"]
    steps = math.ceil((rows - round2["phase_one_rows"]) / config["gradient_accumulation"])
    if (progress.get("pilot") or progress.get("stage") != 2 or progress.get("epoch") != round2["epochs"]
            or progress.get("next_row") != 0 or progress.get("optimizer_steps") != steps):
        raise ValueError("Expected a completed round 2 phase 2 checkpoint (E2 or EB2)")
    expected = config | {"epochs_per_stage": round2["epochs"], "learning_rate": round2["learning_rate"],
                         "warmup_ratio": round2["warmup_ratio"]}
    for key, value in expected.items():
        if key not in {"local_model_path", "checkpoint_every_steps"} and progress["config"].get(key) != value:
            raise ValueError(f"Checkpoint recipe mismatch: {key}")


def distinct_maps(data):
    """[(split, row)] once per map, in first-seen order."""
    seen, kept = set(), []
    for split, row in data:
        if map_key(split, row) not in seen:
            seen.add(map_key(split, row))
            kept.append((split, row))
    return kept


@torch.no_grad()
def dump(model, processor, data_root, items, context_layer, target_layers):
    """One unmasked prompt pass per map: the latents at `context_layer` and each cell's state at each
    of `target_layers`, in half precision on the CPU."""
    model.eval()
    model.config.livr_stage = 2
    merge, image_id = model.config.vision_config.spatial_merge_size, model.config.image_token_id
    contexts, cells = [], {layer: [] for layer in target_layers}
    for count, (split, row) in enumerate(items):
        prompt = to_device(encode(processor, data_root, split, row, training=False, **latent_kwargs(model)),
                           model.device)
        positions = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
        with amp(model.device):
            hidden = model(**prompt, output_hidden_states=True, use_cache=False).hidden_states
        contexts.append(hidden[context_layer][0, positions].float().half().cpu())
        for layer in target_layers:
            cells[layer].append(cell_states(hidden, layer, prompt["input_ids"], image_id, prompt["image_grid_thw"][0],
                                            merge, len(row["map_desc"])).half().cpu())
        if count % 200 == 0:
            print(f"dump {count + 1}/{len(items)}", flush=True)
    return contexts, cells


def balanced_accuracy(predicted, truth):
    scores = [(predicted[truth == k] == k).double().mean() for k in (0, 1) if (truth == k).any()]
    return float(sum(scores) / len(scores))


def hole_labels(row):
    desc = row["map_desc"]
    return torch.tensor([float(v == -1) for line in desc for v in line])


def layer_probe(cells, rows, fit, test, device):
    """Per grid size, one ridge probe from a cell's state to hole or not: fitted on the `fit` maps'
    cells, scored on the `test` maps' cells (balanced accuracy)."""
    result = {}
    for n in sorted({len(rows[i]["map_desc"]) for i in test}):
        f = [i for i in fit if len(rows[i]["map_desc"]) == n]
        t = [i for i in test if len(rows[i]["map_desc"]) == n]
        x_fit = torch.cat([cells[i] for i in f]).to(device)
        y_fit = torch.cat([hole_labels(rows[i]) for i in f])[:, None].to(device)
        scores, _ = ridge(x_fit, y_fit, torch.cat([cells[i] for i in t]).to(device))
        truth = torch.cat([hole_labels(rows[i]) for i in t]).long()
        result[str(n)] = balanced_accuracy((scores[:, 0].cpu() > 0.5).long(), truth)
    return result


def centres_of(cells, rows, keep):
    """Per grid size, the mean cell state at each position over the `keep` maps."""
    by_size = {}
    for i in keep:
        by_size.setdefault(len(rows[i]["map_desc"]), []).append(cells[i].float())
    return {n: torch.stack(c).mean(0) for n, c in by_size.items()}


def heads_error(heads, contexts, cells, sizes, device):
    """The share of target variance unexplained over these maps (`unexplained`), heads in evaluation
    mode, no gradient."""
    training = heads.training
    heads.eval()
    with torch.no_grad():
        predicted = torch.cat([heads.predict(heads.context(c.to(device).float()), n) for c, n in zip(contexts, sizes)])
        target = torch.cat([heads.target(t.to(device).float()) for t in cells])
    heads.train(training)
    return unexplained(predicted, target)


def warm_start(heads, sigreg, train, validation, settings, device):
    """Fit the heads on the start checkpoint's frozen features before any language-model update, so
    the first updates do not follow a random predictor. `train`, `validation`: (contexts, centred
    cells, sizes). Batches of maps; SIGReg on each batch's embeddings. Keeps the epoch with the
    lowest validation error. Returns the history and the last batch's embeddings for the queues."""
    plan, lam = settings["warm_start"], settings["sigreg_weight"]
    generator = torch.Generator().manual_seed(settings["seed"])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=plan["learning_rate"], weight_decay=settings["head_weight_decay"])
    history, best = [], (heads_error(heads, *validation, device), 0, {k: v.clone() for k, v in heads.state_dict().items()})
    contexts, cells, sizes = train
    for epoch in range(1, plan["epochs"] + 1):
        heads.train()
        order = torch.randperm(len(sizes), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), plan["maps_per_batch"]):
            batch = order[start:start + plan["maps_per_batch"]]
            z_c = [heads.context(contexts[i].to(device).float()) for i in batch]
            z_t = [heads.target(cells[i].to(device).float()) for i in batch]
            error = torch.stack([(heads.predict(c, sizes[i]) - t).square().mean() for c, t, i in zip(z_c, z_t, batch)]).mean()
            if lam:
                reg = (sigreg(torch.cat(z_c), generator) + sigreg(torch.cat(z_t), generator)) / 2
                loss = (1 - lam) * error + lam * reg
            else:
                loss = error
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        record = {"epoch": epoch, "loss": sum(losses) / len(losses),
                  "train_error": heads_error(heads, *[x[:256] for x in train], device),
                  "validation_error": heads_error(heads, *validation, device)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_error"] < best[0]:
            best = (record["validation_error"], epoch, {k: v.clone() for k, v in heads.state_dict().items()})
    heads.load_state_dict(best[2])
    heads.eval()
    with torch.no_grad():
        recent = order[-settings["queue_target"]:]
        queue_c = torch.cat([heads.context(contexts[i].to(device).float()) for i in recent])[:settings["queue_context"]]
        queue_t = torch.cat([heads.target(cells[i].to(device).float()) for i in recent])[:settings["queue_target"]]
    heads.train()
    return ({"history": history, "best_epoch": best[1], "best_validation_error": best[0]},
            {"context": queue_c.cpu(), "target": queue_t.cpu()})


def gradient_norms(model, aux, processor, data_root, rows, stage, config):
    """Per row: the norms of the answer loss's and the latent JEPA term's gradients with respect to
    the language model's trainable parameters, from one forward, as phase 1 trains."""
    model.train()
    model.visual.eval()
    model.config.livr_stage = stage
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for name, p in model.named_parameters() if p.requires_grad]
    heads = list(aux.head.parameters())
    aux.bind(rows)
    aux.prepare(processor.tokenizer, [row for _, row in rows])
    norms = []
    size = lambda grads: float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None)))
    for index, (split, row) in enumerate(rows):
        batch = to_device(encode(processor, data_root, split, row, **latent_kwargs(model)), model.device)
        with amp(model.device):
            out = model(**batch, use_cache=False, output_hidden_states=True)
            term, _ = aux.loss(out.hidden_states, index, batch)
        answer_grads = torch.autograd.grad(out.loss, params, retain_graph=True, allow_unused=True)
        term_grads = torch.autograd.grad(term, params + heads, allow_unused=True)
        norms.append({"answer": size(answer_grads), "jepa": size(term_grads[:len(params)]),
                      "jepa_heads": size(term_grads[len(params):]), "answer_loss": float(out.loss),
                      "jepa_term": float(term)})
        del out, answer_grads, term_grads
    model.zero_grad(set_to_none=True)
    return norms


def swap_check(model, processor, evaluator, data_root, split, rows, seed, max_new_tokens):
    """Greedy, unmasked: each map answered with its own latents and with another map's (same size,
    different layout; every layer's states at the four latent positions). Counts how often the
    swapped answer is the donor's own route, as the latent battery's swap."""
    model.eval()
    model.config.livr_stage = 2
    settings = GREEDY | {"max_new_tokens": max_new_tokens}
    patch = LatentPatch(model)
    ids = model.config.livr_latent_ids
    states, own = [], []
    try:
        for row in rows:
            prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
            positions = latent_positions(prompt["input_ids"], ids)
            states.append(prompt_states(model, patch, prompt, positions)[:, positions].half().cpu())
            own.append(answer(model, patch, processor, evaluator, row, prompt, None, None, settings)[0])
        identity = []
        for i in range(min(4, len(rows))):
            prompt = to_device(encode(processor, data_root, split, rows[i], training=False), model.device)
            positions = latent_positions(prompt["input_ids"], ids)
            again = answer(model, patch, processor, evaluator, rows[i], prompt, positions, states[i].float(), settings)[0]
            identity.append(again["route"] == own[i]["route"])
        donors = swap_donors(rows, seed)
        swapped = []
        for i, row in enumerate(rows):
            if i not in donors:
                continue
            prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
            positions = latent_positions(prompt["input_ids"], ids)
            result = answer(model, patch, processor, evaluator, row, prompt, positions,
                            states[donors[i]].float(), settings)[0]
            donor_route = own[donors[i]]["route"] or ""
            swapped.append({"row": i, "donor": donors[i], "grid_size": len(row["map_desc"]),
                            "follows_donor": result["route"] == own[donors[i]]["route"],
                            "first_two_follow_donor": len(donor_route) >= 2
                            and (result["route"] or "")[:2] == donor_route[:2],
                            "keeps_own": result["route"] == own[i]["route"], "success": result["success"]})
    finally:
        patch.remove()
    sizes = sorted({s["grid_size"] for s in swapped})
    return {"identity": identity, "maps": len(swapped), "follows_donor": sum(s["follows_donor"] for s in swapped),
            "first_two_follow_donor": sum(s["first_two_follow_donor"] for s in swapped),
            "keeps_own": sum(s["keeps_own"] for s in swapped), "own_success": sum(o["success"] for o in own),
            "by_size": {str(n): {"maps": sum(s["grid_size"] == n for s in swapped),
                                 "follows_donor": sum(s["follows_donor"] for s in swapped if s["grid_size"] == n)}
                        for n in sizes}}, swapped


def first_per_size(rows, count):
    kept, counts = [], Counter()
    for row in rows:
        n = len(row["map_desc"])
        if counts[n] < count:
            counts[n] += 1
            kept.append(row)
    return kept


def gate_items(data_root, round2, round1_routes, search_run, validation_count):
    """Every map the gate and the arm read, once: the training maps (as round 2 builds its rows), the
    pool, the held-out maps and the validation maps. Returns (items, groups, round 2's rows, counts)."""
    data, counts = training_rows(data_root, round2, round1_routes, search_run)
    trained = distinct_maps(data)
    pool = [(round2["pool_split"], r) for r in records(data_root, round2["pool_split"])]
    held = [(round2["heldout_split"], r) for r in records(data_root, round2["heldout_split"])]
    validation = validation_maps(data_root, round2, jsonl(Path(search_run) / "routes.jsonl"), validation_count)
    items = distinct_maps(trained + pool + held + validation)
    return items, {"training": trained, "pool": pool, "heldout": held, "validation": validation}, data, counts


def validation_maps(data_root, round2, routes, count):
    """Pool maps for the heads' validation: unsolved by round 2's search, so never trained on, and
    beyond the seen maps evaluated descriptively."""
    pool = records(data_root, round2["pool_split"])
    unsolved = {r["row_index"] for r in routes if not r["solved"]}
    return [(round2["pool_split"], pool[i]) for i in range(round2["seen_maps"], len(pool)) if i in unsolved][:count]


def run_gate(model, processor, evaluator, config, data_root, output, settings, round2, round1_routes, search_run,
             load_e2, run_manifest):
    """The gate job (docs/PERCEPTION_JEPA_SPEC.md). Stops early, with its reading recorded, if a gate
    fails. `load_e2()` returns E2's model for its descriptive swap check."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    device = model.device
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "round2_settings": round2,
                                                         "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    summary = {"settings": settings, "valid": False}

    def finish(reading):
        summary.update(reading=reading, seconds=time.time() - began,
                       peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)
        json_write(output / "summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)
        return summary

    # 1. Features of every training map, the pool and the held-out maps.
    items, groups, data, counts = gate_items(data_root, round2, round1_routes, search_run, settings["validation_maps"])
    trained, pool, held, validation = (groups[k] for k in ("training", "pool", "heldout", "validation"))
    index = {map_key(s, r): i for i, (s, r) in enumerate(items)}
    rows = [r for _, r in items]
    seed_all(config["seed"])
    started = time.time()
    contexts, cells = dump(model, processor, data_root, items, settings["context_layer"], settings["target_layers"])
    summary["dump"] = {"maps": len(items), "training_maps": len(trained), "seconds": time.time() - started}

    # 2. Gate: the image tokens separate holes at 7 x 7 and 8 x 8; the target layer.
    fit = [index[map_key(s, r)] for s, r in pool]
    test = [index[map_key(s, r)] for s, r in held]
    probes = {str(layer): layer_probe(cells[layer], rows, fit, test, device) for layer in settings["target_layers"]}
    passing = [layer for layer in settings["target_layers"]
               if all(v >= settings["separates"] for v in probes[str(layer)].values())]
    summary["image_token_probe"] = {"by_layer": probes, "threshold": settings["separates"], "passing": passing}
    if not passing:
        return finish("gate failed: the image tokens do not separate holes at 7 x 7 and 8 x 8; stop")
    layer = passing[0]
    summary["target_layer"] = layer

    # 3. Centring means from the training maps and the pool (never the held-out maps); heads' warm start.
    keep = sorted({index[map_key(s, r)] for s, r in trained + pool})
    centres = centres_of(cells[layer], rows, keep)
    train_ids = [index[map_key(s, r)] for s, r in trained]
    val_ids = [index[map_key(s, r)] for s, r in validation]
    centred = {i: cells[layer][i].float() - centres[len(rows[i]["map_desc"])] for i in train_ids + val_ids}
    pack = lambda ids: ([contexts[i] for i in ids], [centred[i] for i in ids], [len(rows[i]["map_desc"]) for i in ids])
    torch.manual_seed(settings["seed"])
    heads = JepaHeads(model.config.hidden_size, dim=settings["dim"], width=settings["width"],
                      predictor_width=settings["predictor_width"], dropout=settings["dropout"],
                      frequencies=settings["frequencies"]).to(device)
    sigreg = SIGReg(settings["directions"]).to(device)
    started = time.time()
    start_error = heads_error(heads, *pack(val_ids), device)
    warm, queues = warm_start(heads, sigreg, pack(train_ids), pack(val_ids), settings, device)
    summary["warm_start"] = warm | {"start_validation_error": start_error, "seconds": time.time() - started,
                                    "validation_maps": len(val_ids)}
    del cells

    # 4. The loss weight: equal median gradient norms into the language model (phase 1's first rows).
    one, _ = phases(data, round2, config["gradient_accumulation"])
    jepa = settings | {"target_layer": layer, "weight": 1.0}
    aux = LatentJepa(jepa, heads, centres, model.config.num_hidden_layers, model.config.vision_config.spatial_merge_size,
                     model.config.image_token_id, model.config.livr_latent_ids, queues)
    started = time.time()
    norms = gradient_norms(model, aux, processor, data_root, one[:settings["pilot_rows"]], 1, config)
    beta = statistics.median(n["answer"] for n in norms) / statistics.median(n["jepa"] for n in norms)
    low, high = settings["beta_bounds"]
    summary["weight"] = {"norms": norms, "beta": beta, "bounds": [low, high], "seconds": time.time() - started}
    jepa["weight"] = beta
    final = LatentJepa(jepa, heads, centres, model.config.num_hidden_layers,
                       model.config.vision_config.spatial_merge_size, model.config.image_token_id,
                       model.config.livr_latent_ids, queues)
    final.save(output / HEADS_FILE, {"gate": {"target_layer": layer, "beta": beta}})
    if not low <= beta <= high:
        return finish(f"gate held: the loss weight {beta:.4g} is outside {low} to {high}; a decision is needed")

    # 5. Gate: on E the route still follows the latents. Then E2, descriptive.
    swap_rows = first_per_size(records(data_root, round2["heldout_split"]), settings["swap_maps_per_size"])
    seed_all(config["seed"])
    started = time.time()
    swap_e, rows_e = swap_check(model, processor, evaluator, data_root, round2["heldout_split"], swap_rows,
                                config["seed"], settings["max_new_tokens"])
    summary["swap_e"] = swap_e | {"seconds": time.time() - started}
    with (output / "swap-e.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in rows_e)
    share = swap_e["follows_donor"] / max(1, swap_e["maps"])
    if not all(swap_e["identity"]):
        return finish("withheld: re-inserting a map's own latents changed its answer")
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    e2 = load_e2()  # E stays loaded beside it; two 3B models fit an 80 GB card
    seed_all(config["seed"])
    started = time.time()
    swap_e2, rows_e2 = swap_check(e2, processor, evaluator, data_root, round2["heldout_split"], swap_rows,
                                  config["seed"], settings["max_new_tokens"])
    summary["swap_e2"] = swap_e2 | {"seconds": time.time() - started}
    with (output / "swap-e2.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in rows_e2)
    if share < settings["swap_follows"]:
        return finish(f"gate failed: E's swapped answers follow the donor on {share:.0%} of maps, below "
                      f"{settings['swap_follows']:.0%}; the route is not decided in the latents; stop")
    summary["valid"] = True
    return finish("gates passed: run EJ2")


# The retry with fixed targets (EJF) --------------------------------------------------------------

def run_features(model, processor, config, data_root, output, settings, round2, round1_routes, search_run,
                 run_manifest):
    """Job 1 of the retry: E's latents at the context layer and each cell's state at the target layer,
    for every map the gate and the arm read, saved for gate 3 on CPU. Nothing is read on the GPU."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    items, groups, _, counts = gate_items(data_root, round2, round1_routes, search_run, settings["validation_maps"])
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "round2_settings": round2,
                                                         "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    seed_all(config["seed"])
    layer = settings["target_layer"]
    contexts, cells = dump(model, processor, data_root, items, settings["context_layer"], [layer])
    torch.save({"context_layer": settings["context_layer"], "target_layer": layer,
                "keys": [map_key(s, r) for s, r in items], "map_desc": [r["map_desc"] for _, r in items],
                "groups": {name: [map_key(s, r) for s, r in g] for name, g in groups.items()},
                "contexts": torch.stack(contexts), "cells": cells[layer]}, output / FEATURES_FILE)
    summary = {"maps": len(items), "groups": {name: len(g) for name, g in groups.items()}, "counts": counts,
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def fixed_projection(cells, components, chunk=4096):
    """[components, hidden]: each channel divided by its standard deviation over `cells` (centred cell
    states, [cells, hidden]); the top principal components of the result; one scale, so the average
    projected variance is 1. Linear: a position whose mean state is zero keeps a zero mean target."""
    std = cells.double().std(0).clamp(min=1e-6)
    second = torch.zeros(cells.shape[1], cells.shape[1], dtype=torch.float64)
    for start in range(0, len(cells), chunk):
        z = cells[start:start + chunk].double() / std
        second += z.T @ z
    values, vectors = torch.linalg.eigh(second / (len(cells) - 1))
    values, vectors = values.flip(0).clamp(min=0), vectors.flip(1)
    kept = values[:components]
    matrix = (vectors[:, :components] / std[:, None]).T / kept.mean().sqrt().clamp(min=1e-12)
    return matrix.float(), {"components": components, "variance_kept": float(kept.sum() / values.sum())}


def hole_share(targets, holes):
    """The share of the targets' variance that the hole label explains: between-class over total sum of
    squares, over the given maps' cells."""
    t, h = torch.cat(targets).double(), torch.cat(holes).bool()
    total = (t - t.mean(0)).square().sum()
    within = sum((t[h == k] - t[h == k].mean(0)).square().sum() for k in (False, True) if (h == k).any())
    return float(1 - within / total.clamp(min=1e-12))


def within_position_share(values):
    """[maps, cells, dim]: the share of the variance that differs between maps at the same cell."""
    x = values.double()
    total = (x - x.mean((0, 1), keepdim=True)).square().sum()
    return float((x - x.mean(0, keepdim=True)).square().sum() / total.clamp(min=1e-12))


def holes_read(values, rows, fit, test, device="cpu"):
    """Per grid size, a ridge probe from a cell's value (`values[i]`: [cells, dim] for map i), minus that
    position's mean over the `fit` maps, to hole or not: fitted on the `fit` maps, scored on the `test`
    maps (balanced accuracy)."""
    result = {}
    for n in sorted({len(rows[i]["map_desc"]) for i in test}):
        f = [i for i in fit if len(rows[i]["map_desc"]) == n]
        t = [i for i in test if len(rows[i]["map_desc"]) == n]
        mean = torch.stack([values[i].float() for i in f]).mean(0)
        x_fit = torch.cat([values[i].float() - mean for i in f]).to(device)
        y_fit = torch.cat([hole_labels(rows[i]) for i in f])[:, None].to(device)
        scores, _ = ridge(x_fit, y_fit, torch.cat([values[i].float() - mean for i in t]).to(device))
        truth = torch.cat([hole_labels(rows[i]) for i in t]).long()
        result[str(n)] = balanced_accuracy((scores[:, 0].cpu() > 0.5).long(), truth)
    return result


def hole_direction_share(values, targets, rows, fit, test):
    """Per grid size: the targets' hole direction (hole minus ice mean target over the `fit` maps, unit
    length), and the share of the `test` maps' target variance along it that `values` explain."""
    result = {}
    for n in sorted({len(rows[i]["map_desc"]) for i in test}):
        f = [i for i in fit if len(rows[i]["map_desc"]) == n]
        t = [i for i in test if len(rows[i]["map_desc"]) == n]
        fitted, holes = torch.cat([targets[i].float() for i in f]), torch.cat([hole_labels(rows[i]) for i in f]).bool()
        d = fitted[holes].mean(0) - fitted[~holes].mean(0)
        d = d / d.norm().clamp(min=1e-12)
        truth = torch.cat([targets[i].float() for i in t]) @ d
        guess = torch.cat([values[i].float() for i in t]) @ d
        result[str(n)] = float(1 - (truth - guess).square().sum() / (truth - truth.mean()).square().sum().clamp(min=1e-12))
    return result


def donors(sizes):
    """Each map's donor: the next map of the same size, in order. A map alone at its size has none."""
    groups = {}
    for i, n in enumerate(sizes):
        groups.setdefault(n, []).append(i)
    return {g[k]: g[(k + 1) % len(g)] for g in groups.values() if len(g) > 1 for k in range(len(g))}


def dependence(predict, targets, sizes):
    """The share of target variance left unexplained with each map's own latents (all maps), and the
    shuffled gap: the share with a same-size donor's latents minus the share with its own (maps with a
    donor). `predict(i, j)`: map j's latents' prediction of map i's cells."""
    own = unexplained(torch.cat([predict(i, i) for i in range(len(sizes))]), torch.cat(targets))
    pairs = donors(sizes)
    if not pairs:
        return {"unexplained": own, "shuffled_gap": None}
    keep = sorted(pairs)
    target = torch.cat([targets[i] for i in keep])
    mine = unexplained(torch.cat([predict(i, i) for i in keep]), target)
    other = unexplained(torch.cat([predict(i, pairs[i]) for i in keep]), target)
    return {"unexplained": own, "shuffled_gap": other - mine}


def linear_ceiling(contexts, targets, rows, fit, test, folds=2):
    """Per grid size, one ridge from a map's four latents (flattened) to all its cells' targets at once,
    fitted on the `fit` maps and scored on the `test` maps as gate 3c. The `fit` maps' own predictions,
    for the hole probe, are cross-fitted over `folds`."""
    result = {}
    for n in sorted({len(rows[i]["map_desc"]) for i in test}):
        f = [i for i in fit if len(rows[i]["map_desc"]) == n]
        t = [i for i in test if len(rows[i]["map_desc"]) == n]
        if len(f) < 2 * folds or len(t) < 2:
            result[str(n)] = None  # too few maps to fit
            continue
        x = lambda ids: torch.stack([contexts[i].float().flatten() for i in ids])
        y = lambda ids: torch.stack([targets[i].float().flatten() for i in ids])
        predicted = {}
        for k in range(folds):
            out = f[k::folds]
            inside = [i for i in f if i not in set(out)]
            scores, _ = ridge(x(inside), y(inside), x(out))
            predicted.update({i: v.float().view(n * n, -1) for i, v in zip(out, scores)})
        scores, _ = ridge(x(f), y(f), x(t))
        predicted.update({i: v.float().view(n * n, -1) for i, v in zip(t, scores)})
        found = dependence(lambda i, j: predicted[t[j]], [targets[i].float() for i in t], [n] * len(t))
        result[str(n)] = found | {"holes_from_predictions": holes_read(predicted, rows, f, t)[str(n)],
                                  "hole_direction_share": hole_direction_share(predicted, targets, rows, f, t)[str(n)]}
    return result


def passes_3c(by_size, settings):
    """The shuffled gap at every size and, unless the threshold is None (the amendment after job 7116395),
    holes from the predictions."""
    holes = settings["holes_from_predictions"]
    return all(v is not None and v["shuffled_gap"] is not None and v["shuffled_gap"] >= settings["shuffled_gap"]
               and (holes is None or v["holes_from_predictions"] >= holes) for v in by_size.values())


def run_fixed_gate(features_run, output, settings, device="cpu"):
    """Gate 3, on CPU (the spec's "Retry with fixed targets"): fixed targets from E's saved cells; 3a and
    3b on them; the heads' warm start; 3c. Saves the heads with every map's targets, pass or fail."""
    if settings["sigreg_weight"] != 0:
        raise ValueError("The fixed-target retry has no SIGReg: sigreg_weight must be 0")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    saved = torch.load(Path(features_run) / FEATURES_FILE, weights_only=False)
    if (saved["context_layer"], saved["target_layer"]) != (settings["context_layer"], settings["target_layer"]):
        raise ValueError("The features were saved at other layers than the settings name")
    keys = [tuple(k) for k in saved["keys"]]
    index = {k: i for i, k in enumerate(keys)}
    rows = [{"map_desc": d} for d in saved["map_desc"]]
    sizes = [len(d) for d in saved["map_desc"]]
    groups = {name: [index[tuple(k)] for k in g] for name, g in saved["groups"].items()}
    contexts, cells = saved["contexts"], saved["cells"]
    pool, held = groups["pool"], groups["heldout"]
    by_size = {n: [i for i in held if sizes[i] == n] for n in sorted({sizes[i] for i in held})}
    summary = {"settings": settings, "features_run": str(features_run), "maps": len(keys), "valid": False}

    def finish(reading):
        summary.update(reading=reading, seconds=time.time() - began)
        json_write(output / "summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)
        return summary

    # The targets: centring means from the training and pool maps (never the held-out maps), then 3a, 3b.
    keep = sorted(set(groups["training"]) | set(pool))
    centres = centres_of(cells, rows, keep)
    centred = [cells[i].float() - centres[sizes[i]] for i in range(len(keys))]
    summary["targets"] = []
    for components in (settings["components"], settings["fallback_components"]):
        matrix, spectrum = fixed_projection(torch.cat([centred[i] for i in keep]), components)
        targets = [c @ matrix.T for c in centred]
        record = spectrum | {
            "hole_probe": holes_read(targets, rows, pool, held, device),
            "hole_share": {str(n): hole_share([targets[i] for i in ids], [hole_labels(rows[i]) for i in ids])
                           for n, ids in by_size.items()},
            "between_position_share": {str(n): 1 - within_position_share(torch.stack([targets[i] for i in ids]))
                                       for n, ids in by_size.items()}}
        summary["targets"].append(record)
        print(json.dumps(record), flush=True)
        if not all(v >= settings["separates"] for v in record["hole_probe"].values()):
            return finish(f"gate 3a failed: the targets at {components} components do not show the holes; stop")
        if all(v >= settings["hole_share"] for v in record["hole_share"].values()):
            break
    else:
        return finish("gate 3b failed: the hole label explains too little of the targets' variance at "
                      f"{settings['components']} and {settings['fallback_components']} components; stop")
    summary["components"] = components

    # The heads' warm start on E's latents.
    torch.manual_seed(settings["seed"])
    heads = JepaHeads(contexts.shape[-1], dim=settings["dim"], width=settings["width"],
                      predictor_width=settings["predictor_width"], dropout=settings["dropout"],
                      frequencies=settings["frequencies"], target="fixed", target_dim=components,
                      standardise=settings.get("standardise_context", False)).to(device)
    if heads.config["standardise"]:
        fitted = contexts[groups["training"]].float()
        heads.context[0].set(fitted.mean(0).to(device), fitted.std(0).to(device))
        summary["context_standardised_on"] = "training maps"
    pack = lambda ids: ([contexts[i] for i in ids], [targets[i] for i in ids], [sizes[i] for i in ids])
    started = time.time()
    start_error = heads_error(heads, *pack(groups["validation"]), device)
    warm, queues = warm_start(heads, None, pack(groups["training"]), pack(groups["validation"]), settings, device)
    summary["warm_start"] = warm | {"start_validation_error": start_error, "seconds": time.time() - started,
                                    "validation_maps": len(groups["validation"])}

    # 3c: the warm-started predictions depend on the latents, on held-out maps.
    heads.eval()
    with torch.no_grad():
        z = {i: heads.context(contexts[i].to(device).float()) for i in set(pool) | set(held)}
        predicted = {i: heads.predict(z[i], sizes[i]).cpu() for i in z}
        latent = {}
        for n, ids in by_size.items():
            found = dependence(lambda i, j: predicted[ids[j]], [targets[i] for i in ids], [n] * len(ids))
            latent[str(n)] = found | {
                "holes_from_predictions": holes_read(predicted, rows, pool, ids, device)[str(n)],
                "hole_direction_share": hole_direction_share(predicted, targets, rows, pool, ids)[str(n)],
                "within_position_share": within_position_share(torch.stack([predicted[i] for i in ids]))}
    summary["latent_dependence"] = latent
    summary["linear_ceiling"] = linear_ceiling(contexts, targets, rows, pool, held)
    save_heads(output / HEADS_FILE, settings | {"weight": 1.0}, heads, centres, queues,
               {k: targets[i].half() for k, i in index.items()},
               {"projection": matrix, "gate": {"valid": passes_3c(latent, settings), "beta": None,
                                               "note": "the arm sets the loss weight from its pilot rows"}})
    if not passes_3c(latent, settings):
        limit = ("the linear ceiling passes: the predictor is the limit" if passes_3c(summary["linear_ceiling"], settings)
                 else "the linear ceiling fails too: E's latents carry too little cell content")
        return finish(f"gate 3c failed: the warm-started predictions do not depend enough on the latents ({limit}); stop")
    summary["valid"] = True
    return finish("gates passed: run EJF")

# The arm -----------------------------------------------------------------------------------------

@torch.no_grad()
def monitor(model, processor, data_root, items, aux):
    """On `items`, with the current language model and heads (unmasked prompt passes, heads in
    evaluation mode): the share of target variance unexplained and the shuffled gap (`dependence`)."""
    model.eval()
    stage, model.config.livr_stage = model.config.livr_stage, 2
    heads, contexts, targets, sizes = aux.head, [], [], []
    training = heads.training
    heads.eval()
    for split, row in items:
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        with amp(model.device):
            hidden = model(**prompt, output_hidden_states=True, use_cache=False).hidden_states
        n = len(row["map_desc"])
        context, cells = aux.terms(hidden, prompt, n, map_key(split, row) if aux.targets is not None else None)
        with torch.autocast(context.device.type, enabled=False):
            contexts.append(heads.context(context.float()))
            targets.append(heads.target(cells.float()))
        sizes.append(n)
    result = dependence(lambda i, j: heads.predict(contexts[j], sizes[i]), targets, sizes)
    heads.train(training)
    model.config.livr_stage = stage
    return result


def add_targets(model, processor, data_root, aux, items):
    """Fixed targets for any of `items` that the gate's heads lack, from the same prompt pass on the
    start checkpoint, the gate's centring means and its projection. Call before any update. Returns
    how many maps were added."""
    missing = distinct_maps([(s, r) for s, r in items if map_key(s, r) not in aux.targets])
    if not missing:
        return 0
    if aux.projection is None:
        raise ValueError("The gate's heads carry no projection, so targets cannot be added")
    layer = aux.settings["target_layer"]
    _, cells = dump(model, processor, data_root, missing, aux.settings["context_layer"], [layer])
    for (split, row), c in zip(missing, cells[layer]):
        centred = c.float() - aux.centres[len(row["map_desc"])].float()
        aux.targets[map_key(split, row)] = (centred @ aux.projection.float().T).half()
    return len(missing)


def run_jepa_arm(model, processor, evaluator, config, data_root, output, round2, gate_run, round1_routes, search_run,
                 run_manifest):
    """EJ2: EB2's two phases (phase 1 under the Stage 1 mask, phase 2 unmasked) on the same rows in
    the same order, with the latent JEPA term in both, then round 2's evaluation and the benchmark.
    EJF, when the gate's heads carry fixed targets: the same, after the loss weight's rule (gate 2b)
    on the first phase-1 rows; it stops before training if the weight is out of bounds."""
    output = Path(output)
    gate = json.loads((Path(gate_run) / "summary.json").read_text())
    if not gate["valid"]:
        raise ValueError(f"The gate job did not pass: {gate['reading']}")
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    aux = from_saved(Path(gate_run) / HEADS_FILE, model)
    fixed = aux.targets is not None
    arm = "ejf" if fixed else "ej2"
    settings = aux.settings
    checks = check_sets(evaluator, data_root, round2)
    data, counts = training_rows(data_root, round2, round1_routes, search_run)
    one, two = phases(data, round2, config["gradient_accumulation"])
    json_write(output / "manifest.json", run_manifest | {"arm": arm, "settings": settings, "round2_settings": round2,
                                                         "checks": checks, "counts": counts, "gate_run": str(gate_run),
                                                         "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    routes = jsonl(Path(search_run) / "routes.jsonl")
    watch = {"validation": validation_maps(data_root, round2, routes, settings["monitor_maps"]),
             "training": distinct_maps(data)[::max(1, len(distinct_maps(data)) // settings["monitor_maps"])]
             [:settings["monitor_maps"]]}
    weight, added = None, None
    if fixed:
        added = add_targets(model, processor, data_root, aux, data + watch["validation"] + watch["training"])
        started = time.time()
        norms = gradient_norms(model, aux, processor, data_root, one[:settings["pilot_rows"]], 1, config)
        beta = statistics.median(n["answer"] for n in norms) / statistics.median(n["jepa"] for n in norms)
        low, high = settings["beta_bounds"]
        weight = {"norms": norms, "beta": beta, "bounds": [low, high], "seconds": time.time() - started}
        if not low <= beta <= high:
            summary = {"arm": arm, "checks": checks, "weight": weight, "targets_added": added,
                       "seconds": time.time() - began,
                       "reading": f"held: the loss weight {beta:.4g} is outside {low} to {high}; a decision is needed"}
            json_write(output / "summary.json", summary)
            print(json.dumps(summary, indent=1), flush=True)
            return summary
        aux.weight = beta
        aux.settings = settings = settings | {"weight": beta}
    errors = {"start": {k: monitor(model, processor, data_root, v, aux) for k, v in watch.items()}}
    tuned = config | {"epochs_per_stage": round2["epochs"], "learning_rate": round2["learning_rate"],
                      "warmup_ratio": round2["warmup_ratio"]}
    started = time.time()
    aux.bind(one)
    train_stage(model, processor, data_root, output / "phase1", tuned, 1, final_weights_only=True, data=one,
                save_final=False, grounding=aux)
    errors["phase1"] = {k: monitor(model, processor, data_root, v, aux) for k, v in watch.items()}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    middle = time.time()
    aux.bind(two)
    train_stage(model, processor, data_root, output / "phase2", tuned, 2, final_weights_only=True, data=two,
                grounding=aux)
    errors["phase2"] = {k: monitor(model, processor, data_root, v, aux) for k, v in watch.items()}
    training = {"phase1": loss_summary(output / "phase1/steps.jsonl") | {"seconds": middle - started, "stage": 1},
                "phase2": loss_summary(output / "phase2/steps.jsonl") | {"seconds": time.time() - middle, "stage": 2},
                "jepa_error": errors}
    seed_all(config["seed"])
    evaluation = evaluate(model, processor, evaluator, data_root, output, round2)
    run_benchmark(model, processor, config, data_root, output / "evaluation",
                  run_manifest | {"checkpoint": f"in-memory {output / 'phase2/final'}", "arm": arm})
    summary = {"arm": arm, "checks": checks, "counts": counts, "weight": weight, "targets_added": added,
               "training": training,
               "evaluation": evaluation,
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# Readings (CPU) ----------------------------------------------------------------------------------

def first_mistake(desc, route):
    """The first move that leaves every shortest safe route: `hole` (straight into an adjacent hole),
    `safe` (safe but off every shortest route), `edge`, `stopped`, `overrun`, or None for a route
    that stays on a shortest route to the goal. With whether a safe shortest move toward the goal
    existed there."""
    dist, goal, cell = distances(desc), find(desc, 2), find(desc, 1)
    for k in range(1, len(route) + 2):
        if dist.get(cell) == 0:
            return ("overrun", False) if k <= len(route) else (None, False)
        here = outcomes(desc, dist, cell)
        toward = any(o == "optimal" and manhattan(step(cell, m), goal) < manhattan(cell, goal) for m, o in here.items())
        if k > len(route):
            return "stopped", toward
        if here[route[k - 1]] != "optimal":
            return {"hole": "hole", "safe": "safe", "edge": "edge"}[here[route[k - 1]]], toward
        cell = step(cell, route[k - 1])
    raise AssertionError("unreachable")


def participation(states, layer, population="heldout78"):
    """Per grid size, the participation ratio of the latents' centred, channel-standardised states."""
    i = states["layers"].index(layer)
    out = {}
    for n in sorted({len(item["map_desc"]) for item in states["items"]}):
        keep = [k for k, item in enumerate(states["items"]) if item["population"] == population
                and len(item["map_desc"]) == n]
        x = states["states"][keep, i].flatten(1).double()
        x = x - x.mean(0)
        x = x / x.std(0).clamp(min=1e-6)
        evals = torch.linalg.eigvalsh(x @ x.T).clamp(min=0)
        out[str(n)] = float(evals.sum() ** 2 / evals.square().sum())
    return out


def compare(data_root, round2, runs, gate_run, livr_outcomes, arm="ej2", manipulation_probes=None, start_check=None,
            took_requires=True):
    """`runs`: {"e": round 2's search run (E's evaluation), "e2", "eb2", and `arm` ("ej2", "ejf" or "ejr"):
    arm runs}. The map layouts come from `data_root`; `livr_outcomes` is LIVR's per-map benchmark outcomes.
    `manipulation_probes`, {arm: predictions, "eb2": predictions} in `probe_predictions`' format,
    replaces round 2's probe as the manipulation check (which is then reported as `all_tokens`);
    `start_check`, with a boolean `matched`, replaces the matched first step as validity;
    `took_requires`, a further condition on "took" (EGR's separation)."""
    label = arm.upper()
    runs = {k: Path(v) for k, v in runs.items()}
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    keys = sorted(greedy["e"])
    if any(sorted(g) != keys for g in greedy.values()):
        raise ValueError("The runs evaluated different maps")
    success = {name: {k: bool(g[k]["success"]) for k in keys} for name, g in greedy.items()}
    held = [k for k in keys if k[0] == "heldout78"]
    old = [k for k in keys if k[0] in ROUND1]
    sizes = {k: greedy["e"][k]["grid_size"] for k in keys}
    tables = {split: records(data_root, split) for split in {r["split"] for r in greedy["e"].values()}}
    desc = {k: tables[greedy["e"][k]["split"]][k[1]]["map_desc"] for k in keys}

    def compared(a, b, among=held):
        return paired(success[a], success[b], among) | {
            "by_size": {str(n): paired(success[a], success[b], [k for k in among if sizes[k] == n])
                        for n in sorted({sizes[k] for k in among})}}

    gate = json.loads((Path(gate_run) / "summary.json").read_text())
    summaries = {name: json.loads((run / "summary.json").read_text()) for name, run in runs.items() if name != "e"}
    arm_summary = summaries[arm]
    if start_check is None:
        first_loss = json.loads((runs[arm] / "phase1/steps.jsonl").read_text().splitlines()[0])["loss"]
        eb2_first = json.loads((runs["eb2"] / "phase1/steps.jsonl").read_text().splitlines()[0])["loss"]
        matched = abs(first_loss - eb2_first) <= 1e-3 * max(1.0, abs(eb2_first))
        start = {arm: first_loss, "eb2": eb2_first, "matched": matched}
    else:
        matched, start = bool(start_check["matched"]), start_check
    valid = gate["valid"] and all(s["checks"]["valid"] for s in summaries.values()) and matched

    states = {name: torch.load(run / "states.pt") for name, run in runs.items()}
    layer = round2["probe_layer"]
    probes = {name: probe_predictions(s, layer) for name, s in states.items()}
    scores = {name: probe_score(p) for name, p in probes.items()}
    check = manipulation_probes or {arm: probes[arm], "eb2": probes["eb2"]}
    check_scores = {name: probe_score(p) for name, p in check.items()}
    difference = check_scores[arm]["hole_balanced_accuracy"] - check_scores["eb2"]["hole_balanced_accuracy"]
    interval = bootstrap_difference(check[arm], check["eb2"], round2["bootstrap"], round2["seed"])
    took = difference >= round2["took_margin"] and interval["low"] > 0 and bool(took_requires)

    def reading(result, better, worse, neither, gated=True):
        if not valid:
            return "withheld: a validity check failed"
        if gated and not took:
            return "the latent JEPA did not take; no reading on perception"
        if result["p"] < 0.05:
            return better if result["only_first"] > result["only_second"] else worse
        return neither

    primary, secondary = compared(arm, "eb2"), compared(arm, "e2")
    mistakes = {}
    for name in runs:
        counter = Counter()
        for k in held:
            if not success[name][k]:
                kind, toward = first_mistake(desc[k], greedy[name][k]["route"] or "")
                counter[f"{kind}{' (safe toward-goal move existed)' if kind == 'hole' and toward else ''}"] += 1
        mistakes[name] = dict(counter)
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    test_detours = {k[1] for k in keys if k[0] == "test_detour"}
    rows = sorted(livr_bench)
    benchmark = {}
    for name in summaries:
        if not (runs[name] / "evaluation/responses.jsonl").exists():
            benchmark[name] = None  # the arm did not run the benchmark (EGR)
            continue
        mine = {r["row_index"]: bool(r["success"]) for r in jsonl(runs[name] / "evaluation/responses.jsonl")}
        benchmark[name] = {"all": paired(mine, livr_bench, rows),
                           "detour": paired(mine, livr_bench, [r for r in rows if r in test_detours]),
                           "no_detour": paired(mine, livr_bench, [r for r in rows if r not in test_detours])}
    return {
        "validity": {"gate": gate["valid"], "set_checks": {n: s["checks"]["valid"] for n, s in summaries.items()},
                     "matched_start": start, "valid": valid},
        "manipulation": ({"layer": layer, **{n: scores[n] for n in ("eb2", arm, "e2", "e")}} if manipulation_probes is None
                         else {"layer": layer, "check": "count-matched (one token per block)",
                               **{n: check_scores[n] for n in (arm, "eb2")},
                               "all_tokens": {n: scores[n] for n in ("eb2", arm, "e2", "e")}})
        | {"difference": difference, "interval_95": interval, "margin": round2["took_margin"], "took": took},
        "primary": primary | {"reading": reading(primary, "the latent JEPA adds to expert iteration",
                                                 "the latent JEPA harms expert iteration", "no detectable difference")},
        "secondary": secondary | {"reading": reading(secondary, f"{label} solves more than E2",
                                                     f"{label} solves fewer than E2", "no detectable difference",
                                                     gated=False)},
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs},
            "first_mistakes": mistakes,
            "participation_ratio": {name: participation(s, layer) for name, s in states.items()},
            "jepa_error": arm_summary["training"]["jepa_error"],
            "round1_heldout_vs_e": {name: compared(name, "e", among=old) for name in summaries},
            "benchmark_vs_livr": benchmark,
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held)) for name in runs},
            "training": {name: s["training"] for name, s in summaries.items()}},
        "gate": {k: gate[k] for k in ("image_token_probe", "target_layer", "warm_start", "swap_e", "swap_e2",
                                      "targets", "components", "latent_dependence", "linear_ceiling") if k in gate}
        | {"beta": (arm_summary.get("weight") or gate.get("weight") or {}).get("beta")}}
