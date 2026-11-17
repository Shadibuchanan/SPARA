"""Region latents (docs/PERCEPTION_JEPA_SPEC.md, "Region latents: EJR"): one latent token per 2 x 2 block
of map cells, and one linear readout, shared by every token, from a token's layer-12 state to its own
block's fixed targets (EJF's). E's four latents are near-copies of one another; giving each token
its own part of the map is meant to break that.

- Tokens 0 to 3 are LIVR's; 4 to 15 are added (`extend_latents`), each starting as a copy of token
  k mod 4 plus noise. A map of size n gets one token per block: 4 at 3 x 3 and 4 x 4, 9 at 5 x 5
  and 6 x 6, 16 at 7 x 7 and 8 x 8.
- The targets are gate 3's (job 7118222), keyed by `perception_jepa.map_key`. Image tokens precede
  the latents, so the layer-4 cell states they come from do not depend on how many latents follow.
- `run_region_arm`: the start block (the small-map identity check, the extension, the per-token
  standardisation and the readout's ridge warm start, the loss weight's rule), then EB2's two phases
  with the readout loss, round 2's evaluation and the benchmark. `compare_region` applies the readings.
- The "global+patch" layout (the spec's "Global + patch latents"): LIVR's four, then one latent per tile
  of each image (4 x 4 tiles, up to 4 images), masked to its tile (`restrict_patches`). It needs no map
  grid, for perception questions with any images.
"""
import contextlib
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .detour_scaling import check_sets, jsonl, paired
from .expert_round2 import (ALPHAS, evaluate, loss_summary, phases, probe_predictions, probe_score, ridge,
                            training_rows)
from .data import records
from .explore_probe import GREEDY, score
from .expert_round2 import latent_states, masked, state_items
from .detour_scaling import greedy_eval
from .latent_battery import LatentPatch, latent_positions, prompt_states, swap_donors
from .model import LATENT_TOKENS
from .perception_jepa import (HEADS_FILE, add_targets, compare, distinct_maps, gradient_norms, hole_labels,
                              map_key, validation_maps)
from .runtime import json_write, seed_all
from .serialization import encode, latent_kwargs
from .training import amp, to_device, train_stage

BLOCK = 2
REGION_TOKENS = LATENT_TOKENS + [f"<|livr_{i}|>" for i in range(4, 16)]  # EJR: LIVR's four, then twelve more
BLOCK_TOKENS = [f"<|livr_region_{i}|>" for i in range(16)]  # EGR: one per block, after LIVR's four
PATCH_GRID = 4  # global+patch: tiles per side of each image
MAX_IMAGES = 4  # global+patch: images a prompt may carry (BLINK's most)
PATCH_TOKENS = [f"<|livr_patch_{i}|>" for i in range(PATCH_GRID * PATCH_GRID * MAX_IMAGES)]  # one per (image, tile)
PATCH_TEXT = ("all", "prefix")  # what text a patch latent sees: the whole prompt, or only before the first image
READOUT_FILE = "region_readout.pt"
SIZES = range(3, 9)
LAYOUTS = ("region", "global+region", "global+patch")


def blocks(n, block=BLOCK):
    """Row-major blocks of `block` x `block` cells. Each block is a list of (cell, slot): the cell's
    row-major index in the map and its row-major place in a full block. Edge blocks at odd sizes are
    smaller and keep their cells' places."""
    count = math.ceil(n / block)
    return [[(r * n + c, (r - br * block) * block + (c - bc * block))
             for r in range(br * block, min(n, (br + 1) * block)) for c in range(bc * block, min(n, (bc + 1) * block))]
            for br in range(count) for bc in range(count)]


def tokens_by_size(sizes=SIZES, layout="region"):
    """{grid size, as a string: the latent tokens its prompt carries}. "region" (EJR): one token per
    block, LIVR's four first. "global+region" (EGR): LIVR's four, then one block token per block."""
    if layout == "region":
        out = {str(n): REGION_TOKENS[:len(blocks(n))] for n in sizes}
    elif layout == "global+region":
        out = {str(n): LATENT_TOKENS + BLOCK_TOKENS[:len(blocks(n))] for n in sizes}
    else:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if any(len(blocks(int(n))) > 16 for n in out):
        raise ValueError("More blocks than region tokens")
    return out


def globals_of(layout):
    """How many latents precede the block or patch latents: 4 for "global+region" and "global+patch",
    none for "region"."""
    return 4 if layout in ("global+region", "global+patch") else 0


def patch_latents(grid=PATCH_GRID, max_images=MAX_IMAGES):
    """`encode`'s latents for the global+patch layout: LIVR's four, then one token per tile of each image,
    image-major and row-major within an image. Each slot has its own token, because only a latent token's
    first occurrence may read the image in Stage 1 (`model.latent_queries`)."""
    per = grid * grid
    if per * max_images > len(PATCH_TOKENS):
        raise ValueError("More patch slots than patch tokens")
    return {"globals": list(LATENT_TOKENS), "per_image": [PATCH_TOKENS[m * per:(m + 1) * per] for m in range(max_images)]}


def extend_latents(model, processor, seed=0, layout="region", mask=False, grid=PATCH_GRID, max_images=MAX_IMAGES,
                   text="all"):
    """Add the layout's new tokens and switch the model to region latents. "region": tokens 4 to 15;
    "global+region": sixteen block tokens after LIVR's four; "global+patch": `grid` x `grid` patch tokens per
    image for up to `max_images` images, after LIVR's four. Each new embedding row starts as LIVR's token
    k mod 4 plus Gaussian noise at the model's initializer range. With `mask`, each block latent sees only
    its block's image tokens (`restrict_regions`), and each patch latent only its tile's (`restrict_patches`),
    with the text `text` names. Qwen keeps spare embedding rows, so the table is resized only when the
    tokenizer outgrows it."""
    tokenizer = processor.tokenizer
    if getattr(model.config, "livr_region_latents", None):
        raise ValueError("Region latents are already installed")
    if list(model.config.livr_latent_ids) != tokenizer.convert_tokens_to_ids(LATENT_TOKENS):
        raise ValueError("Expected LIVR's four latent tokens")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if mask and layout == "region":
        raise ValueError("The block mask is defined for the global+region and global+patch layouts")
    if text not in PATCH_TEXT:
        raise ValueError(f"text must be one of {PATCH_TEXT}")
    if layout == "global+patch":
        every = LATENT_TOKENS + [t for tokens in patch_latents(grid, max_images)["per_image"] for t in tokens]
    else:
        every = REGION_TOKENS if layout == "region" else LATENT_TOKENS + BLOCK_TOKENS
    new = every[4:]
    if any(t in tokenizer.get_vocab() for t in new):
        raise ValueError("Region tokens already exist in the tokenizer")
    tokenizer.add_tokens(new, special_tokens=True)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    ids = tokenizer.convert_tokens_to_ids(every)
    generator = torch.Generator().manual_seed(seed)
    tables = [model.get_input_embeddings().weight]
    if model.get_output_embeddings().weight.data_ptr() != tables[0].data_ptr():
        tables.append(model.get_output_embeddings().weight)
    with torch.no_grad():
        for table in tables:
            for k in range(4, len(ids)):
                noise = torch.randn(table.shape[1], generator=generator) * model.config.initializer_range
                table[ids[k]] = table[ids[k % 4]] + noise.to(table.device, table.dtype)
    model.config.livr_latent_ids = ids
    if layout == "global+patch":
        model.config.livr_latent_tokens = patch_latents(grid, max_images)
        model.config.livr_region_latents = {"tokens": every, "layout": layout, "globals": 4, "mask": bool(mask),
                                            "grid": grid, "max_images": max_images, "text": text}
        model.config.livr_serialization = (
            f"user-end,4-latents,patch-latents({grid}x{grid} per image, image-major"
            + (f", masked to its tile and {text} text" if mask else "") + "),assistant-header-no-newline,answer,im-end")
        return ids
    model.config.livr_latent_tokens = tokens_by_size(layout=layout)
    model.config.livr_region_latents = {"block": BLOCK, "tokens": every, "layout": layout,
                                        "globals": globals_of(layout), "mask": bool(mask)}
    model.config.livr_serialization = (
        "user-end,region-latents(one per 2x2 block),assistant-header-no-newline,answer,im-end" if layout == "region"
        else "user-end,4-latents,block-latents(one per 2x2 block" + (", masked to its block" if mask else "")
        + "),assistant-header-no-newline,answer,im-end")
    return ids


def block_image_tokens(n, grid_thw, merge):
    """[blocks, image tokens] boolean: the merged image tokens that overlap each block's cells at all
    (tokens on a boundary belong to both blocks)."""
    from .latent_battery import cell_weights
    weights = cell_weights(n, grid_thw, merge) > 0
    return torch.stack([weights[[c for c, _ in cells]].any(0) for cells in blocks(n)])


def restrict_regions(allowed, full_ids, query_start, config, grid_sizes, grid_thw):
    """Narrow each block latent's row of the boolean mask [B, Q, K] to the prompt before the image, its
    block's image tokens and itself. Only the block latents' first prompt occurrences are restricted."""
    region = config.livr_region_latents
    block_ids = config.livr_latent_ids[region["globals"]:]
    merge = config.vision_config.spatial_merge_size
    allowed = allowed.clone()
    for b in range(full_ids.shape[0]):
        ids = full_ids[b]
        image = (ids == config.image_token_id).nonzero().squeeze(-1)
        if not len(image):
            continue
        n = int(grid_sizes[b])
        visible = block_image_tokens(n, grid_thw[b], merge).to(ids.device)
        if visible.shape[1] != len(image):
            raise ValueError("The image tokens do not match the image grid")
        for k, token in enumerate(block_ids[:len(visible)]):
            where = (ids == token).nonzero().squeeze(-1)
            if not len(where) or int(where[0]) < query_start:
                continue
            q = int(where[0])
            row = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
            row[:int(image[0])] = True
            row[image[visible[k]]] = True
            row[q] = True
            allowed[b, q - query_start] &= row
    return allowed


def patch_tiles(grid_thw, merge, grid):
    """The tile of each merged image token of one image, in token order (row-major, as `cell_weights`): the
    merged grid is cut into `grid` x `grid` tiles of near-equal height and width. Tiles are empty only when
    the merged grid has fewer than `grid` rows or columns."""
    frames, height, width = (int(v) for v in grid_thw)
    if frames != 1:
        raise ValueError("Patch latents are defined for still images")
    rows, cols = height // merge, width // merge
    r, c = torch.arange(rows) * grid // rows, torch.arange(cols) * grid // cols
    return (r[:, None] * grid + c[None, :]).flatten()


def image_runs(ids, image_id):
    """The positions of each image's tokens, one tensor per image in prompt order. Qwen wraps every image in
    vision start and end tokens, so the images' tokens form separate runs."""
    positions = (ids == image_id).nonzero().squeeze(-1)
    if not len(positions):
        return []
    breaks = ((positions[1:] != positions[:-1] + 1).nonzero().squeeze(-1) + 1).tolist()
    return list(torch.tensor_split(positions, breaks))


def restrict_patches(allowed, full_ids, query_start, config, grid_thw):
    """Narrow each patch latent's row of the boolean mask [B, Q, K]. Patch latent m * tiles + t (image m,
    tile t) sees its image's tile-t tokens, itself, and text: every prompt token before it that is neither an
    image token nor a latent ("all"), or only those before the first image ("prefix"). It sees no other
    image, no other tile and no other latent. `grid_thw` lists every image of the batch, row by row."""
    region = config.livr_region_latents
    grid = region["grid"]
    per = grid * grid
    patch_ids = config.livr_latent_ids[region["globals"]:]
    merge = config.vision_config.spatial_merge_size
    latents = torch.tensor(config.livr_latent_ids, device=full_ids.device)
    if grid_thw is None and query_start > 0:
        return allowed  # cached decoding: the patch latents are prompt tokens, never queries here
    allowed, offset = allowed.clone(), 0
    for b in range(full_ids.shape[0]):
        ids = full_ids[b]
        runs = image_runs(ids, config.image_token_id)
        if not runs:
            continue
        if grid_thw is None or len(grid_thw) < offset + len(runs):
            raise ValueError("Patch latents need every image's grid (image_grid_thw)")
        if len(runs) * per > len(patch_ids):
            raise ValueError("More images than the patch latents cover")
        position = torch.arange(ids.shape[0], device=ids.device)
        text = (ids != config.image_token_id) & ~torch.isin(ids, latents)
        if region.get("text", "all") == "prefix":
            text &= position < int(runs[0][0])
        for m, run in enumerate(runs):
            tiles = patch_tiles(grid_thw[offset + m], merge, grid).to(ids.device)
            if len(tiles) != len(run):
                raise ValueError("The image tokens do not match the image grid")
            for t in range(per):
                where = (ids == patch_ids[m * per + t]).nonzero().squeeze(-1)
                if not len(where) or int(where[0]) < query_start:
                    continue
                q = int(where[0])
                row = text & (position < q)
                row[run[tiles == t]] = True
                row[q] = True
                allowed[b, q - query_start] &= row
        offset += len(runs)
    return allowed


def hide_globals(allowed, full_ids, query_start, config, flags):
    """For the rows whose flag is set, no query that is not a latent may attend to the global latents'
    prompt occurrences: the answer's only map is the block latents (EGD)."""
    globals_ = config.livr_latent_ids[:config.livr_region_latents["globals"]]
    latents = config.livr_latent_ids
    allowed = allowed.clone()
    for b in range(full_ids.shape[0]):
        if not bool(flags[b]):
            continue
        ids = full_ids[b]
        keys = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        for token in globals_:
            where = (ids == token).nonzero().squeeze(-1)
            if len(where):
                keys[int(where[0])] = True
        queries = ~torch.isin(ids, torch.tensor(latents, device=ids.device))[query_start:]
        allowed[b] &= ~(queries[:, None] & keys[None, :])
    return allowed


def hidden(data):
    """Copies of (split, row) pairs flagged so that the answer may not see the global latents."""
    return [(split, row | {"livr_hide_globals": True}) for split, row in data]


def block_targets(targets, n):
    """[blocks, block cells, dim] and a [blocks, block cells] mask: each block's cells' targets in their
    places, zero where an edge block has no cell."""
    found = blocks(n)
    out = torch.zeros(len(found), BLOCK * BLOCK, targets.shape[-1], dtype=targets.dtype)
    mask = torch.zeros(len(found), BLOCK * BLOCK, dtype=torch.bool)
    for b, cells in enumerate(found):
        for cell, slot in cells:
            out[b, slot], mask[b, slot] = targets[cell], True
    return out, mask


class Readout(torch.nn.Module):
    """A fixed per-token standardisation (set once, `set_standardisation`), then one linear map from a
    token's state to its block's cells' targets, shared by every token and size."""

    def __init__(self, hidden, tokens=len(REGION_TOKENS), target_dim=256):
        super().__init__()
        self.register_buffer("mean", torch.zeros(tokens, hidden))
        self.register_buffer("std", torch.ones(tokens, hidden))
        self.linear = torch.nn.Linear(hidden, BLOCK * BLOCK * target_dim)
        self.target_dim = target_dim
        self.config = {"hidden": hidden, "tokens": tokens, "target_dim": target_dim}

    def set_standardisation(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))

    def standardise(self, states):
        k = len(states)
        return (states - self.mean[:k]) / self.std[:k]

    def forward(self, states):
        """states: [tokens, hidden] for tokens 0 to k - 1; returns [tokens, block cells, dim]."""
        return self.linear(self.standardise(states)).view(len(states), BLOCK * BLOCK, self.target_dim)


class RegionLoss:
    """The readout loss for spara.training.train_stage: each token's block, squared error over the
    block's cells, averaged. `targets`, {map key: [cells, dim]}; `bind` the rows before each stage."""

    released_only = False
    clip_separately = True
    loss_key = "region_loss"
    metric_keys = ("region_loss", "region_unexplained", "shuffled_gap")
    head_file = READOUT_FILE

    def __init__(self, settings, head, targets, latent_ids, centres=None, projection=None):
        if settings["weight"] < 0 or settings["head_learning_rate"] <= 0:
            raise ValueError("Need weight >= 0 and a positive head learning rate")
        self.settings, self.weight, self.head = settings, settings["weight"], head
        self.targets, self.latent_ids = targets, latent_ids
        self.centres, self.projection = centres, projection  # for `perception_jepa.add_targets`
        self.keys = self.sizes = None
        self.generator = torch.Generator().manual_seed(settings.get("seed", 0))
        self.others = {}  # per size, recent maps' latent states, for the shuffled gap
        self.scale = None

    def set_scale(self):
        """The targets' average per-dimension variance over the stored maps: errors as a share of it."""
        stacked = torch.cat([t.float() for t in self.targets.values()])
        self.scale = float(stacked.var(0).mean())

    def bind(self, data):
        self.keys = [map_key(split, row) for split, row in data]

    def prepare(self, tokenizer, rows):
        self.sizes = [len(row["map_desc"]) for row in rows]
        if self.keys is None or [k[1] for k in self.keys] != [row["image_input"] for row in rows]:
            raise ValueError("Bind the rows' splits (RegionLoss.bind) before training")
        missing = [k for k in self.keys if k not in self.targets]
        if missing:
            raise ValueError(f"No fixed targets for {len(missing)} rows, such as {missing[0]}")
        if self.scale is None:
            self.set_scale()
        return {"rows": len(rows), "sizes": {str(s): self.sizes.count(s) for s in sorted(set(self.sizes))}}

    def parameter_group(self):
        return {"params": list(self.head.parameters()), "lr": self.settings["head_learning_rate"],
                "weight_decay": self.settings.get("head_weight_decay", 0.0)}

    def states(self, hidden_states, batch, n):
        """The block latents' states: after the global latents; detached when the readout only
        monitors (`readout_only`), so nothing reaches the language model."""
        skip = self.settings.get("globals", 0)
        positions = latent_positions(batch["input_ids"], self.latent_ids)
        if len(positions) != skip + len(blocks(n)):
            raise ValueError(f"A {n} x {n} prompt carries {len(positions)} latents, not {skip + len(blocks(n))}")
        states = hidden_states[self.settings["context_layer"]][0, positions[skip:]]
        return states.detach() if self.settings.get("readout_only") else states

    def error(self, states, key, n):
        target, mask = block_targets(self.targets[key].float(), n)
        target, mask = target.to(states.device), mask.to(states.device)
        with torch.autocast(states.device.type, enabled=False):
            return (self.head(states.float()) - target)[mask].square().mean()

    def loss(self, hidden_states, index, batch):
        n, key = self.sizes[index], self.keys[index]
        states = self.states(hidden_states, batch, n)
        error = self.error(states, key, n)
        gap = None
        pool = self.others.setdefault(n, deque(maxlen=self.settings.get("shuffle_pool", 64)))
        if pool:
            other = pool[torch.randint(len(pool), (1,), generator=self.generator).item()].to(states.device)
            with torch.no_grad():
                gap = (float(self.error(other, key, n)) - float(error)) / self.scale
        pool.append(states.detach().float().cpu())
        return error, {"region_unexplained": float(error) / self.scale, "shuffled_gap": gap}

    def save(self, path, extra=None):
        torch.save({"settings": self.settings, "config": self.head.config,
                    "state_dict": {k: v.cpu() for k, v in self.head.state_dict().items()}} | (extra or {}), path)


def ridge_fit(x, y):
    """Ridge from x [rows, features] to y [rows, outputs], strength chosen by leave-one-out error over
    round 2's ALPHAS (as `expert_round2.ridge`, which returns predictions only): (weight [features,
    outputs], bias [outputs], alpha)."""
    x, y = x.double(), y.double()
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    s2, uy = s.square(), u.T @ yc
    best = None
    for alpha in ALPHAS:
        lam = alpha * s2.sum() / len(xc)
        shrink = s2 / (s2 + lam)
        leverage = (u.square() * shrink).sum(-1)
        loo = ((yc - u @ (shrink[:, None] * uy)) / (1 - leverage).clamp(min=1e-9)[:, None]).square().mean()
        if best is None or loo < best[0]:
            best = (loo, lam, alpha)
    weight = vh.T @ ((s / (s2 + best[1]))[:, None] * uy)
    return weight.float(), (y_mean - x_mean @ weight).float(), best[2]


# The start block --------------------------------------------------------------------------------

@torch.no_grad()
def answer_passes(model, processor, data_root, rows, stage):
    """Each row's answer loss and final-layer states, one forward under `stage`'s mask, with the
    model's current latents."""
    model.eval()
    before, model.config.livr_stage = model.config.livr_stage, stage
    losses, states, ids = [], [], []
    for split, row in rows:
        batch = to_device(encode(processor, data_root, split, row, **latent_kwargs(model)), model.device)
        with amp(model.device):
            out = model(**batch, use_cache=False, output_hidden_states=True)
        losses.append(float(out.loss))
        states.append(out.hidden_states[-1][0].float().cpu())
        ids.append(batch["input_ids"][0].cpu())
    model.config.livr_stage = before
    return losses, states, ids


def start_check(layout, small, large, before, after, tolerance, global_ids):
    """`before`, `after`: answer_passes on E and on the extended model, over `small` + `large` rows.
    "region": the small rows' prompts are unchanged, so their final-layer states must be.
    "global+region": every prompt keeps E's prefix through the global latents (`global_ids`), so every
    row's states over that prefix must be. The embeddings are tied, so the new rows also enter the
    output softmax and answer losses can move a little; their changes are reported."""
    (loss_b, states_b, ids_b), (loss_a, states_a, _) = before, after
    checked = len(small) if layout == "region" else len(small) + len(large)
    relative = []
    for i in range(checked):
        a, b = states_a[i], states_b[i]
        if layout != "region":
            end = int(latent_positions(ids_b[i][None], global_ids)[-1]) + 1
            a, b = a[:end], b[:end]
        relative.append(float((a - b).abs().max() / b.abs().max().clamp(min=1e-12)))
    change = [x - y for x, y in zip(loss_a, loss_b)]
    return {"small_rows": len(small), "large_rows": len(large), "rows_checked": checked,
            "loss_before": loss_b, "loss_after": loss_a,
            "max_relative_state_difference": max(relative) if relative else None, "tolerance": tolerance,
            "matched": bool(relative) and max(relative) <= tolerance,
            "small_loss_change": statistics.mean(change[:len(small)]) if small else None,
            "insertion_disturbance": statistics.mean(change[len(small):]) if large else None,
            "mean_loss_change": statistics.mean(change) if change else None}


def check_rows(one, count):
    """Phase 1's first `count` rows at 3 x 3 or 4 x 4, and its first `count` at 5 x 5 or above."""
    small = [r for r in one if len(r[1]["map_desc"]) <= 4][:count]
    large = [r for r in one if len(r[1]["map_desc"]) >= 5][:count]
    return small, large


@torch.no_grad()
def token_states(model, processor, data_root, items, layer):
    """Per map, the region tokens' states at `layer` (unmasked prompt pass), [tokens, hidden], CPU."""
    model.eval()
    before, model.config.livr_stage = model.config.livr_stage, 2
    out = []
    for count, (split, row) in enumerate(items):
        prompt = to_device(encode(processor, data_root, split, row, training=False, **latent_kwargs(model)),
                           model.device)
        positions = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
        with amp(model.device):
            hidden = model(**prompt, output_hidden_states=True, use_cache=False).hidden_states
        out.append(hidden[layer][0, positions].float().cpu())
        if count % 100 == 0:
            print(f"token states {count + 1}/{len(items)}", flush=True)
    model.config.livr_stage = before
    return out


def warm_maps(data, counts):
    """The start block's maps: every distinct training map at 7 x 7 and 8 x 8 and the first ones at
    smaller sizes, `counts` = {"large": n at 7 and 8 (all if None), "middle": n at 5 and 6, "small": n
    at 3 and 4}."""
    maps = distinct_maps(data)
    size = lambda r: len(r[1]["map_desc"])
    large = [r for r in maps if size(r) >= 7][:counts.get("large")]
    middle = [r for r in maps if size(r) in (5, 6)][:counts["middle"]]
    small = [r for r in maps if size(r) <= 4][:counts["small"]]
    return large + middle + small


def warm_start(head, aux, items, states):
    """The per-token standardisation from `states` (each token over the maps that carry it), then the
    readout's ridge fit from every token's standardised state to its block's targets."""
    tokens = head.mean.shape[0]
    stacked = [[s[t] for s in states if len(s) > t] for t in range(tokens)]
    mean = torch.stack([torch.stack(v).mean(0) if v else torch.zeros(head.mean.shape[1]) for v in stacked])
    std = torch.stack([torch.stack(v).std(0) if len(v) > 1 else torch.ones(head.mean.shape[1]) for v in stacked])
    head.set_standardisation(mean.to(head.mean.device), std.to(head.std.device))
    xs, ys = [], []
    for (split, row), s in zip(items, states):
        n = len(row["map_desc"])
        target, _ = block_targets(aux.targets[map_key(split, row)].float(), n)
        xs.append(head.standardise(s.to(head.mean.device)).cpu())
        ys.append(target.flatten(1))
    weight, bias, alpha = ridge_fit(torch.cat(xs), torch.cat(ys))
    with torch.no_grad():
        head.linear.weight.copy_(weight.T.to(head.linear.weight.device))
        head.linear.bias.copy_(bias.to(head.linear.bias.device))
    return {"maps": len(items), "rows": sum(len(s) for s in states), "alpha": alpha,
            "maps_per_token": [len(v) for v in stacked]}


@torch.no_grad()
def monitor(model, processor, data_root, items, aux):
    """On `items`, with the current model and readout: the unexplained share (over all maps) and the
    shuffled gap (each map's targets against the next same-size map's latents)."""
    skip = aux.settings.get("globals", 0)
    states = [s[skip:] for s in token_states(model, processor, data_root, items, aux.settings["context_layer"])]
    head = aux.head
    training = head.training
    head.eval()
    sizes = [len(row["map_desc"]) for _, row in items]
    keys = [map_key(s, r) for s, r in items]
    own = [float(aux.error(states[i].to(head.mean.device), keys[i], sizes[i])) for i in range(len(items))]
    groups = {}
    for i, n in enumerate(sizes):
        groups.setdefault(n, []).append(i)
    gaps = []
    for ids in groups.values():
        for k, i in enumerate(ids):
            if len(ids) > 1:
                j = ids[(k + 1) % len(ids)]
                gaps.append(float(aux.error(states[j].to(head.mean.device), keys[i], sizes[i])) - own[i])
    head.train(training)
    return {"unexplained": statistics.mean(own) / aux.scale,
            "shuffled_gap": statistics.mean(gaps) / aux.scale if gaps else None}


def loss_weight(model, aux, processor, data_root, one, settings, config):
    """The equal-gradient rule on phase 1's first rows; sets the weight unless it is out of bounds."""
    started = time.time()
    norms = gradient_norms(model, aux, processor, data_root, one[:settings["pilot_rows"]], 1, config)
    beta = statistics.median(n["answer"] for n in norms) / statistics.median(n["jepa"] for n in norms)
    low, high = settings["beta_bounds"]
    held = (None if low <= beta <= high
            else f"held: the loss weight {beta:.4g} is outside {low} to {high}; a decision is needed")
    if not held:
        aux.weight = beta
        aux.settings = aux.settings | {"weight": beta}
    return {"norms": norms, "beta": beta, "bounds": [low, high], "held": held, "seconds": time.time() - started}


def load_targets(gate_run):
    """Gate 3's targets, centring means and projection (for missing maps), and its settings."""
    saved = torch.load(Path(gate_run) / HEADS_FILE, map_location="cpu", weights_only=False)
    if saved.get("targets") is None or saved.get("projection") is None:
        raise ValueError("The gate's heads carry no fixed targets")
    centres = {int(k): v for k, v in saved["centres"].items()}
    return dict(saved["targets"]), centres, saved["projection"], saved["settings"]


# The arm ----------------------------------------------------------------------------------------

def run_region_arm(model, processor, evaluator, config, data_root, output, settings, round2, gate_run,
                   round1_routes, search_run, run_manifest):
    """EJR or EGR (the spec's "Region latents: EJR" and "Global + region latents: EGR"), by the
    settings' `layout`, `mask`, `readout_only` and `benchmark`."""
    arm, layout = settings.get("arm", "ejr"), settings.get("layout", "region")
    output = Path(output)
    gate = json.loads((Path(gate_run) / "summary.json").read_text())
    if not gate["valid"]:
        raise ValueError(f"Gate 3 did not pass: {gate['reading']}")
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    targets, centres, projection, gate_settings = load_targets(gate_run)
    if gate_settings["target_layer"] != settings["target_layer"]:
        raise ValueError("The gate's targets come from another layer")
    checks = check_sets(evaluator, data_root, round2)
    data, counts = training_rows(data_root, round2, round1_routes, search_run)
    one, two = phases(data, round2, config["gradient_accumulation"])
    routes = jsonl(Path(search_run) / "routes.jsonl")
    watch = {"validation": validation_maps(data_root, round2, routes, settings["monitor_maps"]),
             "training": distinct_maps(data)[::max(1, len(distinct_maps(data)) // settings["monitor_maps"])]
             [:settings["monitor_maps"]]}
    json_write(output / "manifest.json", run_manifest | {"arm": arm, "settings": settings, "round2_settings": round2,
                                                         "checks": checks, "counts": counts, "gate_run": str(gate_run),
                                                         "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    summary = {"arm": arm, "checks": checks, "counts": counts}

    def stop(reading):
        summary.update(reading=reading, seconds=time.time() - began)
        json_write(output / "summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)
        return summary

    # 1-3. The identity check around the extension (`start_check`).
    small, large = check_rows(one, settings["check_rows"])
    global_ids = list(model.config.livr_latent_ids)
    before = answer_passes(model, processor, data_root, small + large, 1)
    ids = extend_latents(model, processor, settings["seed"], layout, settings.get("mask", False))
    after = answer_passes(model, processor, data_root, small + large, 1)
    summary["start_check"] = start_check(layout, small, large, before, after, settings["identity_tolerance"],
                                         global_ids)
    if not summary["start_check"]["matched"]:
        return stop("withheld: the checked rows' states changed when the tokens were added")

    # 4. Targets for every map read, the standardisation and the readout's warm start.
    head = Readout(model.config.hidden_size, target_dim=projection.shape[0]).to(model.device)
    aux = RegionLoss(settings | {"weight": 1.0, "globals": globals_of(layout)}, head, targets, ids, centres,
                     projection)
    aux.settings["target_layer"] = settings["target_layer"]
    added = add_targets(model, processor, data_root, aux, data + watch["validation"] + watch["training"])
    summary["targets_added"] = added
    started = time.time()
    items = warm_maps(data, settings["warm_maps"])
    skip = globals_of(layout)
    states = [s[skip:] for s in token_states(model, processor, data_root, items, settings["context_layer"])]
    summary["warm_start"] = warm_start(head, aux, items, states) | {"seconds": time.time() - started}
    aux.set_scale()

    # 5. The loss weight. A monitor-only readout sends nothing into the model, so it has none.
    if settings.get("readout_only"):
        summary["weight"] = None
    else:
        summary["weight"] = loss_weight(model, aux, processor, data_root, one, settings, config)
        if summary["weight"]["held"]:
            return stop(summary["weight"]["held"])

    # EB2's two phases, with the readout.
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
    summary["training"] = {
        "phase1": loss_summary(output / "phase1/steps.jsonl") | {"seconds": middle - started, "stage": 1},
        "phase2": loss_summary(output / "phase2/steps.jsonl") | {"seconds": time.time() - middle, "stage": 2},
        "jepa_error": errors}
    seed_all(config["seed"])
    summary["evaluation"] = evaluate(model, processor, evaluator, data_root, output, round2)
    if settings.get("benchmark", True):
        run_benchmark(model, processor, config, data_root, output / "evaluation",
                      run_manifest | {"checkpoint": f"in-memory {output / 'phase2/final'}", "arm": arm})
    summary["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    return stop("complete")


# The swap test (the spec's "EGR: does the answer read the block latents?") ----------------------

def first_maps(rows, sizes, count):
    """(row index, row) for the first `count` rows of each size, in order."""
    out = []
    for n in sizes:
        out += [(i, r) for i, r in enumerate(rows) if len(r["map_desc"]) == n][:count]
    return out


def run_latent_swap(model, processor, evaluator, data_root, output, settings, round2, run_manifest, recorded=None):
    """Masked greedy on the first held-out maps of each size, under four conditions: own; the block
    latents from a same-size donor (every layer); the block latents replaced by their mean over the maps
    of that size; the global latents from the donor. `recorded`: masked.jsonl of the arm run, for the
    reproduction check."""
    region = getattr(model.config, "livr_region_latents", None) or {}
    if not region.get("mask") or region.get("layout") != "global+region":
        raise ValueError("The swap test reads a masked global + region checkpoint (EGR)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    split = round2["heldout_split"]
    chosen = first_maps(records(data_root, split), settings["sizes"], settings["maps_per_size"])
    if settings.get("hide_globals"):  # EGD: the answer may not see the global latents
        chosen = [(i, row | {"livr_hide_globals": True}) for i, row in chosen]
    rows = [row for _, row in chosen]
    donors = swap_donors(rows, settings["seed"])
    skip = region["globals"]
    budget = {"max_new_tokens": round2["max_new_tokens"]}
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "round2_settings": round2,
                                                         "maps": [i for i, _ in chosen]})
    patch = LatentPatch(model)
    results = []
    try:
        model.eval()
        with masked(model):
            prompts, positions, states = [], [], []
            for _, row in chosen:
                prompt = to_device(encode(processor, data_root, split, row, training=False, **latent_kwargs(model)),
                                   model.device)
                where = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
                prompts.append(prompt)
                positions.append(where)
                states.append(prompt_states(model, patch, prompt, where)[:, where].half().cpu())
            means = {}
            for k, row in enumerate(rows):
                means.setdefault(len(row["map_desc"]), []).append(states[k][:, skip:].float())
            means = {n: torch.stack(v).mean(0) for n, v in means.items()}

            def run(k, where=None, values=None):
                with patch.apply(where, values), torch.no_grad(), amp(model.device):
                    generated = model.generate(**prompts[k], **GREEDY, **budget, use_cache=True)
                (route, success), = score(evaluator, processor, rows[k], generated)
                return {"route": route, "success": bool(success)}

            conditions = settings.get("conditions", ("block_swap", "block_mean", "global_swap"))
            for k, (index, row) in enumerate(chosen):
                donor = donors.get(k)
                record = {"row_index": index, "grid_size": len(row["map_desc"]), "donor": donor,
                          "own": run(k)}
                if donor is not None:
                    record["block_swap"] = run(k, positions[k][skip:], states[donor][:, skip:].float())
                    if "global_swap" in conditions:
                        record["global_swap"] = run(k, positions[k][:skip], states[donor][:, :skip].float())
                record["block_mean"] = run(k, positions[k][skip:], means[len(row["map_desc"])])
                if k < 4:
                    record["identity"] = run(k, positions[k][skip:], states[k][:, skip:].float())
                results.append(record)
                if k % 20 == 0:
                    print(f"swap {k + 1}/{len(chosen)}", flush=True)
    finally:
        patch.remove()
    for r in results:
        if r["donor"] is not None:
            r["donor_route"] = results[r["donor"]]["own"]["route"]
    with (output / "swaps.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in results)
    summary = {"settings": settings, "maps": len(results), "seconds": time.time() - began,
               "identity": [r["identity"]["route"] == r["own"]["route"] for r in results if "identity" in r],
               "conditions": swap_readings(results)}
    if recorded is not None:
        before = {r["row_index"]: r["route"] for r in jsonl(recorded) if r["population"] == "heldout78"}
        same = sum(before.get(r["row_index"]) == r["own"]["route"] for r in results)
        summary["reproduced"] = {"same_route": same, "maps": len(results), "share": same / len(results)}
    summary["reading"] = swap_reading(summary, settings)
    summary["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def swap_readings(results):
    """Per condition: maps, maps solved, routes unchanged from own, and (swaps) routes equal to the donor's."""
    out = {"own": {"maps": len(results), "solved": sum(r["own"]["success"] for r in results)}}
    for name in ("block_swap", "block_mean", "global_swap"):
        rows = [r for r in results if name in r]
        if not rows:
            continue
        out[name] = {"maps": len(rows), "solved": sum(r[name]["success"] for r in rows),
                     "unchanged": sum(r[name]["route"] == r["own"]["route"] for r in rows)}
        if name.endswith("swap"):
            out[name]["donor_route"] = sum(r[name]["route"] == r["donor_route"] for r in rows)
    return out


def swap_reading(summary, settings):
    """The pre-declared reading (the spec's swap-test table)."""
    if not all(summary["identity"]) or summary.get("reproduced", {"share": 1.0})["share"] < settings["reproduce"]:
        return "withheld: own routes did not reproduce"
    c = summary["conditions"]
    mean_unchanged = c["block_mean"]["unchanged"] / max(1, c["block_mean"]["maps"])
    swap_unchanged = c["block_swap"]["unchanged"] / max(1, c["block_swap"]["maps"])
    swap_donor = c["block_swap"]["donor_route"] / max(1, c["block_swap"]["maps"])
    if mean_unchanged >= 0.9 and swap_donor <= 0.1:
        return "the answer ignores the block latents"
    if mean_unchanged < 0.7 or swap_unchanged < 0.7:
        return "the answer reads the block latents, but their content does not shape its moves"
    return "mixed"


# Global dropout from EGR (the spec's "Global dropout from EGR: EGD") ------------------------------

def every_other(data):
    """Phase 2's rows with the globals hidden on every other row (even positions)."""
    return [(split, row | {"livr_hide_globals": True}) if k % 2 == 0 else (split, row)
            for k, (split, row) in enumerate(data)]


def run_global_dropout(model, processor, evaluator, config, data_root, output, settings, round2, round1_routes,
                       search_run, run_manifest):
    """EGD: from EGR's weights, round 2's two phases with the global latents hidden from the answer (every
    row in phase 1, every other row in phase 2), then masked greedy with the globals hidden and shown,
    unmasked greedy, the latent states, and the swap test with the globals hidden."""
    region = getattr(model.config, "livr_region_latents", None) or {}
    if not region.get("mask") or region.get("layout") != "global+region":
        raise ValueError("Global dropout starts from a masked global + region checkpoint (EGR)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    checks = check_sets(evaluator, data_root, round2)
    data, counts = training_rows(data_root, round2, round1_routes, search_run)
    one, two = phases(data, round2, config["gradient_accumulation"])
    json_write(output / "manifest.json", run_manifest | {"arm": "egd", "settings": settings, "round2_settings": round2,
                                                         "checks": checks, "counts": counts,
                                                         "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    tuned = config | {"epochs_per_stage": round2["epochs"], "learning_rate": round2["learning_rate"],
                      "warmup_ratio": round2["warmup_ratio"]}
    started = time.time()
    train_stage(model, processor, data_root, output / "phase1", tuned, 1, final_weights_only=True, data=hidden(one),
                save_final=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    middle = time.time()
    train_stage(model, processor, data_root, output / "phase2", tuned, 2, final_weights_only=True,
                data=every_other(two))
    training = {"phase1": loss_summary(output / "phase1/steps.jsonl") | {"seconds": middle - started, "stage": 1},
                "phase2": loss_summary(output / "phase2/steps.jsonl") | {"seconds": time.time() - middle, "stage": 2}}
    seed_all(config["seed"])
    model.eval()
    model.config.livr_stage = 2
    split = round2["heldout_split"]
    items = [("heldout78", split, i, r) for i, r in enumerate(records(data_root, split))]
    shown, without = items, [(p, sp, i, r | {"livr_hide_globals": True}) for p, sp, i, r in items]
    evaluation = {}
    for name, rows, use_mask in (("masked_hidden", without, True), ("masked", shown, True), ("greedy", shown, False)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as log, \
                (masked(model) if use_mask else contextlib.nullcontext()):
            results = greedy_eval(model, processor, evaluator, data_root, rows, log, round2["max_new_tokens"])
        evaluation[name] = [sum(r["success"] for r in results), len(results)]
    latent_states(model, processor, data_root, state_items(data_root, round2), round2["state_layers"],
                  output / "states.pt")
    swap = run_latent_swap(model, processor, evaluator, data_root, output / "swap",
                           settings["swap"] | {"hide_globals": True, "conditions": ("block_swap", "block_mean")},
                           round2, run_manifest, output / "masked_hidden.jsonl")
    summary = {"arm": "egd", "checks": checks, "counts": counts, "training": training, "evaluation": evaluation,
               "swap": {k: swap[k] for k in ("identity", "conditions", "reproduced", "reading")},
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def compare_dropout(runs, settings, swap_run=None):
    """`runs`: {"egd", "egr", "eb2", "e2"}. The fixed readings: whether EGD's answer uses the block latents
    (its swap test), and exact McNemar on the held-out maps, masked (globals shown) and unmasked.
    `swap_run`: a separate swap-test job's output, when the arm's own did not finish (job 7124258); the
    set checks then come from the arm's manifest."""
    runs = {k: Path(v) for k, v in runs.items()}
    load = lambda run, name: {r["row_index"]: bool(r["success"]) for r in jsonl(run / name)
                              if r["population"] == "heldout78"}
    if swap_run is None:
        egd = json.loads((runs["egd"] / "summary.json").read_text())
        swap = egd["swap"]
    else:
        egd = {"checks": json.loads((runs["egd"] / "manifest.json").read_text())["checks"]}
        swap = json.loads((Path(swap_run) / "summary.json").read_text())
    c = swap["conditions"]
    uses = (c["block_mean"]["unchanged"] / max(1, c["block_mean"]["maps"]) < 1 - settings["uses_change"]
            or c["block_swap"]["donor_route"] / max(1, c["block_swap"]["maps"]) >= settings["uses_donor"])
    valid = (egd["checks"]["valid"] and all(swap["identity"])
             and swap["reproduced"]["share"] >= settings["reproduce"])
    rows = sorted(load(runs["eb2"], "greedy.jsonl"))
    tests = {}
    for mode, file in (("masked", "masked.jsonl"), ("greedy", "greedy.jsonl")):
        mine = load(runs["egd"], file)
        others = ("egr", "eb2") if mode == "masked" else ("egr", "eb2", "e2")
        tests[mode] = {name: paired(mine, load(runs[name], file), rows) for name in others}
    improves = any(t["p"] < 0.05 and t["only_first"] > t["only_second"] for t in
                   (tests["masked"]["egr"], tests["greedy"]["egr"]))
    reading = ("withheld: a validity check failed" if not valid
               else "hiding the globals did not force reliance on the block latents" if not uses
               else "making the answer use the latents helps routes" if improves
               else "reading the map from the latents does not fix the policy; the limit is the decision process")
    hidden_count = load(runs["egd"], "masked_hidden.jsonl")
    return {"valid": valid, "uses_block_latents": uses, "swap": swap, "tests": tests,
            "masked_hidden_solved": sum(hidden_count.values()), "reading": reading}


# Readings (CPU) ---------------------------------------------------------------------------------

def block_probe(states, layer, token_for, populations=("pool78", "heldout78")):
    """Per grid size, each block's holes from one latent token's state (`token_for(n, block)`), by
    ridge fitted on pool maps and scored on held-out maps, in `expert_round2.probe_predictions`' format."""
    index = states["layers"].index(layer)
    items, x = states["items"], states["states"][:, index].float()
    out = {}
    for n in sorted({len(i["map_desc"]) for i in items}):
        fit = [k for k, i in enumerate(items) if i["population"] == populations[0] and len(i["map_desc"]) == n]
        test = [k for k, i in enumerate(items) if i["population"] == populations[1] and len(i["map_desc"]) == n]
        if not fit or not test:
            continue
        truth = torch.stack([hole_labels(items[k]) for k in test]) > 0.5
        y_fit = torch.stack([hole_labels(items[k]) for k in fit])
        predicted = torch.zeros_like(truth)
        for b, cells in enumerate(blocks(n)):
            idx = [c for c, _ in cells]
            t = token_for(n, b)
            scores, _ = ridge(x[fit, t], y_fit[:, idx], x[test, t])
            predicted[:, idx] = scores > 0.5
        out[n] = {"predicted": predicted, "truth": truth, "alpha": None, "rows": [items[k]["row_index"] for k in test],
                  "goal_accuracy": None, "start_accuracy": None}
    return out


def token_similarity(states, layer, offset=0, population=("pool78", "heldout78")):
    """Per grid size: each block latent's per-cell hole map (correlation, on held-out maps, of a ridge from
    that latent's state to every cell's hole) and the mean similarity between latents' maps. Also each
    latent's mean correlation on its own block against the other blocks."""
    index = states["layers"].index(layer)
    items, x = states["items"], states["states"][:, index].float()
    out = {}
    for n in sorted({len(i["map_desc"]) for i in items}):
        fit = [k for k, i in enumerate(items) if i["population"] == population[0] and len(i["map_desc"]) == n]
        test = [k for k, i in enumerate(items) if i["population"] == population[1] and len(i["map_desc"]) == n]
        if not fit or not test:
            continue
        y_fit = torch.stack([hole_labels(items[k]) for k in fit])
        y_test = torch.stack([hole_labels(items[k]) for k in test])
        maps = []
        for t in range(offset, offset + len(blocks(n))):
            scores, _ = ridge(x[fit, t], y_fit, x[test, t])
            centred, truth = scores.float() - scores.float().mean(0), y_test - y_test.mean(0)
            maps.append((centred * truth).sum(0) / (centred.norm(dim=0) * truth.norm(dim=0)).clamp(min=1e-9))
        maps = torch.stack(maps)
        similarity = torch.corrcoef(maps)
        own = [float(maps[b, [c for c, _ in cells]].mean()) for b, cells in enumerate(blocks(n))]
        other = [float(maps[b, [c for c in range(n * n) if c not in {c for c, _ in cells}]].mean())
                 for b, cells in enumerate(blocks(n))]
        out[str(n)] = {"similarity": float(similarity[~torch.eye(len(maps), dtype=torch.bool)].mean()),
                       "own_block": statistics.mean(own), "other_blocks": statistics.mean(other)}
    return out


def compare_region(data_root, round2, runs, gate_run, livr_outcomes, arm="ejr", offset=0, separation=None):
    """`runs`: {"e", "e2", "eb2", `arm`}. The fixed readings: `perception_jepa.compare` with the
    count-matched manipulation check (each block from its own latent, `offset` + block, against EB2's
    token 0) and the start check in place of the matched first step. With `separation` (EGR), "took"
    also needs the block latents' mean similarity below it, and masked greedy is compared with EB2's.
    Also round 2's probe on the first four latents alone, and (with an offset) on the block latents."""
    runs = {k: Path(v) for k, v in runs.items()}
    layer = round2["probe_layer"]
    arm_states = torch.load(runs[arm] / "states.pt")
    eb2_states = torch.load(runs["eb2"] / "states.pt")
    probes = {arm: block_probe(arm_states, layer, lambda n, b: offset + b),
              "eb2": block_probe(eb2_states, layer, lambda n, b: 0)}
    summary = json.loads((runs[arm] / "summary.json").read_text())
    similarity = token_similarity(arm_states, layer, offset)
    mean_similarity = statistics.mean(v["similarity"] for v in similarity.values())
    separated = True if separation is None else mean_similarity < separation
    result = compare(data_root, round2, runs, gate_run, livr_outcomes, arm=arm, manipulation_probes=probes,
                     start_check=summary["start_check"], took_requires=separated)
    result["manipulation"]["separation"] = {"by_size": similarity, "mean_similarity": mean_similarity,
                                            "threshold": separation, "separated": separated}
    first_four = dict(arm_states) | {"states": arm_states["states"][:, :, :4]}
    result["descriptive"]["probe_tokens_0_to_3"] = probe_score(probe_predictions(first_four, layer))
    if offset:
        block_only = dict(arm_states) | {"states": arm_states["states"][:, :, offset:]}
        result["descriptive"]["probe_block_latents"] = probe_score(probe_predictions(block_only, layer))
    if separation is not None:
        masked = {name: {r["row_index"]: bool(r["success"]) for r in jsonl(runs[name] / "masked.jsonl")
                         if r["population"] == "heldout78"} for name in (arm, "eb2")}
        rows = sorted(masked["eb2"])
        if sorted(masked[arm]) != rows:
            raise ValueError("The masked evaluations scored different maps")
        test = paired(masked[arm], masked["eb2"], rows)
        took, valid = result["manipulation"]["took"], result["validity"]["valid"]
        test["reading"] = ("withheld: a validity check failed" if not valid
                           else "did not take; no reading" if not took
                           else (f"{arm.upper()} solves more with the image path cut" if test["only_first"] > test["only_second"]
                                 else f"{arm.upper()} solves fewer with the image path cut") if test["p"] < 0.05
                           else "no detectable difference with the image path cut")
        result["discriminating_masked"] = test
    return result
