"""Phase 0 latent battery: what a final LIVR Stage 2 checkpoint's latents carry
and whether its answers use them.

These are diagnostics, not the benchmark protocol. Decoding is greedy, so each
paired difference comes from the intervention rather than from sampling. Every
test map is answered under two attention modes and four latent conditions:

- unmasked (Stage 2, as deployed) or bottleneck (the Stage 1 mask: only the
  latents read the image);
- own latents; latents swapped in from another test map of the same size and a
  different layout; the mean latent of training maps of that size; or no
  latent tokens at all.

Swap and mean overwrite every decoder layer's output at the four latent
positions, so later tokens read the replacement through attention exactly as
they would read the model's own latent; positional encoding stays the
recipient's. Test rows have no gold answer, so each condition also scores the
reference answer, the model's own unmasked greedy answer, by teacher forcing.

A feature pass first stores hidden states for CPU probes: the latents, the last
prompt token, and each map cell's overlap-weighted image tokens.
"""
import contextlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from .data import layout_key, records
from .evaluation import ReferenceEvaluator
from .model import latent_queries
from .runtime import json_write, seed_all
from .serialization import encode
from .training import amp, to_device

MODES = {"unmasked": 2, "bottleneck": 1}
CONDITIONS = ("own", "swap", "mean", "removed")


def moves(answer):
    """The reference parser's view of a route: every U, D, R or L character."""
    return "".join(ch for ch in answer.upper() if ch in "UDRL")


def latent_positions(ids, latent_ids):
    return latent_queries(ids, latent_ids)[0].nonzero().squeeze(-1)


def probe_layers(count):
    """Every fourth decoder layer's output, ending at the last."""
    return list(range(count - 1, -1, -4))[::-1]


class LatentPatch:
    """Forward hooks on the decoder layers. Inside `apply`, a forward that covers
    `positions` (a prompt or teacher-forced pass, not a cached decode step)
    records each layer's output, or overwrites it at those positions with
    `values[layer]`."""

    def __init__(self, model):
        self.positions = self.values = self.recorded = None
        self.handles = [layer.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(model.model.layers)]

    def _hook(self, index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.positions is None or hidden.shape[1] <= int(self.positions[-1]):
                return None
            if self.recorded is not None:
                self.recorded.append(hidden[0].detach())
            if self.values is None:
                return None
            hidden = hidden.clone()
            hidden[0, self.positions] = self.values[index].to(hidden.device, hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    @contextlib.contextmanager
    def apply(self, positions, values=None, record=False):
        self.positions = positions if positions is not None and len(positions) else None
        self.values, self.recorded = values, [] if record else None
        try:
            yield self
        finally:
            self.positions = self.values = None

    def remove(self):
        for handle in self.handles:
            handle.remove()


def prompt_states(model, patch, prompt, positions):
    """Every decoder layer's output over the prompt: [layers, length, hidden]."""
    with patch.apply(positions, record=True), torch.no_grad(), amp(model.device):
        model(**prompt, use_cache=False)
    return torch.stack(patch.recorded)


def cell_weights(size, grid_thw, merge):
    """[size*size, tokens]: each map cell's overlap with each merged image token,
    rows summing to one. The map fills the image; tokens tile it row-major."""
    _, height, width = (int(x) for x in grid_thw)
    rows, cols = height // merge, width // merge

    def overlap(cells, tokens):
        c, t = torch.arange(cells)[:, None], torch.arange(tokens)[None, :]
        return (torch.minimum((c + 1) / cells, (t + 1) / tokens)
                - torch.maximum(c / cells, t / tokens)).clamp(min=0)

    weights = overlap(size, rows)[:, None, :, None] * overlap(size, cols)[None, :, None, :]
    weights = weights.reshape(size * size, rows * cols)
    return weights / weights.sum(-1, keepdim=True)


def swap_donors(rows, seed):
    """Each test row's donor: a row of the same grid size with a different layout,
    taken from a seeded cycle so donors spread across the group."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row["map_desc"])].append(index)
    rng, donors = random.Random(seed), {}
    for size in sorted(groups):
        cycle = groups[size][:]
        rng.shuffle(cycle)
        for position, index in enumerate(cycle):
            for step in range(1, len(cycle)):
                other = cycle[(position + step) % len(cycle)]
                if layout_key(rows[other]) != layout_key(rows[index]):
                    donors[index] = other
                    break
    return donors


def first_per_size(rows, limit):
    """The first `limit` rows of each grid size, in their original order."""
    counts, kept = defaultdict(int), []
    for row in rows:
        counts[len(row["map_desc"])] += 1
        if counts[len(row["map_desc"])] <= limit:
            kept.append(row)
    return kept


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def teacher(prompt, completion):
    """Teacher-forced batch scoring `completion` after `prompt`."""
    ids = torch.cat([prompt["input_ids"], completion[None]], dim=-1)
    labels = ids.clone()
    labels[:, :prompt["input_ids"].shape[-1]] = -100
    return prompt | {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def answer(model, patch, processor, evaluator, row, prompt, positions, values, settings, reference=None):
    """Greedy answer and the reference answer's mean token NLL under one condition."""
    length = prompt["input_ids"].shape[-1]
    with patch.apply(positions, values), torch.no_grad(), amp(model.device):
        generated = model.generate(**prompt, **settings, use_cache=True)
        completion = generated[0, length:]
        reference = completion if reference is None else reference
        nll = model(**teacher(prompt, reference), use_cache=False).loss.item()
    text = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
    scored = evaluator.score(row, text)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    return {"answer": scored["answer"], "route": moves(scored["extracted_answer"]),
            "success": bool(scored["success"]), "reference_nll": nll, "tokens": len(completion),
            "truncated": not any(int(t) in eos for t in completion)}, completion, text


def feature_pass(model, processor, data_root, split, rows, patch, output, layers):
    """Stores probe features and returns per-mode latents for every layer:
    a per-grid mean for train, the per-row states for test."""
    tokens, ids = model.config.livr_latent_tokens, model.config.livr_latent_ids
    merge = model.config.vision_config.spatial_merge_size
    features = {mode: {"latent": [], "last": []} for mode in MODES}
    cells, sums, states_by_mode = [], {mode: {} for mode in MODES}, {mode: [] for mode in MODES}
    for index, row in enumerate(rows):
        prompt = to_device(encode(processor, data_root, split, row, training=False, latents=tokens), model.device)
        positions = latent_positions(prompt["input_ids"], ids)
        for mode, stage in MODES.items():
            model.config.livr_stage = stage
            states = prompt_states(model, patch, prompt, positions)
            latent = states[:, positions].float()
            if split == "train":
                total, count = sums[mode].get(len(row["map_desc"]), (0, 0))
                sums[mode][len(row["map_desc"])] = (total + latent, count + 1)
            else:
                states_by_mode[mode].append(latent.cpu())
            features[mode]["latent"].append(latent[layers].half().cpu())
            features[mode]["last"].append(states[layers, -1].half().cpu())
            if mode == "unmasked":
                # Image states precede the latents, so the bottleneck leaves them unchanged.
                image = prompt["input_ids"][0] == model.config.image_token_id
                weights = cell_weights(len(row["map_desc"]), prompt["image_grid_thw"][0], merge)
                cells.append(torch.einsum("ct,ltd->cld", weights.to(states.device),
                                          states[layers][:, image].float()).half().cpu())
        if index % 100 == 0:
            print(f"features {split} {index}/{len(rows)}", flush=True)
    torch.save({"layers": layers, "cells": cells,
                "latent": {m: torch.stack(f["latent"]) for m, f in features.items()},
                "last": {m: torch.stack(f["last"]) for m, f in features.items()},
                "rows": [{"map_id": r["map_id"], "map_desc": r["map_desc"],
                          "text_output": r.get("text_output")} for r in rows]},
               Path(output) / f"features-{split}.pt")
    if split == "train":
        return {mode: {size: total / count for size, (total, count) in by_size.items()}
                for mode, by_size in sums.items()}
    return states_by_mode


def summarize_battery(results):
    summary = {}
    for mode in MODES:
        own = {r["row_index"]: r["modes"][mode]["own"] for r in results}
        for condition in CONDITIONS:
            cells = [(r, r["modes"][mode][condition]) for r in results if condition in r["modes"][mode]]
            if not cells:
                continue
            successes = [c["success"] for _, c in cells]
            base = [own[r["row_index"]]["success"] for r, _ in cells]
            lost = sum(b and not s for b, s in zip(base, successes))
            gained = sum(s and not b for b, s in zip(base, successes))
            entry = {
                "count": len(cells), "successes": sum(successes),
                "per_grid": {str(size): sum(c["success"] for r, c in cells if r["grid_size"] == size)
                             for size in sorted({r["grid_size"] for r, _ in cells})},
                "route_changed": sum(c["route"] != own[r["row_index"]]["route"] for r, c in cells),
                "mean_reference_nll": sum(c["reference_nll"] for _, c in cells) / len(cells),
                "truncated": sum(c["truncated"] for _, c in cells),
                "vs_own": {"lost": lost, "gained": gained, "mcnemar_p": mcnemar(lost, gained)}}
            if condition == "swap":
                # Donors answered so far; all of them once the battery completes.
                answered = [c for _, c in cells if c["donor_index"] in own]
                entry["donor_map_successes"] = sum(c["donor_success"] for _, c in cells)
                entry["matches_donor_route"] = sum(c["route"] == own[c["donor_index"]]["route"] for c in answered)
                entry["donors_answered"] = len(answered)
            summary[f"{mode}/{condition}"] = entry
    return summary


def run_battery(model, processor, config, data_root, output, run_manifest, limit=None, max_new_tokens=128):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    evaluator = ReferenceEvaluator(data_root)
    settings = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                "max_new_tokens": max_new_tokens}
    layers = probe_layers(model.config.num_hidden_layers)
    train, test = records(data_root, "train"), records(data_root, "test")
    if limit:
        # A pilot keeps every grid size, so each has a mean latent and swap donors.
        train, test = first_per_size(train, limit), first_per_size(test, limit)
    donors = swap_donors(test, config["seed"])
    json_write(output / "manifest.json", run_manifest | {
        "evaluator": evaluator.provenance, "probe_layers": layers, "limit": limit,
        "cell_features": "overlap-weighted image-token states; merged tokens taken as row-major "
                         "over the merged grid and the map as filling the whole image",
        "effective_generation": model.generation_config.to_dict() | settings,
        "swap_donors": {str(k): v for k, v in sorted(donors.items())}})
    seed_all(config["seed"])
    patch = LatentPatch(model)
    tokens, ids = model.config.livr_latent_tokens, model.config.livr_latent_ids

    # Before spending the feature pass: replacing a row's latents with its own
    # recorded states must not change its answer.
    identity = []
    for index in range(min(4, len(test))):
        prompt = to_device(encode(processor, data_root, "test", test[index], training=False, latents=tokens), model.device)
        positions = latent_positions(prompt["input_ids"], ids)
        for mode, stage in MODES.items():
            model.config.livr_stage = stage
            own = prompt_states(model, patch, prompt, positions)[:, positions].float()
            plain, completion, _ = answer(model, patch, processor, evaluator, test[index], prompt, None, None, settings)
            patched, _, _ = answer(model, patch, processor, evaluator, test[index], prompt, positions,
                                   own, settings, completion)
            identity.append({"row_index": index, "mode": mode, "same_route": plain["route"] == patched["route"],
                             "nll_difference": patched["reference_nll"] - plain["reference_nll"]})
    json_write(output / "identity.json", identity)
    if not all(i["same_route"] for i in identity):
        raise RuntimeError("Replacing latents with their own states changed an answer; see identity.json")

    means = feature_pass(model, processor, data_root, "train", train, patch, output, layers)
    states = feature_pass(model, processor, data_root, "test", test, patch, output, layers)
    results = []
    with (output / "rows.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(test):
            size = len(row["map_desc"])
            prompt = to_device(encode(processor, data_root, "test", row, training=False, latents=tokens), model.device)
            removed = to_device(encode(processor, data_root, "test", row, training=False, latents=[]), model.device)
            positions = latent_positions(prompt["input_ids"], ids)
            record = {"row_index": index, "map_id": row["map_id"], "grid_size": size, "modes": {}}
            reference = None
            for mode, stage in MODES.items():
                model.config.livr_stage = stage
                cells = {}
                replacements = {"own": None, "mean": means[mode][size]}
                if index in donors:
                    replacements["swap"] = states[mode][donors[index]]
                for condition, values in replacements.items():
                    cells[condition], completion, text = answer(
                        model, patch, processor, evaluator, row, prompt, positions, values, settings, reference)
                    if reference is None:
                        reference = completion  # unmasked own answer: scored under every condition
                    if condition == "swap":
                        cells[condition] |= {"donor_index": donors[index],
                                             "donor_success": bool(evaluator.score(test[donors[index]], text)["success"])}
                # The latent-free prompt is shorter, so the reference is re-appended to it.
                cells["removed"], _, _ = answer(model, patch, processor, evaluator, row, removed, None, None,
                                                settings, reference)
                record["modes"][mode] = cells
            results.append(record)
            log.write(json.dumps(record) + "\n"); log.flush()
            if index % 25 == 0:
                print(f"battery {index + 1}/{len(test)}", json.dumps(summarize_battery(results)["unmasked/own"]), flush=True)
    patch.remove()
    summary = {"rows": len(results), "identity": identity, "conditions": summarize_battery(results)}
    json_write(output / "summary.json", summary)
    return summary
