"""Probe for structured exploration on harder maps (docs/STRUCTURED_EXPLORATION_SPEC.md).

One model load runs three parts on a final LIVR checkpoint:

1. basis: principal directions of the latent states at the adjusted layer across the
   released training maps, centred within grid size, per latent slot;
2. sweep: greedy and sampled success on a fresh map set, per grid size;
3. noise: greedy routes under isotropic and structured adjustments on the fresh maps
   of the smaller sizes, with rescue and break rates.

The rules for choosing a scale, the noise family and the training sizes are fixed in
the specification and applied here.
"""
import json
from pathlib import Path

import torch

from .data import records
from .evaluation import ReferenceEvaluator
from .latent_rl import LatentAdjuster, choose_sigma, moves
from .runtime import json_write
from .serialization import encode
from .text_rl import repeat_prompt
from .training import amp, to_device

GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


@torch.no_grad()
def latent_states(model, processor, data_root, split, rows, adjuster):
    """z at the adjusted layer for each row: (rows, K, d), float32 on CPU."""
    states = []
    for row in rows:
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        with adjuster.capture() as state, amp(model.device):
            model(**prompt, use_cache=False)
        if len(state["priors"]) != 1:
            raise RuntimeError(f"Expected one latent capture per prompt, saw {len(state['priors'])}")
        states.append(state["priors"][0][0].float().cpu())
    return torch.stack(states)


def principal_basis(states, sizes, k):
    """Per slot, the top-k principal directions (K, d, k) of `states` (n, K, d) centred
    within grid size, the standard deviation along each (K, k), and the share of
    variance they capture."""
    states = states.double()
    centred = states.clone()
    groups = sorted(set(sizes))
    for size in groups:
        index = torch.tensor([i for i, s in enumerate(sizes) if s == size])
        centred[index] -= states[index].mean(0)
    directions, scales, captured = [], [], []
    for slot in range(states.shape[1]):
        _, singular, vh = torch.linalg.svd(centred[:, slot], full_matrices=False)
        variance = singular.square() / (len(states) - len(groups))
        directions.append(vh[:k].T)
        scales.append(variance[:k].sqrt())
        captured.append(float(variance[:k].sum() / variance.sum()))
    return {"directions": torch.stack(directions).float(), "scales": torch.stack(scales).float(),
            "captured": captured, "k": k, "maps": len(states)}


def structured_offsets(basis, scale, count, generator, device):
    """delta = scale * sum_j u_j s_j eps_j per slot: (count, K, d)."""
    directions, scales = basis["directions"].to(device), basis["scales"].to(device)
    noise = torch.randn((count, *scales.shape), generator=generator, device=device)
    return scale * torch.einsum("sdk,sk,nsk->nsd", directions, scales, noise)


def isotropic_offsets(slots, size, scale, count, generator, device):
    return scale * torch.randn((count, slots, size), generator=generator, device=device)


def score(evaluator, processor, row, sequences):
    results = [evaluator.score(row, processor.tokenizer.decode(s, skip_special_tokens=False)) for s in sequences]
    return [(moves(r["extracted_answer"]), bool(r["success"])) for r in results]


def mixed(successes):
    return 0 < sum(successes) < len(successes)


@torch.no_grad()
def success_sweep(model, processor, data_root, split, rows, samples, generation, evaluator, log, max_new_tokens):
    """Greedy answer and `samples` sampled answers per map."""
    model.eval()
    results = []
    for count, (index, row) in enumerate(rows):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        with amp(model.device):
            greedy = model.generate(**prompt, **GREEDY, max_new_tokens=max_new_tokens, use_cache=True)
            sampled = model.generate(**repeat_prompt(prompt, samples), **generation, use_cache=True)
        (route, success), = score(evaluator, processor, row, greedy)
        record = {"row_index": index, "grid_size": len(row["map_desc"]), "prompt_tokens": prompt["input_ids"].shape[-1],
                  "greedy_route": route, "greedy_success": success,
                  "sample_successes": [s for _, s in score(evaluator, processor, row, sampled)],
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 25 == 0:
            print(f"sweep {count + 1}/{len(rows)}", flush=True)
    return results


def summarise_sweep(results, margin=0.05):
    per_size = {}
    for size in sorted({r["grid_size"] for r in results}):
        cells = [r for r in results if r["grid_size"] == size]
        failures = [r for r in cells if not r["greedy_success"]]
        peaks = [r["peak_gpu_gib"] for r in cells if r["peak_gpu_gib"] is not None]
        per_size[str(size)] = {
            "maps": len(cells),
            "greedy_success_rate": sum(r["greedy_success"] for r in cells) / len(cells),
            "sample_success_rate": sum(sum(r["sample_successes"]) for r in cells) / sum(len(r["sample_successes"]) for r in cells),
            "mixed_share": sum(mixed(r["sample_successes"]) for r in cells) / len(cells),
            "all_fail_share": sum(not any(r["sample_successes"]) for r in cells) / len(cells),
            "all_success_share": sum(all(r["sample_successes"]) for r in cells) / len(cells),
            "greedy_failures": len(failures),
            # Text sampling's own exploration: greedy failures where some sample succeeds.
            "greedy_failures_rescued_by_sampling": (sum(any(r["sample_successes"]) for r in failures) / len(failures)
                                                    if failures else None),
            "prompt_tokens_max": max(r["prompt_tokens"] for r in cells),
            "peak_gpu_gib": max(peaks) if peaks else None}
    best = max(cell["mixed_share"] for cell in per_size.values())
    chosen = [int(size) for size, cell in per_size.items() if cell["mixed_share"] >= best - margin - 1e-9]
    return {"per_size": per_size, "training_sizes": chosen}


@torch.no_grad()
def noise_probe(model, processor, data_root, split, rows, families, draws, evaluator, adjuster, generator, log,
                max_new_tokens):
    """Greedy routes per map with no adjustment and `draws` adjustments of each scale in
    each family. `families` maps a name to (scales, offsets(scale, count))."""
    model.eval()
    results = []
    for count, (index, row) in enumerate(rows):
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        slots = len(model.config.livr_latent_ids)
        plan = [(name, scale) for name, (scales, _) in families.items() for scale in scales]
        offsets = torch.cat([torch.zeros(1, slots, model.config.hidden_size, device=model.device)]
                            + [families[name][1](scale, draws) for name, scale in plan])
        with adjuster.offset(offsets), amp(model.device):
            generated = model.generate(**repeat_prompt(prompt, len(offsets)), **GREEDY,
                                       max_new_tokens=max_new_tokens, use_cache=True)
        scored = score(evaluator, processor, row, generated)
        (route, success), adjusted = scored[0], scored[1:]
        record = {"row_index": index, "grid_size": len(row["map_desc"]), "route": route, "success": success,
                  "families": {name: {} for name in families}}
        for i, (name, scale) in enumerate(plan):
            cells = adjusted[i * draws:(i + 1) * draws]
            record["families"][name][str(scale)] = {"route_changed": [r != route for r, _ in cells],
                                                    "success": [s for _, s in cells]}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 25 == 0:
            print(f"noise {count + 1}/{len(rows)}", flush=True)
    return results


def summarise_noise(results):
    base = sum(r["success"] for r in results) / len(results)
    failed, solved = [r for r in results if not r["success"]], [r for r in results if r["success"]]
    summary = {"rows": len(results), "base_success_rate": base, "base_failures": len(failed), "families": {}}
    for name in results[0]["families"]:
        scales = {}
        for scale in results[0]["families"][name]:
            cell = lambda r: r["families"][name][scale]
            changed = [c for r in results for c in cell(r)["route_changed"]]
            successes = [s for r in results for s in cell(r)["success"]]
            rescued = sum(s for r in failed for s in cell(r)["success"])
            broken = sum(not s for r in solved for s in cell(r)["success"])
            draws = len(cell(results[0])["success"])
            scales[scale] = {
                "route_change_rate": sum(changed) / len(changed), "success_rate": sum(successes) / len(successes),
                "rescue_rate": rescued / (len(failed) * draws) if failed else None,
                "break_rate": broken / (len(solved) * draws) if solved else None,
                # A rescue or a break always changes the route.
                "rescue_share": rescued / sum(changed) if sum(changed) else None,
                "maps_with_rescue": sum(any(cell(r)["success"]) for r in failed),
                "maps_with_break": sum(not all(cell(r)["success"]) for r in solved)}
        summary["families"][name] = {
            "scales": scales, "chosen": choose_sigma({"base_success_rate": base, "sigmas": scales})}
    summary["decision"] = choose_noise(summary)
    return summary


def choose_noise(summary):
    """Structured noise at its chosen scale if one qualifies and its rescue share is at
    least that of the isotropic scale with the closest route-change rate; otherwise
    isotropic noise at its chosen scale (None if neither qualifies)."""
    structured, isotropic = summary["families"]["structured"], summary["families"]["isotropic"]
    fallback = {"family": "isotropic", "scale": isotropic["chosen"]} if isotropic["chosen"] is not None else None
    if structured["chosen"] is None:
        return fallback and fallback | {"reason": "no structured scale met the rule"}
    cell = structured["scales"][str(structured["chosen"])]
    sigma, match = min(isotropic["scales"].items(),
                       key=lambda item: abs(item[1]["route_change_rate"] - cell["route_change_rate"]))
    comparison = {"structured_rescue_share": cell["rescue_share"], "matched_isotropic_sigma": float(sigma),
                  "isotropic_rescue_share": match["rescue_share"]}
    if (cell["rescue_share"] or 0.0) >= (match["rescue_share"] or 0.0):
        return {"family": "structured", "scale": structured["chosen"], **comparison}
    return (fallback or {"family": None, "scale": None}) | comparison | {"reason": "lower rescue share than matched isotropic"}


def explore_probe(model, processor, data_root, output, split, rl, run_manifest, seed, basis_k=16, samples=8,
                  draws=4, isotropic=(1.2, 1.6, 2.0, 2.4), structured=(0.5, 1.0, 2.0, 4.0, 8.0),
                  noise_sizes=(5, 6), max_new_tokens=128):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    layer = rl["latent"]["layer"]
    json_write(output / "manifest.json", run_manifest | {
        "split": split, "layer": layer, "basis_k": basis_k, "samples": samples, "draws": draws,
        "isotropic": list(isotropic), "structured": list(structured), "noise_sizes": list(noise_sizes),
        "max_new_tokens": max_new_tokens, "generation": rl["generation"], "seed": seed})
    evaluator, adjuster = ReferenceEvaluator(data_root), LatentAdjuster(model, layer)
    model.eval()
    train = records(data_root, "train")
    states = latent_states(model, processor, data_root, "train", train, adjuster)
    basis = principal_basis(states, [len(r["map_desc"]) for r in train], basis_k)
    torch.save(basis, output / "basis.pt")
    summary = {"layer": layer, "basis": {k: basis[k] for k in ("captured", "k", "maps")}}
    print(json.dumps(summary["basis"]), flush=True)

    fresh = list(enumerate(records(data_root, split)))
    with (output / "sweep.jsonl").open("w", encoding="utf-8") as log:
        sweep = success_sweep(model, processor, data_root, split, fresh, samples, rl["generation"], evaluator, log,
                              max_new_tokens)
    summary["sweep"] = summarise_sweep(sweep)
    print(json.dumps(summary["sweep"], indent=1), flush=True)

    generator = torch.Generator(device=model.device).manual_seed(seed)
    slots, size = len(model.config.livr_latent_ids), model.config.hidden_size
    families = {"isotropic": (list(isotropic), lambda s, n: isotropic_offsets(slots, size, s, n, generator, model.device)),
                "structured": (list(structured), lambda s, n: structured_offsets(basis, s, n, generator, model.device))}
    chosen_rows = [(i, row) for i, row in fresh if len(row["map_desc"]) in noise_sizes]
    with (output / "noise.jsonl").open("w", encoding="utf-8") as log:
        noise = noise_probe(model, processor, data_root, split, chosen_rows, families, draws, evaluator, adjuster,
                            generator, log, max_new_tokens)
    adjuster.remove()
    summary["noise"] = summarise_noise(noise)
    json_write(output / "summary.json", summary)
    print(json.dumps(summary["noise"], indent=1), flush=True)
    return summary
