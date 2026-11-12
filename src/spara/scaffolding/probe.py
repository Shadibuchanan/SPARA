"""Latent-sensitivity probe: how often does a Gaussian adjustment of scale sigma
change the route a checkpoint writes?

Scaffolding RL can only reward latent exploration that changes routes. Decoding
is greedy after the latent block, so every change comes from the adjustment
rather than from text sampling. Rows are training maps, as in RL.
"""
import json
import random
import statistics
from pathlib import Path

import torch

from ..data import records
from ..evaluation import ReferenceEvaluator
from ..runtime import json_write
from ..training import amp
from . import serialization
from .decoding import LatentDecoder
from .model import suppression
from .policy import masked_kl
from .training import to_device


def moves(answer):
    """The reference parser's view of a route: every U, D, R or L character."""
    return "".join(ch for ch in answer.upper() if ch in "UDRL")


def probe_row(decoder, processor, prompt, row, evaluator, sigmas, draws, generator):
    """Greedy routes for delta = 0 and `draws` isotropic adjustments per sigma."""
    model, ids = decoder.model, prompt["input_ids"]
    _, cache, delta = model.prefill(ids, prompt.get("pixel_values"), prompt.get("image_grid_thw"))
    prior = model.latent_prior(cache, delta).float()
    noise = torch.randn((len(sigmas) * draws, *prior.shape), generator=generator, device=prior.device)
    scale = torch.tensor(sigmas, device=prior.device).repeat_interleave(draws).view(-1, 1, 1)
    actions = torch.cat([prior[None], prior + scale * noise])
    rollouts, logits = decoder.decode_block(ids, cache, delta, actions, decoder.settings.max_new_tokens)
    log_p = suppression(logits.float(), model.config, True).log_softmax(-1)
    # KL(zero adjustment || adjusted) over the first answer token.
    kl = masked_kl(log_p[:1].expand_as(log_p), log_p).tolist()
    scored = []
    for rollout in rollouts:
        text = processor.tokenizer.decode(ids[0].tolist() + rollout.ids, skip_special_tokens=False)
        result = evaluator.score(row, text)
        scored.append((moves(result["extracted_answer"]), bool(result["success"])))
    (route, success), adjusted = scored[0], scored[1:]
    per_sigma = {}
    for i, sigma in enumerate(sigmas):
        draws_i = range(i * draws, (i + 1) * draws)
        per_sigma[str(sigma)] = {"route_changed": [adjusted[j][0] != route for j in draws_i],
                                 "success": [adjusted[j][1] for j in draws_i],
                                 "first_token_kl": [kl[j + 1] for j in draws_i]}
    return {"grid_size": len(row["map_desc"]), "route": route, "success": success,
            "latent_rms": float(prior.square().mean().sqrt()), "sigmas": per_sigma}


def summarize_probe(results, sigmas):
    base = sum(r["success"] for r in results)
    summary = {"rows": len(results), "base_successes": base,
               "latent_rms": statistics.mean(r["latent_rms"] for r in results), "sigmas": {}}
    for sigma in map(str, sigmas):
        cells = [r["sigmas"][sigma] for r in results]
        changed = [c for cell in cells for c in cell["route_changed"]]
        outcomes = [(r["success"], s) for r, cell in zip(results, cells) for s in cell["success"]]
        summary["sigmas"][sigma] = {
            "route_change_rate": sum(changed) / len(changed),
            "rows_with_any_change": sum(any(cell["route_changed"]) for cell in cells),
            "success_rate": sum(s for _, s in outcomes) / len(outcomes),
            "gained": sum(s and not b for b, s in outcomes), "lost": sum(b and not s for b, s in outcomes),
            "median_first_token_kl": statistics.median(k for cell in cells for k in cell["first_token_kl"])}
    return summary


def latent_probe(model, processor, data_root, output, config, sigmas, rows, draws, run_manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "manifest.json", run_manifest | {"sigmas": sigmas, "rows": rows, "draws": draws})
    evaluator = ReferenceEvaluator(data_root)
    data = records(data_root, "train")
    picked = sorted(random.Random(config["seed"]).sample(range(len(data)), min(rows, len(data))))
    # Greedy with the rollouts' token limit and no repetition penalty, as in RL.
    decoder = LatentDecoder(model, {"do_sample": False, "repetition_penalty": 1.0,
                                    "max_new_tokens": config["rl"]["rollout"]["max_new_tokens"]}, "none")
    generator = torch.Generator(device=model.device).manual_seed(config["seed"])
    model.eval()
    results = []
    with (output / "rows.jsonl").open("w", encoding="utf-8") as log:
        for index in picked:
            row = data[index]
            prompt = to_device(serialization.prompt(processor, data_root, "train", row), model.device)
            with torch.no_grad(), amp(model.device):
                result = probe_row(decoder, processor, prompt, row, evaluator, sigmas, draws, generator)
            results.append(result | {"row_index": index})
            log.write(json.dumps(results[-1]) + "\n"); log.flush()
    summary = summarize_probe(results, sigmas)
    json_write(output / "summary.json", summary)
    return summary
