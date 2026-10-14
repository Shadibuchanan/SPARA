"""Bounded actual-model checks and cost measurements for all three phases; not a benchmark.

Checks are recorded rather than aborting, so one allocation reports every result;
`summary.json` sets `passed` only when all of them hold.
"""
import contextlib
import gc
import json
import resource
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from ..data import records
from ..evaluation import ReferenceEvaluator
from ..runtime import json_write, make_optimizer, save_checkpoint
from ..training import amp
from . import serialization
from .decoding import LatentDecoder
from .helper import helper_image
from .model import (ScaffoldingEncoder, ScaffoldingQwen, add_heads, answer_sequence, predictor_sequence,
                    suppression, train_language)
from .training import (answer_loss, check_finite, collect_group, grad_norm, latent_matching, rl_update,
                       scaffold_loss, to_device)


def measured(device, function):
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tick = time.perf_counter()
    result = function()
    if cuda:
        torch.cuda.synchronize()
    return result, {"seconds": time.perf_counter() - tick,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0,
                    "host_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sequence_logits(model, sample, latents):
    prompt = sample["prompt"]
    ids, _ = answer_sequence(prompt["input_ids"], sample["answer"], model.config)
    return model.lm_head(model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], latents)).float()


def prior_of(model, sample):
    prompt, k = sample["prompt"], model.config.scaffold_k
    queries = predictor_sequence(prompt["input_ids"], model.config)
    return model.hidden_states(queries, prompt["pixel_values"], prompt["image_grid_thw"],
                               query_start=queries.shape[1] - k)[0, -k:]


def cache_agreement(model, prompt, precision):
    """Greedy cached decoding through a forced latent block versus one uncached
    scoring pass over the final sequence (log-probabilities of chosen tokens)."""
    c, device = model.config, model.device
    decoder = LatentDecoder(model, {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": 24}, "none")
    context = contextlib.nullcontext() if precision == "fp32" else amp(device)
    with torch.no_grad(), context:
        hidden, cache, delta = model.prefill(prompt["input_ids"], prompt["pixel_values"], prompt["image_grid_thw"])
        state = (model.lm_head(hidden[:, -1]), cache, delta)
        first = (torch.tensor(c.scaffold_bot_id, device=device), torch.tensor(0.0, device=device))
        rollout = decoder.generate(prompt, 24, state, first)
        length = prompt["input_ids"].shape[1]
        ids = torch.cat([prompt["input_ids"], torch.tensor([rollout.ids], device=device)], dim=1)
        positions = length + torch.tensor(rollout.sampled[1:], device=device, dtype=torch.long)
        hidden = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], rollout.latent[None])
        logits = suppression(model.lm_head(hidden[0, positions - 1]).float(), c, positions > length + rollout.block)
        scored = logits.log_softmax(-1).gather(-1, ids[0, positions][:, None]).squeeze(1)
    error = (scored.cpu() - torch.tensor(rollout.log_probs[1:])).abs()
    return {"max_abs_log_prob_error": float(error.max()) if error.numel() else 0.0, "tokens": int(error.numel())}


def run_pilot(config, data_root, output, model, processor, run_manifest=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if run_manifest:
        json_write(output / "manifest.json", run_manifest)
    device, k, rows = model.device, config["latent_tokens"], records(data_root, "train")
    measurements, checks = [], {}

    def record(entry):
        measurements.append(entry)
        print(json.dumps(entry, default=str), flush=True)
        json_write(output / "measurements.json", measurements)

    def check(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}
        print(json.dumps({"check": name, **checks[name]}, default=str), flush=True)
        json_write(output / "checks.json", checks)

    # Expanded training lengths, including the latent block.
    lengths = []
    for row in rows:
        sample = serialization.training_sample(processor, data_root, row, config)
        lengths.append(sample["prompt"]["input_ids"].shape[1] + k + 2 + sample["answer"].shape[1])
    order = sorted(range(len(rows)), key=lengths.__getitem__)
    typical, longest = order[len(order) // 2], order[-1]
    json_write(output / "input-lengths.json", {"tokens": lengths, "typical_row": typical, "longest_row": longest})
    helper_image(data_root, rows[typical], config).save(output / "helper-typical.png")
    samples = {i: to_device(serialization.training_sample(processor, data_root, rows[i], config, True), device)
               for i in (typical, longest)}

    # Stage 1, Scaffolding Phase: only psi trains, through the frozen base VLM.
    phase = config["scaffold"]
    encoder = ScaffoldingEncoder.from_vlm(model, k, phase["pooling_heads"])
    model.requires_grad_(False)
    model.train(); model.visual.eval()
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder.train()
    optimizer = make_optimizer(encoder, config | {"learning_rate": phase["learning_rate"]})
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    frozen_probe = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    def scaffold_update(index, microsteps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(microsteps):
            with amp(device):
                loss, _ = scaffold_loss(model, encoder, samples[index])
            check_finite(loss)
            (loss / microsteps).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        return sum(losses) / len(losses)

    for kind, index in (("typical", typical), ("longest", longest)):
        loss, cost = measured(device, lambda: scaffold_update(index, phase["gradient_accumulation"]))
        record({"phase": "scaffold", "kind": kind, "row": index, "tokens": lengths[index], "loss": loss,
                "microsteps": phase["gradient_accumulation"], **cost})
    fit = [scaffold_update(typical, 1) for _ in range(10)]
    check("scaffold_tiny_fit", fit[-1] < 0.9 * fit[0], fit)
    check("scaffold_base_vlm_frozen", all(p.grad is None for p in model.parameters())
          and torch.equal(frozen_probe, model.model.layers[0].self_attn.q_proj.weight))
    encoder.eval()
    with torch.no_grad(), amp(device):
        targets = {i: encoder(**samples[i]["helper"]).float() for i in samples}
    save_checkpoint(output / "scaffold", encoder, processor, optimizer, scheduler,
                    {"phase": "scaffold", "pilot": True, "config": config})
    reloaded = ScaffoldingEncoder.from_pretrained(output / "scaffold", device=device).eval()
    with torch.no_grad(), amp(device):
        again = reloaded(**samples[typical]["helper"]).float()
    check("scaffold_roundtrip", torch.allclose(targets[typical], again, atol=1e-4, rtol=1e-4))
    encoder = reloaded = optimizer = scheduler = None  # rebinding keeps closures valid
    release()

    # Stage 1, Latent-Prediction Phase: language model trains, vision tower frozen.
    phase = config["predict"]
    train_language(model)
    model.train(); model.visual.eval()
    model.config.scaffold_phase = "predict"
    optimizer = make_optimizer(model, config | {"learning_rate": phase["learning_rate"]})
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    query_ids = model.config.scaffold_query_ids
    queries_before = model.get_input_embeddings().weight[query_ids].detach().clone()
    visual_probe = next(model.visual.parameters()).detach().clone()
    query_gradients = []

    def predict_update(index, microsteps):
        optimizer.zero_grad(set_to_none=True)
        rows_stats = []
        for _ in range(microsteps):
            with amp(device):
                latent, stats = latent_matching(model, samples[index], targets[index])
            check_finite(latent)
            (phase["lambda_latent"] * latent / microsteps).backward()
            with amp(device):
                ce, ce_stats = answer_loss(model, samples[index], targets[index])
            check_finite(ce)
            (phase["lambda_task"] * ce / microsteps).backward()
            rows_stats.append(stats | ce_stats)
        query_gradients.append(bool(model.get_input_embeddings().weight.grad[query_ids].abs().sum(-1).gt(0).all()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["max_grad_norm"],
                                       error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        return {key: sum(s[key] for s in rows_stats) / len(rows_stats) for key in rows_stats[0]}

    for kind, index in (("typical", typical), ("longest", longest)):
        stats, cost = measured(device, lambda: predict_update(index, phase["gradient_accumulation"]))
        record({"phase": "predict", "kind": kind, "row": index, "tokens": lengths[index], **stats,
                "microsteps": phase["gradient_accumulation"], **cost})
    fit = [predict_update(typical, 1) for _ in range(10)]
    weighted = [phase["lambda_latent"] * f["latent_loss"] + phase["lambda_task"] * f["ce_loss"] for f in fit]
    check("predict_tiny_fit", weighted[-1] < 0.9 * weighted[0], fit)
    check("predict_query_gradients", all(query_gradients))
    check("predict_vision_frozen", all(p.grad is None for p in model.visual.parameters())
          and torch.equal(visual_probe, next(model.visual.parameters())))
    check("predict_queries_updated", not torch.equal(queries_before, model.get_input_embeddings().weight[query_ids]))
    model.eval()
    agreement = {precision: cache_agreement(model, samples[typical]["prompt"], precision) for precision in ("fp32", "bf16")}
    check("decode_cache_agreement_fp32", agreement["fp32"]["max_abs_log_prob_error"] < 1e-3, agreement)
    with torch.no_grad():
        expected = sequence_logits(model, samples[typical], targets[typical][None])
    # Weights-only roundtrip: a fresh optimizer keeps the pilot checkpoint small.
    optimizer = scheduler = None
    release()
    fresh = make_optimizer(model, config | {"learning_rate": 0.0})
    save_checkpoint(output / "predict", model, processor, fresh, get_cosine_schedule_with_warmup(fresh, 0, 1),
                    {"phase": "predict", "pilot": True, "config": config})
    del fresh
    reference = ScaffoldingQwen.from_pretrained(output / "predict", attn_implementation=config["attention_backend"],
                                               torch_dtype=getattr(torch, config["rl"]["reference_dtype"])).to(device)
    reference.requires_grad_(False)
    reference.eval()
    with torch.no_grad():
        actual = sequence_logits(reference, samples[typical], targets[typical][None])
    check("predict_roundtrip", torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)
          if reference.dtype == torch.float32 else True, float((expected - actual).abs().max()))

    # Stage 2, Scaffolding RL.
    rl = config["rl"]
    with torch.no_grad():
        prior = prior_of(model, samples[typical])
        stage1 = sequence_logits(model, samples[typical], prior[None])
    if rl["variant"] == "scaffolding":
        add_heads(model, rl)
        with torch.no_grad():
            mean, sigma = model.latent_heads(prior)
            check("heads_initialized", mean.abs().max().item() == 0
                  and torch.allclose(sigma, torch.full_like(sigma, rl["sigma_init"])))
            actions, _, _ = LatentDecoder(model, {"do_sample": False}, "mean").action(prior, 1)
            zero = sequence_logits(model, samples[typical], actions)
            nudged = sequence_logits(model, samples[typical], actions + sigma * torch.randn_like(actions))
            bot = samples[typical]["prompt"]["input_ids"].shape[1]
        check("zero_adjustment_equivalence", torch.equal(zero, stage1))
        check("adjustment_changes_only_later_tokens", torch.equal(nudged[:, :bot + 1], zero[:, :bot + 1])
              and not torch.equal(nudged[:, bot + k + 1:], zero[:, bot + k + 1:]))
    train_language(model)
    model.config.scaffold_phase = "rl"
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, config | {"learning_rate": rl["learning_rate"]})
    decoder = LatentDecoder(model, rl["rollout"], "sample" if rl["variant"] == "scaffolding" else "none")
    evaluator = ReferenceEvaluator(data_root)
    model.eval()
    groups = []
    for kind, index in (("typical", typical), ("longest", longest)):
        group, cost = measured(device, lambda: collect_group(decoder, processor, data_root, rows[index], evaluator, rl))
        groups.append(group)
        record({"phase": "rl", "kind": f"rollout-{kind}", "row": index, "group_size": rl["group_size"],
                "rewards": group["rewards"], "response_tokens": [len(r.sampled) for r in group["rollouts"]],
                "block_rate": sum(r.block is not None for r in group["rollouts"]) / len(group["rollouts"]), **cost})
    model.train(); model.visual.eval()
    old = {}
    optimizer.zero_grad(set_to_none=True)
    update, cost = measured(device, lambda: rl_update(model, reference, groups, rl, old))
    total_norm = grad_norm(parameters)
    # Pilot only: the text term's own gradient norm, at the same theta_old.
    optimizer.zero_grad(set_to_none=True)
    rl_update(model, reference, groups, rl | {"variant": "text_only"}, old)
    text_norm = grad_norm(parameters)
    optimizer.zero_grad(set_to_none=True)
    rl_update(model, reference, groups, rl, old)
    torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    record({"phase": "rl", "kind": "update", "prompts": len(groups), **update, "total_grad_norm": total_norm,
            "text_grad_norm": text_norm, **cost})
    check("rl_ratio_one_at_theta_old", update.get("latent_abs_log_ratio", 0.0) == 0 and update["text_abs_log_ratio"] == 0)
    check("rl_kl_near_zero_at_start", update["kl"] < 1e-3, update["kl"])
    optimizer = reference = None
    release()

    # Evaluation-cost probe on training prompts with the benchmark decoding settings.
    model.eval()
    mode = config["evaluation_latent"] if hasattr(model, "latent_heads") else "none"
    evaluation = LatentDecoder(model, config["generation"], mode)
    for kind, index in (("typical", typical), ("longest", longest)):
        with torch.no_grad(), amp(device):
            rollout, cost = measured(device, lambda: evaluation.generate(samples[index]["prompt"]))
        record({"phase": "evaluate", "kind": kind, "train_row": index, "diagnostic_only": True,
                "generated_tokens": len(rollout.sampled), "latent_block": rollout.block is not None,
                "response": processor.tokenizer.decode(rollout.ids, skip_special_tokens=False), **cost})

    def seconds(phase, kind):
        return next(m["seconds"] for m in measurements if m["phase"] == phase and m["kind"] == kind)

    per_row = {name: seconds(name, "typical") / config[name]["gradient_accumulation"] for name in ("scaffold", "predict")}
    rollout_seconds = (seconds("rl", "rollout-typical") + seconds("rl", "rollout-longest")) / 2
    estimate = {
        "scaffold": per_row["scaffold"] * config["scaffold"]["epochs"] * len(rows) / 3600,
        "predict": per_row["predict"] * config["predict"]["epochs"] * len(rows) / 3600,
        "rl": (rollout_seconds + seconds("rl", "update") / len(groups)) * rl["prompts_per_step"] * rl["steps"] / 3600,
        "evaluate": (seconds("evaluate", "typical") + seconds("evaluate", "longest")) / 2 * 400 / 3600,
    }
    json_write(output / "summary.json", {
        "passed": all(c["passed"] for c in checks.values()), "pilot_only": True, "checks": checks,
        "estimated_gpu_hours": estimate | {"total": sum(estimate.values())},
        "estimate_caveat": "Typical-row microsteps and two RL prompts; excludes loading, checkpoint I/O, "
                           "queue time and the shift in rollout length as RL trains",
        "peak_gpu_allocated_bytes": max(m["peak_gpu_allocated_bytes"] for m in measurements),
        "total_seconds": time.perf_counter() - started})
