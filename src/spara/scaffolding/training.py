"""Stage 1 (Scaffolding and Latent-Prediction Phases) and Stage 2 (Scaffolding RL)."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from ..data import records
from ..runtime import json_write, make_optimizer, restore_training, save_checkpoint
from ..training import amp
from . import serialization
from .decoding import LatentDecoder
from .model import answer_sequence, predictor_sequence, suppression, train_language
from .policy import clipped_fraction, clipped_surrogate, gaussian_log_prob, group_advantages, masked_kl
from .runtime import recipe


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device) if isinstance(value, torch.Tensor) else value


def answer_logits(model, hidden, positions):
    return model.lm_head(hidden[0, positions - 1]).float()


def answer_positions(ids, bot, k):
    return torch.arange(bot + k + 2, ids.shape[1], device=ids.device)


def scaffold_loss(vlm, encoder, sample):
    """Eq. 1: answer cross-entropy of the frozen VLM reading z* = g_psi(h)."""
    target = encoder(**sample["helper"])
    prompt = sample["prompt"]
    ids, bot = answer_sequence(prompt["input_ids"], sample["answer"], vlm.config)
    hidden = vlm.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], target[None])
    positions = answer_positions(ids, bot, vlm.config.scaffold_k)
    loss = F.cross_entropy(answer_logits(vlm, hidden, positions), ids[0, positions])
    return loss, {"loss": loss.item(), "target_norm": target.detach().float().norm(dim=-1).mean().item()}


def latent_matching(model, sample, target):
    """Eq. 2, first term: the predictor pass regresses z (final-norm hidden states at
    the K query positions) onto the frozen target z*."""
    prompt, k = sample["prompt"], model.config.scaffold_k
    queries = predictor_sequence(prompt["input_ids"], model.config)
    z = model.hidden_states(queries, prompt["pixel_values"], prompt["image_grid_thw"],
                            query_start=queries.shape[1] - k)[0, -k:].float()
    target = target.float()
    loss = (z - target).square().sum(-1).mean()
    with torch.no_grad():
        stats = {"latent_loss": loss.item(),
                 # Below 1 means z explains part of z*; z = 0 would score exactly 1.
                 "latent_loss_normalized": (loss / target.square().sum(-1).mean()).item(),
                 "latent_cosine": F.cosine_similarity(z, target, dim=-1).mean().item()}
    return loss, stats


def answer_loss(model, sample, target):
    """Eq. 2, second term: the answer pass reads the teacher-forced z*; BOT is supervised."""
    prompt, k = sample["prompt"], model.config.scaffold_k
    ids, bot = answer_sequence(prompt["input_ids"], sample["answer"], model.config)
    hidden = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], target[None])
    positions = torch.cat([torch.tensor([bot], device=ids.device), answer_positions(ids, bot, k)])
    loss = F.cross_entropy(answer_logits(model, hidden, positions), ids[0, positions])
    return loss, {"ce_loss": loss.item()}


@torch.no_grad()
def scaffold_targets(encoder, processor, data_root, rows, config, device):
    """z* for every training row, from the frozen scaffolding encoder."""
    encoder.eval()
    targets = []
    for row in rows:
        helper = to_device(serialization.helper(processor, data_root, row, config), device)
        with amp(device):
            targets.append(encoder(**helper).float().cpu())
    return torch.stack(targets)


def schedule(optimizer, total, config):
    return get_cosine_schedule_with_warmup(optimizer, math.ceil(total * config["warmup_ratio"]), total)


def begin(output, phase, config, optimizer, scheduler, resume, lineage):
    output.mkdir(parents=True, exist_ok=False)
    progress = {"phase": phase, "epoch": 0, "next_row": 0, "rl_steps": 0, "optimizer_steps": 0,
                "completed": False, "config": config, "source": str(resume) if resume else None, **lineage}
    if resume:
        progress = restore_training(resume, optimizer, scheduler)
        if progress["phase"] != phase or progress["completed"] or recipe(progress["config"], phase) != recipe(config, phase):
            raise ValueError("Resume must continue an unfinished checkpoint of the same phase and recipe")
    return progress


def check_finite(*losses):
    if not all(torch.isfinite(loss) for loss in losses):
        raise FloatingPointError("Nonfinite loss")


def supervised_epochs(rows, phase, config, output, microstep, optimizer, scheduler, progress, parameters, save):
    accumulation = phase["gradient_accumulation"]
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(progress["epoch"], phase["epochs"]):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
            start = progress["next_row"] if epoch == progress["epoch"] else 0
            for offset in range(start, len(rows), accumulation):
                selected = order[offset:offset + accumulation]
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                stats = [microstep(index, len(selected)) for index in selected]
                norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress.update(epoch=epoch, next_row=offset + len(selected),
                                optimizer_steps=progress["optimizer_steps"] + 1)
                row = {"phase": progress["phase"], "epoch": epoch, "step": progress["optimizer_steps"],
                       **{key: sum(s[key] for s in stats) / len(stats) for key in stats[0]},
                       "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                       "seconds": time.perf_counter() - started}
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                every = config.get("checkpoint_every_steps", 0)
                if every and progress["optimizer_steps"] % every == 0:
                    save(output / f"step-{progress['optimizer_steps']}")
            progress.update(epoch=epoch + 1, next_row=0)
    progress["completed"] = True
    save(output / "final")
    json_write(output / "summary.json", progress)
    return output / "final"


def train_scaffold(vlm, encoder, processor, data_root, output, config, resume=None):
    """Scaffolding Phase: update only psi through the frozen base VLM's answer loss."""
    phase, rows, output = config["scaffold"], records(data_root, "train"), Path(output)
    vlm.requires_grad_(False)
    vlm.train(); vlm.visual.eval()
    vlm.config.use_cache = False
    if config["gradient_checkpointing"]:
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder.train()
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = make_optimizer(encoder, config | {"learning_rate": phase["learning_rate"]})
    scheduler = schedule(optimizer, phase["epochs"] * math.ceil(len(rows) / phase["gradient_accumulation"]), config)
    progress = begin(output, "scaffold", config, optimizer, scheduler, resume, {})

    def microstep(index, count):
        sample = to_device(serialization.training_sample(processor, data_root, rows[index], config, True), vlm.device)
        with amp(vlm.device):
            loss, stats = scaffold_loss(vlm, encoder, sample)
        check_finite(loss)
        (loss / count).backward()
        return stats

    save = lambda path: save_checkpoint(path, encoder, processor, optimizer, scheduler, progress)
    return supervised_epochs(rows, phase, config, output, microstep, optimizer, scheduler, progress, parameters, save)


def train_predict(model, processor, targets, data_root, output, config, resume=None, encoder_path=None):
    """Latent-Prediction Phase: fine-tune the language model (vision tower frozen)."""
    phase, rows, output = config["predict"], records(data_root, "train"), Path(output)
    if targets.shape[0] != len(rows):
        raise ValueError("Expected one scaffold target per training row")
    train_language(model)
    model.train(); model.visual.eval()
    model.config.use_cache = False
    model.config.scaffold_phase = "predict"
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, config | {"learning_rate": phase["learning_rate"]})
    scheduler = schedule(optimizer, phase["epochs"] * math.ceil(len(rows) / phase["gradient_accumulation"]), config)
    progress = begin(output, "predict", config, optimizer, scheduler, resume, {"encoder": str(encoder_path)})

    def microstep(index, count):
        sample = to_device(serialization.training_sample(processor, data_root, rows[index], config), model.device)
        target = targets[index].to(model.device)
        # Backward each pass separately so only one graph is alive at a time.
        with amp(model.device):
            latent, latent_stats = latent_matching(model, sample, target)
        check_finite(latent)
        (phase["lambda_latent"] * latent / count).backward()
        with amp(model.device):
            ce, ce_stats = answer_loss(model, sample, target)
        check_finite(ce)
        (phase["lambda_task"] * ce / count).backward()
        return latent_stats | ce_stats

    save = lambda path: save_checkpoint(path, model, processor, optimizer, scheduler, progress)
    return supervised_epochs(rows, phase, config, output, microstep, optimizer, scheduler, progress, parameters, save)


def prompt_indices(step, count, total, seed):
    """Consecutive prompts from a fresh shuffle of the training rows each epoch."""
    orders, indices = {}, []
    for position in range(step * count, (step + 1) * count):
        epoch, offset = divmod(position, total)
        if epoch not in orders:
            orders[epoch] = torch.randperm(total, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        indices.append(orders[epoch][offset])
    return indices


@torch.no_grad()
def collect_group(decoder, processor, data_root, row, evaluator, rl):
    """G rollouts for one prompt, rewarded by reference-evaluator route success."""
    model = decoder.model
    prompt = to_device(serialization.prompt(processor, data_root, "train", row), model.device)
    with amp(model.device):
        rollouts = decoder.sample_group(prompt, rl["group_size"], rl["rollout"]["max_new_tokens"])
    rewards = []
    for rollout in rollouts:
        text = processor.tokenizer.decode(prompt["input_ids"][0].tolist() + rollout.ids, skip_special_tokens=False)
        rewards.append(float(bool(evaluator.score(row, text)["success"])))
    return {"prompt": prompt, "rollouts": rollouts, "rewards": rewards, "advantages": group_advantages(rewards)}


def grad_norm(parameters):
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def rl_update(model, reference, groups, rl, old):
    """Accumulate gradients of eq. 6 over every rollout in the step. `old` caches
    log-probabilities from the first pass over these rollouts (theta_old)."""
    c, device, k = model.config, model.device, model.config.scaffold_k
    count = sum(len(group["rollouts"]) for group in groups)
    temperature = rl["rollout"].get("temperature", 1.0)
    epsilon = rl["clip_epsilon"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    stats = defaultdict(list)
    if rl["variant"] == "scaffolding":
        for g, group in enumerate(groups):
            prompt, by_prefix = group["prompt"], defaultdict(list)
            for i, rollout in enumerate(group["rollouts"]):
                if rollout.block is not None:
                    by_prefix[tuple(rollout.ids[:rollout.block])].append(i)
            # Rollouts with the same text before BOT share one predictor pass.
            for prefix, members in by_prefix.items():
                prefix_ids = torch.tensor([list(prefix)], dtype=torch.long, device=device)
                queries = predictor_sequence(torch.cat([prompt["input_ids"], prefix_ids], dim=1), c)
                with amp(device):
                    prior = model.hidden_states(queries, prompt["pixel_values"], prompt["image_grid_thw"],
                                                query_start=queries.shape[1] - k)[0, -k:]
                mean, sigma = model.latent_heads(prior)
                loss = 0.0
                for i in members:
                    rollout = group["rollouts"][i]
                    log_prob = gaussian_log_prob(rollout.latent.to(device), prior, mean, sigma,
                                                 rl["latent_logprob_reduction"])
                    log_ratio = log_prob - old.setdefault(("latent", g, i), log_prob.detach())
                    loss = loss - clipped_surrogate(log_ratio, group["advantages"][i].to(device), epsilon)
                    stats["latent_abs_log_ratio"].append(log_ratio.abs().item())
                    stats["latent_clipped"].append(clipped_fraction(log_ratio, epsilon).item())
                (loss / count).backward()
                stats["sigma_mean"].append(sigma.mean().item())
                stats["mean_abs_mu"].append(mean.abs().mean().item())
        stats["latent_grad_norm"].append(grad_norm(parameters))
    for g, group in enumerate(groups):
        prompt = group["prompt"]
        length = prompt["input_ids"].shape[1]
        for i, rollout in enumerate(group["rollouts"]):
            ids = torch.cat([prompt["input_ids"], torch.tensor([rollout.ids], device=device)], dim=1)
            positions = length + torch.tensor(rollout.sampled, device=device)
            after = (positions > length + rollout.block) if rollout.block is not None else False
            latents = None if rollout.block is None else rollout.latent[None].to(device)
            with amp(device):
                hidden = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], latents)
            log_probs = suppression(answer_logits(model, hidden, positions) / temperature, c, after).log_softmax(-1)
            with torch.no_grad(), amp(device):
                reference_hidden = reference.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], latents)
                reference_log_probs = suppression(answer_logits(reference, reference_hidden, positions) / temperature,
                                                  c, after).log_softmax(-1)
            token = log_probs.gather(-1, ids[0, positions][:, None]).squeeze(1)
            log_ratio = token - old.setdefault(("text", g, i), token.detach())
            surrogate = clipped_surrogate(log_ratio, group["advantages"][i].to(device), epsilon).mean()
            kl = masked_kl(log_probs, reference_log_probs).mean()
            check_finite(surrogate, kl)
            ((rl["kl_beta"] * kl - surrogate) / count).backward()
            stats["kl"].append(kl.item())
            stats["text_clipped"].append(clipped_fraction(log_ratio, epsilon).item())
            stats["text_abs_log_ratio"].append(log_ratio.abs().mean().item())
    return {key: sum(values) / len(values) for key, values in stats.items()}


def rollout_stats(groups):
    rewards = [reward for group in groups for reward in group["rewards"]]
    rollouts = [rollout for group in groups for rollout in group["rollouts"]]
    sigmas = [rollout.sigma for rollout in rollouts if rollout.sigma is not None]
    stats = {"reward_mean": sum(rewards) / len(rewards),
             "reward_std_within_group": sum(torch.tensor(g["rewards"]).std().item() for g in groups) / len(groups),
             "pass_at_group": sum(any(g["rewards"]) for g in groups) / len(groups),
             "block_rate": sum(r.block is not None for r in rollouts) / len(rollouts),
             "response_tokens": sum(len(r.sampled) for r in rollouts) / len(rollouts)}
    if sigmas:
        stats |= {"sigma_min": min(float(s.min()) for s in sigmas), "sigma_max": max(float(s.max()) for s in sigmas)}
    return stats


def train_rl(model, reference, processor, data_root, output, config, evaluator, resume=None, source=None):
    """Scaffolding RL: GRPO over sampled residual latent actions and text tokens."""
    rl, rows, output = config["rl"], records(data_root, "train"), Path(output)
    train_language(model)
    reference.requires_grad_(False)
    reference.eval()
    model.config.use_cache = False
    model.config.scaffold_phase = "rl"
    model.config.scaffold_rl_variant = rl["variant"]
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decoder = LatentDecoder(model, rl["rollout"], "sample" if rl["variant"] == "scaffolding" else "none")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, config | {"learning_rate": rl["learning_rate"]})
    scheduler = schedule(optimizer, rl["steps"] * rl["ppo_epochs"], config)
    progress = begin(output, "rl", config, optimizer, scheduler, resume, {"reference": str(source)})
    save = lambda path: save_checkpoint(path, model, processor, optimizer, scheduler, progress)
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
        for step in range(progress["rl_steps"], rl["steps"]):
            started = time.perf_counter()
            model.eval()
            groups = [collect_group(decoder, processor, data_root, rows[index], evaluator, rl)
                      for index in prompt_indices(step, rl["prompts_per_step"], len(rows), config["seed"])]
            rollout_seconds = time.perf_counter() - started
            model.train(); model.visual.eval()
            old = {}
            for _ in range(rl["ppo_epochs"]):
                optimizer.zero_grad(set_to_none=True)
                update = rl_update(model, reference, groups, rl, old)
                norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress["optimizer_steps"] += 1
            progress["rl_steps"] = step + 1
            row = {"phase": "rl", "step": step + 1, **rollout_stats(groups), **update, "grad_norm": float(norm),
                   "lr": scheduler.get_last_lr()[0], "rollout_seconds": rollout_seconds,
                   "seconds": time.perf_counter() - started}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            every = config.get("checkpoint_every_steps", 0)
            if every and progress["rl_steps"] % every == 0:
                save(output / f"step-{progress['rl_steps']}")
    progress["completed"] = True
    save(output / "final")
    json_write(output / "summary.json", progress)
    return output / "final"
