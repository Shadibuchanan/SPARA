"""GRPO from the final LIVR checkpoint. Condition B (text only) leaves the latents
unperturbed; with an `rl["latent"]` section the same loop runs condition C, adding
Gaussian adjustments of the latent states (spara.latent_rl).
"""
import contextlib
import json
import math
import time
from pathlib import Path

import torch

from .data import records
from .runtime import make_optimizer, save_checkpoint, restore_training, json_write
from .serialization import encode
from .training import amp, to_device

PHASE = "text-rl"


def repeat_prompt(batch, copies):
    # Identical prompts need no padding; Qwen concatenates image patches across rows.
    return {key: value.repeat(copies, 1) for key, value in batch.items()}


def completion_mask(completions, eos_ids):
    # Keep tokens through the first end token; generate pads finished rows after it.
    ends = torch.isin(completions, torch.tensor(eos_ids, device=completions.device))
    return (ends.cumsum(-1) - ends.long()).eq(0)


def group_advantages(rewards):
    # Group-relative advantage; identical rewards give zero advantage (no signal).
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def token_logprobs(model, prompt, completion, temperature):
    """Per-token log-probabilities of one completion under the rollout temperature."""
    ids = torch.cat([prompt["input_ids"], completion[None]], dim=-1)
    inputs = dict(prompt, input_ids=ids, attention_mask=torch.ones_like(ids))
    with amp(ids.device):
        logits = model(**inputs, use_cache=False).logits[0, prompt["input_ids"].shape[-1] - 1:-1]
    logprobs = (logits.float() / temperature).log_softmax(-1)
    return logprobs.gather(-1, completion[:, None]).squeeze(-1)


def sequence_loss(policy, reference, advantage, rl):
    """Eq. 6 text term of Scaffolding Minds: clipped ratio and k3 KL, mean over tokens.

    One policy update per rollout batch, so the old policy equals the current
    detached policy: the ratio is one in value and the clip is inactive.
    """
    ratio = (policy - policy.detach()).exp()
    clipped = ratio.clamp(1 - rl["clip_epsilon"], 1 + rl["clip_epsilon"])
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    difference = reference - policy
    kl = difference.exp() - difference - 1
    return -(surrogate - rl["kl_coefficient"] * kl).mean(), kl.detach().mean()


@torch.no_grad()
def rollout(model, prompt, rl, eos_ids, adjuster=None):
    """G sampled completions and, for condition C, each row's latent action (G, K, d)."""
    model.eval()
    with (adjuster.sample() if adjuster else contextlib.nullcontext()) as state, amp(prompt["input_ids"].device):
        generated = model.generate(**repeat_prompt(prompt, rl["group_size"]),
                                   **rl["generation"], use_cache=True)
    completions = generated[:, prompt["input_ids"].shape[-1]:]
    lengths = completion_mask(completions, eos_ids).sum(-1)
    actions = None
    if adjuster:
        if len(state["records"]) != 1:
            raise RuntimeError(f"Expected one adjusted prefill per group, saw {len(state['records'])}")
        actions = state["records"][0]["actions"]
    return [completions[i, :lengths[i]] for i in range(len(completions))], actions


def steps_per_epoch(rows, rl):
    return math.ceil(rows / rl["prompts_per_step"])


def train_text_rl(model, reference, processor, data_root, output, config, rl,
                  source, resume=None, scorer=None, final_weights_only=False, max_steps=None,
                  save_final=True, adjusters=None):
    """GRPO on the training records (`rl["train_split"]`, default the released split).
    `reference` is the frozen LIVR checkpoint.

    Condition C: `rl["latent"]` holds the latent policy and `adjusters` the
    (policy, reference) LatentAdjusters; the model must carry `latent_heads`.
    """
    latent = rl.get("latent")
    if bool(latent) != bool(adjusters):
        raise ValueError("Latent RL needs both rl['latent'] and the latent adjusters")
    if latent:
        from .latent_rl import PHASE as phase, latent_objective, replaying
        if getattr(model, "latent_heads", None) is None:
            raise ValueError("Latent RL requires Gaussian heads on the policy")
        policy_adjuster, reference_adjuster = adjusters
    else:
        phase, policy_adjuster, reference_adjuster = PHASE, None, None
        replaying = lambda adjuster, actions: contextlib.nullcontext({})
    if scorer is None:
        from .evaluation import ReferenceEvaluator
        evaluator = ReferenceEvaluator(data_root)
        scorer = lambda row, text: float(evaluator.score(row, text)["success"])
    if model.config.livr_stage != 2:
        raise ValueError("Text-only RL starts from final Stage 2 LIVR weights")
    # B and C train on the released training split; B' and C' on a fresh/<name> set.
    split = rl.get("train_split", "train")
    rows = records(data_root, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference.eval().requires_grad_(False)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = make_optimizer(model, config | {"learning_rate": rl["learning_rate"]},
                               {"latent_heads.": latent["head_learning_rate"]} if latent else None)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    temperature = rl["generation"].get("temperature", 1.0)
    progress = {"phase": phase, "stage": 2, "epoch": 0, "next_row": 0, "optimizer_steps": 0,
                "config": config, "rl": rl, "source_checkpoint": str(source)}
    # A step-limited pilot measures cost and signal; its checkpoint never counts as complete.
    pilot = max_steps is not None
    if resume:
        progress = restore_training(resume, optimizer, scheduler)
        if (progress.get("phase") != phase or progress["config"] != config or progress["rl"] != rl
                or progress["source_checkpoint"] != str(source)):
            raise ValueError("Resume must keep the same LIVR source and RL recipe")
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log, \
            (output / "rollouts.jsonl").open("w", encoding="utf-8") as rollout_log:
        for epoch in range(progress["epoch"], rl["epochs"]):
            if pilot and progress["optimizer_steps"] >= max_steps:
                break
            generator = torch.Generator().manual_seed(rl.get("seed", config["seed"]) + epoch)
            order = torch.randperm(len(rows), generator=generator).tolist()
            start = progress["next_row"] if epoch == progress["epoch"] else 0
            for offset in range(start, len(rows), rl["prompts_per_step"]):
                if pilot and progress["optimizer_steps"] >= max_steps:
                    break
                selected = order[offset:offset + rl["prompts_per_step"]]
                started = time.perf_counter()
                groups = []
                for index in selected:
                    prompt = to_device(encode(processor, data_root, split, rows[index], training=False), model.device)
                    completions, actions = rollout(model, prompt, rl, eos_ids, policy_adjuster)
                    texts = [processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], c]), skip_special_tokens=False)
                             for c in completions]
                    rewards = torch.tensor([scorer(rows[index], text) for text in texts])
                    groups.append((index, prompt, completions, rewards, group_advantages(rewards), actions))
                    rollout_log.write(json.dumps({"step": progress["optimizer_steps"] + 1, "row": index,
                                                  "rewards": rewards.tolist(), "lengths": [len(c) for c in completions]}) + "\n")
                generation_seconds = time.perf_counter() - started
                model.train()
                model.visual.eval()
                optimizer.zero_grad(set_to_none=True)
                count = sum(len(g[2]) for g in groups)
                loss_sum = kl_sum = 0.0
                latent_stats = []
                for index, prompt, completions, rewards, advantages, actions in groups:
                    for i, (completion, advantage) in enumerate(zip(completions, advantages.tolist())):
                        action = None if actions is None else actions[i:i + 1]
                        # The reference reads the same latent action, so KL covers the text only.
                        with torch.no_grad(), replaying(reference_adjuster, action):
                            reference_logprobs = token_logprobs(reference, prompt, completion, temperature)
                        # Keep the replay open through backward: checkpointing re-runs the hook.
                        with replaying(policy_adjuster, action) as state:
                            policy = token_logprobs(model, prompt, completion, temperature)
                            loss, kl = sequence_loss(policy, reference_logprobs, advantage, rl)
                            if latent:
                                latent_loss, stats = latent_objective(model, state["prior"], action, advantage,
                                                                      latent, rl["clip_epsilon"])
                                loss = loss + latent_loss
                                latent_stats.append(stats)
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f"Nonfinite RL loss at row {index}")
                            (loss / count).backward()
                        loss_sum += loss.item(); kl_sum += kl.item()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress.update(epoch=epoch, next_row=offset + len(selected), optimizer_steps=progress["optimizer_steps"] + 1)
                rewards = torch.stack([g[3] for g in groups])
                row = {"phase": phase, "epoch": epoch, "step": progress["optimizer_steps"],
                       "reward": rewards.mean().item(), "zero_variance_groups": int((rewards.std(-1) == 0).sum()),
                       "groups": len(groups), "loss": loss_sum / count, "kl": kl_sum / count,
                       "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                       "mean_completion_tokens": sum(len(c) for g in groups for c in g[2]) / count,
                       "generation_seconds": generation_seconds, "seconds": time.perf_counter() - started,
                       "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
                if latent:
                    row |= {key: sum(s[key] for s in latent_stats) / len(latent_stats) for key in latent_stats[0]}
                    row["head_lr"] = scheduler.get_last_lr()[-1]
                log.write(json.dumps(row) + "\n"); log.flush(); rollout_log.flush()
                print(json.dumps(row), flush=True)
                every = rl.get("checkpoint_every_steps", 0)
                if every and progress["optimizer_steps"] % every == 0:
                    save_checkpoint(output / f"step-{progress['optimizer_steps']}", model, processor, optimizer, scheduler, progress)
            else:
                progress.update(epoch=epoch + 1, next_row=0)
    model.eval()
    if pilot or not save_final:
        # Pilot weights are not an experiment result; unsaved runs are evaluated by the caller.
        json_write(output / "summary.json", progress | ({"pilot": True} if pilot else {}))
        return None
    save_checkpoint(output / "final", model, processor, optimizer, scheduler, progress, not final_weights_only)
    json_write(output / "summary.json", progress)
    return output / "final"
