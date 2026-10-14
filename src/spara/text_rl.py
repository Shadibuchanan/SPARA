"""Text-only GRPO from the final LIVR checkpoint (condition B); latents are not perturbed."""
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
def rollout(model, prompt, rl, eos_ids):
    model.eval()
    with amp(prompt["input_ids"].device):
        generated = model.generate(**repeat_prompt(prompt, rl["group_size"]),
                                   **rl["generation"], use_cache=True)
    completions = generated[:, prompt["input_ids"].shape[-1]:]
    lengths = completion_mask(completions, eos_ids).sum(-1)
    return [completions[i, :lengths[i]] for i in range(len(completions))]


def steps_per_epoch(rows, rl):
    return math.ceil(rows / rl["prompts_per_step"])


def train_text_rl(model, reference, processor, data_root, output, config, rl,
                  source, resume=None, scorer=None, final_weights_only=False, max_steps=None):
    """GRPO on the original training records. `reference` is the frozen LIVR checkpoint."""
    if scorer is None:
        from .evaluation import ReferenceEvaluator
        evaluator = ReferenceEvaluator(data_root)
        scorer = lambda row, text: float(evaluator.score(row, text)["success"])
    if model.config.livr_stage != 2:
        raise ValueError("Text-only RL starts from final Stage 2 LIVR weights")
    rows = records(data_root, "train")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference.eval().requires_grad_(False)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = make_optimizer(model, config | {"learning_rate": rl["learning_rate"]})
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    temperature = rl["generation"].get("temperature", 1.0)
    progress = {"phase": PHASE, "stage": 2, "epoch": 0, "next_row": 0, "optimizer_steps": 0,
                "config": config, "rl": rl, "source_checkpoint": str(source)}
    # A step-limited pilot measures cost and signal; its checkpoint never counts as complete.
    pilot = max_steps is not None
    if resume:
        progress = restore_training(resume, optimizer, scheduler)
        if (progress.get("phase") != PHASE or progress["config"] != config or progress["rl"] != rl
                or progress["source_checkpoint"] != str(source)):
            raise ValueError("Resume must keep the same LIVR source and RL recipe")
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log, \
            (output / "rollouts.jsonl").open("w", encoding="utf-8") as rollout_log:
        for epoch in range(progress["epoch"], rl["epochs"]):
            if pilot and progress["optimizer_steps"] >= max_steps:
                break
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            order = torch.randperm(len(rows), generator=generator).tolist()
            start = progress["next_row"] if epoch == progress["epoch"] else 0
            for offset in range(start, len(rows), rl["prompts_per_step"]):
                if pilot and progress["optimizer_steps"] >= max_steps:
                    break
                selected = order[offset:offset + rl["prompts_per_step"]]
                started = time.perf_counter()
                groups = []
                for index in selected:
                    prompt = to_device(encode(processor, data_root, "train", rows[index], training=False), model.device)
                    completions = rollout(model, prompt, rl, eos_ids)
                    texts = [processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], c]), skip_special_tokens=False)
                             for c in completions]
                    rewards = torch.tensor([scorer(rows[index], text) for text in texts])
                    groups.append((index, prompt, completions, rewards, group_advantages(rewards)))
                    rollout_log.write(json.dumps({"step": progress["optimizer_steps"] + 1, "row": index,
                                                  "rewards": rewards.tolist(), "lengths": [len(c) for c in completions]}) + "\n")
                generation_seconds = time.perf_counter() - started
                model.train()
                model.visual.eval()
                optimizer.zero_grad(set_to_none=True)
                count = sum(len(g[2]) for g in groups)
                loss_sum = kl_sum = 0.0
                for index, prompt, completions, rewards, advantages in groups:
                    for completion, advantage in zip(completions, advantages.tolist()):
                        with torch.no_grad():
                            reference_logprobs = token_logprobs(reference, prompt, completion, temperature)
                        policy = token_logprobs(model, prompt, completion, temperature)
                        loss, kl = sequence_loss(policy, reference_logprobs, advantage, rl)
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
                row = {"phase": PHASE, "epoch": epoch, "step": progress["optimizer_steps"],
                       "reward": rewards.mean().item(), "zero_variance_groups": int((rewards.std(-1) == 0).sum()),
                       "groups": len(groups), "loss": loss_sum / count, "kl": kl_sum / count,
                       "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                       "mean_completion_tokens": sum(len(c) for g in groups for c in g[2]) / count,
                       "generation_seconds": generation_seconds, "seconds": time.perf_counter() - started,
                       "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
                log.write(json.dumps(row) + "\n"); log.flush(); rollout_log.flush()
                print(json.dumps(row), flush=True)
                every = rl.get("checkpoint_every_steps", 0)
                if every and progress["optimizer_steps"] % every == 0:
                    save_checkpoint(output / f"step-{progress['optimizer_steps']}", model, processor, optimizer, scheduler, progress)
            else:
                progress.update(epoch=epoch + 1, next_row=0)
    model.eval()
    if pilot:
        # Pilot weights are not an experiment result; record measurements only.
        json_write(output / "summary.json", progress | {"pilot": True})
        return None
    save_checkpoint(output / "final", model, processor, optimizer, scheduler, progress, not final_weights_only)
    json_write(output / "summary.json", progress)
    return output / "final"
