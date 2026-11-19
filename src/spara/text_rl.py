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
from .runtime import make_optimizer, offloaded_optimizer, save_checkpoint, restore_training, json_write
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
def rollout(model, prompt, rl, eos_ids, adjuster=None, offsets=None):
    """G sampled completions and, for condition C, each row's latent action (G, K, d). With
    `offsets` (C″), each row's latent states get that fixed offset instead (no heads)."""
    model.eval()
    if offsets is not None:
        context = adjuster.offset(offsets)
    else:
        context = adjuster.sample() if adjuster else contextlib.nullcontext()
    with context as state, amp(prompt["input_ids"].device):
        generated = model.generate(**repeat_prompt(prompt, rl["group_size"]),
                                   **rl["generation"], use_cache=True)
    completions = generated[:, prompt["input_ids"].shape[-1]:]
    lengths = completion_mask(completions, eos_ids).sum(-1)
    actions = None
    if adjuster and offsets is None:
        if len(state["records"]) != 1:
            raise RuntimeError(f"Expected one adjusted prefill per group, saw {len(state['records'])}")
        actions = state["records"][0]["actions"]
    return [completions[i, :lengths[i]] for i in range(len(completions))], actions


def steps_per_epoch(rows, rl):
    return math.ceil(rows / rl["prompts_per_step"])


def stream_batch(rows, size, seed, position):
    """`size` row indices from `position` of a stream of seeded permutations, one per epoch
    (seed + epoch), and the next position. Every batch is full; epochs run into each other."""
    selected = []
    while len(selected) < size:
        epoch, offset = divmod(position, rows)
        order = torch.randperm(rows, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        taken = order[offset:offset + size - len(selected)]
        selected += taken
        position += len(taken)
    return selected, position


def train_text_rl(model, reference, processor, data_root, output, config, rl,
                  source, resume=None, scorer=None, final_weights_only=False, max_steps=None,
                  save_final=True, adjusters=None, noise=None):
    """GRPO on the training records (`rl["train_split"]`, default the released split).
    `reference` is the frozen LIVR checkpoint.

    Condition C: `rl["latent"]` holds the latent policy and `adjusters` the
    (policy, reference) LatentAdjusters; the model must carry `latent_heads`.

    B′ and C′ (docs/STRUCTURED_EXPLORATION_SPEC.md, 2026-11-17 amendment): `rl["steps"]`
    runs that many optimizer steps of exactly `prompts_per_step` maps from a stream of
    seeded permutations, and `rl["skip_uniform_groups"]` averages each step's loss over the
    rollouts of groups whose rewards differ, skipping the rest (their advantage is zero). A
    step with no such group updates nothing but still counts, so every arm of a seed sees
    the same maps.

    C″ (docs/LATENT_DIAGNOSTICS_SPEC.md, C″ amendment): `rl["noise"]` and `noise`, a
    norm-matched principal basis at the adjusters' layer. Each group's rollouts get fixed
    latent offsets drawn at `rl["noise"]["scale"]` from a generator seeded by the run's seed;
    the update replays each rollout's offset on the recomputed latent states (a = z + offset),
    for the policy and the reference. No heads and no latent term: at scale 0 this is B′.
    """
    latent, noisy = rl.get("latent"), rl.get("noise")
    if latent and noisy:
        raise ValueError("C′'s learned latent policy and C″'s fixed latent noise are exclusive")
    if bool(noisy) != (noise is not None):
        raise ValueError("Latent noise needs both rl['noise'] and its basis")
    if bool(latent or noisy) != bool(adjusters):
        raise ValueError("Latent RL and latent noise need both their settings and the latent adjusters")
    if latent:
        from .latent_rl import PHASE as phase, latent_objective, replaying
        if getattr(model, "latent_heads", None) is None:
            raise ValueError("Latent RL requires Gaussian heads on the policy")
        policy_adjuster, reference_adjuster = adjusters
    elif noisy:
        from .latent_rl import noise_offsets
        phase, (policy_adjuster, reference_adjuster) = PHASE, adjusters
        replaying = lambda adjuster, offsets: adjuster.offset(offsets)
        expected = (noisy["scale"] ** 2 * noise["weights"].double().square().sum(-1)).tolist()
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
    # offload_optimizer: AdamW's state on the CPU, for GPUs under the ~70 GiB that B and C peaked at.
    build = offloaded_optimizer if rl.get("offload_optimizer") else make_optimizer
    optimizer = build(model, config | {"learning_rate": rl["learning_rate"]},
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
    seed, size = rl.get("seed", config["seed"]), rl["prompts_per_step"]
    # C″'s noise has its own generator, so token sampling draws from the global stream as in B′.
    noise_generator = torch.Generator(device=model.device).manual_seed(seed) if noisy else None

    def batches():
        """(maps, position after them) per step: one seeded permutation per epoch, or with
        `steps`, a stream of permutations cut into full batches until that many steps."""
        if rl.get("steps"):
            position = progress["next_row"]
            while progress["optimizer_steps"] < rl["steps"]:
                selected, position = stream_batch(len(rows), size, seed, position)
                yield selected, {"epoch": position // len(rows), "next_row": position}
            return
        for epoch in range(progress["epoch"], rl["epochs"]):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            start = progress["next_row"] if epoch == progress["epoch"] else 0
            for offset in range(start, len(rows), size):
                selected = order[offset:offset + size]
                yield selected, {"epoch": epoch, "next_row": offset + len(selected)}
            progress.update(epoch=epoch + 1, next_row=0)

    skip_uniform = rl.get("skip_uniform_groups", False)
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log, \
            (output / "rollouts.jsonl").open("w", encoding="utf-8") as rollout_log:
        for selected, place in batches():
            if pilot and progress["optimizer_steps"] >= max_steps:
                break
            started = time.perf_counter()
            groups, noise_sizes = [], []
            for index in selected:
                prompt = to_device(encode(processor, data_root, split, rows[index], training=False), model.device)
                eps = None
                if noisy:
                    actions, eps = noise_offsets(noise, noisy["scale"], rl["group_size"], noise_generator, model.device)
                    completions, _ = rollout(model, prompt, rl, eos_ids, policy_adjuster, offsets=actions)
                    measured = actions.double().square().sum(-1).mean(0).tolist()
                    noise_sizes.append([m / max(e, 1e-30) for m, e in zip(measured, expected)])
                else:
                    completions, actions = rollout(model, prompt, rl, eos_ids, policy_adjuster)
                texts = [processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], c]), skip_special_tokens=False)
                         for c in completions]
                rewards = torch.tensor([scorer(rows[index], text) for text in texts])
                groups.append((index, prompt, completions, rewards, group_advantages(rewards), actions))
                rollout_log.write(json.dumps({
                    "step": progress["optimizer_steps"] + 1, "row": index, "rewards": rewards.tolist(),
                    "lengths": [len(c) for c in completions],
                    # The completions let the rewards be rescored on CPU (B′ and C′ validity).
                    "completions": [processor.tokenizer.decode(c, skip_special_tokens=False) for c in completions]}
                    # C″: each rollout's noise coordinates (G, K, k), for later readings.
                    | ({"eps": [[[round(float(x), 3) for x in slot] for slot in row] for row in eps.cpu()]}
                       if eps is not None else {})) + "\n")
            generation_seconds = time.perf_counter() - started
            model.train()
            model.visual.eval()
            optimizer.zero_grad(set_to_none=True)
            # Equal rewards give every rollout zero advantage: no policy gradient, only KL.
            trained = [g for g in groups if g[3].std() > 0] if skip_uniform else groups
            count = sum(len(g[2]) for g in trained)
            loss_sum = kl_sum = 0.0
            latent_stats = []
            for index, prompt, completions, rewards, advantages, actions in trained:
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
            norm = 0.0
            if count:
                norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                            config["max_grad_norm"], error_if_nonfinite=True))
                optimizer.step()
            scheduler.step()
            progress.update(**place, optimizer_steps=progress["optimizer_steps"] + 1)
            rollouts = sum(len(g[2]) for g in groups)
            progress["verifier_calls"] = progress.get("verifier_calls", 0) + rollouts
            rewards = torch.stack([g[3] for g in groups])
            row = {"phase": phase, "epoch": place["epoch"], "step": progress["optimizer_steps"],
                   "reward": rewards.mean().item(), "zero_variance_groups": int((rewards.std(-1) == 0).sum()),
                   "groups": len(groups), "trained_groups": len(trained), "updated": bool(count),
                   "loss": loss_sum / count if count else None, "kl": kl_sum / count if count else None,
                   "grad_norm": norm, "lr": scheduler.get_last_lr()[0],
                   "mean_completion_tokens": sum(len(c) for g in groups for c in g[2]) / rollouts,
                   "verifier_calls": progress["verifier_calls"],
                   "generation_seconds": generation_seconds, "seconds": time.perf_counter() - started,
                   "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
            if latent:
                if latent_stats:
                    row |= {key: sum(s[key] for s in latent_stats) / len(latent_stats) for key in latent_stats[0]}
                row["head_lr"] = scheduler.get_last_lr()[-1]
            if noisy:  # measured / expected squared size of the noise per slot, over the step's groups
                row["noise_size_ratio"] = [round(sum(s) / len(s), 4) for s in zip(*noise_sizes)]
            log.write(json.dumps(row) + "\n"); log.flush(); rollout_log.flush()
            print(json.dumps(row), flush=True)
            every = rl.get("checkpoint_every_steps", 0)
            if every and progress["optimizer_steps"] % every == 0:
                save_checkpoint(output / f"step-{progress['optimizer_steps']}", model, processor, optimizer, scheduler, progress)
    model.eval()
    if pilot or not save_final:
        # Pilot weights are not an experiment result; unsaved runs are evaluated by the caller.
        json_write(output / "summary.json", progress | ({"pilot": True} if pilot else {}))
        return None
    save_checkpoint(output / "final", model, processor, optimizer, scheduler, progress, not final_weights_only)
    json_write(output / "summary.json", progress)
    return output / "final"
