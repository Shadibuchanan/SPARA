"""Single-GPU, two-stage training with explicit stage/resume semantics."""
import contextlib
import math
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from .data import records
from .runtime import make_optimizer, save_checkpoint, restore_training, json_write
from .serialization import encode
from .model import LATENT_TOKENS


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def epoch_order(config, epoch, size):
    # order_cycle replays the per-stage permutations so a one-stage run sees
    # LIVR's step-by-step data order.
    cycle = config.get("order_cycle", config["epochs_per_stage"])
    generator = torch.Generator().manual_seed(config["seed"] + epoch % cycle)
    return torch.randperm(size, generator=generator).tolist()


def train_stage(model, processor, data_root, output, config, stage, resume=None, final_weights_only=False):
    rows = records(data_root, "train")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.config.livr_stage = stage
    model.train()
    model.visual.eval()
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = make_optimizer(model, config)
    accumulation = config["gradient_accumulation"]
    steps_per_epoch = math.ceil(len(rows) / accumulation)
    total = steps_per_epoch * config["epochs_per_stage"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total * config["warmup_ratio"]), total)
    progress = {"stage": stage, "epoch": 0, "next_row": 0, "optimizer_steps": 0,
                "config": config, "source_checkpoint": str(resume) if resume else None}
    if resume:
        progress = restore_training(resume, optimizer, scheduler)
        if progress["stage"] != stage or progress["config"] != config:
            raise ValueError("Resume must keep the same stage and recipe; stage transitions load weights only")
    latents = getattr(model.config, "livr_latent_tokens", LATENT_TOKENS)
    # LIVR keeps the default call; only runs without its latents override them.
    serialization = {} if latents == LATENT_TOKENS else {"latents": latents}
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
        import json
        for epoch in range(progress["epoch"], config["epochs_per_stage"]):
            order = epoch_order(config, epoch, len(rows))
            start = progress["next_row"] if epoch == progress["epoch"] else 0
            for offset in range(start, len(rows), accumulation):
                selected = order[offset:offset + accumulation]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                started = time.perf_counter()
                for index in selected:
                    batch = to_device(encode(processor, data_root, "train", rows[index], **serialization), model.device)
                    with amp(model.device):
                        loss = model(**batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at row {index}")
                    (loss / len(selected)).backward()
                    loss_sum += loss.item()
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress.update(epoch=epoch, next_row=offset + len(selected), optimizer_steps=progress["optimizer_steps"] + 1)
                row = {"stage": stage, "epoch": epoch, "step": progress["optimizer_steps"],
                       "loss": loss_sum / len(selected), "grad_norm": float(norm),
                       "lr": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - started}
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                every = config.get("checkpoint_every_steps", 0)
                if every and progress["optimizer_steps"] % every == 0:
                    save_checkpoint(output / f"step-{progress['optimizer_steps']}", model, processor, optimizer, scheduler, progress)
            progress.update(epoch=epoch + 1, next_row=0)
    save_checkpoint(output / "final", model, processor, optimizer, scheduler, progress, not final_weights_only)
    json_write(output / "summary.json", progress)
    return output / "final"
