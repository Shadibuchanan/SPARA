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


def train_stage(model, processor, data_root, output, config, stage, resume=None, final_weights_only=False,
                grounding=None, data=None, save_final=True):
    """`grounding` adds an auxiliary loss: spara.grounding.StepGrounding, or another object with
    the same methods (spara.perception_jepa.LatentJepa); without it this is LIVR's stage training
    unchanged. `data`, a list of (split, row) with `text_output`, replaces the released training
    split (docs/DETOUR_SCALING_SPEC.md). `save_final` False keeps the trained weights in memory
    only (the first phase of docs/EXPERT_ITERATION_ROUND2_SPEC.md)."""
    if data is not None and grounding is not None and getattr(grounding, "released_only", True):
        raise ValueError("Step grounding trains on the released training split only")
    if config.get("train_records"):
        import json as _json
        rows = [_json.loads(line) for line in Path(config["train_records"]).read_text(encoding="utf-8").splitlines()]
        data = [("train", row) for row in rows]
    else:
        data = data if data is not None else [("train", row) for row in records(data_root, "train")]
        rows = [row for _, row in data]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.config.livr_stage = stage
    model.train()
    model.visual.eval()
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = make_optimizer(model, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    head_params = []  # clipped on their own when the auxiliary objective asks, so they cannot shrink LM updates
    if grounding is not None:
        if resume:
            raise ValueError("Resuming a step-grounding run is not supported; its head has no saved optimizer state")
        grounding.head.to(model.device)
        grounding_counts = grounding.prepare(processor.tokenizer, rows)
        optimizer.add_param_group(grounding.parameter_group())  # before the scheduler records initial rates
        if getattr(grounding, "clip_separately", False):
            head_params = list(grounding.head.parameters())
        else:
            trainable += list(grounding.head.parameters())
    accumulation = config["gradient_accumulation"]
    steps_per_epoch = math.ceil(len(rows) / accumulation)
    total = steps_per_epoch * config["epochs_per_stage"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total * config["warmup_ratio"]), total)
    progress = {"stage": stage, "epoch": 0, "next_row": 0, "optimizer_steps": 0,
                "config": config, "source_checkpoint": str(resume) if resume else None}
    if grounding is not None:
        progress["grounding"] = grounding.settings | {"labels": grounding_counts}
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
                extra = {key: [] for key in getattr(grounding, "metric_keys",
                                                    ("grounding_loss", "cell_accuracy", "outcome_accuracy"))}
                started = time.perf_counter()
                for index in selected:
                    split, row = data[index]
                    batch = to_device(encode(processor, data_root, split, row, **serialization), model.device)
                    with amp(model.device):
                        if grounding is None:
                            loss = total_loss = model(**batch, use_cache=False).loss
                        else:
                            out = model(**batch, use_cache=False, output_hidden_states=True)
                            loss = out.loss
                            aux, metrics = grounding.loss(out.hidden_states, index, batch)
                            total_loss = loss + grounding.weight * aux
                            del out  # logits and all layers' states; the graph keeps what backward needs
                            extra[getattr(grounding, "loss_key", "grounding_loss")].append(aux.item())
                            for key, value in metrics.items():
                                if value is not None:
                                    extra[key].append(value)
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError(f"Nonfinite loss at row {index}")
                    (total_loss / len(selected)).backward()
                    loss_sum += loss.item()
                norm = torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"], error_if_nonfinite=True)
                head_norm = (torch.nn.utils.clip_grad_norm_(head_params, config["max_grad_norm"], error_if_nonfinite=True)
                             if head_params else None)
                optimizer.step()
                scheduler.step()
                progress.update(epoch=epoch, next_row=offset + len(selected), optimizer_steps=progress["optimizer_steps"] + 1)
                row = {"stage": stage, "epoch": epoch, "step": progress["optimizer_steps"],
                       "loss": loss_sum / len(selected), "grad_norm": float(norm),
                       "lr": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - started}
                if grounding is not None:
                    row |= {key: sum(v) / len(v) if v else None for key, v in extra.items()}
                if head_params:
                    row["head_grad_norm"] = float(head_norm)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                every = config.get("checkpoint_every_steps", 0)
                if every and progress["optimizer_steps"] % every == 0:
                    save_checkpoint(output / f"step-{progress['optimizer_steps']}", model, processor, optimizer, scheduler, progress)
            progress.update(epoch=epoch + 1, next_row=0)
    if not save_final:
        json_write(output / "summary.json", progress)
        return None
    save_checkpoint(output / "final", model, processor, optimizer, scheduler, progress, not final_weights_only)
    if grounding is not None:
        from .grounding import HEAD_FILE
        grounding.save(output / getattr(grounding, "head_file", HEAD_FILE))
    json_write(output / "summary.json", progress)
    return output / "final"
