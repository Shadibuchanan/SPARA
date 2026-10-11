"""Bounded actual-model correctness and resource measurements, not a benchmark."""
import gc
import json
import resource
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from .data import records
from .runtime import json_write, load_model, make_optimizer, manifest, save_checkpoint, restore_training
from .serialization import encode
from .training import amp, to_device
from .validation import cache_agreement


def run_pilot(config, data_root, output, continue_from=None):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pilot requires an allocated CUDA GPU with BF16 support")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = Path(continue_from) if continue_from else None
    if source:
        progress = json.loads((source / "roundtrip/complete.json").read_text())["progress"]
        if not progress.get("pilot") or progress["stage"] != 1 or progress["config"] != config:
            raise ValueError("Pilot continuation needs the same recipe's completed Stage 1 roundtrip")
        if not (source / "stage-1-generation.json").is_file():
            raise ValueError("Stage 1 checks must be complete before continuing Stage 2")
    model, processor = load_model(config, checkpoint=source / "roundtrip" if source else None, device="cuda")
    run_manifest = manifest(config, model, processor, data_root)
    run_manifest["continuation_of"] = str(source) if source else None
    json_write(output / "manifest.json", run_manifest)
    rows = records(data_root, "train")
    # Measure actual expanded lengths, including original processor resizing.
    lengths, visual_lengths = [], []
    for row in rows:
        batch = encode(processor, data_root, "train", row)
        lengths.append(batch["input_ids"].shape[-1])
        visual_lengths.append(int(batch["input_ids"].eq(model.config.image_token_id).sum()))
    ordered = sorted(range(len(rows)), key=lambda i: lengths[i])
    typical, longest = ordered[len(ordered) // 2], ordered[-1]
    json_write(output / "input-lengths.json", {"tokens": lengths, "visual_tokens": visual_lengths,
                                               "typical_row": typical, "longest_row": longest})
    fixture = encode(processor, data_root, "train", rows[typical])
    json_write(output / "token-fixture.json", {"row": typical, "ids": fixture["input_ids"][0].tolist(),
        "labels": fixture["labels"][0].tolist(), "tokens": processor.tokenizer.convert_ids_to_tokens(fixture["input_ids"][0].tolist()),
        "image_grid_thw": fixture["image_grid_thw"].tolist()})
    measurements = [x for x in json.loads((source / "measurements.json").read_text()) if x["stage"] == 1] if source else []
    visual_before = next(model.visual.parameters()).detach().clone()
    language_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    latent_before = model.get_input_embeddings().weight[model.config.livr_latent_ids].detach().clone()
    tiny_losses = [x["loss"] for x in measurements if x["kind"] == "tiny-fit"]
    checkpoint = (source if source else output) / "roundtrip"
    if source:
        model.eval()
        prompt = to_device(encode(processor, data_root, "train", rows[typical], training=False), model.device)
        json_write(output / "stage-1-recheck.json", cache_agreement(model, prompt))
        del prompt
    for stage in ((2,) if source else (1, 2)):
        model.config.livr_stage = stage
        model.train(); model.visual.eval()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = make_optimizer(model, config)
        assert not optimizer.state  # new state at each stage, not a resumed Stage 1 optimizer
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
        # Stage 1 also overfits one train record as a diagnostic. No test examples.
        jobs = [("typical", typical), ("longest", longest)]
        if stage == 1:
            jobs += [("tiny-fit", typical)] * 10
        for kind, index in jobs:
            optimizer.zero_grad(set_to_none=True)
            batch = to_device(encode(processor, data_root, "train", rows[index]), model.device)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            tick = time.perf_counter(); losses = []
            for _ in range(config["gradient_accumulation"]):
                with amp(model.device):
                    loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite pilot loss")
                (loss / config["gradient_accumulation"]).backward()
                losses.append(loss.item())
            latent_grad = model.get_input_embeddings().weight.grad[model.config.livr_latent_ids].abs().sum(dim=1)
            assert latent_grad.gt(0).all(), "All four latent embeddings must receive gradients"
            assert all(p.grad is None for p in model.visual.parameters())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            measurement = {"stage": stage, "kind": kind, "row": index, "tokens": lengths[index],
                "loss": sum(losses) / len(losses), "accumulation": config["gradient_accumulation"],
                "seconds": time.perf_counter() - tick, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "host_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            measurements.append(measurement)
            if stage == 1 and kind == "tiny-fit": tiny_losses.append(measurement["loss"])
            print(json.dumps(measurement), flush=True)
            json_write(output / "measurements.json", measurements)
        if stage == 1:
            if not tiny_losses[-1] < tiny_losses[0] * 0.9:
                raise AssertionError(f"Tiny-subset learning check failed: {tiny_losses}")
            torch.testing.assert_close(visual_before, next(model.visual.parameters()), atol=0, rtol=0)
            assert not torch.equal(language_before, model.model.layers[0].self_attn.q_proj.weight)
            assert not torch.equal(latent_before, model.get_input_embeddings().weight[model.config.livr_latent_ids])
            model.eval(); model.gradient_checkpointing_disable()
            sample = to_device(encode(processor, data_root, "train", rows[typical], training=False), model.device)
            with torch.no_grad(), amp(model.device):
                expected = model(**sample, use_cache=False).logits[:, -1].float().cpu()
            save_checkpoint(checkpoint, model, processor, optimizer, scheduler,
                            {"stage": 1, "pilot": True, "config": config, "optimizer_steps": len(jobs)})
            optimizer.zero_grad(set_to_none=True)
            del model, optimizer, scheduler, batch, sample, loss
            gc.collect(); torch.cuda.empty_cache()
            model, processor = load_model(config, checkpoint=checkpoint, device="cuda")
            model.eval()
            sample = to_device(encode(processor, data_root, "train", rows[typical], training=False), model.device)
            with torch.no_grad(), amp(model.device):
                actual = model(**sample, use_cache=False).logits[:, -1].float().cpu()
            torch.testing.assert_close(expected, actual, atol=0.02, rtol=0.02)
            # Actual optimizer-state reload before intentionally discarding it for Stage 2.
            optimizer = make_optimizer(model, config)
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
            restored = restore_training(checkpoint, optimizer, scheduler)
            assert restored["optimizer_steps"] == len(jobs) and optimizer.state
            del optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
        model.eval(); model.gradient_checkpointing_disable()
        prompt = to_device(encode(processor, data_root, "train", rows[typical], training=False), model.device)
        agreement = cache_agreement(model, prompt)
        with torch.no_grad(), amp(model.device):
            generated = model.generate(**prompt, max_new_tokens=32, do_sample=False, use_cache=True)
        json_write(output / f"stage-{stage}-generation.json", {"train_row": typical,
            "diagnostic_only": True, "raw_response": processor.tokenizer.decode(generated[0], skip_special_tokens=False),
            "cache_agreement": agreement})
        del generated, prompt
        if stage == 2:
            del optimizer, scheduler
    typical_times = [x["seconds"] for x in measurements if x["kind"] == "typical"]
    json_write(output / "summary.json", {"passed": True, "pilot_only": True,
        "total_seconds": time.perf_counter() - start, "continuation_of": str(source) if source else None, "tiny_fit_losses": tiny_losses,
        "estimated_20_epoch_gpu_hours": sum(typical_times) * 125 * 10 / 3600,
        "estimate_caveat": "Two typical accumulated steps; excludes checkpoint/queue overhead and dataset length variation",
        "checkpoint": json.loads((checkpoint / "complete.json").read_text()),
        "peak_gpu_allocated_bytes": max(x["peak_gpu_allocated_bytes"] for x in measurements)})
