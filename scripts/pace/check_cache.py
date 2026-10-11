"""Reproduce cache numerical differences on an actual saved LIVR checkpoint."""
import argparse
import contextlib
import json
from pathlib import Path
import torch
from spara.data import records
from spara.runtime import load_model, read_config, seed_all, json_write
from spara.serialization import encode
from spara.training import amp, to_device

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare-precision", action="store_true")
args = parser.parse_args()
config = read_config("configs/livr.json")
seed_all(config["seed"])
model, processor = load_model(config, args.checkpoint, device="cuda")
model.eval()
prompt = to_device(encode(processor, args.root / "data/vsp", "train", records(args.root / "data/vsp", "train")[742], training=False), model.device)
results = []
for stage in ([1, 2] if args.compare_precision else [1]):
    model.config.livr_stage = stage
    for precision in (["bf16", "fp32"] if args.compare_precision else ["bf16"]):
        with torch.no_grad(), amp(model.device) if precision == "bf16" else contextlib.nullcontext():
            first = model(**prompt, use_cache=True)
            token = first.logits[:, -1].argmax(dim=-1, keepdim=True)
            full = torch.cat([prompt["input_ids"], token], dim=-1)
            cached = model(input_ids=token, attention_mask=torch.ones_like(full), livr_full_input_ids=full,
                           cache_position=torch.tensor([prompt["input_ids"].shape[-1]], device=model.device),
                           past_key_values=first.past_key_values, use_cache=True).logits[:, -1].float()
            uncached = model(**dict(prompt, input_ids=full, attention_mask=torch.ones_like(full)), use_cache=False).logits[:, -1].float()
            difference = (cached - uncached).abs()
            p, q = cached.log_softmax(-1), uncached.log_softmax(-1)
            result = {"stage": stage, "precision": precision, "token": int(token),
                      "max_abs_error": difference.max().item(), "rms_error": difference.square().mean().sqrt().item(),
                      "mean_abs_error": difference.mean().item(),
                      "outside_original_tolerance": int((difference > 0.15 + 0.02 * uncached.abs()).sum()),
                      "cached_argmax": int(cached.argmax()), "uncached_argmax": int(uncached.argmax()),
                      "kl_cached_to_uncached": (p.exp() * (p - q)).sum().item()}
            results.append(result); print(json.dumps(result), flush=True)
        del first
json_write(args.output, results)
if not args.compare_precision:
    torch.testing.assert_close(cached, uncached, atol=0.15, rtol=0.02)
