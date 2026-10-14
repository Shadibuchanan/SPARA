"""Forensic probe: how many pinned Qwen2.5-VL-3B language weights can one AdamW step
move when parameters are stored in BF16 (Mirage's training setup)?

Downloads only byte ranges (the leading rows of sampled tensors) at the pinned
revision; it loads no model and changes no project artifact. An Adam step moves
each weight by about the learning rate, so the probe applies `w - lr * sign(g)` and
counts weights whose BF16 value is unchanged after round-to-nearest.
"""
import argparse
import json
import re
import struct
import urllib.request
from collections import defaultdict

import torch

REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
BASE = f"https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/{REVISION}/"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
PARTS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
         "post_attention_layernorm.weight", "self_attn.q_proj.bias")


def read_range(shard, start, end):
    request = urllib.request.Request(BASE + shard, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 9, 18, 27, 35])
    parser.add_argument("--max-bytes", type=int, default=4 << 20, help="Leading bytes sampled per tensor")
    parser.add_argument("--rates", type=float, nargs="+", default=[1e-5, 1e-6])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    headers = {}
    for shard in SHARDS:
        length = struct.unpack("<Q", read_range(shard, 0, 8))[0]
        headers[shard] = (json.loads(read_range(shard, 8, 8 + length)), 8 + length)
    kind = lambda name: re.sub(r"layers\.\d+\.", "layers.N.", name)
    totals = defaultdict(int)
    for header, _ in headers.values():
        for name, info in header.items():
            if name.startswith("model."):
                totals[kind(name)] += torch.Size(info["shape"]).numel()
    names = [f"model.layers.{layer}.{part}" for layer in args.layers for part in PARTS]
    names += ["model.embed_tokens.weight", "model.norm.weight"]  # embeddings are tied to the output head
    stuck, sampled = defaultdict(lambda: [0] * len(args.rates)), defaultdict(int)
    for name in names:
        shard = next(s for s, (h, _) in headers.items() if name in h)
        header, offset = headers[shard]
        if header[name]["dtype"] != "BF16":
            raise ValueError(f"Unexpected dtype for {name}")
        start, end = header[name]["data_offsets"]
        weights = torch.frombuffer(bytearray(read_range(shard, offset + start, offset + min(end, start + args.max_bytes))),
                                   dtype=torch.bfloat16)
        for i, rate in enumerate(args.rates):
            moved = (weights.float() - rate * torch.sign(torch.randn(weights.shape))).to(torch.bfloat16)
            stuck[kind(name)][i] += int((moved == weights).sum())
        sampled[kind(name)] += weights.numel()
    covered = sum(totals[k] for k in sampled)
    result = {"revision": REVISION, "layers": args.layers, "max_bytes_per_tensor": args.max_bytes,
              "covered_language_parameters": covered,
              "language_parameters": sum(v for k, v in totals.items() if not k.startswith("model.visual")),
              "per_kind": {k: {"parameters": totals[k], "sampled": sampled[k],
                               "unchanged": {str(r): stuck[k][i] / sampled[k] for i, r in enumerate(args.rates)}}
                           for k in sorted(sampled)},
              "parameter_weighted_unchanged": {str(r): sum(totals[k] * stuck[k][i] / sampled[k] for k in sampled) / covered
                                               for i, r in enumerate(args.rates)}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
