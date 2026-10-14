"""Diagnostic: sample each test map K times per checkpoint to estimate per-map success rates.

Not the benchmark protocol (one sample per map); it measures evaluation variance.
"""
import argparse
from pathlib import Path

import torch

from spara.benchmark import sample_test_maps
from spara.runtime import load_model, read_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/livr.json")
parser.add_argument("--data-root", required=True)
parser.add_argument("--checkpoint", action="append", required=True, help="name=path; repeatable")
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = read_config(args.config)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
for spec in args.checkpoint:
    name, path = spec.split("=", 1)
    model, processor = load_model(config, path, device="cuda")
    sample_test_maps(model, processor, config, args.data_root, args.samples, output / f"{name}.jsonl")
    del model
    torch.cuda.empty_cache()
