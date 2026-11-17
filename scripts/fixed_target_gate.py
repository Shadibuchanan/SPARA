"""Gate 3 of the fixed-target retry (docs/PERCEPTION_JEPA_SPEC.md, "Retry with fixed targets"), on CPU.

Reads the features job's saved latents and cells (`features.pt`); no model and no GPU. Writes
summary.json and the heads file, with every map's fixed targets, for the EJF arm.

Usage: python scripts/fixed_target_gate.py FEATURES_RUN OUTPUT [--settings configs/perception_jepa_fixed.json]
"""
import argparse
import json
from pathlib import Path

import torch

from spara.perception_jepa import run_fixed_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_run")
    parser.add_argument("output")
    parser.add_argument("--settings", default="configs/perception_jepa_fixed.json")
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    torch.manual_seed(settings["seed"])
    summary = run_fixed_gate(args.features_run, args.output, settings)
    raise SystemExit(0 if summary["valid"] else 3)


if __name__ == "__main__":
    main()
