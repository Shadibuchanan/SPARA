"""Prepare rebalanced training datasets for Phase 1C.

Separates training maps into detour and no-detour maps using exact grid BFS,
then creates subsets with controllable detour ratios:
  - natural (approx 10% detour maps, matching the test benchmark)
  - balanced (25% or 50% detour maps)
  - original (unfiltered)
"""
import argparse
from collections import deque
import json
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spara.data import records
from analyze_map_detours import analyze_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp", help="Root of Mirage VSP dataset")
    parser.add_argument("--output-dir", default="data/rebalanced", help="Output directory for generated datasets")
    parser.add_argument("--detour-ratio", type=float, default=0.10, help="Target fraction of detour maps (e.g. 0.10 for natural)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    train_rows = records(args.data_root, "train")
    
    detour_rows = []
    nodetour_rows = []
    
    for r in train_rows:
        analysis = analyze_map(r["map_desc"])
        if analysis["is_detour"]:
            detour_rows.append(r)
        else:
            nodetour_rows.append(r)
            
    print(f"Original train set: {len(nodetour_rows)} no-detour rows, {len(detour_rows)} detour rows")
    
    # Calculate sample counts for target detour ratio
    # If ratio = p, then n_detour / (n_detour + n_nodetour) = p
    # n_detour = p * n_total, n_nodetour = (1-p) * n_total
    # If we keep all nodetour_rows (902), n_detour = 902 * p / (1 - p)
    p = args.detour_ratio
    target_detour_count = int(round(len(nodetour_rows) * p / (1.0 - p)))
    target_detour_count = min(target_detour_count, len(detour_rows))
    
    sampled_detours = random.sample(detour_rows, target_detour_count)
    rebalanced = nodetour_rows + sampled_detours
    random.shuffle(rebalanced)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_ratio_{int(p*100)}pct.jsonl"
    
    with out_path.open("w", encoding="utf-8") as f:
        for row in rebalanced:
            f.write(json.dumps(row) + "\n")
            
    print(f"\nCreated rebalanced dataset: {out_path}")
    print(f"  Total rows: {len(rebalanced)}")
    print(f"  No-detour rows: {len(nodetour_rows)} ({len(nodetour_rows)/len(rebalanced)*100:.1f}%)")
    print(f"  Detour rows: {len(sampled_detours)} ({len(sampled_detours)/len(rebalanced)*100:.1f}%)")


if __name__ == "__main__":
    main()
