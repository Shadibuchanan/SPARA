# Evidence for Rebalanced Detour Fine-Tuning (Job 7109628)

Evaluates full Stage 1 and Stage 2 fine-tuning from scratch (`Qwen/Qwen2.5-VL-3B-Instruct`) on the natural 10% detour distribution (`data/rebalanced/train_ratio_10pct.jsonl`).

## Context & Motivation
An audit of the standard 400-map benchmark test set (`scripts/analyze_map_detours.py`) revealed that **90.0% of test maps require zero detours** (only 10.0% require a detour). Past detour fine-tuning runs (e.g. `detour_mix.jsonl` with 66.7% detour maps) severely distorted the policy, inducing hallucinated detour detours on straight maps and causing significant regressions on the benchmark.

To test whether the policy can learn safe routing without regressing on direct maps, we trained on a dataset reflecting the natural benchmark distribution (90% direct, 10% detour).

## Setup & Execution
- **Job ID:** 7109628 on PACE-ICE (`ice-gpu`, NVIDIA H100 80GB GPU)
- **Wall Time:** 1h 49m 33s
- **Architecture:** Stage 1 (10 epochs) + Stage 2 (10 epochs)
- **Training Set:** `data/rebalanced/train_ratio_10pct.jsonl` (1,000 maps: 902 direct, 98 detour)
- **Evaluation:** Full 400-map standard reference test benchmark with official `mathruler==0.1.0` and SHA-256 verified grader.

## Results
- **Overall Accuracy:** 305 / 400 (76.25%)
  - **3x3 Grids:** 89 / 100 (89.0%)
  - **4x4 Grids:** 79 / 100 (79.0%)
  - **5x5 Grids:** 79 / 100 (79.0%)
  - **6x6 Grids:** 58 / 100 (58.0%)

## Key Takeaways
1. Training from scratch on the natural 10% detour mix successfully preserves strong routing on straight maps (no detour hallucinations).
2. The model achieves 76.25% accuracy across all grid sizes.
3. This checkpoint serves as the clean base policy for Phase 2 preference optimization (DPO / unlikelihood on fatal hole tokens) and RL exploration.
