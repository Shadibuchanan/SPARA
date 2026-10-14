# Evidence for vanilla SFT arm A (6259811)

Vanilla SFT control for [LIVR run 6189457](../livr-6189457/): `configs/sft.json`, no latent
tokens and no Stage-1 bottleneck, 2,500 optimizer updates. See
[the plan](../../plans/VANILLA_SFT_CONTROL_PLAN.md) and [ADR-0004](../../adr/0004-reinstate-vanilla-sft-control.md).
Score: 320/400 (80.00%); 312/392
(79.59%) excluding the eight train/test overlapping rows. Hardware: NVIDIA H100 80GB HBM3.

| File | Contents |
| --- | --- |
| evaluation-summary.json | Overall, per-grid and overlap-excluded reference success counts |
| evaluation-outcomes.jsonl | All 400 generated answers, extracted routes and outcomes in original test order; row_index is zero-based |
| `train-steps.jsonl` | Optimizer-step logs in the LIVR format |
| provenance.json | Recipe, evaluator/data hashes, hardware, GPU seconds, completed checks, runtime source hashes and evidence checksums |

As in the LIVR bundle, raw responses omit the repeated prompt prefix, and no weights,
optimizer states, images or credentials are included. Regenerate the comparison chart with
`python scripts/plot_sft_comparison.py --sft-a <arm A bundle> --sft-c <arm C bundle>`.
