# Evidence for latent RL run 6879761 (condition C)

Latent RL from the LIVR rerun `6253300` (`stage2/final`) with
[configs/latent_rl.json](../../../configs/latent_rl.json) (σ_init 1.6, adjusted
layer 16), one epoch on one H200, then the reference benchmark in memory with the
mean adjustment, a sampled adjustment and no adjustment. See
[the results](../../LATENT_RL_RESULTS.md) and [the specification](../../LATENT_RL_SPEC.md).

| File | Contents |
| --- | --- |
| evaluation-outcomes-{mean,sample,zero}.jsonl | All 400 test maps in original order (row_index zero-based): grid size, success and extracted route for each latent mode |
| evaluation-summary-{mean,sample,zero}.json | Overall and per-grid reference success counts |
| steps.jsonl | Per-step training log: reward, zero-variance groups, loss, KL, gradient norm, timing, peak memory, mean σ and mean absolute μ |
| rollouts.jsonl | Per-map rewards and completion lengths for every training group, in the same map order as `text-rl-6256423/rollouts.jsonl` |
| probe-6879148-summary.json | First σ probe (σ 0.01 to 2, 2 draws per map) on 100 training maps |
| probe-6879676-summary.json | Second σ probe (σ 1.2 to 1.8, 4 draws per map), which chose σ = 1.6 |
| pilot-6879676-steps.jsonl | The 10-step pilot at σ = 1.6 |

Raw responses and run manifests remain on PACE scratch under
`results/latent-rl-6879761` and `results/latent-rl-pilot-*`. The final weights were
deleted from scratch on 2026-11-08 to free space. These files contain no model
weights, images or credentials.
