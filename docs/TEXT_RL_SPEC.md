# Text-only RL (condition B)

Status: one full run completed, scoring 326/400 (81.5%) against 328/400 for its
starting checkpoint. See [the results](TEXT_RL_RESULTS.md).

Condition B continues from the final Stage 2 LIVR checkpoint with GRPO on the
generated route text. LIVR's latent tokens stay in the prompt but are never
perturbed or sampled. This isolates the main comparison in the
[project plan](SPARA_PROJECT_PLAN.md): SPARA (C) differs from B only by its
latent adjustments.

## Method

For each training map, sample a group of routes from the current policy, score
each with the pinned reference evaluator (1 for success, 0 otherwise), and
normalize rewards within the group. The loss is the text term of Scaffolding
Minds Eq. 6: a clipped importance ratio times the group advantage, minus a
k3 KL penalty toward the frozen LIVR checkpoint. Each sequence's tokens are
averaged, then sequences are averaged equally, matching the paper and the
LIVR baseline's per-example loss weighting.

The model keeps Stage 2 (normal) attention, the frozen vision encoder/projector
and full language-backbone training. Rollouts use cached decoding;
`tests/test_text_rl.py` checks their logits against the uncached training forward.

## Settings and sources

| Setting | Value | Source |
| --- | --- | --- |
| Group size G | 8 | Scaffolding Minds D.1 |
| Clip epsilon | 0.2 | Scaffolding Minds D.1 |
| Learning rate | 1e-6 | Scaffolding Minds D.1 |
| KL coefficient | 0.001 against the starting checkpoint | Scaffolding Minds D.1 |
| Reward | Reference evaluator success, 0/1 | Project plan; Scaffolding Minds D.1 |
| Training maps | Original 1,000 training records | Project plan ("match maps") |
| Rollout decoding | Temperature 0.7, top-p 0.9, 1,024 new tokens | Assumption: same as evaluation |
| Prompts per optimizer step | 8 (64 rollouts) | Assumption |
| Epochs | 1 (125 optimizer steps) | Assumption |
| Policy updates per rollout batch | 1, so the ratio is 1 and clipping is inactive | Assumption (common GRPO default) |
| Log-probabilities | Logits divided by rollout temperature; top-p ignored | Assumption (TRL convention) |
| Optimizer | Fresh AdamW, LIVR betas/epsilon/weight decay, constant LR, clip 1.0 | Assumption |
| Advantage | (r - mean) / (sample std + 1e-4) | Assumption (GRPO/TRL convention) |
| Checkpoint | Final RL weights; no validation selection | Same policy as the LIVR baseline |

Settings are in [configs/text_rl.json](../configs/text_rl.json). Condition C
should reuse them unchanged apart from its latent policy.

## Running

```bash
python -m spara.cli train-rl --data-root /path/to/data/vsp \
  --checkpoint /path/to/stage2/final --output /path/to/text-rl
python -m spara.cli evaluate --data-root /path/to/data/vsp \
  --checkpoint /path/to/text-rl/final --output /path/to/text-rl-evaluation
```

On PACE, set `SPARA_LIVR_CHECKPOINT` and submit
[scripts/pace/text_rl.sbatch](../scripts/pace/text_rl.sbatch) with the agreed
account and GPU flags. `--resume` continues from a `step-N` checkpoint written
when `checkpoint_every_steps` is nonzero.

Each run writes `steps.jsonl` (mean reward, zero-variance groups, KL, loss,
gradient norm, completion length and timing per step) and `rollouts.jsonl`
(per-map rewards and lengths).

## Open issues

- **Weak learning signal on the training maps.** The LIVR checkpoint reached a
  final training loss near 0.01, so most training maps may produce eight
  successful rollouts. Those groups have zero advantage and contribute only the
  KL term. Check `zero_variance_groups` early in a pilot before spending a full
  budget. If most groups are uniform, options include higher rollout temperature,
  more rollouts per map, or new or larger maps; each changes the agreed scope.
- **Unmeasured cost.** The run holds a second FP32 copy of the 3B model as the
  KL reference (about 15 GB more than the 55 GiB baseline peak) and generates
  64 rollouts per step. Profile a short pilot before requesting a full allocation.
- **Single seed and no validation split**, as in the baseline.
