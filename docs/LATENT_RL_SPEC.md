# Latent RL (condition C)

Status: one full run completed (job `6879761`). With its learned mean adjustment it scored
323/400, against 326/400 for B (McNemar p = 0.63). See [the results](LATENT_RL_RESULTS.md).

Condition C continues from the same final Stage 2 LIVR checkpoint as text-only RL
(condition B, the rerun `6253300`) and runs B's recipe unchanged, adding one thing:
a learned Gaussian adjustment of LIVR's latent states, adapted from Scaffolding RL.
The [latent battery](LATENT_BATTERY_RESULTS.md) showed that LIVR's route answer
follows its four latent states, so adjusting them acts on what sets the route.

## Method

During each forward pass, the four latent states at the output of decoder layer 16
(of 36), z, are replaced by a = z + δ with δ ~ N(μ(z), σ(z)²) per dimension.
Scaffolding's Gaussian heads give μ and σ. Layers 17 to 36, and every token after
the latents, read a. Zero adjustment reproduces LIVR exactly. The adjustment cannot
go at the last layer: later tokens read a position only through the layers before
the last, so a final-layer change would have no effect.

Each rollout draws its own δ; the group of eight shares the prompt. The loss adds
Scaffolding's latent term (eq. 6) to B's text term: the clipped ratio of the
Gaussian log-probability of the rollout's action, times the group advantage. Its
gradient reaches the heads and, through z, the layers below the adjustment. The
KL reference (the frozen LIVR checkpoint) reads the same latent action, so the KL
penalty covers the text only. Everything else, including sampling, advantages,
the text loss and the optimizer, is B's code path; with no latent settings the
same loop runs B.

Code: [src/spara/latent_rl.py](../src/spara/latent_rl.py) and the shared loop in
[src/spara/text_rl.py](../src/spara/text_rl.py). The tests in
[tests/test_latent_rl.py](../tests/test_latent_rl.py) check that:

- zero adjustment gives LIVR's logits;
- a nonzero adjustment changes the answer logits but not earlier tokens;
- replaying a rollout's latent action reproduces its cached sampling logits;
- gradients are the same with and without gradient checkpointing;
- the heads survive a save and reload.

## Settings and sources

B's settings ([TEXT_RL_SPEC.md](TEXT_RL_SPEC.md)) apply unchanged. The latent
settings are in [configs/latent_rl.json](../configs/latent_rl.json).

| Setting | Value | Source |
| --- | --- | --- |
| Adjusted layer | Output of decoder layer 16 of 36 | Assumption: map content peaks in layers 8 to 20 (latent battery), leaving 20 layers to plan from the adjustment |
| Latent action | a = z + δ, δ ~ N(μ, σ²) per dimension over 4 x 2048 | Scaffolding Minds eqs. 3 to 5 |
| Heads | Two-layer MLPs for μ and log σ (hidden 2048), μ starting at zero | Scaffolding implementation |
| Head input | z normalised to unit RMS per latent | Assumption: layer-16 channels span about 1 to 280, while Scaffolding's heads read final-norm states |
| σ at initialisation | 1.6: the smallest probed σ that changes at least 10% of greedy routes on 100 training maps without lowering success by more than 5 points (see below) | The Scaffolding rerun's rule, fixed before its probe |
| log σ bounds | [-9.21, 1.8563], a ceiling of 4 x σ_init = 6.4 | The Scaffolding rerun's ceiling of 4 x its σ_init |
| Latent log-probability | Mean over the 4 x 2048 dimensions | Scaffolding rerun; the paper's sum swamped the text term |
| Head learning rate | 2e-5; the rest of the model keeps B's 1e-6 | Assumption: Scaffolding's heads did not move at 1e-6; its head-rate run used 2e-5 |
| Evaluation | Reference protocol with δ = μ, with sampled δ, and with δ = 0 | Assumption; amended before the full run (below) |

## σ probe and pilot

Across training maps of one grid size, layer-16 latents differ by a per-dimension
RMS of only 0.04 to 0.25, depending on the slot, so the first probe grid ran from
0.01 to 2 (job `6879148`, 2 draws per map and σ). Random directions turned out to
matter far less than those differences. Greedy routes changed at a rate of at most
1% up to σ = 0.5, 4% at σ = 1 (success 93% against 94% unadjusted) and 35.5% at
σ = 2 (success 64%). No grid value met the rule. Keeping the rule, a second probe
between 1 and 2 (job `6879676`, 4 draws) found:

| σ | Routes changed | Success (94% unadjusted) |
| --- | ---: | ---: |
| 1.2 | 3.0% | 93.75% |
| 1.4 | 5.0% | 93.0% |
| 1.6 | 11.25% | 89.25% |
| 1.8 | 19.25% | 80.25% |

The rule chose σ = 1.6, which is close to the success limit: a 4.75-point drop. The
same job ran 10 RL steps at that σ. 17 of 80 groups (21%) had mixed rewards, the
same share as B's 10-step pilot (20%), so latent exploration did not visibly add
learning signal. Mean reward was 0.89, steps took about 23 seconds, and peak memory
was 70.3 GiB on an H200. Over the 10 steps the mean adjustment grew from 0 to 0.033
in mean absolute value, and σ from 1.600 to 1.603. The two probes and the pilot
used 12.3 GPU-minutes.

## Plan and budget

The agreed ceiling is about 1.5 GPU-hours on account `coc`.

1. **Probe and pilot (one job, about 15 minutes):** run the σ probe, then 10 RL
   steps at the chosen σ. Check the share of groups with mixed rewards (B: 9.9%),
   KL, peak memory and step time.
2. **Full run (about 70 minutes):** one epoch of 125 steps (8 maps x 8 rollouts),
   final weights only, then the benchmark with δ = μ, with sampled δ and with δ = 0,
   in memory.

## How the result will be read

These criteria were fixed before any run.

- **Primary:** C with δ = μ against B (326/400, job `6256423`), map by map on the
  400 test maps with an exact McNemar test. C counts as different from B only if
  p < 0.05. For scale, two LIVR runs with identical settings disagree on 31 maps
  (18 against 13), and B disagrees with its starting checkpoint on 20 (9 against 11).
- **Secondary:** C with δ = μ against C with δ = 0, the same weights, which shows
  whether the learned adjustment matters at test time. C with sampled δ is
  reported too.

*Amendment, made after the probe and pilot and before any C evaluation:* the
primary comparison originally used sampled δ. At σ = 1.6, sampled δ alone costs
about 5 points of greedy success on training maps, so a sampled comparison with B
would mostly measure exploration noise. δ = μ is C's learned adjustment without
that noise.
- **Diagnostic:** the share of informative groups in training, against B's 9.9%,
  and whether μ and σ move from their initial values.

There is one seed and no validation split, as for A and B.

## Running

```bash
python -m spara.cli latent-probe --rl-config configs/latent_rl.json --data-root /path/to/data/vsp \
  --checkpoint /path/to/stage2/final --output /path/to/probe
python -m spara.cli train-rl --rl-config configs/latent_rl.json --data-root /path/to/data/vsp \
  --checkpoint /path/to/stage2/final --output /path/to/latent-rl --sigma-init SIGMA \
  --final-weights-only --evaluate-latent-modes
python -m spara.cli evaluate --rl-config configs/latent_rl.json --data-root /path/to/data/vsp \
  --checkpoint /path/to/latent-rl/final --latent-mode sample --output /path/to/evaluation
```

On PACE, submit [scripts/pace/latent_rl.sbatch](../scripts/pace/latent_rl.sbatch)
with `SPARA_LIVR_CHECKPOINT` and `SPARA_STEP=probe-pilot` or `full`.
