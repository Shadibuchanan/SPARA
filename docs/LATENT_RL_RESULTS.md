# Latent RL results (condition C)

Updated 2026-11-08. One seed. The comparisons below were fixed in the
[specification](LATENT_RL_SPEC.md) before any C evaluation.

## Summary

Condition C is text-only RL (condition B) plus a learned Gaussian adjustment of
LIVR's latent states at decoder layer 16. With its learned mean adjustment (the
primary comparison) it scored **323/400**. B scored 326/400, and the LIVR
checkpoint both started from 328/400. **C is not different from B:** the two
disagree on 17 maps (7 against 10, exact McNemar p = 0.63). Two LIVR runs with
identical settings disagree on 31.

The learned adjustment has no effect at test time. The same weights score 324/400
with no adjustment, and the two evaluations disagree on 9 maps (4 against 5).

Latent exploration did increase the learning signal. 16.8% of training groups had
mixed rewards, against 9.9% for B on the same maps in the same order (97 groups
mixed only in C, 28 only in B; p = 4e-10). The extra signal came mostly from noise
breaking maps the model already solved, and it did not change test success.

| Grid | LIVR rerun (6253300) | Text-only RL, B (6256423) | C, δ = μ (primary) | C, sampled δ | C, δ = 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 x 3 | 92 | 92 | 93 | 93 | 93 |
| 4 x 4 | 88 | 84 | 86 | 88 | 87 |
| 5 x 5 | 86 | 86 | 84 | 83 | 83 |
| 6 x 6 | 62 | 64 | 60 | 62 | 61 |
| Overall | 328/400 | 326/400 | **323/400** | 326/400 | 324/400 |

| Paired comparison | Maps only the first solves | Only the second | Exact McNemar p |
| --- | ---: | ---: | ---: |
| C (δ = μ) vs B | 7 | 10 | 0.63 |
| C (δ = μ) vs C (δ = 0) | 4 | 5 | 1.0 |
| C (sampled) vs B | 9 | 9 | 1.0 |
| C (δ = 0) vs B | 9 | 11 | 0.82 |
| C (δ = μ) vs LIVR rerun | 7 | 12 | 0.36 |

## Run

Job `6879761` trained one epoch (125 steps, 8 maps x 8 rollouts) from
`baseline-6253300/stage2/final` with σ_init 1.6, then ran the reference benchmark
in memory three times. It took 3,530 seconds on one H200 under account `coc`.
Together with the two σ probes and the 10-step pilot (jobs `6879148` and `6879676`,
736 seconds), condition C used 4,266 GPU-seconds (1.19 GPU-hours) of the 1.5
approved. Steps took 24.7 seconds on average (B: 24.0), and peak memory was 70.3 GiB
(B: 69.8). The final weights (15 GB) were deleted from PACE scratch on 2026-11-08 to
free space; rerunning the job would retrain them in about an hour. Compact evidence is in
[experiments/latent-rl-6879761](experiments/latent-rl-6879761/README.md).

## Training signal

C and B trained on the same 1,000 maps in the same order (same seed), so their
groups pair up.

| | C | B |
| --- | ---: | ---: |
| Groups with mixed rewards | 168 (16.8%) | 99 (9.9%) |
| Groups where all eight rollouts succeeded | 795 | 863 |
| Groups where all eight failed | 37 | 38 |
| Rollout success | 89.5% | 91.2% |
| Mean KL to the LIVR checkpoint, per token | 0.053 | 0.023 |

The extra mixed groups replaced all-success groups (68 fewer); all-fail groups did
not change. In most of the added signal, then, the adjustment made the model fail a
map it otherwise solves. The resulting gradient favours robustness to the noise
rather than better routes. That fits the test results, but it is an interpretation,
not a measurement.

The Gaussian heads moved little. Mean σ rose from 1.600 to 1.621, and the mean
absolute adjustment μ from 0 to 0.098, small against σ and against the scale at
which random directions change routes (σ above 1). Scaffolding's heads also
learned no useful adjustment at this head learning rate.

## Noise at test time

Before RL, adding the σ = 1.6 noise lowered greedy success on training maps by 4.75
points. After RL, sampled δ scored 326/400 on the test maps against 324/400 with
δ = 0. These two measurements differ in decoding (greedy against sampling at
temperature 0.7, which already perturbs routes) and in maps (training against
test), so the missing test-time drop is not itself evidence that C became robust
to the noise. Rerunning the σ probe on C's weights (about 5 GPU-minutes) is the
like-for-like test of whether training with the noise made the later layers
insensitive to it. Those weights have since been deleted, so the test would first
need C retrained (about 1 GPU-hour).

## Caveats

- One seed, and evaluation samples answers at temperature 0.7, so differences of a
  few maps are noise.
- σ = 1.6 met the selection rule with little margin: a 4.75-point success drop
  against the 5-point limit.
- The adjusted layer (16), the head learning rate (2e-5) and the heads' normalised
  input are assumptions; none was tuned, by design.
- Isotropic noise puts most of its power in directions the model ignores. The
  latents of different maps vary in only about 10 to 18 effective directions
  ([latent battery](LATENT_BATTERY_RESULTS.md)).

## Options

Each of these is a scope decision for the team.

- **Structured exploration:** draw the adjustment in the few directions in which
  maps' latents actually differ, so the noise acts on what the model reads.
- **A harder training set:** new or larger maps on which the model is not already
  saturated. This was B's main open issue too, and it would apply to B and C alike.
- **A robustness probe of C's weights:** the σ probe on C's final weights, to test
  the explanation above. The weights were deleted on 2026-11-08, so this now needs
  C retrained first.
