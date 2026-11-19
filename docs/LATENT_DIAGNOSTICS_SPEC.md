# Latent-exploration diagnostics after C′

Status, 2026-11-19: run (jobs 7148362, 7148373, 7148374, 7148376, 7148638, 7148708); result below. The one usable qualifying configuration was confirmed (job 7151591), and C″ built on it showed no detectable difference from B′ (job 7154355); both are in the amendments at the end.

## Why

In the fair rerun C′ ([STRUCTURED_EXPLORATION_SPEC.md](STRUCTURED_EXPLORATION_SPEC.md), amendment of
2026-11-17) solved fewer of the 240 held-out maps than B′: 77 and 86 against 92 and 96, seeds 42 and 43.
Its learned mean changed nothing at test time (77 vs 78, 86 vs 85 with δ = 0). The run files show:

- **D1, breaks not rescues.** Paired by map, C′'s extra mixed groups come mostly from breaking maps that
  B′ solves every time.
- **D2, heads that cannot move.** σ stayed at 3.9–4.1 and |μ| at about 0.1 (σ starts at 4). At head
  learning rate 2e-5, Adam moves a parameter at most about 2.5e-3 in 125 steps.
- **D3, a swamped gradient.** C′'s gradient norm has a median of about 1,150, against about 5.5 for B′,
  under one clip at 1.0.
- **D4, a partial cut.** Replaying the detached action cuts the text reward off from the latent state
  that later layers read.
- **D5, map directions.** The noise lies in the directions in which *maps* differ, at a layer where map
  content peaks.

These diagnostics measure, in one submission, what is needed to choose the next step:
- C″, a fixed C′;
- latent search inside expert iteration;
- or dropping latent exploration.

Nothing here trains a model.

## Questions and measurements

On the LIVR rerun (`baseline-6253300`), unless stated otherwise.

| # | Question | Measurement |
| --- | --- | --- |
| Q1 | Where does C′'s gradient go? | Three real C′ batches (seed 42's first maps, P from the pilot, C′'s heads at σ = 4). Gradient norms per group: heads; embeddings; layers 1–8, 9–16, 17–24, 25–36; final norm and output layer. Measured for the text term and the latent term (as built, and with the prior detached). Also: the clip factor of their sum, and the share of entries below 1e-7 after clipping (where Adam's ε damps the update), against B′'s update on the same maps |
| Q2 | How much does absolute replay cut? | B′'s rollouts with a zero adjustment: the text gradient with the state replayed absolutely and as an offset, against no adjustment. Also C′'s text gradient with offset against absolute replay |
| Q3 | Can the heads learn from this signal? | Offline (CPU), C′'s heads trained by the latent term alone on recorded (prior, coefficients, advantage) groups. The groups come from the sweep at layer 16, c ∈ {1, 4}, and from Q1's rollouts. Head learning rates {2e-5, 1e-4, 1e-3, 1e-2} × {125, 500} steps. Report σ and \|μ\| over training, the held-out change of the objective, and the cosine between μ and the advantage-weighted mean coefficient on held-out maps |
| Q4 | Does any latent perturbation add exploration over T = 1 text sampling? | The sweep below |
| Q5 | Is C′'s basis "which map"? | Per layer, linear probes on the 2,000 pool maps' latents (1,600 fit, 400 test) for goal, start, holes, first move and route length. The map-content subspace by iterated nullspace projection, with the probe accuracy at each iteration. The share of C′'s noise energy inside it |
| Q6 | Are latents a better search operator than sampling? | On the pilot maps LIVR's greedy answer fails, each with a budget of 64: text pass@64; final-verdict-only search (64 policy calls); and latent random search (64 greedy answers under C′'s noise, 16 at each of c = 1, 2, 4, 8), at layers 16, 24 and 32 |
| Q7 | Did RL change how much the model relies on its latents? | LIVR, B′ 42/43 and C′ 42/43 (δ = 0) on the 240 held-out maps: greedy success, label-route log-probability and first-move probability. On every second map: the latent swap test (all layers, as in the battery) and greedy route change under C′'s noise (layer 16, c = 1 and 4, four draws) |
| Q8 | Where did each arm's weights move? | The norm of each run's weights minus LIVR's, per layer group (CPU) |

### The sweep (Q4)

- **Maps:** the pilot's 200 pool maps, seed 42's first 200, which are the maps training sees first.
- **Baseline:** LIVR's greedy answer and 64 rollouts at T = 1 with the fair sampler.
- **Perturbations:**
  - **Shape:** each perturbs the four latent states at one layer's output, with 8 rollouts per
    configuration under the same sampler.
  - **Scales:** c ∈ {0.25, 0.5, 1, 2, 4}.
  - **Equal size across families:** every family is scaled so a draw at c has the expected squared size
    of C′'s noise at c in every slot.

| Family | Directions | Layers |
| --- | --- | --- |
| `pca` (C′'s) | top 16 principal directions per slot, centred within grid size | 16, 24, 32 |
| `pca_perp_map` | the same, after projecting out the map-content subspace | 16, 24, 32 |
| `map_only` | the principal directions inside the map-content subspace (control: should break most) | 16, 24, 32 |
| `decision` | per map, the span of the gradients of: the label route's and the greedy route's log-probability, and each move's log-probability where they diverge | 16, 24, 32 |
| `isotropic` | uniform over each slot | 16, 24, 32 |
| `answer` | per step, at the last prompt token and each generated token, in the principal basis of last-prompt-token states; c ∈ {0.5, 1, 2} | 16, 24 |

**Continuous metrics.** For each configuration, four draws per map score the label route's
teacher-forced log-probability and the four-move distribution at its first move.

### Rule, fixed before the run

All metrics are over the 200 maps, with the map as the unit.

**Definitions:**
- **Mixed-share gain:** the configuration's share of maps whose 8 rollouts are mixed, minus the share
  of text sampling.
  - Text's share is estimated without bias from its 64 rollouts, for groups of 8.
  - The 95% interval comes from a map bootstrap of 2,000 rounds.
- **Rescue:** a map that text never solves in 64 rollouts, where the configuration solves at least one
  of 8.
- **Break:** a map text solves at least half the time, where the configuration's success rate is lower
  by at least 0.5.
- **Null:** text's own first 8 rollouts scored against its other 56. This gives the rate of false
  rescues and breaks.

**A configuration adds exploration only if all hold:**
- the mixed-share gain is at least 0.02 and its interval is above zero;
- rescues ≥ breaks;
- rescues − breaks ≥ the null's rescues − breaks;
- rollout success is at most 0.02 below text sampling's.

**What follows:**
- **If no configuration qualifies,** latent noise does not add exploration over text sampling on these
  maps. C″ is not run, and latent exploration's budget moves to expert iteration.
- **If some do,** C″ is specified in an amendment, using the best qualifying configuration by
  rescues − breaks. That choice follows this rule, not training outcomes.

## Validity

- **CPU tests:** `tests/test_latent_diagnostics.py`. They cover:
  - every family has the target size;
  - leave-one-out ridge matches refits;
  - iterated nullspace projection captures planted map content and the perpendicular basis avoids it;
  - offset replay at zero gives exactly B′'s gradient, while absolute replay does not;
  - group norms add up;
  - answer noise touches only the last position;
  - every part runs end to end on a tiny model.
- **Smoke pass:** each GPU job first runs a smoke pass on a few maps.
- **Read back:**
  - Q1's summed gradient norm should fall in the range C′ 42 logged (about 400–2,500), and B′'s in
    B′'s (about 3–10).
  - `pca` at layer 16, c = 4, under greedy decoding is the pilot's noise probe. Q7's LIVR row at c = 4
    should agree with it within noise.

## Budget

| Job | Content | Estimate (Blackwell slice) |
| --- | --- | --- |
| A | Sweep at layers 16 and 24, text baseline | about 1.2 h |
| B | Sweep at layer 32, answer noise at 16 and 24 | about 0.8 h |
| C | Q1–Q3 tuples, then Q6 | about 0.8 h |
| D | Q7 on five checkpoints | about 0.7 h |
| CPU | Q8; analysis with Q3's fits | — |

About 3.5 slice-hours, run as four parallel jobs.

## Running

```bash
export SPARA_PYTHON=$ROOT/.venv-cu128/bin/python SPARA_PILOT_RUN=$ROOT/results/fair-pilot-7142592 \
       SPARA_LIVR_CHECKPOINT=$ROOT/code/spara_results/baseline-6253300/stage2/final
SPARA_PART=sweep SPARA_LAYERS=16,24 SPARA_TEXT_BASELINE=1 sbatch ... scripts/pace/latent_diagnostics.sbatch
SPARA_PART=sweep SPARA_LAYERS=32 SPARA_ANSWER_NOISE=1 sbatch ... scripts/pace/latent_diagnostics.sbatch
SPARA_PART=gradients sbatch ... scripts/pace/latent_diagnostics.sbatch
SPARA_PART=checkpoints SPARA_RUNS='{"livr": ..., "b42": ..., "b43": ..., "c42": ..., "c43": ...}' sbatch ...
SPARA_PART=weights SPARA_RUNS=... sbatch ...   # CPU
python -m spara.cli latent-diagnostics --part analyse --sweeps A,B --gradients-run C --checkpoints-run D \
  --weights-run W --runs '{...}' --output $ROOT/results/latent-diag-analysis   # CPU
```

## Result (2026-11-19)

All parts ran on one Blackwell 48 GB slice each (weights on the CPU). The analysis is
`latent-diag-analysis/` on PACE scratch.

**What did not go to plan:**
- The sweeps were stopped at their time limits, as agreed, rather than extended. Layers 16 and 24
  cover the pilot's first **96** maps; layer 32 and answer noise cover the first **156**. A separate
  job (7148708) ran the text baseline on all 200 maps, and every configuration is compared with
  that baseline.
- The `decision` family uses the **label route** to choose its directions (see the Q4 table). RL
  here has a verifier and no labels, so this family is an oracle. It shows what latent noise
  could do in good directions; it is not a candidate for C″. The spec did not say this
  beforehand.

**Text sampling (T = 1), 200 maps:** rollout success 0.165; the share of groups of 8 with mixed
rewards is 0.194; 121 maps are never solved in 64 rollouts; pass@64 is 0.395; LIVR's greedy
answer solves 30.

### Q1 and Q2: C′ as built is confounded

On three real C′ batches (seed 42's first maps, C′'s heads at σ = 4):

| | B′ | C′ as built |
| --- | --- | --- |
| Total gradient norm | 4.7–6.4 | 515–1,237 |
| Global clip factor | 0.16–0.21 | 0.0008–0.002 |
| Entries below 1e-7 after the clip, layers 17–36 | 2–7% | 71–96% |

- The latent term's gradient goes mostly to the embeddings and layers 1–8 (norm 316–1,192) and
  layers 9–16 (68–90). The heads take 31–45. Its cosine with the text gradient is about 0.
- Replaying the absolute latent state changes the text gradient by 13–23% in layers 1–16 and
  1–4% later. Offset replay at zero equals B′'s gradient within 0.6%.
- Q8 agrees: C′'s weights moved 20–35% less than B′'s in layers 17–36.

### Q4: does latent noise add exploration over text sampling?

The rule as fixed (mixed-share gain ≥ 0.02 with its interval above 0; rescues ≥ breaks; rescues −
breaks ≥ the null's, which is 0 − 0; success drop ≤ 0.02). Configurations that pass, and the
nearest others:

| Configuration | Maps | Mixed-share gain (95% CI) | Rescues | Breaks | Success drop | Passes | Usable in RL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `decision/32/2` | 156 | +0.088 (+0.031, +0.148) | 8 | 0 | 0.017 | yes | no (label) |
| `decision/32/1` | 156 | +0.094 (+0.046, +0.145) | 7 | 1 | 0.014 | yes | no (label) |
| `decision/32/0.5`, `/0.25` | 156 | +0.043, +0.036 | 4, 2 | 0, 0 | ≤ 0.005 | yes | no (label) |
| `pca/24/0.25` | 96 | +0.085 (+0.008, +0.165) | 6 | 1 | 0.006 | **yes** | **yes** |
| `pca_perp_map/16/0.25` | 96 | +0.044 (+0.009, +0.084) | 0 | 0 | −0.002 | yes | yes, but no rescues |
| `decision/16/2` | 96 | +0.200 (+0.120, +0.286) | 9 | 1 | 0.023 | no (drop) | no (label) |
| `pca/24/0.5` | 96 | +0.137 (+0.046, +0.232) | 8 | 7 | 0.045 | no | yes |
| `pca/16/4` (C′ as run) | 96 | +0.044 (−0.006, +0.097) | 2 | 0 | 0.009 | no | yes |

- **Label-free noise at layers 16 and 32** (`pca`, `pca_perp_map`, `map_only`, `isotropic`), and
  answer-token noise at 16 and 24, changes little at any scale. Apart from the pass below, none
  passes. The most rescues is 6, from `pca/32/2`, which has 2 breaks and an interval that
  includes 0.
- **At layer 24**, from c = 0.5 or 1 up, every family breaks about as many maps as it rescues, or
  more, and success falls by 0.04–0.14.
- **Only `pca/24/0.25` passes with rescues** among the usable configurations. It is marginal:
  - its interval's lower end is +0.008;
  - it did not pass on the first 81 maps (+0.078, interval −0.009 to +0.167);
  - about 80 configurations were tested;
  - the next scale up already breaks maps.
- `pca_perp_map/16/0.25` passes without rescuing or breaking anything, and the next scale up has a
  significantly negative gain (−0.071). With about 80 tests, this is the kind of pass expected
  by chance.

### Q3: can the heads learn from this signal?

Not answered. The offline fits diverged: |μ| reached 10³–10⁷, σ hit its bounds, and the held-out
objective changed by many orders of magnitude at every learning rate. The latent objective has no lower limit
on 13–18 training groups with mixed rewards (4–6 held out), and the fit does not regularise it.
The held-out alignment was −0.02 to +0.05 at c = 1 and −0.11 to −0.15 at c = 4, on 4–6 groups:
no sign of a learnable mean, but too few groups to rule one out.

### Q5: is C′'s basis "which map"?

Mostly. 90–99.9% of C′'s principal noise energy lies inside the 192-dimensional map-content
subspace per slot, at layers 16 and 24. After 192 dimensions are projected out at layer 16, holes
fall to chance, but goal and start remain decodable on larger grids in some slots (up to 0.97). So
`pca_perp_map` is not free of map content.

### Q6: are latents a better search operator?

On LIVR's 170 greedy failures among the pilot maps, with a budget of 64 each:

| Search | Maps solved |
| --- | --- |
| Final-verdict search (≤ 64 policy calls; verifier calls: mean 5.7 on solved maps, 17.6 on unsolved) | 110 |
| Text pass@64 (T = 1) | 49 |
| Latent random search, layer 16 / 24 / 32 (64 verifier calls) | 21 / 78 / 19 |
| Final-verdict search plus latent search at layer 24 | 117 |

Latent search at layer 24 finds more than text sampling, but fewer maps than final-verdict search,
at more verifier calls. It adds 7 maps. By idea C10's own rule it is dropped.

### Q7: did RL change how much the model relies on its latents?

No. In the swap test, the answer follows the donor's latents on 90 (LIVR), 87 and 84 (B′ 42/43),
and 84 and 86 (C′ 42/43) of 120 maps. C′'s noise at layer 16, c = 4, changes 16–20% of greedy routes
for every checkpoint. Greedy success on the 240 held-out maps matches the earlier evaluation
(42; 92, 96; 78, 85).

### Reading

- B′'s win over C′ stands as run. C′ was not a clean test of latent exploration, for the reasons
  in Q1, Q2, Q3 and Q5.
- Latent noise in good directions can add exploration (`decision`), but those directions came
  from the label.
- Of the noise RL could use, only `pca/24/0.25` passes, and only marginally. It is confirmed or
  rejected on fresh maps below before any C″ is specified.

## Amendment: confirmation of `pca/24/0.25` (2026-11-19, fixed before the run)

**Why:** `pca/24/0.25` was picked as the best of about 80 configurations on the pilot's first 96
maps, so its reading there is biased upward. The pilot's maps 97–200 were not used at layer 24.

**What runs** (one GPU job; `configs/latent_confirm.json`):
- **Model and sampler:** LIVR and the fair sampler, as in Q4.
- **Maps:** pilot maps 97–200, i.e. stream indices 96–199, in the pilot's order.
- **Noise:** the `pca` family at layer 24 loaded from sweep 7148362's `bases.pt`. The run records its
  sha256, so the directions are exactly the ones tested.
- **Draws:**
  - two independent groups of 8 rollouts at c = 0.25, the primary;
  - two at c = 0.1, secondary;
  - noise seed 43.
- **Text baseline:** the 64 rollouts per map from job 7148708, independent of this run.

**Rule:** `pca/24/0.25` is confirmed only if, on maps 97–200:
- the mixed-share gain is at least 0.02 and its 95% map-bootstrap interval (2,000 rounds) is above
  0. The gain is averaged per map over the two groups of 8: the same quantity as in Q4, measured
  with less noise;
- rescues ≥ breaks, from the **first** group of 8 only, defined exactly as in Q4. Using all 16
  rollouts would make rescues easier;
- rescues − breaks ≥ the null's, on the same maps;
- the success drop is at most 0.02, over all 16 rollouts;
- all 104 maps completed.

**Not used for the decision:**
- c = 0.1;
- the pooled first-group reading over all 200 maps;
- the soft metrics.

**Checks:**
- the measured squared size of the noise per slot, over the expected c² × slot energy (about 1);
- the maps equal stream indices 96–199;
- LIVR's greedy answers match job 7148708's on the same maps.

**Power:** if the true gain were the +0.085 seen on the first 96 maps, this test would pass about
70% of the time. The first estimate is likely inflated, so a failure means "too small to be
worth C″", not "zero".

**What follows:**
- **Confirmed:** C″ is specified in a further amendment. It uses:
  - layer 24, C′'s basis, σ starting at 0.25 with an upper bound below 0.5;
  - offset replay;
  - the latent term training only the heads;
  - a head learning rate that can move σ.

  No training runs before the user agrees.
- **Not confirmed:** latent exploration is dropped, and its budget moves to expert iteration.

```bash
SPARA_PART=sweep SPARA_LAYERS=24 SPARA_SETTINGS=configs/latent_confirm.json \
  SPARA_BASES_FROM=$ROOT/results/latent-diag-sweep-L16-24-7148362 sbatch ... scripts/pace/latent_diagnostics.sbatch
python -m spara.cli latent-diagnostics --part confirm --settings configs/latent_confirm.json \
  --sweeps $ROOT/results/latent-diag-sweep-L24-<job> --text-run $ROOT/results/latent-diag-sweep-Lnone-7148708 \
  --reference-run $ROOT/results/latent-diag-sweep-L16-24-7148362 --output $ROOT/results/latent-confirm   # CPU
```

### Result of the confirmation (job 7151591)

**Confirmed.** On maps 97–200:
- mixed-share gain +0.121 (95% CI +0.065 to +0.180);
- rescues 3, breaks 3 (first group; null 0 and 0);
- success 0.150 against text's 0.167 (drop 0.017).

All checks passed: 104 maps in order; 0 greedy mismatches; noise size 0.98–1.00 of expected per slot.

**Secondary:**
- c = 0.1: +0.044 (+0.003, +0.084), 3 rescues, 0 breaks, no success drop.
- Pooled first-group reading over all 200 maps: +0.086 (+0.032, +0.137), 9 rescues, 4 breaks.

**Where the gain comes from:** +0.048 of the +0.121 comes from the 60 maps text never solves (noise
success 0.018), and +0.070 from the 15 maps text solves at least half the time, where noise lowers
success from 0.925 to 0.729. The guards held narrowly. Details:
[docs/experiments/latent-diagnostics-7148362](experiments/latent-diagnostics-7148362/README.md).
By this amendment, C″ is specified next and runs only with the user's agreement.

## Amendment: C″ (2026-11-19, fixed before the runs)

**What C″ is.** B′ plus fixed latent noise in its training rollouts. It is read against the existing
B′ runs (seeds 42 and 43) by the same readings as C′. Every difference from C′:

| | C′ | C″ |
| --- | --- | --- |
| Layer | Output of decoder layer 16 | Output of layer 24 |
| Basis | Pilot's top 16 principal directions per slot at layer 16 | The confirmation's `pca` family at layer 24 (sweep 7148362's `bases.pt`, sha256 `c848988a…8418a`, checked at load) |
| Size | σ = 4, learned | c = 0.25, fixed (the confirmed setting) |
| Heads and latent loss | Gaussian heads trained by a latent policy-gradient term | None: one draw ε ~ N(0, 1) per rollout, offset = c Σⱼ uⱼ wⱼ εⱼ per slot |
| Replay in the update | The recorded latent state replaces z (absolute) | z is recomputed and the same offset added, for the policy and the KL reference |
| Evaluation | δ = μ (≈ 0) | No noise |

**Unchanged from B′:**
- the LIVR start point and the pilot (`fair-pilot-7142592`: 24 maps per step);
- the map stream per seed;
- 8 rollouts per map, 125 steps, the learning rate, clipping, the KL term and the sampler;
- low-memory mode on a 48 GB Blackwell slice;
- the evaluation populations and the benchmark.

The noise has its own generator, seeded by the run's seed. So at c = 0, C″ is B′ exactly: the same
rollouts, losses and weights (`tests/test_fair_rl.py`).

**Logged:** each rollout's ε (`rl/rollouts.jsonl`), and per step the measured over expected squared
size of the noise per slot (`noise_size_ratio`, expected about 1).

**Readings, fixed:** `fair_rl.compare(runs, pilot, settings, livr_outcomes, data_root, evaluator,
other="n")`, the rules of the 2026-11-17 amendment with C″ in C′'s place.
- **Primary:** greedy successes on round 1's 240 held-out detour maps, summed over the two seeds;
  exact sign test against B′. "C″ better" needs p < 0.05 and both seeds the same way.
- **Validity:**
  - both runs complete;
  - the same maps in the same order as B′ (pairing);
  - every rollout reward matches a CPU rescore;
  - the pilot's checks pass;
  - the policy moved from LIVR (at least 16 of the 200 pilot maps).
- **Secondary:** transfer to the 400 maps of 7 × 7 and 8 × 8; the benchmark against LIVR by map
  kind; each arm against LIVR.
- **Diagnostics:**
  - the training mixed share (B′: 0.29 and 0.26);
  - groups paired with B′'s by step and map: mixed for C″ where B′'s all failed (rescues), or
    all succeeded (breaks);
  - KL to LIVR.

**What it can claim:**
- **If C″ is better:** fixed latent noise at layer 24 adds exploration that RL turns into held-out
  gains beyond text sampling, in this recipe.
- **If not:** the confirmed exploration gain (about 3 more mixed groups per step, about 1 on maps
  text never solves) is too small, or too mixed with breaks, to help at 125 steps.
- **Either way,** it says nothing about learned latent policies, which C″ does not have.

**Budget:** one job on one 48 GB Blackwell slice, seeds 42 then 43, each a run and its evaluation.
B′'s runs took about 2 hours each, so about 4–4.5 hours, with a 5.5-hour limit.

```bash
SPARA_PILOT_RUN=$ROOT/results/fair-pilot-7142592 SPARA_NOISE_BASIS=$ROOT/results/latent-diag-sweep-L16-24-7148362 \
  SPARA_LOW_MEMORY=1 sbatch --gres=gpu:rtx_pro_6000_blackwell:1 ... scripts/pace/fair_rl_noise.sbatch
```

### Result of C″ (job 7154355)

All validity checks passed: both runs complete, pairing, 96,000 rollouts rescored, the policy moved
(76 and 77 maps). The noise's measured size averaged 0.99–1.01 of the expected size per slot.

**Primary: no detectable difference.** On the 240 held-out detour maps:
- C″ 98 and 98, against B′ 92 and 96;
- C″ ahead on 22 maps, B′ on 15 (p = 0.32).

**Secondary:**
- Transfer: 74 and 61 against 73 and 67 (p = 0.68).
- Benchmark: 289 and 312 against 305 and 298; like B′, C″ gains on detour maps and loses on
  no-detour maps.

**Diagnostics:**
- C″'s training groups were mixed more often (0.35 and 0.37 against 0.29 and 0.26).
- Paired with B′'s, the extra mixed groups split evenly between maps where B′ failed throughout
  (206, 276) and maps where it always succeeded (224, 267).
- Mean KL to LIVR matched B′'s (0.32).

**Reading:**
- Removing C′'s mechanical problems removed its harm.
- The confirmed latent noise still did not turn into a detectable held-out gain.

Details: [docs/experiments/fair-rl-n-7154355](experiments/fair-rl-n-7154355/README.md).
