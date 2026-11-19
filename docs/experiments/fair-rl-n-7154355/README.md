# C″ against B′ (job 7154355)

The readings fixed in [LATENT_DIAGNOSTICS_SPEC.md](../../LATENT_DIAGNOSTICS_SPEC.md), section
"Amendment: C″", applied by `fair_rl.compare(..., other="n")`. C″ is B′ plus fixed latent noise at
layer 24 (C′'s principal basis at that layer, c = 0.25), confirmed by the latent diagnostics
([latent-diagnostics-7148362](../latent-diagnostics-7148362/README.md)). It has no heads and no
latent loss, replays the noise as an offset, and is evaluated without noise.

The B′ runs are reused unchanged from [fair-rl-7142592](../fair-rl-7142592/README.md).
`comparison.json` holds every reading. `n42/` and `n43/` hold each run's settings, step log
(including the noise's measured size), greedy answers and benchmark answers. Weights and rollouts
stay on PACE scratch (`sft-run/results/fair-n-seed{42,43}-7154355`).

## What ran

| Job | What | GPU time |
| --- | --- | --- |
| 7154355 | C″ seed 42, then seed 43, each training and evaluation, on one 48 GB Blackwell slice (low-memory mode, as B′) | 4 h 26 min |

- **Stack:** the same as B′'s: the same pilot (`fair-pilot-7142592`, 24 maps per step), map
  stream, settings and low-memory mode. Peak GPU memory was 38.7 GiB.
- **The noise was the size intended:** measured over expected squared size per slot averaged
  0.99–1.01 in both runs (per step 0.88–1.20).
- **The basis was the confirmed one:** its sha256 was checked at load.

## Validity: all checks passed

- Both runs completed 125 steps.
- Pairing: within each seed, C″ trained on B′'s maps in B′'s order.
- Rescoring: all 96,000 rollout rewards (B′ and C″) match a CPU rescore.
- The pilot's checks passed.
- Policy moved: C″ differs from LIVR on 76 and 77 of the 200 pilot maps (threshold 16).

## Primary: no detectable difference

On round 1's 240 held-out detour maps (greedy), summing each map's successes over the two seeds:

| | Seed 42 | Seed 43 |
| --- | --- | --- |
| C″ | 98 | 98 |
| B′ | 92 | 96 |
| C′ (for reference) | 77 | 86 |
| LIVR | 42 | 42 |

- C″ is ahead on 22 maps and B′ on 15 (exact sign test, p = 0.32).
- Both seeds lean C″'s way (+6 and +2).
- The fixed reading: **"no detectable difference: no gain of about 7 points or more from latent
  exploration in this recipe."**

## Secondary

- **Transfer to the 400 held-out 7 × 7 and 8 × 8 maps:** C″ 74 and 61, B′ 73 and 67. B′ is ahead
  on 28 maps and C″ on 24 (p = 0.68). No difference.
- **The benchmark against LIVR (328 of 400), split by map kind:**

| Run | All 400 | Detour (40; LIVR 9) | No detour (360; LIVR 319) |
| --- | --- | --- | --- |
| B′ 42 | 305 | 18 | 287 |
| B′ 43 | 298 | 17 | 281 |
| C″ 42 | 289 | 16 | 273 |
| C″ 43 | 312 | 18 | 294 |

  Like B′, C″ gains on detour maps and loses on no-detour maps (pool of detour maps only). The two
  C″ seeds bracket B′'s. The benchmark is one sampled answer per map.

## Diagnostics

| | B′ 42 | B′ 43 | C″ 42 | C″ 43 |
| --- | --- | --- | --- | --- |
| Share of training groups with mixed rewards | 0.29 | 0.26 | **0.35** | **0.37** |
| Mean KL to LIVR | 0.32 | 0.29 | 0.32 | 0.32 |
| Steps that updated | 124 | 124 | 125 | 125 |

- **C″ got more learning signal.** Its share of mixed groups was 6–11 points above B′'s, as the
  confirmation predicted.
- **The extra signal is half rescues and half breaks.** Paired with B′'s groups (same step, same
  map), C″'s groups were mixed where B′'s all failed in 206 and 276 groups, and where B′'s all
  succeeded in 224 and 267, per seed of 3,000. C′'s were 201/221 and 223/240.
- **C″'s text policy moved as much as B′'s** (mean KL 0.32 against 0.29–0.32), unlike C′ (0.24–0.26).
  The KL over the last ten steps was higher for C″ (0.64 and 1.06).

## What this says

- **Fixing C′'s mechanics removed its harm.**
  - C′ lost to B′ (77 and 86 against 92 and 96, p = 0.0016). C″ does not (98 and 98).
  - This comparison is descriptive, not fixed in advance. But C′ and C″ share the pilot, the maps
    and B′. They differ in layer, scale, heads, latent loss and replay.
- **The latent noise still did not detectably help.**
  - It added training signal, but about as much from breaking solved maps as from reaching unsolved
    ones.
  - The held-out gain (+4 maps per seed on average) is within noise. This design detects about 7
    points.
- **Latent exploration in this recipe** (LIVR's four latent tokens, fixed noise, 125 steps from
  LIVR) is not a lever worth more GPU time. Expert iteration, RL from E, and a mixed
  detour/no-detour pool remain the open directions.

## Limits

- **Two seeds per arm, 125 steps.** One layer, basis and scale (fixed by the confirmation).
- **No learned latent policy.** C″ does not test whether heads that learn where to explore would
  help; Q3 of the diagnostics could not show that they can learn here.
- **The benchmark is one sampled answer per map.**
