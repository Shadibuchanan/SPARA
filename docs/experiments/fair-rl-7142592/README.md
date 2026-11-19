# B′ and C′ under fair conditions (jobs 7142592, 7143007, 7143008)

The readings fixed in [STRUCTURED_EXPLORATION_SPEC.md](../../STRUCTURED_EXPLORATION_SPEC.md),
section "Amendment: rerun under fair conditions (2026-11-17)", applied by `fair_rl.compare`.
`comparison.json` holds every reading below; each run's folder holds its settings, step log,
greedy answers and benchmark answers. Weights stay on PACE scratch.

## What ran, and what changed from the plan

| Job | What | GPU time |
| --- | --- | --- |
| 7142592 | Pilot, B′ seed 42, then C′ seed 43 (claimed after the other jobs took C′ 42 and B′ 43) | 4 h 39 min |
| 7143007 | C′ seed 42 | 2 h 13 min |
| 7143008 | B′ seed 43 | 1 h 55 min |
| 7145429 | LIVR's benchmark on the arms' stack | 4 min |
| 7135654 | An earlier Blackwell attempt, cancelled 2 minutes into its first run (its pilot matched this one) | 31 min |

- **Hardware and memory.** All runs used one 48 GB MIG slice of an RTX PRO 6000 Blackwell
  (2g.48gb), PyTorch 2.7.1 + CUDA 12.8, transformers 4.51.3. B and C peaked at about 70 GiB on
  H200, so the runs used `fair-rl --low-memory`: AdamW's master weights and moments on the CPU
  (the same update; `tests/test_offloaded_optimizer.py`) and the KL reference in bfloat16. Peak
  GPU memory was 38.7–38.8 GiB. Both arms ran the same way, so B′ against C′ is unaffected; the
  bfloat16 reference changes only the KL term (weight 0.001), and the run settings record it.
- **LIVR on the same stack.** Its benchmark gives 328 of 400 here, as on H200; the two disagree
  on 2 maps. The benchmark readings use this run's answers.
- **The pilot** ran in job 7142592 on the same stack. Its rules gave 24 maps per step and C′'s
  scale c = 4; the mixed share was 0.175, nothing reached the token cap, and the sampler
  matched the loss exactly (largest total variation 0.0).

## Validity: all checks passed

- Pilot set and sampler checks; all four runs completed 125 steps.
- Pairing: within each seed, B′ and C′ trained on the same maps in the same order.
- C′'s untrained heads left LIVR's routes unchanged on 20 pool maps (both seeds).
- Rescoring: all 96,000 rollout rewards match a CPU rescore with the reference evaluator.
- Policy moved: the runs differ from LIVR on 73, 70, 66 and 63 of the 200 pilot maps
  (threshold 16), so no run was extended.

## Primary: C′ is worse than B′

On round 1's 240 held-out detour maps (greedy; C′ with δ = μ), summing each map's successes
over the two seeds:

| | Seed 42 | Seed 43 |
| --- | --- | --- |
| B′ | 92 | 96 |
| C′ | 77 | 86 |
| LIVR | 42 | 42 |

B′ is ahead on 34 maps and C′ on 12 (exact sign test, p = 0.0016), and both seeds point the
same way. The fixed reading: **"C′ worse: latent exploration hurt in this recipe."**

## Secondary

- **Transfer to the 400 held-out 7 x 7 and 8 x 8 maps:** B′ 73 and 67, C′ 61 and 54, LIVR 19.
  B′ ahead on 38 maps, C′ on 18 (p = 0.010). C′ worse here too.
- **δ = μ against δ = 0 (C′):** 77 against 78 (seed 42) and 86 against 85 (seed 43). The learned
  mean changes one map per seed: at test time C′ is its text policy. Sampled δ: 77 and 82.
- **Each arm against LIVR (240 maps):** every run gains, p < 10⁻⁵ (B′ 58 and 61 maps only it
  solves against 8 and 7; C′ 44 and 51 against 9 and 7).
- **The benchmark against LIVR (328 of 400), split by map kind:**

| Run | All 400 | Detour (40; LIVR 9) | No detour (360; LIVR 319) |
| --- | --- | --- | --- |
| B′ 42 | 305 (20 against 43, p = 0.005) | 18 (9 against 0) | 287 (11 against 43, p < 10⁻⁴) |
| B′ 43 | 298 (18 against 48, p = 0.0003) | 17 (9 against 1) | 281 (9 against 47, p < 10⁻⁶) |
| C′ 42 | 315 (17 against 30, p = 0.08) | 15 (6 against 0) | 300 (11 against 30, p = 0.004) |
| C′ 43 | 309 (15 against 34, p = 0.009) | 13 (6 against 2) | 296 (9 against 32, p = 0.0004) |

  Every run gains on detour maps and loses on no-detour maps, so all four lose on the benchmark
  overall. The pool holds only detour maps; the benchmark is 90% no-detour maps.

## Diagnostics

| | B′ 42 | B′ 43 | C′ 42 | C′ 43 |
| --- | --- | --- | --- | --- |
| Share of groups with mixed rewards in training | 0.29 | 0.26 | 0.33 | 0.33 |
| Mean KL to LIVR | 0.32 | 0.29 | 0.26 | 0.24 |
| σ, first to last | | | 4.00 → 3.86 | 4.00 → 4.13 |
| Mean \|μ\|, first to last | | | 0 → 0.11 | 0 → 0.15 |

- **C′'s noise found more signal and broke as much as it found.** Per seed, of 3,000 paired
  groups, C′ had mixed rewards in 201 and 223 groups where B′'s were all failures, and in 221
  and 240 where B′'s were all successes.
- **Its heads barely learned:** σ stayed near its start and μ near zero, as in C.
- **Its text policy moved less** (lower KL), which fits both its smaller detour gains and its
  smaller no-detour losses.

## What this says

- **Text-only RL works under fair conditions:** sampling from the full distribution, on detour
  maps, with uniform groups skipped, B′ more than doubles LIVR's held-out detour success (42 to
  92–96 of 240) in 125 steps. Round 1's expert iteration reached 88.
- **Structured latent exploration hurts in this recipe.** It adds reward variance, but most of
  it comes from breaking routes the text policy already solves, the heads do not learn to
  steer it, and the text policy learns less.
- **Detour-only training costs the easy maps,** for both arms. A pool or mix with no-detour
  maps is the obvious next change for any benchmark claim.

## Limits

- Two seeds per arm; one scale (c = 4, set by the pilot's rule) and one basis size (k = 16).
- The benchmark is one sampled answer per map.
- The low-memory mode and the Blackwell stack differ from B's and C's H200 runs; comparisons
  with those runs are indicative only. LIVR's numbers here come from the same stack.
