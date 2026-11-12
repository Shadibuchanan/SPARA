# Evidence for steps 1a′ and 1a″: search without an exact step simulator (CPU)

Step 1a′ of the [guided-search spec](../../GUIDED_SEARCH_SPEC.md#step-1a-search-without-an-exact-step-simulator-cpu-fixed-before-the-run).
How does search cost change when the exact step simulator is replaced by nothing (final
verdict only) or by a judge that errs?

- **Maps:** the 97 unseen detour maps (40 test, 57 `fresh/probe`).
- **Priors:** the two model-free priors. No model and no GPU.
- **Run:** CPU only, on 2026-11-10, in 42 seconds; the exploratory variant took 30
  seconds more. Reproduce with
  `PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_sweep.py --data-root data/vsp --output DIR`;
  add `--modes in_grid` for the exploratory variant.
- **Reproducible:** a second run of the final script gave an identical summary (timings
  aside).

## Validity: all checks passed

- **Final verdict only:** every accepted answer passes the reference evaluator, and
  popped costs never decrease, for both priors.
- **Noisy judge:** at ε = 0, both noise structures reproduce the exact search on all 97
  maps, with the same routes and policy calls. Step 1a's dry run also reproduces
  unchanged after the stand-in fix:
  - uniform 9 / 18 / 25 calls;
  - toward-goal 6 / 11 / 20 calls;
  - 6,892 of 6,892 simulator outcomes agree.
- **Tests:** determinism, nesting across rates and budget independence are covered in
  `tests/test_search.py` and `tests/test_route_search.py`.

## Mode 1: final verdict only

| Prior | Solved within 64 / 1,000 / 20,000 calls | Policy calls, median / p90 | Verifier calls on solved maps, median / p90 |
| --- | --- | --- | --- |
| Exact judge, toward-goal (step 1a) | 97 / 97 / 97 | 6 / 11 | 1 |
| Uniform | 0 / 9 / 45 | 20,001 / 20,001 | 1,304 / 3,062 |
| Toward-goal | 0 / 45 / 72 | 1,455 / 20,001 | 181 / 1,823 |

By grid size, solved within 20,000 calls:

| Prior | 3 x 3 (2) | 4 x 4 (10) | 5 x 5 (14) | 6 x 6 (36) | 7 x 7 (18) | 8 x 8 (17) |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 2 | 7 | 11 | 18 | 6 | 1 |
| Toward-goal | 1 | 7 | 14 | 25 | 13 | 12 |

Unsolved maps all stopped at the budget.

**Exploratory, added after the run: priors that rule out moves off the grid.**
- **Why.** The toward-goal prior left one 3 x 3 and three 4 x 4 maps unsolved. The
  cause is that both fixed priors make moves off the grid cheap. Such a move leaves the
  agent in place and costs the toward-goal prior only e⁻³; at a corner goal it ties
  with ending.
- **What it did.** The search padded refused routes with these moves. On test/168
  (3 x 3), 7,379 of the 7,918 refused answers contain three or more of them.
- **The variant.** It gives those moves a score of −1,000.

| Prior, off-grid moves ruled out | Solved within 64 / 1,000 / 20,000 | Policy calls, median / p90 | Verifier calls on solved, median / p90 |
| --- | --- | --- | --- |
| Uniform | 0 / 14 / 60 | 9,151 / 20,001 | 977 / 4,370 |
| Toward-goal | 9 / 67 / 90 | 329 / 10,044 | 108 / 2,249 |

Even without the padding, the toward-goal prior needs a median of 329 policy calls
against 6 with the exact judge.

## Mode 2: noisy judge

ε₉₀ is the largest swept ε such that, at it and at every smaller rate, the search
solves at least 90% of maps within 64 policy calls (mean over 5 seeds). It is the same
for both priors and at both budgets (64 and 1,000). Per-cell misses sit on the
threshold: at ε = 0.1 the means are 88.2 and 87.6 maps against the 87.3 required.

| Structure | Misses | False alarms | Both |
| --- | --- | --- | --- |
| Per cell (misread map) | 0.1 | 0.05 | 0.02 |
| Per transition | 0.05 | 0.05 | 0.02 |

Maps solved within 64 calls, toward-goal prior (mean of 5 seeds; uniform is within 4
maps of it everywhere):

| ε | Cell: misses | Cell: false alarms | Cell: both | Transition: misses | Transition: false alarms | Transition: both |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 97 | 97 | 97 | 97 | 97 | 97 |
| 0.01 | 96.0 | 95.0 | 94.0 | 94.2 | 96.2 | 93.4 |
| 0.02 | 95.6 | 93.6 | 92.2 | 92.8 | 95.8 | 92.0 |
| 0.05 | 92.6 | 88.0 | 83.6 | 89.8 | 92.2 | 85.2 |
| 0.1 | 88.2 | 75.6 | 67.6 | 84.0 | 80.4 | 69.6 |
| 0.2 | 79.6 | 50.6 | 39.2 | 72.0 | 56.6 | 43.2 |
| 0.3 | 70.0 | 28.4 | 21.8 | 63.8 | 37.0 | 26.0 |
| 0.5 | 57.0 | 5.8 | 3.6 | 51.2 | 8.2 | 6.4 |

**No search ran out of budget.**
- **Budget makes no difference.** Every condition solves the same maps within 1,000
  calls as within 64.
- **Every unsolved search emptied its frontier.** By construction, misses can only
  poison (they never rule out a safe move) and false alarms can only block (they never
  hide a hole). The finding is how often each happens:
  - **Misses:** at ε = 0.5, 40–46 maps poisoned (the search closed a state it had
    reached through a missed hole). Misses also cost refused answers, at most about 1.7
    per map.
  - **False alarms:** at a per-transition rate of 0.1, 16.6 maps per seed blocked (the
    judge ruled out every truly safe route).

**Anchor point:** per transition, misses at 0.5 and false alarms at 0.1. These are
roughly the step-grounding probe's rates on G's states.

| Prior | Solved within 64 (5 seeds) | Unsolved: poisoned / blocked (of 485 searches) |
| --- | --- | --- |
| Uniform | 42.6 (42–44) | 189 / 83 |
| Toward-goal | 39.8 (37–44) | 203 / 83 |

## Post-hoc observations (not readings)

- **Even a poor judge beats none at a small budget.** At the anchor, about 40 maps are
  solved within 64 calls, against 0 (or 9, with off-grid moves ruled out) with the
  final verdict only.
- **A poor judge loses at a large budget.** Hard pruning by a noisy judge caps success,
  because more budget does not help. Final-verdict-only search keeps improving with
  budget: it solves 72 maps within 20,000 calls, or 90 with off-grid moves ruled out.
- **The losses under misses come from the search, not from the judge.** A search that
  stopped trusting closed states after a refused answer would turn misses into extra
  verifier calls rather than failures.
- **False alarms block because pruning is absolute.** A judge used as a penalty on a
  move, rather than a veto, would let the search recover.
- **The prior barely matters once the judge errs.** Uniform and toward-goal are within
  4 maps of each other everywhere in mode 2.

## Step 1a″: a search that tolerates a noisy judge

[Step 1a″](../../GUIDED_SEARCH_SPEC.md#step-1a-a-search-that-tolerates-a-noisy-judge-cpu-fixed-before-the-run)
reran mode 2 with seven search variants against the hard veto:
- re-opening states after a refused answer;
- a soft veto at λ = 1, 3 and 10 nats;
- each λ with re-opening.

The readings and tables are in the spec's
[step 1a″ result](../../GUIDED_SEARCH_SPEC.md#step-1a-result-cpu-2026-11-11).

- **Run:** CPU only, on 2026-11-11, in 7,832 seconds. Reproduce with
  `PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_sweep.py --data-root data/vsp --output DIR --modes robust`.
- **Validity:** all checks passed.
  - The hard veto reproduces this folder's summary.json exactly, timings aside.
  - Re-opening, soft veto λ = 10, and both reproduce step 1a at ε = 0, with the same
    routes and policy calls on all 97 maps.
  - No accepted answer fails the reference evaluator.
- **The carried setting.** robust-summary.json's `carried` field is the run's output,
  `"variant": "reopen"`. That is list order: the registered rule tied re-opening, soft
  veto λ = 10, and both.
  - The current `carried()` in `scripts/search_sweep.py`, applied to the same file,
    reports the tie and breaks it by the whole solved-within-1,000 curve.
  - It gives soft veto λ = 10 with re-opening. That tie-break was added after the run.

**Guard check (CPU, after the sweep, before the JEPA job).** The JEPA job's search
first had a guard of 1,280 expansions. This check reran the tied variants with the
toward-goal prior and a per-transition noisy judge with both error types. Settings:
budget 64, depth 64, seeds 0–4, ε from 0 to 0.5.
- **1,280 lost one or two maps in some seeds** for λ = 10 with re-opening, at every
  ε from 0.02 to 0.5. Re-opening alone and λ = 10 alone matched the sweep at 1,280
  (checked up to ε = 0.1).
- **50,000, the sweep's guard, reproduces the sweep exactly.**
- **Passes stay low even so.** Re-expansions reuse cached prefixes, so a larger guard
  adds almost no LIVR passes. Distinct prefixes per map: mean 10.8 at ε = 0 to 39.1 at
  ε = 0.5, at most 288.
- **The JEPA job uses 50,000.**

## Files

| File | Contents |
| --- | --- |
| summary.json | Both modes, as fixed in the spec: per prior and condition, solved within each budget (per seed for mode 2), policy and verifier calls, refused answers, stop reasons, ε₉₀, the anchor point, and validity |
| verdict-only-maps.jsonl | Mode 1, one row per prior and map: solved, policy and verifier calls, stop, route length |
| exploratory-in-grid.json | The exploratory variant of mode 1, in the same format |
| exploratory-in-grid-maps.jsonl | Its per-map rows |
| robust-summary.json | Step 1a″: for each variant (veto, re-opening), mode 2's full report per prior: the exact-judge baseline, zero-noise reproduction checks, every condition's curve and ε₉₀, and the anchor point. Plus the run's `carried` field (see above) |
| guard-check.json | The guard check: solved maps per seed under each guard against the sweep, stop reasons, and distinct prefixes per map |
