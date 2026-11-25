# Fair-RL outcomes, seeds 42 and 43 (B′, C′, C″)

Per-map greedy outcomes and per-step gradient norms of the fair-RL runs, restored for
verification. They were removed from git in commit `a9db9a9` ("Keep run records out of git")
and are copied here, unchanged in content, from its parent: `fair-rl-7142592` (B′ and C′,
seeds 42 and 43) and `fair-rl-n-7154355` (C″, seeds 42 and 43).

- `greedy-outcomes.jsonl`: one row per arm, seed and map (840 maps each): `arm` (`b` B′,
  `c` C′ with its mean latent, `n` C″), `seed`, `population`, `row_index`, `route`,
  `success`. C′ uses `greedy-mean`, as its primary reading did.
- `grad-norms.jsonl`: each training step's total gradient norm before the global clip at 1.0.
- `comparison-*.json`: each run's own recorded readout.

**Reproduce:** `python scripts/verify_fair_rl_outcomes.py` recomputes both primary readings
with `spara.fair_rl.sign_test` on the 240 primary maps and checks them against the recorded
readouts (`tests/test_rebuttal_evidence.py` runs it):

| | First arm | B′ | Maps first ahead | B′ ahead | p |
| --- | --- | --- | --- | --- | --- |
| C′ against B′ | 77, 86 | 92, 96 | 12 | 34 | 0.0016 |
| C″ against B′ | 98, 98 | 92, 96 | 22 | 15 | 0.32 |

Median gradient norm (clip factor): B′ 5.4 and 6.1 (0.19, 0.16); C′ 1,133 and 1,128
(0.0009); C″ 3.8 and 3.6 (0.26, 0.28).

**Not here:** seeds 44 to 47 of the six-seed C″ comparison (jobs 7175728–7175735 and
7178552–7178555) are on a teammate's PACE storage; their counts are in
`docs/LATENT_DIAGNOSTICS_SPEC.md`.
