# Evidence for gate 3, amended (CPU jobs 7117800 and 7118222)

This is gate 3 of the fixed-target retry, rerun after
[the amendment](../../PERCEPTION_JEPA_SPEC.md#amendment-after-gate-3-2026-11-16).
- **The change:** the heads standardise each latent channel before the context projector.
- **Inputs:** the same saved features, from job 7116225.
- **The original gate 3** (job 7116395) failed; see
  [its evidence](../jepa-fixed-gate-7116395/README.md).

- **Compute:** one `ice-cpu` node, 4 min 30 s. No GPU.
- **Standardisation:** fitted on E's latents over the 1,486 training maps, per latent position and
  channel.
- **Source:** the hashes of the files it reads were checked against the local copies before
  submission.

## Result: gates passed

**3a and 3b** are unchanged: holes read from the targets at 1.00 at both sizes; the hole label
explains 21% and 23% of their variance.

**3c, amended** (shuffled gap at least 0.02 at both sizes):

| Held-out maps | 7 x 7 | 8 x 8 |
| --- | --- | --- |
| Shuffled gap | 0.278 | 0.027 |
| Share of target variance unexplained | 0.871 | 0.993 |
| Hole-direction share explained (descriptive) | 0.118 | 0.005 |
| Linear ceiling's hole-direction share | 0.121 | 0.061 |
| Holes from the predictions (no longer a gate) | 0.52 | 0.50 |

- **The warm start:** validation error from 1.003 to 0.947, at the last-but-one epoch (29 of 30).
- **Against the post hoc diagnosis** (gap 0.26 and 0.038; hole-direction share 0.107 and 0.008),
  these are close. The diagnosis standardised the latents outside the heads and stored them in half
  precision.
- **The 8 x 8 margin is slim:** 0.027 against 0.02. At 8 x 8 the heads capture almost none of the
  hole direction the latents allow.

## Rerun on re-keyed features (CPU job 7118222)

The first EJF job (7117889) showed that map IDs repeat within the released training split; see
[the amendment](../../PERCEPTION_JEPA_SPEC.md#amendment-after-gate-3-2026-11-16).
- **Re-keying:** the saved features were re-keyed by split and image on CPU, with
  `scripts/rekey_features.py` (rekey.json).
  - All 2,185 saved maps were found again by ID and layout. The tensors are unchanged.
  - 174 of the released training split's 1,000 rows share an ID.
- **The rerun:** gate 3 on the re-keyed features, one `ice-cpu` node, 2 min 29 s. It passed.
- **Identical to job 7117800:** the targets' checks, the warm start's history, 3c and the linear
  ceiling.
- **Only the keys differ.** EJF uses this run's heads.

## Files

| File | Contents |
| --- | --- |
| gate-summary.json | Gate 3's output: the targets' checks, the warm start's history, 3c with the hole-direction shares, the linear ceiling, the reading |
| gate-summary-7118222.json | The same, rerun on the re-keyed features |
| rekey.json | The re-keying: source file and hash, maps, rows sharing an ID per split |
