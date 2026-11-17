# Evidence for the hole check on the exact map (jobs 7125957 and 7125958)

Specified and read in the [spec](../../HOLE_CHECK_TEXT_SPEC.md#result-jobs-7125957-and-7125958).
Does the hole check work once the map is also given exactly, as text?

- **Starting point:** B (job 7088134) from
  [round 3's evidence](../expert-iteration-round3-7088134/README.md). The readout also
  compares with S3 in [the hole check's evidence](../hole-check-7108600/README.md) and
  with EB3 in round 3's.
- **Jobs:** one H200 each, 2026-11-16.

| Path | Job | Contents | Seconds |
| --- | --- | --- | --- |
| `s3t/` | 7125957 | S3T: S3 with the text map in every prompt; B's zero-shot answers, greedy with the answer text, training steps | 2,218 |
| `o3/` | 7125958 | O3: EB3 with the text map in every prompt; the same | 1,378 |
| `text_check.json` | | The output of `spara.expert_round2.text_check`, with every reading | |

- **No masked read, latent states, benchmark or weights.** Neither arm is deployable,
  because the prompt holds the layout.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
