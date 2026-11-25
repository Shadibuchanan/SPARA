# Evidence for the reading rows (job 7132192)

Specified and read in the [hole check spec](../../HOLE_CHECK_SPEC.md#result-job-7132192).
Can the model learn to list the holes next to a cell when every cell is labelled, and does
the hole check then help?

- **Starting point:** B (job 7088134) from
  [round 3's evidence](../expert-iteration-round3-7088134/README.md). The readout compares
  S3R with EB3 there and with S3 in [the hole check's evidence](../hole-check-7108600/README.md).
- **Job:** one H200, 2026-11-17, 5,701 seconds by the job's own clock.

| Path | Contents |
| --- | --- |
| `s3r/` | S3R: a phase of reading rows, then S3's recipe; the reading probe after phase 0 and after phase 2, greedy (with the answer text), masked greedy, training steps, benchmark outcomes |
| `reading_check.json` | The output of `spara.expert_round2.reading_check`, with every fixed reading |

- **Benchmark outcomes** keep the row, grid size, success, extracted answer and status.
  Prompts and raw text are dropped.
- **Not committed:** `states.pt` (latent states). No weights were saved.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folder named above stays local.
