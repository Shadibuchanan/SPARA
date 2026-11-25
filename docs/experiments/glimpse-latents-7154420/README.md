# Evidence for the glimpse latents (job 7154420)

Specified and read in the [hole check spec](../../HOLE_CHECK_SPEC.md#result-job-7154420).
With four latents after each line's cell, each masked to the neighbouring cell in its
direction, do the hole check's lines read the map, and do the routes get safer?

- **Starting point:** B (job 7088134) from
  [round 3's evidence](../expert-iteration-round3-7088134/README.md). The readout compares
  G3 with EB3 there and with S3 in [the hole check's evidence](../hole-check-7108600/README.md).
- **Job:** one H200, 2026-11-18, 5,155 seconds by the job's own clock.

| Path | Contents |
| --- | --- |
| `g3/` | G3: greedy (with the answer text), the blind pass, masked greedy, training steps, benchmark outcomes |
| `glimpse_check.json` | The output of `spara.expert_round2.glimpse_check`, with every fixed reading |

- **Benchmark outcomes** keep the row, grid size, success, extracted answer and status.
  Prompts and raw text are dropped.
- **Not committed:** `states.pt` (latent states). No weights were saved.
- **The mask check on real images** in the spec is post hoc, from a scratch script, and is
  not in the readout.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folder named above stays local.
