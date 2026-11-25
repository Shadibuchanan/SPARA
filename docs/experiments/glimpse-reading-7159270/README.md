# Evidence for the reading rows with glimpses (job 7159270)

Specified and read in the [hole check spec](../../HOLE_CHECK_SPEC.md#result-job-7159270).
With a label on every cell, do glimpse latents read the holes around a cell that only its
coordinates name?

- **Starting point:** B (job 7088134) from
  [round 3's evidence](../expert-iteration-round3-7088134/README.md). The readout sets
  G3R's probe beside S3R's in
  [the reading rows' evidence](../hole-check-reading-7132192/README.md).
- **Job:** one H200, 2026-11-18, 2,631 seconds by the job's own clock.

| Path | Contents |
| --- | --- |
| `g3r/` | G3R: the probe and the blind probe (with the answer text), training steps, the manifest |
| `glimpse_reading.json` | The output of `spara.expert_round2.glimpse_reading`, with every fixed reading |

- **No route phases and no benchmark** were run, and no weights were saved.
- **Post hoc figures** in the spec (identical answers between the two probes, the loss by
  eighths) come from the files in `g3r/` and are not in the readout.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folder named above stays local.
