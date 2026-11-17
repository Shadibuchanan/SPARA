# Evidence for the hole check (job 7108600)

Specified and read in the [hole check spec](../../HOLE_CHECK_SPEC.md#result-job-7108600).
Does writing the holes around each cell before each move help the policy?

- **Starting point:** B (job 7088134) and EB3 (job 7088136) from
  [round 3's evidence](../expert-iteration-round3-7088134/README.md). The readout compares
  S3 with EB3's and B's records there.
- **Job:** one H200, 2026-11-16, 3,406 seconds by the job's own clock.

| Path | Contents |
| --- | --- |
| `s3/` | S3: EB3's recipe from B with the hole check in every answer; greedy (with the answer text), masked greedy, training steps, benchmark outcomes |
| `hole_check.json` | The output of `spara.expert_round2.hole_check`, with every fixed reading |
| `reading.json` | Post hoc: how the lines read the map, from `scripts/hole_reading.py` |

- **Benchmark outcomes** keep the row, grid size, success, extracted answer and status.
  Prompts and raw text are dropped.
- **Not committed:** `states.pt` (latent states) and the weights. S3's weights are on PACE
  under `results/round3-s3-7108600/phase2/final`. The states are in the local, git-ignored
  `results/hole-check/`.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py` and `scripts/hole_reading.py`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
