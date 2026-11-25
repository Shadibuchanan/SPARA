# Evidence for the oracle check (jobs 7154267 and 7155886)

Specified and read in the [hole check spec](../../HOLE_CHECK_SPEC.md#result-jobs-7154267-and-7155886).
Does a right reading of the holes make a safe route? S3's saved weights answer the 400
held-out maps line by line, with nothing forced, with the true hole lists written for
them, and with the cells written as well. Nothing is trained.

- **Weights:** S3 (job 7108600) from [the hole check's evidence](../hole-check-7108600/README.md).
  The readout compares the passes with S3's recorded answers there and with EB3 in
  [round 3's evidence](../expert-iteration-round3-7088134/README.md).
- **Jobs:** one H200 each, 2026-11-18. Job 7154267 reached its 90-minute limit with 396 of
  the last pass's 400 maps written. Job 7155886 resumed it and wrote the last 4 in 49
  seconds, which is the `seconds` in `oracle-summary.json`.

| Path | Contents |
| --- | --- |
| `oracle/` | The three passes (`oracle-free.jsonl`, `oracle-holes.jsonl`, `oracle-cell.jsonl`, each with the answer text), the summary and the manifest |
| `oracle_read.json` | The output of `spara.expert_round2.oracle_read`, with every fixed reading |

- **Validity:** three checks fail, so the fixed primary reading is withheld. The spec
  lists each and what was found.
- **Post hoc readings** in the spec (where stops-short routes end, steps into listed
  holes, capped answers that loop) came from scratch scripts and are not in the readout.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folder named above stays local.
