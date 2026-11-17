# Evidence for the perception ceiling (jobs 7076349 and 7077465)

Specified and read in the [perception ceiling spec](../../PERCEPTION_CEILING_SPEC.md).
How much of round 2's failure on 7 x 7 and 8 x 8 maps is perception?

| Folder | Job | Contents | Elapsed |
| --- | --- | --- | --- |
| `o2/` | 7076349 | E2's recipe with the map also as text; E's zero-shot read with the text map | 1,553 s |
| `t2/` | 7077465 | The same with the image blanked; E's zero-shot read with a blank image | 1,484 s |

- **Both start from E** (job 7063729) and train on search 7065168's 3,000 rows, as E2
  (job 7065175, in [round 2's evidence](../expert-iteration-round2-7065168/README.md)).
- **`ceiling.json`:** the output of `spara.expert_round2.ceiling` with both arms.
- **`zero-shot.jsonl`:** E's greedy answers on the 400 new held-out maps with the arm's
  prompt, before training.
- **No weights were saved,** and there is no benchmark, masked read or latent states:
  these arms are diagnostics, not candidate models.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
