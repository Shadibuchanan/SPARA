# Evidence for expert iteration round 3 (jobs 7088134–7088136)

Specified and read in the [round 3 spec](../../EXPERT_ITERATION_ROUND3_SPEC.md#result-jobs-70881347088136).
Does a third round of expert iteration still add on 7 x 7 and 8 x 8 maps?

- **Starting point:** E from the [round 1 rerun](../expert-iteration-rerun-7063156/README.md)
  (job 7063729), and round 2's search 7065168 in
  [round 2's evidence](../expert-iteration-round2-7065168/README.md).
- **Jobs:** one H200 each, 2026-11-15. B's first submission (7086894) failed in its first
  second on a shell quoting error, before any work; the three jobs below completed on the
  resubmission.

| Folder | Job | Contents | Seconds |
| --- | --- | --- | --- |
| `b/` | 7088134 | B: EB2's recipe from E, rerun with its weights saved; greedy, masked greedy, benchmark | 1,635 |
| `search/` | 7088135 | Checks; B's greedy and masked greedy; the frontier and new-pool search with B | 6,202 |
| `eb3/` | 7088136 | EB3: EB2's recipe from B on rounds 1 to 3's routes; the same evaluation | 1,617 |

- **`round3.json`:** the output of `spara.expert_round2.round3`, with every reading.
- **The search ran one request per pass.** Batched policy reads differed from single passes
  by 0.060, above the 0.01 limit, so the search fell back as the spec allows.
- **Benchmark outcomes** keep the row, grid size, success, extracted answer and status.
  Prompts and raw text are dropped.
- **Not committed:** each run's `states.pt` (latent states) and the weights. B's and EB3's
  weights are on PACE, under `results/round3-eb2-7088134/phase2/final` and
  `results/round3-eb3-7088136/phase2/final`. The states are in the local, git-ignored
  `results/ei-round3/`.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
