# Evidence for expert iteration round 2 (jobs 7063726–7065176)

Specified and read in the [round 2 spec](../../EXPERT_ITERATION_ROUND2_SPEC.md#result-jobs-70651687065176).
Does a second round of expert iteration carry to 7 x 7 and 8 x 8 maps, and does training
under LIVR's Stage 1 mask add to it?

- **Starting point:** E from the [round 1 rerun](../expert-iteration-rerun-7063156/README.md)
  (job 7063729).
- **Jobs:** one H200 each, except the batching check on `ice-cpu`. All completed on the
  first submission, 2026-11-13 to 2026-11-14.

| Folder | Job | Contents | Elapsed |
| --- | --- | --- | --- |
| `livr/` | 7063726 | LIVR 6189457's greedy answers on the 400 new held-out maps | 241 s |
| `batching/` | 7064652 | Batched against single policy reads of E, fp32 and bf16, on CPU | 1,595 s (CPU) |
| `search/` | 7065168 | Checks; E's greedy and masked greedy; frontier and new-pool search | 3,529 s |
| `e2/` | 7065175 | Two unmasked phases; greedy, masked greedy, benchmark | 1,718 s |
| `eb2/` | 7065176 | Phase 1 under the Stage 1 mask, phase 2 unmasked; the same evaluation | 1,748 s |

- **`comparison.json`:** the output of `spara.expert_round2.compare`, with every reading.
- **Benchmark outcomes** keep the row, grid size, success, extracted answer and status.
  Prompts and raw text are dropped.
- **Not committed:** each run's `states.pt` (latent states, 43 MB) and the arms' weights,
  which were not saved. The states are in the local, git-ignored `results/ei-round2/`.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **Regenerate** with `scripts/export_expert_evidence.py`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
