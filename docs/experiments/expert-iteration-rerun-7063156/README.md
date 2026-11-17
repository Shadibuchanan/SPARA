# Evidence for the expert iteration round 1 rerun (jobs 7063156–7063729)

Read in the [expert iteration spec](../../EXPERT_ITERATION_SPEC.md#rerun-jobs-70631567063729).
This reruns [round 1](../expert-iteration-6961874/README.md) from LIVR 6189457, so that
round 2 has E's weights to start from.

| Folder | Job | Contents | Elapsed |
| --- | --- | --- | --- |
| `routes/` | 7063156 | Checks, the 97 detour maps, the 500 pool maps | 2,077 s |
| `l/` | 7063157 | LIVR 6189457's greedy answers on the 240 held-out maps | 161 s |
| `e/` | 7063729 | Fine-tune on found routes; greedy; benchmark | 1,072 s |
| `m/` | 7063168 | Fine-tune on the same maps' labels; greedy | 1,092 s |

- **`comparison.json`:** the output of `spara.expert_iteration.compare`.
- **V500 was not run.** The first run showed it ties M.
- **E's final weights** stay on PACE scratch for round 2; M's were not saved.
- **Paths** on PACE are written as `$SPARA_REMOTE_ROOT`.
- **In git:** this README and the readout JSON only. The run folders named above stay local.
