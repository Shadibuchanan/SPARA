# PACE-ICE execution context

Read this before connecting to PACE-ICE, preparing jobs, or handling remote artifacts.

## Approved access and storage

The user authorizes agent access through the local SSH alias `pace-ice`, using
existing SSH configuration and identity-file authentication. This alias belongs
to this workstation/account; teammates configure their own target. Scripts must
accept a target and remote root rather than embed a username or identity-file path.
Recommended parameter names are `SPARA_SSH_TARGET` and `SPARA_REMOTE_ROOT`.

Resolve the default root as `$HOME/scratch/SPARA` on the remote host. It existed
empty and writable at the initial 2026-10-11 inspection. The agent created `logs/`
and `results/` and saved `logs/pace-preflight-2026-10-11.json`. Keep subsequent
run outputs, checkpoints and caches under this root as needed.

Use the login node for lightweight coordination, Slurm for GPU computation,
and SSH to inspect job logs. No GPU job has been submitted by this planning work.

## Pilot budget and selection

The user accepted one GPU for at most one allocated hour after environment/data
preparation. Count consumed time across retries rather than treating every retry
as a fresh allowance. Queue wait is separate from allocated GPU time. The full-run budget approved below is separate from this pilot allowance.

Select available, permitted hardware compatible with the pinned stack and full
training memory needs; no particular GPU model is required. Recheck availability
before submission, and use one appropriate request rather than competing duplicate
reservations. An idle AMD or older low-memory GPU is not automatically suitable
for the intended CUDA/BF16 run.

## Observed account state, not team defaults

Read-only inspection on 2026-10-11 established:

- The SSH alias connects to an ICE login node without a password prompt.
- The current user has Slurm account `coc` and QoS `coc-ice`.
- Partition maximum walltime is 16 hours for `ice-gpu`/`coc-gpu` and 18 hours for
  `ice-bw-gpu`. The approved pilot still has a one-hour ceiling.
- Visible GPU families include A40, A100 with 40/80 GB feature labels, L40S,
  H100/H200, older V100/RTX6000, and RTX Pro Blackwell. Verify actual allocated
  hardware, VRAM and compatibility inside the job. Visibility and allow-lists
  are not guarantees of scheduler admission or immediate availability.
- The scratch quota query reports a 300 GB hard limit, about 157.2 GB used, and
  a one-million-file limit. Shared-filesystem free space is not the user's quota.
- No jobs were active for this user at inspection; availability changes over time.

Scratch retention and durable backup policy remain unverified. Before large runs,
recheck quota and establish a durable export destination for completed results
and reproducibility artifacts. Keep these account observations separate from
portable team configuration.

## Implementation pilot, 2026-10-11

The [pilot report](../LIVR_PILOT_REPORT.md) records successful checks and all
preceding numerical-validation failures. Six one-H200 allocations consumed
818 seconds total; `logs/pilot-budget.json` records their IDs and elapsed time.
Do not reset the one-hour allowance when retrying. No jobs remain active from
this pilot. The user subsequently approved the full-training allocation below.

The measured configuration peaks near 55.2 GiB allocated GPU memory, so target
80 GB or larger compatible GPUs. The user approved one H200, four CPUs,
96 GiB RAM and a four-GPU-hour ceiling on 2026-10-11. Job `6189457` completed
the unchanged 10+10-epoch recipe and reference evaluation in 5,664 GPU-seconds,
with Slurm exit code `0:0`. The score is 323/400 (80.75%), verified by local
reference rescoring. Stage 1 backup is fully verified; Stage 2 transfer remains
in progress. See [the results report](../LIVR_BASELINE_RESULTS.md).
`logs/baseline-budget.json` records this separate 14,400-GPU-second allowance;
count any retries cumulatively. Scratch usage before submission was 209.5/300 GiB,
and local free space was about 209 GiB. Export its
completed results/checkpoints to the local workstation and verify hashes so
durability does not depend on scratch retention. Recheck local free space and
remote quota before starting; dispose of only this task's verified duplicate
pilot checkpoint payloads when reclaiming space.
