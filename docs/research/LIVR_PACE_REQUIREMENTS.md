# PACE-ICE requirements for the LIVR pilot

Research date: 2026-10-11. Resolves the public-documentation investigation in [issue #3](https://github.com/Shadibuchanan/SPARA/issues/3). No cluster connection, job submission, environment installation, or allocation inspection was performed.

## Conclusion

PACE-ICE is the agreed execution platform. Prepare the full-fine-tuning pilot, but select its GPU request, walltime, storage location, and software environment from the user's actual ICE access and measured behavior. Public inventory does not establish account entitlement, free capacity, queue delay, or that this model fits a particular GPU. Do not substitute Phoenix account/QoS examples for ICE configuration.

## Official evidence and its limits

| Topic | Verified evidence | Consequence for this project |
|---|---|---|
| Access | ICE is an instructional cluster; faculty request access for courses. College of Computing or cross-listed course requests go through its academic resources route. Terminal and Open OnDemand access are supported. [ICE overview](https://pace.gatech.edu/ice-cluster/), [PACE participation](https://pace.gatech.edu/participation/) | Confirm the student's course/project entitlement and access end date. General GT membership does not establish that this account has ICE access. |
| Connectivity | Official ICE workshop instructions require GT VPN unless using a wired campus connection, and identify `https://ondemand-ice.pace.gatech.edu/`. [PACE Python workshop connection guide](https://docs.pace.gatech.edu/training/img/py101-connect.pdf) | The human operator should establish their own authenticated connection. Credentials are not needed in this repository or research note. |
| Scheduler | PACE confirms ICE migrated to Slurm in 2023; its May 2026 maintenance notice lists a Slurm 25.11 upgrade. [Migration confirmation](https://pace.gatech.edu/2023/08/31/upcoming-firebird-slurm-migration-announcement/), [May 2026 maintenance notice](https://pace.gatech.edu/posts/page/2/) | Prepare for Slurm, not old PBS/Moab tutorials. Exact allowed partition, QoS, GPU count, concurrent-job limits and maximum walltime remain unverified. |
| GPUs and software compatibility | A 2025 official facilities document lists H100/H200 alongside older GPU families. The May 2026 refresh notice says ICE V100-32GB nodes will be removed and V100-16GB/Quadro RTX 6000 GPUs removed from the default GPU pool because they require older CUDA. [Facilities document, section 1.2.3](https://pace.gatech.edu/files/2025/05/GT_Facilities_Latest.pdf), [2026 refresh notice](https://pace.gatech.edu/posts/page/2/) | Historical inventory is not a current scheduling guarantee. Record the allocated GPU model, VRAM, driver, CUDA runtime, and package versions. Do not select an old CUDA recipe merely because an older ICE example uses it. |
| Current authoritative instructions | The ICE overview links to ServiceNow articles for [Getting Started on ICE, KB0042102](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102), [ICE Resources, KB0042095](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095), and [Open OnDemand, KB0042133](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133). The older documentation explicitly says content moved to ServiceNow. [Migration notice](https://docs.pace.gatech.edu/faqs/accessingClusters/) | The relevant current articles could not be read in this session: browser fetch failed, and direct requests returned a GT SSO/guest redirect. Therefore no current numerical quota, storage retention, module version, or scheduling limit is asserted here. |

The 2023 ICE orientation predates the storage/scheduler migration and the 2026 hardware refresh. Its historical 15 GB home quota is **not** a verified current quota. Confirm storage capacity and retention from the current resource article and account inspection before downloading weights or saving optimizer checkpoints. [Historical orientation](https://docs.pace.gatech.edu/ice_cluster/img/ICE_orientation_spring2023.pdf), [May 2023 upgrade notice](https://pace.gatech.edu/posts/?m=202304)

## Account preflight needed before execution

The operator should record the following nonsecret facts in the eventual run manifest. These are verification requirements, not claimed findings:

- Course/project access, its expiration date, allowed Slurm account/partition/QoS, GPU constraints and maximum job/concurrency limits.
- Available GPU model/VRAM under that entitlement; allocated CPU cores and host memory. Queue visibility alone does not prove permission to request a resource.
- Writable paths and current byte/inode quotas; scratch purge/backup policy; durable destination for checkpoints, manifests, and logs. Check model cache, environment, and optimizer-state storage as well as the small VSP dataset.
- Permitted environment mechanism and compatible modules, driver/runtime, Python, PyTorch, Transformers, and attention implementation. Record exact versions rather than relying on mutable defaults.
- Data/model download route, any compute-node network restrictions, and checkpoint transfer path.

Current documentation and the account can settle these facts; the user does not need to guess them in advance.

## Recommended measured pilot (proposed procedure, not cluster policy)

The target supplied by this project's baseline plan is Qwen2.5-VL-3B, four latent tokens, frozen vision encoder/projector, full language-model training, and two ten-epoch stages on VSP. Resource optimization must preserve that method and its effective batch size; a LoRA substitution would be a separate scope decision.

1. After account preflight, start with one eligible high-memory GPU, if permitted, and microbatch one. Keep effective batch eight through accumulation. Pin processor/image limits and inspect training-only image/token-length extremes before profiling; do not tune on test outcomes.
2. Exercise both the Stage 1 restricted attention mask and Stage 2 ordinary causal attention. Include initialization, forward/backward, an optimizer step, checkpoint save and reload, and a short decoding pass. An inference-only memory measurement cannot establish full-training feasibility.
3. After warmup, measure a bounded sample of optimizer steps across typical and longest training examples. Record peak allocated/reserved GPU memory, host-memory peak, tokens/images per example, step time, checkpoint bytes/time and runtime settings. Report the actual attention backend; a custom mask can change memory and kernel behavior.
4. If the run fits, extrapolate each stage separately, including evaluation and checkpoint overhead. With 1,000 training examples and effective batch eight, the nominal schedule is 125 optimizer steps per epoch and 1,250 per stage, subject to exact sampler/batching behavior. Compute GPU-hours from measured elapsed time times allocated GPUs; distinguish queue wait from runtime.
5. If it does not fit, measure activation checkpointing or optimizer/parameter sharding while preserving the training objective and batch semantics. Do not silently reduce image resolution, truncate labels, change the optimizer, or adopt LoRA. Report a resource blocker if fidelity cannot be maintained within the agreed budget.
6. Check that save/resume works within the verified walltime and storage constraints before a full run. Retain a concise feasibility report with the chosen request, evidence, estimated full-run cost, and unresolved assumptions.

## Minimal human decisions still needed

1. **Execution responsibility:** who will authenticate, run the account preflight, submit/monitor the pilot, and return its logs? Recommended initial workflow: user submits the prepared job through their own ICE session; the agent prepares and reviews the artifacts.
2. **Pilot ceiling:** maximum GPU count, total allocated GPU-hours, and deadline. A proposed initial ceiling is one GPU for one hour after environment/data preparation, if ICE entitlement permits; this is a budgeting suggestion, not a prediction that the pilot or full training will finish in that time. Expansion after a failure needs an explicit budget decision.

Full-training budget and feasibility remain unmeasured. This research establishes how to obtain them; it does not authorize or claim cluster execution.
