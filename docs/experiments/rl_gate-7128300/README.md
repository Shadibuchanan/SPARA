# Evidence for RL Feasibility Gate (Job 7128300)

Evaluates whether the fine-tuned model checkpoint on the rebalanced dataset (`results/rebalanced-7109628/stage2/final`) exhibits sufficient policy rollout diversity to benefit from Reinforcement Learning (GRPO / PPO).

## Setup & Execution
- **Job ID:** 7128300 on PACE-ICE (`ice-gpu`, NVIDIA H100 GPU)
- **Wall Time:** 4 min 52 s (19m28s CPU time)
- **Rollout Sampling:** $K = 8$ stochastic rollouts per map (`do_sample=True, temperature=0.7, top_p=0.9`) across 100 test maps.
- **Reference Grader:** `ReferenceEvaluator` with pinned `mathruler==0.1.0`.

## Results
- **Total Maps Evaluated:** 100 (800 total rollouts)
- **All-Success Groups:** 71 (71.0%) $\implies$ Zero intra-group reward variance (advantage = 0)
- **All-Failure Groups:** 19 (19.0%) $\implies$ Zero intra-group reward variance (advantage = 0)
- **Mixed Outcome Groups:** 10 (10.0%) $\implies$ Non-zero reward variance providing learning gradient

## Decision Gate
- **Threshold:** $\ge 10.0\%$ mixed groups
- **Verdict:** **PASSED (10.0%)**
- **Finding:** In mixed groups, rollouts clearly separate valid routes from fatal hole entries on the exact same map, providing clean preference pairs ($y_w, y_l$) for DPO / unlikelihood training or targeted RL updates.
