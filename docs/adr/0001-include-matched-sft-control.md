---
status: superseded by ADR-0002; reinstated by ADR-0004
---

# Include a matched direct-SFT control

The user withdrew this additional experiment on 2026-10-11 to prioritize the
published LIVR planning baseline. See [the current scope](0002-reproduce-livr-reference-protocol.md).
The paragraph below records the earlier decision, not an active requirement.

Before comparing reinforcement-learning methods, compare direct supervised
fine-tuning with LIVR using the same backbone, parameter-training regime, maps,
and total training updates. The user accepted this control on 2026-10-11 so that
the project can measure whether LIVR helps on the selected task, rather than
treating ordinary SFT only as a pipeline smoke test. This adds training cost but
provides an internal comparison alongside the attempted published-result reproduction.
