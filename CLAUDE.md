## Agent usage logging

Use the `agent-usage` skill for every task.

Before making changes, record the initial project state and track relevant
human requests, instructions, corrections, and supplied work.

Before finishing, append a factual entry to `AGENT_USAGE.MD` in the project
root using the skill's format. Include human input, AI contributions,
files changed, folders created, checks performed, and unfinished work.

Preserve existing entries. Do not claim pre-existing changes as your own
or invent human authorship, review, or approval.

If the skill is unavailable, follow these logging requirements directly.

## Agent skills

### Issue tracker

Track work in GitHub Issues for Shadibuchanan/SPARA. Before creating, reading,
or updating issues, specs, or wayfinder maps, read docs/agents/issue-tracker.md.

### Triage labels

Use the five default triage labels. Before triaging or changing issue status,
read docs/agents/triage-labels.md.

### Domain docs

Use a single root glossary and docs/adr/. Before exploring project concepts
or making design decisions, read docs/agents/domain.md.

### PACE-ICE execution

Before connecting to PACE-ICE, preparing jobs, or handling remote artifacts,
read docs/agents/pace-ice.md for access, storage, and the pilot resource ceiling.
