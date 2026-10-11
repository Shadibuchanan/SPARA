# Domain docs

SPARA uses one domain context.

## Before exploring

Read GLOSSARY.md at the repository root and any ADRs in docs/adr/ relevant
to the work. If either is absent, proceed; domain-modeling creates these
documents when terms or decisions are resolved.

If GLOSSARY-MAP.md is introduced later, follow it to the relevant contexts.

## Layout

- GLOSSARY.md: project concepts and their canonical names.
- docs/adr/: architectural decisions, created only when warranted by a
  meaningful trade-off that is costly to reverse and needs context.

## Vocabulary and decisions

Use glossary terms in issues, designs, code, and tests. Identify missing
domain concepts for domain-modeling rather than silently inventing synonyms.

If a proposal conflicts with an ADR, name and link the conflicting decision
and explain why it may need to be revisited.
