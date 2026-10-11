---
name: agent-usage
description: Record human input and AI agent contributions for professors and graders. Use when coursework contribution logging is enabled, when asked to document AI assistance, or when asked to summarize prompts, human decisions, agent edits, and created files.
---

# Agent usage

Keep a concise, factual record of human input and your contribution to the
current task. Follow any course-provided disclosure format when available.

## Track your work

1. Locate the project root and read existing project instructions.
2. Before making changes, note the user's request and initial project state.
   In Git projects, inspect relevant status and diffs. Treat existing changes
   as a baseline, not as your contribution.
3. Track your edits, created files and folders, deletions, renames,
   and any delegated contributions.
4. If invoked after the work, use available conversation and edit history.
   Do not claim every change in a diff as your own.
5. Write one entry per task before finishing or handing back incomplete work.

## Record human input

- Summarize the user's initial request, follow-up instructions, constraints,
  corrections, supplied ideas or code, and explicit decisions.
- Preserve the order of meaningful inputs and explain how each affected
  the work. Use sequence numbers, not timestamps.
- Label each source as "user message," "user-provided code/file,"
  "user-reported edit," or "observed human edit with confirmed attribution."
- Use a person's name only when provided. Otherwise use "User."
- Separate a human request from the resulting AI action. Asking for a
  function does not mean the human wrote that function.
- Record human-written code or manual edits only when supplied, explicitly
  reported, or observed with confirmed attribution.
- Label reported work as user-reported and uncertain authorship as unknown.
- Do not infer human approval or review from silence or continued conversation.
- Summarize prompts by default. Include exact wording only when requested
  or required by the course.
- Record only inputs available for the current task. Do not invent earlier
  conversations or unseen edits.

## Save the record

Append every entry to `AGENT_USAGE.MD` in the project root.
Use this exact filename and capitalization.

Create it with the heading `# Agent usage log` if it does not exist.
Preserve all previous entries.

For each entry:

- Use only the current date in `YYYY-MM-DD` format, using the user's timezone
  when known. Do not include an exact time or timezone in the entry.
- Create a separate entry for each task, even when several tasks share a date.
- Identify the agent/tool and model only when known. Otherwise write
  "Not provided."
- Summarize the substantive work actually performed in 2–4 sentences.
- Include advice or analysis that materially influenced the work, even if
  no project files changed.
- List exact project-relative file paths with an action:
  created, modified, deleted, or renamed.
- Briefly explain each file's purpose or change. For renamed files, include
  both the old and new paths.
- List folders you created separately. Editing a file does not mean you
  created its parent folder.
- Attribute delegated work only when supported by the other agents' reports.
- Keep authorship uncertainty next to the affected contribution, if needed.
- Do not include a Verification section or a Limitations and handoff section.
- Do not include check commands, test results, or routine status checks.
- Do not describe creating the usage-log entry in the AI contribution summary.
- Do not list `AGENT_USAGE.MD` or folders created solely for logging among
  the task's file or folder contributions.
- Do not classify maintaining the log as generation, editing, or verification
  for the substantive task.
- If logging is the only action, say "No substantive project changes."
- Do not include credentials, unrelated private data, hidden reasoning,
  or full chat transcripts.
- Coordinate concurrent logging through a single writer or safe serialized
  append. Preserve other agents' entries.
- Do not commit, push, publish, or send the log unless separately authorized.
- If writing fails, provide the intended entry in your response and explain
  that it was not saved.

## Entry format

Use only the sections below. Write "None" or "Not provided" where appropriate.
Replace all placeholders with actual information.

```markdown
## <YYYY-MM-DD> — <short task title>

**Agent/tool:** <tool and model, if known>
**User request:** <brief task description>
**AI contribution:** <2–4 sentences describing substantive work performed>
**Contribution type:** <generation, editing, debugging, review, or advice>

### Human input and contributions

| Order    | Source                         | Input or contribution          | Effect on the work              |
| -------- | ------------------------------ | ------------------------------ | ------------------------------- |
| <number> | <person if known; source type> | <actual input or contribution> | <resulting direction or change> |

### Files

| Path                    | Action                             | Contribution        |
| ----------------------- | ---------------------------------- | ------------------- |
| <project-relative path> | <created/modified/deleted/renamed> | <brief explanation> |

### Folders created

- <exact project-relative path and purpose, or None>
```

If there are no substantive file changes, write "None" under Files instead
of an empty table.

## Finish

Briefly tell the user that the entry was saved to `AGENT_USAGE.MD`.
Claim it was saved only after a successful write.

Treat the log as an agent-authored disclosure, not independent proof of
authorship or human review.
