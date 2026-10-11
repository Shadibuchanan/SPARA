# Issue tracker: GitHub

Issues and specs live in GitHub Issues for Shadibuchanan/SPARA.
Use the authenticated gh CLI from this repository. Pass
--repo Shadibuchanan/SPARA when running outside the checkout.

## Conventions

- Create an issue: gh issue create --title "Title" --body-file <file>.
- Read an issue: gh issue view <number> --json number,title,body,labels,comments,assignees.
- List issues: gh issue list --state open --json number,title,body,labels,assignees.
- Comment: gh issue comment <number> --body-file <file>.
- Edit a body: gh issue edit <number> --body-file <file>.
- Add or remove labels: gh issue edit <number> --add-label <label> / --remove-label <label>.
- Close: gh issue close <number>.

Write multiline issue bodies and comments to UTF-8 temporary files and use
--body-file. Refer to issues by linked titles in human-facing output.

## Pull requests as a triage surface

PRs as a request surface: no.

## Skill operations

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read the issue and its comments.

## Wayfinding operations

- Map: one issue labelled wayfinder:map. Its body holds Destination, Notes,
  Decisions so far, Not yet specified, and Out of scope.
- Child ticket: an issue with one label, wayfinder:research,
  wayfinder:prototype, wayfinder:grilling, or wayfinder:task.
  Create with gh issue create --parent <map-number>, or attach with
  gh issue edit <map-number> --add-sub-issue <child-number>.
- Blocking: use native dependencies. Set
  gh issue edit <child-number> --add-blocked-by <blocker-number>.
- Frontier: read the map's sub-issues in order; select open, unassigned
  tickets with no open blockers. Fetch children with
  gh api --paginate repos/Shadibuchanan/SPARA/issues/<map-number>/sub_issues.
  Use each child's issue_dependencies_summary.blocked_by for the open
  blocker count; inspect dependencies when the summary is unavailable.
- Claim: assign the selected ticket to the driving developer before work;
  gh issue edit <number> --add-assignee @me for the authenticated driver.
- Resolve: post the answer as a comment, close the ticket, then append a
  linked-title pointer and one-line gist to the map's Decisions so far.
- Wayfinder maps and tickets carry wayfinder labels only, not triage labels.

Use gh help to check available commands. If the installed CLI lacks native
flags, use GitHub's API. If the repository itself lacks sub-issues, use a
child task list in the map and a "Part of #<map-number>" line in each child.
If native dependencies are unavailable, use "Blocked by: #<number>" lines
and check each referenced issue's state when finding the frontier.
