---
name: handoff
description: Write a handoff document for the next session or another agent - where things stand (verified versus in progress), decisions and why, files touched, how to verify, open questions, exact next steps, and the command to resume. Use at the end of a long session, when context is getting long, or when pausing orchestrated work.
argument-hint: "[output path; default .claude/orchestration/<slug>/handoff.md]"
allowed-tools:
  - Read
  - Grep
  - Glob
  - Write
  - TaskList
  - Bash(git status *)
  - Bash(git diff *)
  - Bash(git log *)
  - Bash(git branch *)
---

# /handoff

Output path: $ARGUMENTS

1. **Gather the facts**, do not recall them: `TaskList`; `git status`, `git diff --stat`, `git log --oneline -10`, the current branch; the plan file under `.claude/orchestration/` if one exists; the last verification output you actually saw.
2. **Fill `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/handoff.md`.** Only items with verification evidence go under Done. Anything else is In progress or Not started. Every decision carries its reason and the alternative that was rejected. Next steps are concrete enough to start from cold, with exact commands.
3. **Write the file** to the given path, or to `.claude/orchestration/<slug>/handoff.md` (create the directory). End it with the resume line: `/orchestrate <path>`.
4. **Tell the user** the path and the top three next steps. Commit the handoff only if the user asked for commits.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
