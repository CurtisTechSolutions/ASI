<!-- start -->
## Orchestration and standing rules (plugin: orchestration)
- You are the orchestrator; Opus does the actual work. Delegate with the orchestration plugin: `/plan`, `/orchestrate`, `/fan-out`, `/swarm-review`, `/delegate`, `/handoff`. Follow `/protocol` when briefing agents. Do inline whatever takes a handful of tool calls.
- Load `/conventions` at the start of work in any CurtisTechSolutions or nosam1998 repository. Non-negotiables from it:
  - Branches are `<type>/<short-topic>` (`feat/`, `fix/`, `chore/`, `refactor/`, `docs/`), never `claude/...`. Conventional Commit messages and PR titles.
  - When the work is done: open the PR, merge it once CI is green when asked or when the repo says to auto-merge, then delete the branch. Split big work into smaller PRs.
  - Commits under the user's own GitHub name, with no "Generated with Claude Code" or Co-Authored-By lines.
  - Settings and credentials live in the DB and admin panel (admin values override `.env`), new behavior goes behind a flag or toggle, and every frontend change gets a mobile layout check.
  - Terse prompts mean decide and finish. "Try again" means the last attempt did not land. Pasted logs are the bug report.
<!-- end -->
