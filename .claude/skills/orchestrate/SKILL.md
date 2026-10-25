---
name: orchestrate
description: Run a multi-unit change end to end - intake, recon, a plan with disjoint file ownership, parallel implementation by subagents, independent verification, integration, review, and a report. Use when the user asks to orchestrate, parallelize, or "use agents" for a task, or when a change clearly splits into two or more independent units. Not for small tasks.
argument-hint: "<goal, or path to a plan or handoff file>"
allowed-tools:
  - Agent
  - SendMessage
  - TaskCreate
  - TaskUpdate
  - TaskList
  - TaskGet
  - Read
  - Grep
  - Glob
---

# /orchestrate

Goal or plan: $ARGUMENTS

You are the orchestrator. If the protocol is not already in context, read `${CLAUDE_PROJECT_DIR}/.claude/skills/protocol/SKILL.md` first. Say in one line what you are about to do, then work through the phases. Make reasonable assumptions and state them; do not stop to ask unless proceeding under any assumption would waste the work.

## Phase 0: Intake

- If the argument is a path to an existing plan or handoff file, read it and resume at the phase it indicates. Treat its "Done" items as claims: spot-check one.
- Otherwise restate the goal and the acceptance criteria. Pull constraints from the conversation: branch, whether to commit, style rules, deadlines.
- Load `/conventions` if it is not in context: it carries the user's standing rules (branch naming, PR and merge loop, authorship, config in the DB) and the per-repo facts that go into every brief.
- **Gate:** if the whole task is a handful of tool calls, say so and do it inline. Orchestration is for work that splits.

## Phase 1: Recon, only what the plan needs

- If the repo is unfamiliar, spawn `researcher` agents in one message, one question each: build/lint/test commands, conventions, where the relevant code lives, what already exists. Keep the conclusions and the `file:line` facts; drop the file dumps.
- These facts go into every later brief.

## Phase 2: Plan

- Spawn `planner` with: goal, acceptance criteria, recon facts, constraints, and anything ruled out. Ask for the structure in `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/plan.md`.
- Check the plan yourself before using it: every edited file in exactly one unit; reserved files listed; every unit has a verify command; waves respect dependencies; units are real units, not one tangled change split arbitrarily. Send it back to the planner with `SendMessage` to fix, rather than proceeding with a broken plan.
- Write the plan to `.claude/orchestration/<slug>/plan.md` in the project (create the directory). Create one task per unit with `TaskCreate`.
- Tell the user, in one line, the unit count and the agent count you are about to use.

## Phase 3: Build, one wave at a time

- For each wave, one message with one `implementer` Agent call per unit. Each brief follows `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/brief.md` and stands alone: the unit section verbatim, the plan's Context section, the reserved-file list under "Do not touch", and the report format. Mark the task `in_progress`.
- When reports come back: read `git diff --stat` and the changed files. A `done` without pasted verification output is `partial`.
- A unit that failed verification gets one repair round: send the failure output to the same implementer with `SendMessage`. A second failure stops that unit; leave its task open and record it for the report.
- Collect every `Needs` item for the integrator before starting the next wave; if a later wave depends on a reserved-file change, apply that change yourself (or via the integrator) between waves.

## Phase 4: Integrate

- Spawn `integrator` with the plan, every unit report, the `Needs` list, and the full check commands. Wait for it.
- Red checks caused by one unit's logic: one repair round with that unit's implementer (`SendMessage`), then re-run the checks.

## Phase 5: Verify and review

- Run the full checks once yourself, or read the integrator's pasted output and re-run at least one command to confirm it. Green is a requirement, not a hope.
- Spawn `reviewer` on the full diff against the starting commit. For each blocker or major finding you cannot confirm in a couple of tool calls, spawn `verifier` agents in parallel, one per finding, briefed with the finding and the code paths. Fix confirmed findings through the owning implementer (`SendMessage`) or inline when trivial, then re-run the checks.
- When the repo expects tests or docs to move with code: `tester` for new behavior without tests, `docs-writer` for public-facing changes.

## Phase 6: Report

- Mark completed tasks `completed`. Write the report in the shape of `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/report.md`: outcome, what changed per unit, verification evidence, review findings fixed and open, what is not done, agents used.
- If work remains, run `/handoff` so the next session can resume from the plan.
- Finish the delivery loop the conventions describe when the user asked for it: a `<type>/<topic>` branch (never `claude/`), commits under the user's name with no Claude attribution, a PR, a merge once CI is green, branch deleted.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
