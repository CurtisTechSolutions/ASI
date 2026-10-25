---
name: plan
description: Produce a work breakdown for a goal - independent units with disjoint file ownership, waves, a verification command per unit, and reserved shared files - without changing any code. Use when the user asks for a plan, or before /orchestrate on a large change.
argument-hint: "<goal>"
allowed-tools:
  - Agent
  - SendMessage
  - Read
  - Grep
  - Glob
  - TaskCreate
---

# /plan

Goal: $ARGUMENTS

This skill changes no code. It produces a plan file other agents can execute.

1. **Restate** the goal and the acceptance criteria in a few lines. Collect constraints from the conversation (branch, no commits, style, deadlines, what the user has already rejected).
2. **Fill the gaps first.** If the plan needs facts you do not have (build/lint/test commands, where the relevant code lives, existing patterns), spawn `researcher` agents in one message, one question each, or an `Explore` agent for a broad sweep. Keep the conclusions with `file:line` references.
3. **Spawn `planner`** with a complete brief: goal, acceptance criteria, the facts from step 2, constraints, what is ruled out, and the plan structure in `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/plan.md`.
4. **Check the plan** before accepting it: every edited file in exactly one unit; reserved shared files listed; every unit has a verify command; waves respect dependencies; each unit section stands alone. If it fails any of these, send the specific problems back to the planner with `SendMessage` rather than patching it silently.
5. **Write** the plan to `.claude/orchestration/<slug>/plan.md` in the project (create the directory). Show the user the Units table, the reserved files, and the risks, then the path. One closing line: `Run /orchestrate <path> to execute.`

Do not create tasks or start implementation from this skill.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
