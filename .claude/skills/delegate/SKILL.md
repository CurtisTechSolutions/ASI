---
name: delegate
description: Hand one task to one subagent with a complete brief (goal, context, owned files, constraints, done criteria, report format), then verify the result. Use for a single well-scoped side task, or to run one of this plugin's agents by name.
argument-hint: "<agent: planner|researcher|implementer|reviewer|verifier|tester|integrator|docs-writer> <task>"
allowed-tools:
  - Agent
  - SendMessage
  - Read
  - Grep
  - Glob
---

# /delegate

Input: $ARGUMENTS

1. **Pick the agent.** If the input names one, use `<name>`. Otherwise choose by the task: a question → `researcher`; a code change → `implementer`; check a claim → `verifier`; review a diff → `reviewer`; tests → `tester`; docs → `docs-writer`; a breakdown → `planner`; cohere parallel work → `integrator`.
2. **Gate.** If the task is a handful of tool calls with a known target, do it inline and say so. Delegation is for work that would otherwise flood this context.
3. **Write the brief** from `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/brief.md`, all seven parts, including the lines from `/conventions` that apply (branch naming, authorship, config in the DB, mobile check). Put in the facts you already have (commands, conventions, `file:line` references, what was ruled out). If the agent would have to rediscover something you could find in two tool calls, find it first. Give the agent the code to look at, not your conclusion.
4. **Launch** the agent. Foreground when your next step depends on the result; background when you have other work to do meanwhile.
5. **Verify the result** before relaying it: read the diff for a change, re-run the verification command, or spawn `verifier` for a high-stakes claim. A second round goes to the same agent through `SendMessage` so it keeps its context.
6. **Report** in at most ten lines: outcome, evidence, what is not done.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
