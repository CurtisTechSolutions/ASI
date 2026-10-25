---
name: protocol
description: The orchestration protocol this plugin's skills and subagents follow - when to delegate versus work inline, how to brief an agent, disjoint file ownership for parallel work, verifying agent claims, the report format, and cost rules. Load when coordinating several agents, or when asked how orchestration works here.
user-invocable: true
---

# Orchestration protocol

You, the main session, are the orchestrator. Agents are workers with no memory of this conversation. Every rule below exists because a fresh agent knows only what its brief says, and you see only the summary it sends back.

## 1. Decide whether to delegate at all

| Situation | Do |
|---|---|
| Known target, a handful of tool calls | Do it yourself. An agent costs more than it looks. |
| One question whose answer needs reading many files | One `researcher` (or the built-in `Explore`). Keep the conclusion, not the file dumps. |
| Two or more changes that touch disjoint files | Parallel `implementer` agents, one per unit, launched in a single message. |
| The same transformation over N targets | `/fan-out`. A workflow if the user opted in and N is large. |
| A claim you are about to act on (a finding, a "done" report) | `verifier`, briefed with the code, never with your conclusion. |
| Discovery of unknown size (every call site, every bug) | Loop until two rounds find nothing new. The `review-sweep` and `fix-until-green` workflows do this. |
| Work that spans sessions | A plan file plus `/handoff`. |

Never: delegate a task and then also do it yourself; spawn an agent to check something you could run inline; hand an agent your hypothesis and ask it to confirm it (it will).

## 2. Roles this plugin provides

| Agent | Mutates files | Use for |
|---|---|---|
| `planner` | no | Work breakdown with disjoint ownership, waves, verify commands |
| `researcher` | no | One question, answered with `file:line` evidence |
| `implementer` | owned files only | One unit of change, verified |
| `reviewer` | no | Adversarial review: defects with failure scenarios |
| `verifier` | no | Confirm or refute one claim with evidence |
| `tester` | test files | Tests that would fail if the change were wrong |
| `integrator` | reserved files + glue | After parallel units land: cohere, full checks |
| `docs-writer` | docs only | Docs that move with the code |

Built-ins still apply: `Explore` for read-only sweeps, `Plan` for a quick architectural take, `fork` when the agent needs your conversation context.

## 3. Brief like a peer

Every brief has the seven parts in `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/brief.md`: goal, context (commands, conventions, facts with `file:line`, what is ruled out), files (owns / reads / do not touch), constraints, done criteria with the verify command, and the exact report format.

- Include the relevant lines from `/conventions` (branch naming, authorship, config in the DB, mobile check) in every brief that touches git or UI. Agents do not see that skill on their own.
- Point at files to read instead of pasting them.
- Give the code, not your conclusion, when you want an honest answer.
- If you cannot write the brief clearly, you do not understand the task well enough to hand it off. Research first.

## 4. File ownership for parallel work

- Every file that will be edited belongs to exactly one unit. Check this yourself before launching a wave.
- Files that several units would touch (lockfiles, barrel or index files, route tables, migrations, shared types, generated code) are **reserved**: no unit edits them. Units report what they need under `Needs`; the integrator applies it.
- Worktree isolation is for experiments and throwaway spikes. Changes made in a worktree do not land in the working tree by themselves.

## 5. Agent output is a claim, not a fact

- Read `git diff --stat` and the changed files rather than trusting a `Changed:` list.
- A `done` without pasted verification output is `partial`.
- Re-run the verification command yourself for anything you will call done, or send it to a verifier.
- Reviewer findings go through a verifier before you spend effort fixing them, unless you can confirm them in a couple of tool calls.
- Mistakes come back in the same confident register as findings. Treat confidence as zero evidence.

## 6. Running agents

- Independent agents go in one message. Dependent ones wait for their inputs.
- Track units with `TaskCreate` / `TaskUpdate` so state survives a long session.
- Do not poll. Background agents notify you when they finish.
- For a second round with the same agent (a repair after failed verification), use `SendMessage` to that agent so it keeps its context. A fresh agent re-learns everything.
- Launch foreground only when your very next action depends on the result and nothing else could usefully happen meanwhile.

## 7. Reporting

Use `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/report.md`. Lead with the outcome. Evidence for every verification. Say plainly what is not done. Count the agents used.

## 8. Cost

- Inline is the default. Before spawning three or more agents, state the agent count in one line. Above eight agents in one orchestration, say why; proceed only when the user asked for orchestration or a thorough result.
- Standing instruction from the user: the orchestrator delegates and **Opus does the actual work**. Every agent in this plugin defaults to `model: opus`; override per call only when the user asks. Raise `effort` for hard units.
- Stop conditions: two failed repair rounds on one unit means stop and escalate with the evidence. A verifier returning `UNVERIFIABLE` means say so; do not guess.

## 9. Workflows versus the Agent tool

Workflows (the `Workflow` tool) run a deterministic script that fans out dozens of agents and keeps intermediate results out of your context. They need an explicit opt-in from the user: the word `ultracode`, or "use a workflow" in their own words. This plugin ships four: `/codebase-map`, `/review-sweep`, `/parallel-build`, `/fix-until-green`. Without the opt-in, the skills do the same shapes with the Agent tool.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
