---
name: fan-out
description: Apply the same operation to many targets (files, modules, packages, test suites) with parallel subagents - enumerate targets deterministically, partition into batches with disjoint files, run identical briefs, collect a results table, verify. Use for "do X to every Y" tasks such as migrations, adding types or tests or docs across modules, and bulk renames.
argument-hint: "<operation> across <targets, e.g. src/**/*.py>"
allowed-tools:
  - Agent
  - SendMessage
  - Read
  - Grep
  - Glob
  - TaskCreate
  - TaskUpdate
---

# /fan-out

Task: $ARGUMENTS

1. **Enumerate targets** with Glob or Grep so the list is deterministic and repeatable. Print the count. Zero targets: stop and say so.
2. **Learn the transformation on one target first.** Do one target inline (or the sample the user named). Record the exact edit pattern, the pitfalls you hit, and the verify command per target. This becomes the worked example in every brief.
3. **Partition.** One target per agent up to 8 agents; beyond that, batches of `ceil(N / 8)` targets each. Each batch owns only its targets' files. Name shared files (barrels, lockfiles, registries) as off limits; collect needed changes to them for a final pass.
   - Above about 20 targets, say in one line that a workflow would cost less per target and the user can opt in with "use a workflow"; then continue with batches unless they do.
4. **Write one brief** from `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/brief.md` with the target list substituted per batch: the exact transformation, the worked example from step 2, the verify command per target, "do not touch any other file", and the report format. Identical briefs give comparable results.
5. **Launch every batch in one message** as `implementer` agents. Create one task per batch with `TaskCreate` and mark them `in_progress`.
6. **Collect.** Build a table: target | status | verification | notes. Spot-check two results by reading their diffs. A batch reporting `partial` or `blocked` gets one repair round through `SendMessage` to the same agent; after that, record it as not done.
7. **Apply shared-file changes** collected in step 3, then run the repo-wide check once (lint, typecheck, tests). Report with `${CLAUDE_PROJECT_DIR}/.claude/templates/orchestration/report.md`: the table, verification evidence, what is not done, agents used.

Do not commit unless the user asked.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
