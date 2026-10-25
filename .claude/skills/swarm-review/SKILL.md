---
name: swarm-review
description: Review a diff, branch, or PR with several independent reviewers (correctness, security, data and contracts, tests, performance), dedupe, then adversarially verify each finding and report only confirmed defects with evidence. Use when the user wants a thorough review, or before merging a large change.
argument-hint: "[diff range, branch, PR number, or path; default: working tree vs default branch]"
allowed-tools:
  - Agent
  - SendMessage
  - Read
  - Grep
  - Glob
  - Bash(git diff *)
  - Bash(git log *)
  - Bash(git status *)
  - Bash(git merge-base *)
---

# /swarm-review

Target: $ARGUMENTS

0. **Workflow route.** If the user opted into workflows in this session (the word `ultracode`, or "use a workflow"), run the `review-sweep` workflow with `args: { target: "<resolved diff description>", thorough: <true if the user asked for thorough> }` and report its result in the format of step 4. Otherwise continue.
1. **Resolve the target** to one concrete diff command and print its `--stat`. Default: uncommitted changes plus commits not on the default branch (`git merge-base` against `main` or `master`). A PR number means fetch that PR's diff. Above roughly 2,000 changed lines, say so and review in rounds split by directory.
2. **Find.** One message, one `reviewer` per lens, each briefed with the diff command, the lens definition below, and the rule "defects only, each with `file:line`, severity, and a concrete failure scenario; no style".
   - correctness: logic errors, off-by-one, wrong conditions, null handling, broken invariants, unhandled error paths, races
   - security: injection, authn/authz gaps, secrets exposure, unsafe deserialization, path traversal, SSRF, unsafe defaults
   - data and contracts: API, schema, or serialization changes that break callers; migrations; backward compatibility
   - tests: behavior changed without tests, tests that cannot fail, weakened assertions, missing edge cases the diff introduces
   - performance: N+1 queries, unbounded loops or memory, blocking calls on hot paths, missing limits or pagination
3. **Verify.** Dedupe findings by `file:line` and summary. For every blocker and major finding, spawn one `verifier` in parallel with the finding and the code paths, instructed to try to refute it. Minor findings: verify only when cheap; otherwise list them as unverified.
4. **Report.**
   - Confirmed (ranked by severity): finding, `file:line`, failure scenario, evidence, suggested fix
   - Refuted: one line each, so the user knows what was checked
   - Unverified: minor findings not sent to a verifier
   - Not reviewed: anything the reviewers reported skipping
5. Do not fix anything unless the user asks. If they do, fix through `implementer` agents grouped by owning files, re-run the checks, and re-verify the fixed findings.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
