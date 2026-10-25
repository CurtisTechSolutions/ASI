# Brief: <unit id or task title>

<!-- Every agent brief has all seven sections. A fresh agent knows only what is written here. -->

## Goal
<One or two sentences describing what "done" looks like, not how to get there.>

## Context you need (do not rediscover)
- Repo: <language, framework, package manager>
- Commands: build `<cmd>` · lint `<cmd>` · typecheck `<cmd>` · test `<cmd>`
- Conventions: <bullets the agent must follow, with an example file to imitate>
- Facts: <`path/file:line` references that matter for this unit>
- Ruled out: <approaches already rejected, and why>

## Files
- Owns (you may edit): `path/a`, `path/b`
- Read for context: `path/c`
- Do not touch: <reserved shared files, other units' files>

## Constraints
- <no new dependencies / keep the public API / no commits / match existing error handling / etc.>

## Done means
- [ ] <acceptance criterion>
- [ ] <acceptance criterion>
- [ ] `<verify command>` passes; paste the output

## Report back in exactly this format
```
Status: done | partial | blocked
Changed: <path — what and why>
Verification: <command → key output lines>
Assumptions: <...>
Needs (edits outside my ownership): <path: exact change — why>
Not done: <...>
```

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
