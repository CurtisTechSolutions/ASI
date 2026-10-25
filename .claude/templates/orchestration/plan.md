# Plan: <goal in one line>

<!-- Produced by planner. Every unit section must stand alone: an implementer reading only its unit and the Context section can do the work. -->

## Goal and acceptance criteria
- <criterion>
- <criterion>

## Context (facts, with file:line)
- Commands: build `<cmd>` · lint `<cmd>` · typecheck `<cmd>` · test `<cmd>`
- Conventions: <bullets>
- Facts: `path/file:line` — <what it shows>
- Ruled out: <approaches rejected and why>

## Units
| id | title | wave | owns | depends on | verify |
|----|-------|------|------|------------|--------|
| U1 | <title> | 1 | `path/a`, `path/b` | - | `<cmd>` |
| U2 | <title> | 1 | `path/c` | - | `<cmd>` |
| U3 | <title> | 2 | `path/d` | U1 | `<cmd>` |

### U1: <title>
**Goal:** <...>
**Owns:** `path/a`, `path/b`
**Reads for context:** `path/c`
**Do not touch:** <reserved files, other units' files>
**Constraints:** <...>
**Done means:**
- [ ] <...>
- [ ] `<verify command>` passes

### U2: <title>
<same structure>

## Reserved files (integrator only)
- `path/index.ts` — <what units will need from it>
- `package-lock.json` — regenerate with the package manager, never by hand

## Integration
- Full checks, in order: `<cmd>`, `<cmd>`, `<cmd>`
- Expected glue: <imports, shared types, registrations>

## Risks and rollback
- <risk> → <mitigation or rollback>

## Out of scope
- <...>

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
