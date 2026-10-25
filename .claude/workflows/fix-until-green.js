export const meta = {
  name: 'fix-until-green',
  description: 'Run a check command, fix the reported errors file by file in parallel, and repeat until the check passes or two rounds make no progress',
  whenToUse: 'A typecheck, lint, or test command with many independent failures. args: { check: "npx tsc --noEmit", maxRounds: 5, maxParallel: 8 }',
  phases: [
    { title: 'Check', detail: 'run the command and parse failures' },
    { title: 'Fix', detail: 'one agent per failing file' },
  ],
}

// Standing instruction from prior sessions: the orchestrator delegates and Opus does the actual work. Override with args.model.
const WORKER_MODEL = (args && args.model) || 'opus'

const check = args && args.check
if (!check) return { error: 'args.check is required, e.g. { check: "npm test" }' }
const maxRounds = (args && args.maxRounds) || 5
const maxParallel = (args && args.maxParallel) || 8

const CHECK_SCHEMA = {
  type: 'object',
  required: ['passed', 'errors'],
  properties: {
    passed: { type: 'boolean' },
    errors: {
      type: 'array',
      items: {
        type: 'object',
        required: ['file', 'message'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          message: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}
const FIX_SCHEMA = {
  type: 'object',
  required: ['status', 'changed_files'],
  properties: {
    status: { type: 'string', enum: ['fixed', 'partial', 'blocked'] },
    changed_files: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

let lastKey = null
let stalled = 0
const history = []
for (let round = 1; round <= maxRounds; round++) {
  phase('Check')
  const run = await agent(
    `Run exactly this command in the repository root: ${check}\n` +
      `Report passed=true if it exits 0 with no failures. Otherwise list every reported failure with its file path (relative to the repo root, or "GENERAL" when no file is named), line if given, and message. ` +
      `Do not fix anything. Do not modify files.`,
    { model: WORKER_MODEL, label: `check:${round}`, phase: 'Check', schema: CHECK_SCHEMA },
  )
  if (!run) return { passed: false, rounds: round, error: 'check agent returned no result', history }
  history.push({ round, passed: run.passed, errors: run.errors.length, summary: run.summary || '' })
  if (run.passed) return { passed: true, rounds: round, history }

  const key = JSON.stringify(run.errors.slice(0, 50).map((e) => `${e.file}:${e.line || 0}:${e.message}`))
  if (key === lastKey) {
    stalled++
    if (stalled >= 2) return { passed: false, rounds: round, stopped: 'no progress for two rounds', remaining: run.errors, history }
  } else {
    stalled = 0
  }
  lastKey = key

  const byFile = new Map()
  for (const e of run.errors) {
    const f = e.file || 'GENERAL'
    if (!byFile.has(f)) byFile.set(f, [])
    byFile.get(f).push(e)
  }
  const files = Array.from(byFile.keys())
  const batch = files.slice(0, maxParallel)
  if (files.length > batch.length) log(`Round ${round}: ${files.length} failing files, fixing ${batch.length} this round`)
  else log(`Round ${round}: ${run.errors.length} errors in ${files.length} files`)

  phase('Fix')
  const fixes = await parallel(
    batch.map((file) => () =>
      agent(
        (file === 'GENERAL'
          ? `Fix these failures reported by "${check}" that do not name a file: ${JSON.stringify(byFile.get(file))}. Find the responsible code first. `
          : `Fix these failures reported by "${check}" in ${file} only: ${JSON.stringify(byFile.get(file))}. Edit only ${file} unless a fix is impossible without touching another file, and then say so under notes. `) +
          `Other agents are fixing other files at the same time. Fix the root cause: do not skip, disable, or weaken tests or checks, and do not add lint or type suppressions. ` +
          `Do not run the full command; it will be re-run after this round. Do not commit.`,
        { model: WORKER_MODEL, label: `fix:${file}`, phase: 'Fix', schema: FIX_SCHEMA },
      ),
    ),
  )
  const blocked = fixes.filter(Boolean).filter((f) => f.status === 'blocked')
  if (blocked.length) log(`Round ${round}: ${blocked.length} files blocked`)
}
return { passed: false, rounds: maxRounds, stopped: 'max rounds reached', history }

// synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh
