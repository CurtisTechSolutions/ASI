export const meta = {
  name: 'parallel-build',
  description: 'Plan a goal into independent units with disjoint file ownership, implement them in parallel, verify each independently, repair once, then integrate',
  whenToUse: 'A feature or refactor that splits into two or more independent code changes. args: { goal: "...", context: "facts and constraints", units: [optional pre-made units { id, title, brief, owns, reads, verify }], reserved_files: [], integration_checks: [], repairRounds: 1 }',
  phases: [
    { title: 'Plan', detail: 'units with disjoint ownership' },
    { title: 'Build', detail: 'implement, verify, repair per unit' },
    { title: 'Integrate', detail: 'full checks and glue' },
  ],
}

// Standing instruction from prior sessions: the orchestrator delegates and Opus does the actual work. Override with args.model.
const WORKER_MODEL = (args && args.model) || 'opus'

const goal = (args && args.goal) || ''
const context = (args && args.context) || ''
const repairRounds = args && typeof args.repairRounds === 'number' ? args.repairRounds : 1
if (!goal && !(args && Array.isArray(args.units) && args.units.length)) return { error: 'args.goal or args.units is required' }

const UNIT_PROPS = {
  id: { type: 'string' },
  title: { type: 'string' },
  brief: { type: 'string' },
  owns: { type: 'array', items: { type: 'string' } },
  reads: { type: 'array', items: { type: 'string' } },
  verify: { type: 'string' },
}
const PLAN_SCHEMA = {
  type: 'object',
  required: ['units', 'reserved_files', 'integration_checks'],
  properties: {
    units: { type: 'array', items: { type: 'object', required: ['id', 'title', 'brief', 'owns', 'verify'], properties: UNIT_PROPS } },
    reserved_files: { type: 'array', items: { type: 'string' } },
    integration_checks: { type: 'array', items: { type: 'string' } },
    context: { type: 'string' },
  },
}
const REPORT_SCHEMA = {
  type: 'object',
  required: ['status', 'changed_files', 'verification'],
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    changed_files: { type: 'array', items: { type: 'string' } },
    verification: { type: 'string' },
    assumptions: { type: 'array', items: { type: 'string' } },
    needs: { type: 'array', items: { type: 'string' } },
    not_done: { type: 'array', items: { type: 'string' } },
  },
}
const VERIFY_SCHEMA = {
  type: 'object',
  required: ['passed', 'evidence'],
  properties: {
    passed: { type: 'boolean' },
    evidence: { type: 'string' },
    problems: { type: 'array', items: { type: 'string' } },
  },
}

phase('Plan')
let plan
if (args && Array.isArray(args.units) && args.units.length) {
  plan = {
    units: args.units,
    reserved_files: (args && args.reserved_files) || [],
    integration_checks: (args && args.integration_checks) || [],
    context,
  }
  log(`Using ${plan.units.length} caller-supplied units`)
} else {
  plan = await agent(
    `Plan the goal "${goal}" as a work breakdown for parallel implementation by independent agents. Context and constraints: ${context || 'none given; discover the build, lint, and test commands yourself'}. ` +
      `Rules: 2 to 8 units; each unit is one agent's job, touches 1 to 6 files, and is independently verifiable. Every file that will be edited appears in exactly one unit's owns list. ` +
      `Files that more than one unit would need (lockfiles, barrel or index files, route tables, migrations, shared types, generated code) go in reserved_files and no unit edits them. ` +
      `Each unit's brief must be executable without conversation context: goal, exact files, constraints, done criteria. Each unit has a verify command. ` +
      `integration_checks lists the repo's full lint, typecheck, and test commands in order. Put repo facts (commands, conventions, file:line references) in context. Do not modify files.`,
    { model: WORKER_MODEL, label: 'plan', phase: 'Plan', schema: PLAN_SCHEMA },
  )
}
if (!plan || !plan.units || !plan.units.length) return { error: 'planning produced no units', plan }
const planContext = plan.context || context

const owners = new Map()
const conflicts = []
for (const u of plan.units) {
  for (const f of u.owns || []) {
    if (owners.has(f)) conflicts.push(`${f}: ${owners.get(f)} and ${u.id}`)
    else owners.set(f, u.id)
  }
}
if (conflicts.length) return { error: 'ownership conflicts; fix the plan before building', conflicts, plan }
log(`${plan.units.length} units, ${(plan.reserved_files || []).length} reserved files`)

const implementPrompt = (u, failure) =>
  `You are implementing one unit of a larger change. Other agents are editing other files in this working tree right now.\n` +
  `Unit ${u.id}: ${u.title}\n${u.brief}\n` +
  `Context: ${planContext}\n` +
  `Owns (the only files you may edit): ${JSON.stringify(u.owns)}\n` +
  `Read for context: ${JSON.stringify(u.reads || [])}\n` +
  `Do not touch (reserved for the integrator): ${JSON.stringify(plan.reserved_files || [])}\n` +
  `Verify with: ${u.verify}\n` +
  (failure ? `A previous attempt failed verification. Problems: ${JSON.stringify(failure.problems || [])}. Evidence: ${failure.evidence}. Fix these.\n` : '') +
  `Rules: no edits outside your owned files (report them under needs instead), no unrelated refactors, no commits. Run the verify command and report its real output.`

const verifyPrompt = (u, report) =>
  `Independently verify unit ${u.id} (${u.title}). The implementer reports: ${JSON.stringify(report)}. ` +
  `Read the diff of ${JSON.stringify(u.owns)}, check that it meets the brief: ${u.brief}. Run: ${u.verify}. ` +
  `Also confirm no files outside ${JSON.stringify(u.owns)} were modified by this unit's work when you can tell. ` +
  `Return passed=true only if the verify command passes and the brief is met. List concrete problems otherwise. Do not modify files.`

phase('Build')
const results = await pipeline(
  plan.units,
  (u) => agent(implementPrompt(u, null), { model: WORKER_MODEL, label: `implement:${u.id}`, phase: 'Build', schema: REPORT_SCHEMA }),
  async (report, u) => {
    let current = report
    for (let round = 0; round <= repairRounds; round++) {
      const v = await agent(verifyPrompt(u, current), { model: WORKER_MODEL, label: `verify:${u.id}${round ? `:r${round}` : ''}`, phase: 'Build', schema: VERIFY_SCHEMA })
      if (v && v.passed) return { unit: u.id, title: u.title, status: 'verified', report: current, verification: v }
      if (round === repairRounds) return { unit: u.id, title: u.title, status: 'failed', report: current, verification: v }
      current = await agent(implementPrompt(u, v || { problems: ['verifier returned no result'], evidence: '' }), {
        model: WORKER_MODEL,
        label: `repair:${u.id}:r${round + 1}`,
        phase: 'Build',
        schema: REPORT_SCHEMA,
      })
    }
    return { unit: u.id, title: u.title, status: 'failed', report: current, verification: null }
  },
)

phase('Integrate')
const built = results.filter(Boolean)
const failed = built.filter((r) => r.status !== 'verified').map((r) => r.unit)
if (built.length < plan.units.length) log(`${plan.units.length - built.length} units returned no result`)
const integration = await agent(
  `All units of the goal "${goal}" are implemented in this working tree by separate agents. Unit results: ${JSON.stringify(
    built.map((r) => ({ unit: r.unit, status: r.status, changed: r.report && r.report.changed_files, needs: r.report && r.report.needs, not_done: r.report && r.report.not_done })),
  )}. ` +
    `Context: ${planContext}. Reserved files that you alone may edit: ${JSON.stringify(plan.reserved_files || [])}. ` +
    `Apply the reserved-file edits the units requested under needs, resolve cross-unit mismatches (imports, shared types, call signatures), then run the integration checks in order: ${JSON.stringify(plan.integration_checks || [])}. ` +
    `Make minimal glue changes only; do not rework a unit's logic, and do not silence failures. Regenerate lockfiles and generated files with the project's tooling. ` +
    `Report: reserved-file edits applied, glue changes, each check command with its real result, and anything still failing with the unit that owns it. Do not commit.`,
  { model: WORKER_MODEL, label: 'integrate', phase: 'Integrate' },
)
return { goal, units: built, failed_units: failed, integration }

// synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh
