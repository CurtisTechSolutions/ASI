export const meta = {
  name: 'codebase-map',
  description: 'Map a repository: parallel readers per subsystem, then one synthesized architecture map',
  whenToUse: 'Before planning a large change in an unfamiliar repo. Optional args: { root: "src", focus: "what you care about", max: 12 }',
  phases: [
    { title: 'Discover', detail: 'enumerate subsystems' },
    { title: 'Read', detail: 'one reader per subsystem' },
    { title: 'Synthesize', detail: 'merge into one map' },
  ],
}

// Standing instruction from prior sessions: the orchestrator delegates and Opus does the actual work. Override with args.model.
const WORKER_MODEL = (args && args.model) || 'opus'

const root = (args && args.root) || '.'
const focus = (args && args.focus) || ''
const max = (args && args.max) || 12

const DISCOVER_SCHEMA = {
  type: 'object',
  required: ['subsystems'],
  properties: {
    subsystems: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'path', 'why'],
        properties: {
          name: { type: 'string' },
          path: { type: 'string' },
          why: { type: 'string' },
        },
      },
    },
    build_and_test: { type: 'string' },
  },
}

const READ_SCHEMA = {
  type: 'object',
  required: ['name', 'purpose', 'entry_points', 'depends_on', 'depended_on_by', 'conventions', 'risks'],
  properties: {
    name: { type: 'string' },
    purpose: { type: 'string' },
    entry_points: { type: 'array', items: { type: 'string' } },
    depends_on: { type: 'array', items: { type: 'string' } },
    depended_on_by: { type: 'array', items: { type: 'string' } },
    conventions: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    key_files: { type: 'array', items: { type: 'string' } },
  },
}

phase('Discover')
const discovered = await agent(
  `Enumerate the subsystems of the repository under "${root}". Use the directory structure, package manifests, build config, and import graph; do not read every file. ` +
    `Return at most ${max} subsystems, each with a short name, its root path, and one line on why it is a unit. ` +
    (focus ? `Prioritize subsystems relevant to: ${focus}. ` : '') +
    `Also report the commands used to build and test the project if you can find them. Do not modify files.`,
  { model: WORKER_MODEL, label: 'discover', phase: 'Discover', schema: DISCOVER_SCHEMA },
)
const allFound = (discovered && discovered.subsystems) || []
const subsystems = allFound.slice(0, max)
if (allFound.length > max) log(`Capped to ${max} subsystems; ${allFound.length - max} dropped`)
if (!subsystems.length) return { error: 'No subsystems discovered', root }
log(`Reading ${subsystems.length} subsystems`)

phase('Read')
const reads = await pipeline(subsystems, (s) =>
  agent(
    `Read the subsystem "${s.name}" rooted at "${s.path}" (${s.why}). ` +
      `Report: purpose, entry points (file:line), what it depends on, what depends on it, conventions an implementer must follow, risks or hotspots, and the key files. ` +
      (focus ? `Focus on anything relevant to: ${focus}. ` : '') +
      `Cite file paths for every claim. Do not modify files.`,
    { model: WORKER_MODEL, label: `read:${s.name}`, phase: 'Read', schema: READ_SCHEMA },
  ),
)
const readOk = reads.filter(Boolean)
if (readOk.length < subsystems.length) log(`${subsystems.length - readOk.length} subsystem reads failed`)

phase('Synthesize')
const buildAndTest = (discovered && discovered.build_and_test) || 'unknown'
const map = await agent(
  `Synthesize one architecture map from these subsystem reports (JSON): ${JSON.stringify(readOk)}. ` +
    `Produce markdown with: 1) a one-paragraph overview, 2) a dependency table (subsystem, depends on), 3) entry points, 4) conventions that apply repo-wide, ` +
    `5) hotspots and shared files that parallel work must not touch concurrently, 6) build and test commands: ${buildAndTest}. ` +
    `Do not add claims that are not in the reports.`,
  { model: WORKER_MODEL, label: 'synthesize', phase: 'Synthesize' },
)
return { root, focus, build_and_test: buildAndTest, subsystems: readOk, map }

// synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh
