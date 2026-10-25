export const meta = {
  name: 'review-sweep',
  description: 'Review a change from several independent lenses, dedupe, then adversarially verify every finding before reporting',
  whenToUse: 'Reviewing a diff, branch, or PR when you want confirmed defects rather than a style pass. args: { target: "<diff description or command>", thorough: false, lenses: ["correctness", "security", "data-and-contracts", "tests", "performance"] }',
  phases: [
    { title: 'Find', detail: 'one reviewer per lens' },
    { title: 'Verify', detail: 'independent refutation per finding' },
  ],
}

// Standing instruction from prior sessions: the orchestrator delegates and Opus does the actual work. Override with args.model.
const WORKER_MODEL = (args && args.model) || 'opus'

const target =
  (args && args.target) ||
  'the uncommitted changes plus the commits not on the default branch (use git status, git diff, and git diff against the merge-base with main or master)'
const thorough = !!(args && args.thorough)

const ALL_LENSES = {
  correctness: 'logic errors, off-by-one, wrong conditions, null or undefined handling, broken invariants, unhandled error paths, race conditions',
  security: 'injection, authn/authz gaps, secrets exposure, unsafe deserialization, path traversal, SSRF, unsafe defaults',
  'data-and-contracts': 'API, schema, or serialization changes that break callers, migrations, backward compatibility',
  tests: 'behavior changed without tests, tests that cannot fail, weakened assertions, missing edge cases the diff introduces',
  performance: 'N+1 queries, unbounded loops or memory, blocking calls on hot paths, missing pagination or limits',
}
const requested = args && Array.isArray(args.lenses) && args.lenses.length ? args.lenses : Object.keys(ALL_LENSES)
const lensKeys = requested.filter((k) => ALL_LENSES[k])
if (!lensKeys.length) return { error: 'no known lenses requested', known: Object.keys(ALL_LENSES) }

const FINDINGS_SCHEMA = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['file', 'line', 'severity', 'summary', 'failure_scenario'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          summary: { type: 'string' },
          failure_scenario: { type: 'string' },
        },
      },
    },
  },
}
const VERDICT_SCHEMA = {
  type: 'object',
  required: ['verdict', 'evidence'],
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'UNVERIFIABLE'] },
    evidence: { type: 'string' },
    suggested_fix: { type: 'string' },
  },
}

phase('Find')
// Barrier is deliberate: findings from different lenses overlap heavily, and dedupe must see all of them before the expensive verify stage.
const found = await parallel(
  lensKeys.map((k) => () =>
    agent(
      `Review ${target} through the "${k}" lens only: ${ALL_LENSES[k]}. ` +
        `Read the full diff and the surrounding code that the diff calls or is called by. Report only defects you can tie to a concrete failure scenario (inputs or state leading to a wrong result). No style comments. ` +
        `For each: file, line, severity (blocker, major, minor), a one-sentence summary, and the failure scenario. Return an empty list if you find nothing real. Do not modify files.`,
      { model: WORKER_MODEL, label: `find:${k}`, phase: 'Find', schema: FINDINGS_SCHEMA },
    ),
  ),
)
const all = found.filter(Boolean).flatMap((r) => r.findings || [])
const seen = new Set()
const unique = []
for (const f of all) {
  const key = `${f.file}:${f.line}:${String(f.summary || '').slice(0, 40).toLowerCase()}`
  if (!seen.has(key)) {
    seen.add(key)
    unique.push(f)
  }
}
log(`${all.length} raw findings, ${unique.length} after dedupe`)
if (!unique.length) return { target, lenses: lensKeys, confirmed: [], refuted: [], unverifiable: [] }

phase('Verify')
const votes = thorough ? 3 : 1
const voteLens = ['does it reproduce on a realistic input', 'is the code path reachable from a real caller', 'is the severity right given the blast radius']
const verified = await pipeline(unique, (f) =>
  parallel(
    Array.from({ length: votes }, (_, i) => () =>
      agent(
        `Independently verify this review finding${votes > 1 ? ` (verifier ${i + 1} of ${votes}; your lens: ${voteLens[i]})` : ''}: ` +
          `${JSON.stringify(f)} in ${target}. Try to REFUTE it: read the code, trace callers, run a test or a one-off script if that settles it. ` +
          `Return CONFIRMED only with concrete evidence (file:line and the path from input to failure), REFUTED with the reason, or UNVERIFIABLE with what you could not check. Do not modify tracked files.`,
        { model: WORKER_MODEL, label: `verify:${f.file}:${f.line}`, phase: 'Verify', schema: VERDICT_SCHEMA },
      ),
    ),
  ).then((vs) => {
    const ok = vs.filter(Boolean)
    const confirmedVotes = ok.filter((v) => v.verdict === 'CONFIRMED').length
    const refutedVotes = ok.filter((v) => v.verdict === 'REFUTED').length
    let verdict = 'UNVERIFIABLE'
    if (ok.length && confirmedVotes > refutedVotes) verdict = 'CONFIRMED'
    else if (ok.length && refutedVotes > confirmedVotes) verdict = 'REFUTED'
    const fix = ok.find((v) => v.suggested_fix)
    return { ...f, verdict, evidence: ok.map((v) => v.evidence), suggested_fix: fix ? fix.suggested_fix : null }
  }),
)
const out = verified.filter(Boolean)
const order = { blocker: 0, major: 1, minor: 2 }
const by = (v) => out.filter((x) => x.verdict === v).sort((a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9))
return { target, lenses: lensKeys, votes, confirmed: by('CONFIRMED'), refuted: by('REFUTED'), unverifiable: by('UNVERIFIABLE') }

// synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh
