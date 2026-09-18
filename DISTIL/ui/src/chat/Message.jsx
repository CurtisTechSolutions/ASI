import React from 'react'
import { colourOf } from '../api.js'
import { COMMANDS } from '../commands.js'
import Answer from './Answer.jsx'
import Detail from './Detail.jsx'
import Recalled from './Recalled.jsx'
import Scatter from '../viz/Scatter.jsx'
import EntropyCurve from '../viz/EntropyCurve.jsx'
import GradeStages from '../viz/GradeStages.jsx'

// One turn in the transcript.
//
// Every capability that used to be a tab renders here instead, so the record of
// what you did and the result of doing it are the same list.
export default function Message({ message, onGrade, onAnswer, onTrace }) {
  if (message.role === 'user') {
    return <div className="turn user"><div className="bubble">{message.text}</div></div>
  }
  return (
    <div className="turn agent">
      <div className="bubble">
        {message.text && <p className="say">{message.text}</p>}
        <Body message={message} onGrade={onGrade} onAnswer={onAnswer} onTrace={onTrace} />
      </div>
    </div>
  )
}

function Body({ message, onGrade, onAnswer, onTrace }) {
  const { kind, data, streaming } = message

  if (kind === 'ask') {
    return <Answer message={message} onGrade={onGrade} onAnswer={onAnswer} onTrace={onTrace} />
  }
  if (kind === 'auto') return <Auto message={message} />
  if (streaming) return <p className="working">working…</p>
  if (kind === 'error') return null            // the text above is the whole message
  if (!data) return null

  switch (kind) {
    case 'welcome': return <Welcome />
    case 'help': return <Help />
    case 'tools': return <Tools data={data} />
    case 'seed': return <Seeded data={data} />
    case 'forge': return <Forged data={data} />
    case 'recall': return <Recalled recall={normaliseRecall(data)} onTrace={onTrace} />
    case 'memory': return <MemoryPlot data={data} onTrace={onTrace} />
    case 'concepts': return <Concepts data={data} onTrace={onTrace} />
    case 'explore': return <Explored data={data} />
    case 'cases': return <Cases data={data} />
    case 'clarify': return <Clarified data={data} onTrace={onTrace} />
    case 'trace': return <TraceGraph data={data} onTrace={onTrace} />
    case 'mcp': return <Mcp data={data} />
    case 'compress': return <Compressed data={data} />
    case 'tune': return <Tuned data={data} />
    case 'system': return <SystemCard data={data} />
    default: return <pre className="raw">{JSON.stringify(data, null, 2)}</pre>
  }
}

// `/recall` returns hits shaped for the old panel; the chat renderer takes the
// flatter shape that `solve` attaches to every answer. One converter beats two
// renderers that drift.
const normaliseRecall = (data) => ({
  query: data.query,
  weights: data.weights,
  hits: (data.hits || []).map((h) => ({
    id: h.trace.id, kind: h.trace.kind, text: h.trace.text, grade: h.trace.grade,
    verified: h.trace.verified, score: h.score, similarity: h.similarity,
    credibility: h.credibility, recency: h.recency,
  })),
})

// A dashboard for a process nobody is watching continuously.
//
// The first version was a scrolling log, which answers "what did it just do?"
// and nothing else -- you cannot tell from forty rows whether an hour of running
// built anything. So the log is last. Above it: what it is doing right now, then
// what the whole run has produced, then what it has learned to spend its time
// on.
//
// The tallies are outcomes, not activity. "63 cycles" is a measure of how long
// it ran; "2 tools built, 9 beliefs found resting on nothing" is a measure of
// whether that was worth it.
function Auto({ message }) {
  const { cycles, strategy, streaming, error } = message
  const current = cycles.length ? cycles[cycles.length - 1] : null
  const tally = summarise(cycles)

  return (
    <div className="auto">
      <p className="auto-head">
        {streaming ? 'running by itself' : 'stopped'}
        {streaming && <span className="pulse" aria-hidden="true" />}
        <span className="auto-sub">
          {cycles.length} cycle{cycles.length === 1 ? '' : 's'} · {tally.bits.toFixed(1)} bits
        </span>
      </p>

      {error && <p className="error">{error}</p>}

      {streaming && current && (
        <p className="auto-now">
          <span className={`move ${current.move}`}>{current.move}</span>
          <span>{current.note}</span>
        </p>
      )}

      <ul className="tallies">
        <Tally n={tally.pursued} one="direction taken" label="directions taken" hint="problems it set itself when it had nothing left to do" good />
        <Tally n={tally.built} one="tool built" label="tools built" hint="written, verified and registered — the only move that adds a capability" />
        <Tally n={tally.shaky} one="belief resting on nothing" label="beliefs resting on nothing" hint="why-chains that bottomed out in an assumption or a circle" good />
        <Tally n={tally.refuted} one="idea refuted" label="ideas refuted" hint="a refuted experiment is a stored result, not a wasted cycle" good />
        <Tally n={tally.grounded} one="belief it could justify" label="beliefs it could justify" hint="why-chains that reached ground" />
        <Tally n={tally.folded} one="memory folded" label="memories folded" hint="cold traces consolidated into digests that keep the detail" />
      </ul>

      {tally.stats && (
        <p className="auto-growth">
          memory is at <strong>{tally.stats.traces}</strong> traces and{' '}
          <strong>{tally.stats.tools.length}</strong> tools
          {tally.grew > 0 && <> — <strong>+{tally.grew}</strong> since this run began</>}
        </p>
      )}

      {strategy && (
        <div className="auto-strategy">
          <h4>what it has learned to spend time on</h4>
          <ul className="strategy">
            {Object.entries(strategy).sort((a, b) => b[1] - a[1]).map(([name, share]) => (
              <li key={name}>
                <span className={`move ${name}`} title={MOVES[name]}>{name}</span>
                <div className="bar"><i style={{ width: `${share * 100}%` }} /></div>
                <strong>{Math.round(share * 100)}%</strong>
              </li>
            ))}
          </ul>
          <p className="axis-note">
            Regret-matched over its six moves, and rewarded for <em>information</em>,
            not success — a move that confirms what it already believed scores near
            zero however cleanly it ran.
          </p>
        </div>
      )}

      <Detail title="everything it did" summary={`${cycles.length} cycles, newest last`}>
        <ol className="cycles">
          {cycles.slice(-60).map((c) => (
            <li key={c.n} className={c.move}>
              <span className="n">{c.n}</span>
              <span className="move">{c.move}</span>
              <span className="note">{c.note}</span>
              <span className={`bits ${c.learned > 0.5 ? 'high' : c.learned > 0 ? 'some' : 'none'}`}
                    title="information, not success">
                {c.learned.toFixed(2)}
              </span>
            </li>
          ))}
        </ol>
        {cycles.length > 60 && (
          <p className="elided">…{cycles.length - 60} earlier cycles not shown</p>
        )}
      </Detail>
    </div>
  )
}

const MOVES = {
  question: 'asked why about something it believed, then attacked the premise',
  experiment: 'ran an experiment it could not call in advance',
  build: 'tried to build a capability it lacked',
  consolidate: 'folded cold memory into digests',
  tune: 're-fitted its own policy against measured outcomes',
}

function Tally({ n, label, one, hint, good }) {
  return (
    <li className={n > 0 && good ? 'good' : n > 0 ? '' : 'zero'} title={hint}>
      <strong>{n}</strong>
      <span>{n === 1 && one ? one : label}</span>
    </li>
  )
}

// Outcomes, read off the cycles. Counted here rather than sent by the server
// because the server would have to keep the same running totals anyway, and a
// number computed in two places is a number that disagrees with itself.
function summarise(cycles) {
  const out = { bits: 0, built: 0, shaky: 0, grounded: 0, refuted: 0, folded: 0,
                tuned: 0, pursued: 0, stats: null, grew: 0 }
  let first = null
  for (const c of cycles) {
    out.bits += c.learned
    const d = c.detail || {}
    if (c.move === 'build' && d.registered) out.built += 1
    if (c.move === 'question') {
      if (d.terminal === 'assumed' || d.terminal === 'circular') out.shaky += 1
      else if (d.terminal === 'grounded') out.grounded += 1
    }
    if (c.move === 'experiment' && d.ran && d.passed === false) out.refuted += 1
    if (c.move === 'consolidate') out.folded += d.freed || 0
    if (c.move === 'tune' && Object.keys(d.changed || {}).length) out.tuned += 1
    if (c.move === 'pursue' && d.direction) out.pursued += 1
    if (c.stats) {
      if (first === null) first = c.stats.traces
      out.stats = c.stats
    }
  }
  if (out.stats && first !== null) out.grew = out.stats.traces - first
  return out
}

function Welcome() {
  return (
    <ul className="examples">
      <li>try <em>build a csv cleaner and compute the median of each column</em></li>
      <li>try <em>make the thing better</em> to watch it refuse to guess</li>
      <li>type <code>/</code> for everything else it can do</li>
    </ul>
  )
}

function Help() {
  return (
    <ul className="help">
      {COMMANDS.map((c) => (
        <li key={c.name}>
          <code>/{c.name}{c.args ? ` ${c.args}` : ''}</code>
          <span>{c.blurb}</span>
        </li>
      ))}
    </ul>
  )
}

function Tools({ data }) {
  const tools = data.tools || []
  if (!tools.length) {
    return <p className="empty">no tools yet — <code>/forge</code> writes one, or <code>/seed</code> plants the starter kit.</p>
  }
  return (
    <ul className="tool-list">
      {tools.map((t) => (
        <li key={t.name}>
          <code>{t.signature || t.name}</code>
          <span className="purpose">{t.purpose}</span>
          <span className="meta">
            <span className={`transport ${t.transport}`}>{t.transport}</span>
            {t.deps && t.deps.length > 0 && <em>builds on {t.deps.join(', ')}</em>}
            {t.grade && <span className={t.grade.score > 0 ? 'good' : 'bad'}>
              {t.grade.score > 0 ? '+' : ''}{t.grade.score.toFixed(2)}
            </span>}
          </span>
        </li>
      ))}
    </ul>
  )
}

// `/api/seed` reports what was planted, not the toolbox. Sending it through the
// tool renderer made the message contradict itself: "I planted 12" directly above
// "no tools yet".
function Seeded({ data }) {
  const planted = data.planted || []
  const already = data.already || []
  if (!planted.length && !already.length) return <p className="empty">nothing to plant.</p>
  return (
    <ul className="tool-list">
      {planted.map(([name, score]) => (
        <li key={name}>
          <code>{name}</code>
          <span className="meta"><span className="good">verified {score.toFixed(2)}</span></span>
        </li>
      ))}
      {already.length > 0 && (
        <li><span className="purpose">already present: {already.join(', ')}</span></li>
      )}
      {(data.rejected || []).map(([name, why]) => (
        <li key={name}><code>{name}</code><span className="meta"><span className="bad">rejected — {why}</span></span></li>
      ))}
    </ul>
  )
}

function Forged({ data }) {
  return (
    <div className="forged">
      <p className={data.registered ? 'verdict ok' : 'verdict no'}>
        <strong>{data.registered ? `registered ${data.name}` : `rejected ${data.name}`}</strong>
        <span>{data.registered
          ? 'it passed every stage, so it is callable and recallable now'
          : 'not stronger than what is already registered — recorded so it is not re-derived'}</span>
      </p>
      {data.grade && <GradeStages grade={data.grade} />}
      {data.source && (
        <Detail title="the source it wrote" summary={`${data.source.split('\n').length} lines`}>
          <pre className="source">{data.source}</pre>
        </Detail>
      )}
      {data.tests && (
        <Detail title="the contract it had to pass" summary="run in the sandbox, twice, under different hash seeds">
          <pre className="source">{data.tests}</pre>
        </Detail>
      )}
    </div>
  )
}

function MemoryPlot({ data, onTrace }) {
  return (
    <div className="memory-plot">
      <p className="say">
        {data.traces.length} memories, projected onto a plane. Distance means
        something; the directions do not.
      </p>
      <Scatter traces={data.traces} onPick={(t) => onTrace && onTrace(t.id)} />
    </div>
  )
}

// A concept is a cluster named and compressed: the heading says what the group
// is, the body is the cluster's context kept in detail -- distinguishing terms,
// the best-graded member verbatim, the spread of grades.
function Concepts({ data, onTrace }) {
  if (!data.concepts.length) {
    return <p className="empty">no structure worth naming yet — it needs a few
      related memories before a cluster forms.</p>
  }
  return (
    <div className="concepts">
      <p className="say">
        {data.concepts.length} cluster(s) over {data.pool} memories
        {data.skipped > 0 && <> ({data.skipped} beyond the pool cap not clustered)</>}
        {data.written.length > 0 && <> — remembered, ungraded</>}
      </p>
      {data.concepts.map((c, i) => {
        const [heading, ...body] = c.text.split('\n')
        return (
          <div key={i} className="concept">
            <h4>{heading}</h4>
            <pre>{body.join('\n')}</pre>
            <p className="concept-meta">
              cohesion {c.detail.cohesion} · {c.detail.size} members
              {c.detail.mean_grade !== null && c.detail.mean_grade !== undefined &&
                <> · mean grade {c.detail.mean_grade}</>}
            </p>
            <div className="edges">
              <ul>
                {c.sources.slice(0, 8).map((id) => (
                  <li key={id}>
                    <button onClick={() => onTrace && onTrace(id)}>{id}</button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Explored({ data }) {
  const ran = data.experiments.filter((e) => e.ran)
  return (
    <div className="explored">
      <p className="say">
        {data.experiments.length} experiments, {ran.length} actually runnable,{' '}
        {ran.filter((e) => e.passed).length} of those passed. A refuted one is a
        stored result, not a wasted cycle.
      </p>
      <ul className="experiments">
        {data.experiments.map((e, i) => (
          <li key={i} className={!e.ran ? 'skipped' : e.passed ? 'ok' : 'no'}>
            <span className="origin">{e.origin}</span>
            <span className="idea">{e.idea}</span>
            <span className="outcome">{!e.ran ? 'not run' : e.passed ? 'held' : 'refuted'}</span>
            {e.detail && <em>{e.detail}</em>}
          </li>
        ))}
      </ul>
      <Detail title="which generator earns its keep" summary="regret-matched mix over idea sources">
        <ul className="strategy">
          {Object.entries(data.strategy).map(([name, share]) => (
            <li key={name}>
              <span>{name}</span>
              <div className="bar"><i style={{ width: `${share * 100}%` }} /></div>
              <strong>{Math.round(share * 100)}%</strong>
            </li>
          ))}
        </ul>
      </Detail>
    </div>
  )
}

function Cases({ data }) {
  if (data.cases) {
    if (!data.cases.length) return <p className="empty">no cases on record yet.</p>
    return (
      <ul className="cases">
        {data.cases.map((c, i) => (
          <li key={i} className={c.grade > 0 ? 'worked' : 'failed'}>
            <span className="mark">{c.grade > 0 ? 'worked' : 'failed'}</span>
            <strong>{c.problem}</strong>
            <span>{c.solution}</span>
          </li>
        ))}
      </ul>
    )
  }
  return (
    <div className="precedent">
      <p className="say">{data.note}</p>
      {data.precedent && (
        <p className="worked"><strong>{data.precedent.problem}</strong> → {data.precedent.solution}</p>
      )}
      {data.avoid.map((a, i) => (
        <p key={i} className="failed"><strong>avoid:</strong> {a.solution}</p>
      ))}
    </div>
  )
}

function Clarified({ data, onTrace }) {
  return (
    <div className="clarified">
      <p className={`verdict ${data.actionable ? 'ok' : 'no'}`}>
        <strong>{data.actionable ? 'clear enough to start' : 'not clear enough yet'}</strong>
        <span>{data.reason}</span>
      </p>
      {data.questions.length > 0 && (
        <ul className="questions flat">
          {data.questions.map((q) => (
            <li key={q.gap}>
              <span className="gap">{q.gap}</span>
              {q.text}
              {q.known && <em>memory already says: {q.known}</em>}
            </li>
          ))}
        </ul>
      )}
      <Detail title="what I already knew" summary={`${data.recall.hits.length} from memory`}>
        <Recalled recall={data.recall} onTrace={onTrace} />
      </Detail>
    </div>
  )
}

// The link graph, finally rendered. Credit propagates along these edges: grading
// a composite well should earn its parts something, and until these were written
// there was nothing for it to travel down.
function TraceGraph({ data, onTrace }) {
  const t = data.trace
  return (
    <div className="trace-graph">
      <p className="trace-head">
        <span className="pill" style={{ borderColor: colourOf(t.kind) }}>{t.kind}</span>
        {t.text}
      </p>
      <dl className="trace-facts">
        <div><dt>credibility</dt><dd>{t.credibility}</dd></div>
        <div><dt>recalled</dt><dd>{t.hits} times</dd></div>
        <div><dt>grades</dt><dd>{t.grades.length
          ? t.grades.map((g) => `${g.score > 0 ? '+' : ''}${g.score} (${g.source})`).join(', ')
          : 'none yet'}</dd></div>
      </dl>
      <Edges title="built on" edges={data.links} onTrace={onTrace}
             empty="nothing — this is a root" />
      <Edges title="used by" edges={data.backlinks} onTrace={onTrace}
             empty="nothing yet — a grade here stays here" />
      {data.missing.length > 0 && (
        <p className="missing">{data.missing.length} linked trace(s) no longer in the store</p>
      )}
    </div>
  )
}

function Edges({ title, edges, onTrace, empty }) {
  return (
    <div className="edges">
      <h4>{title}</h4>
      {edges.length === 0 ? <p className="empty">{empty}</p> : (
        <ul>
          {edges.map((e) => (
            <li key={e.id}>
              <button onClick={() => onTrace && onTrace(e.id)}>
                <span className="pill" style={{ borderColor: colourOf(e.kind) }}>{e.kind}</span>
                {e.text}
                {e.grade !== null && e.grade !== undefined && (
                  <em className={e.grade > 0 ? 'good' : 'bad'}>
                    {e.grade > 0 ? '+' : ''}{e.grade.toFixed(2)}
                  </em>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Mcp({ data }) {
  if (data.attached) {
    return (
      <div>
        <p className={data.failed.length ? 'verdict no' : 'verdict ok'}>
          <strong>{data.attached}</strong>
          <span>{data.failed.length
            ? data.failed[0].error
            : `${data.tools.length} tool(s) embedded — recalled alongside local ones, ungraded until used`}</span>
        </p>
      </div>
    )
  }
  if (data.removed !== undefined) {
    return <p className="say">{data.removed ? 'detached' : 'no such server'} — {data.servers.length} remaining.</p>
  }
  if (!data.servers.length) {
    return <p className="empty">no servers attached. <code>/mcp add fs npx -y @modelcontextprotocol/server-filesystem /tmp</code></p>
  }
  return (
    <ul className="servers">
      {data.servers.map((s) => (
        <li key={s.name}>
          <strong>{s.name}</strong>
          <code>{s.command.join(' ')}</code>
          <span className={s.alive ? 'good' : 'dim'}>{s.alive ? 'running' : 'idle'}</span>
        </li>
      ))}
    </ul>
  )
}

function Compressed({ data }) {
  const r = data.report
  return (
    <p className="say">
      {r.clusters} cold cluster(s); {r.compressed ? `compressed ${r.compressed}` : 'nothing compressed'},
      folding {r.freed} trace(s). Originals are archived first, so this is reversible.
    </p>
  )
}

function Tuned({ data }) {
  const changes = Object.entries(data.result.changed || {})
  if (!changes.length) return <p className="say">nothing moved — the current policy is already the best of the trials.</p>
  return (
    <ul className="tuned">
      {changes.map(([name, [from, to]]) => (
        <li key={name}><code>{name}</code> {from} → <strong>{to}</strong></li>
      ))}
    </ul>
  )
}

function SystemCard({ data }) {
  return (
    <div className="system-card">
      <dl className="trace-facts">
        <div><dt>provider</dt><dd>{data.provider}</dd></div>
        <div><dt>embedder</dt><dd>{data.stats.embedder}</dd></div>
        <div><dt>remembered</dt><dd>{data.stats.traces}</dd></div>
        <div><dt>verified</dt><dd>{data.stats.verified}</dd></div>
        <div><dt>home</dt><dd>{data.home}</dd></div>
      </dl>
      <Detail title="providers" summary={`${data.providers.filter((p) => p.available).length} reachable`}>
        <ul className="providers">
          {data.providers.map((p) => (
            <li key={p.name} className={p.available ? 'on' : 'off'}>
              <strong>{p.name}</strong>
              <span>{p.available ? 'available' : 'not configured'}</span>
              <em>{p.embeds ? 'embeddings' : 'no embeddings — lexical fallback'}</em>
            </li>
          ))}
        </ul>
      </Detail>
      <Detail title="policy" summary="every number it may change about itself">
        <ul className="policy">
          {Object.entries(data.policy).map(([name, value]) => (
            <li key={name}><span>{name.replace(/_/g, ' ')}</span><strong>{value}</strong></li>
          ))}
        </ul>
      </Detail>
    </div>
  )
}
