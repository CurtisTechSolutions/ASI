import React, { useState } from 'react'
import { get, post } from '../api.js'
import { useAsync } from '../useAsync.js'

// Stats, the tunable policy, the casebook and the maintenance actions.
export default function System({ state, refreshState }) {
  const cases = useAsync(() => get('cases'), [])
  const [compressed, setCompressed] = useState(null)
  const [tuned, setTuned] = useState(null)
  const [working, setWorking] = useState(false)

  const act = async (fn) => {
    setWorking(true)
    try { await fn() } finally { setWorking(false); refreshState() }
  }

  if (!state) return <div className="panel"><p className="empty">loading…</p></div>
  const { stats, policy, providers } = state

  return (
    <div className="panel system">
      <section className="card">
        <h3>memory<span className="sub">{state.home}</span></h3>
        <div className="stat-row">
          <Stat label="traces" value={stats.traces} />
          <Stat label="graded" value={stats.graded} />
          <Stat label="verified" value={stats.verified} />
          <Stat label="cases" value={stats.cases} />
          <Stat label="tools" value={stats.tools.length} />
          <Stat label="mean grade" value={stats.mean_grade === null ? '—' : stats.mean_grade.toFixed(3)} />
        </div>
        <div className="kind-bars">
          {Object.entries(stats.by_kind).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
            <div key={k} className="kind-bar">
              <span className="kind-name">{k}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${(n / stats.traces) * 100}%` }} />
              </span>
              <span className="bar-value">{n}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h3>providers<span className="sub">first reachable wins; local-model-first by default</span></h3>
        <ul className="providers">
          {providers.map((p) => (
            <li key={p.name} className={p.available ? 'on' : 'off'}>
              <strong>{p.name}</strong>
              {p.name === state.provider && <span className="active">in use</span>}
              <span>{p.available ? 'available' : 'not configured'}</span>
              <span className="dim">{p.embeds ? 'embeddings' : 'no embeddings — lexical fallback'}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h3>policy<span className="sub">every number the system may change about itself</span></h3>
        <div className="policy-grid">
          {Object.entries(policy).map(([k, v]) => (
            <div key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd>{typeof v === 'number' ? v.toFixed(3) : String(v)}</dd></div>
          ))}
        </div>
        <div className="toolbar">
          <button disabled={working} onClick={() => act(async () => setTuned(await post('upgrade', { trials: 6 })))}>
            tune against measured outcomes
          </button>
        </div>
        {tuned && tuned.result && (
          <p className="axis-note">
            objective {tuned.result.score_before} → {tuned.result.score_after}
            {Object.keys(tuned.result.changed).length === 0
              ? ' · no mutation beat the incumbent'
              : ` · changed ${Object.keys(tuned.result.changed).join(', ')}`}
          </p>
        )}
      </section>

      <section className="card">
        <h3>maintenance<span className="sub">compression is reversible: originals are archived first</span></h3>
        <div className="toolbar">
          <button disabled={working}
                  onClick={() => act(async () => setCompressed(await post('compress', { dry_run: true })))}>
            preview compression
          </button>
          <button disabled={working} className="danger"
                  onClick={() => act(async () => setCompressed(await post('compress', { dry_run: false })))}>
            compress now
          </button>
        </div>
        {compressed && (
          <>
            <p className="axis-note">
              {compressed.report.dry_run ? 'would compress' : 'compressed'}{' '}
              {compressed.report.compressed} cluster(s), folding {compressed.report.freed} trace(s)
            </p>
            {compressed.digests.map((d, i) => (
              <pre key={i} className="digest">{d.text}</pre>
            ))}
          </>
        )}
      </section>

      <section className="card">
        <h3>cases<span className="sub">problems, and what actually solved them</span></h3>
        {cases.error && <div className="error">{cases.error}</div>}
        <ul className="cases">
          {((cases.data && cases.data.cases) || []).map((c, i) => (
            <li key={i} className={c.grade > 0 ? 'worked' : 'failed'}>
              <span className="case-mark">{c.grade > 0 ? 'worked' : 'did not'}</span>
              <div>
                <strong>{c.problem}</strong>
                <p>{c.solution}</p>
              </div>
            </li>
          ))}
          {cases.data && cases.data.cases.length === 0 && (
            <li className="empty">no cases on record yet.</li>
          )}
        </ul>
      </section>
    </div>
  )
}

const Stat = ({ label, value }) => (
  <div className="stat"><span className="stat-value">{value}</span><span className="stat-label">{label}</span></div>
)
