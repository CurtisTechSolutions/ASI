import React, { useState } from 'react'
import { get, post } from '../api.js'
import { useAsync } from '../useAsync.js'
import GradeStages from '../viz/GradeStages.jsx'

// Every capability, whatever process it runs in. A locally forged Python tool
// and one exposed by an MCP server are the same kind of thing here, because at
// recall time the question is "what can act on this?" and the transport is an
// implementation detail.
export default function Tools() {
  const { data, error, busy, run } = useAsync(() => get('tools'), [])
  const [open, setOpen] = useState(null)
  const [goal, setGoal] = useState('compute the median of a list')
  const [forged, setForged] = useState(null)
  const [invoked, setInvoked] = useState({})
  const [working, setWorking] = useState(false)

  const forge = async () => {
    setWorking(true); setForged(null)
    try {
      setForged(await post('forge', { goal }))
      await run()
    } catch (e) {
      setForged({ error: e.message })
    } finally {
      setWorking(false)
    }
  }

  const plant = async () => {
    setWorking(true)
    try { await post('seed', {}); await run() } finally { setWorking(false) }
  }

  const invoke = async (name, raw) => {
    try {
      const args = raw.trim() ? JSON.parse(raw) : []
      const body = Array.isArray(args) ? { name, args } : { name, kwargs: args }
      const out = await post('invoke', body)
      setInvoked({ ...invoked, [name]: out.result })
    } catch (e) {
      setInvoked({ ...invoked, [name]: { ok: false, error: e.message } })
    }
  }

  const tools = (data && data.tools) || []

  return (
    <div className="panel tools">
      <div className="toolbar">
        <form className="ask-form grow" onSubmit={(e) => { e.preventDefault(); forge() }}>
          <input value={goal} onChange={(e) => setGoal(e.target.value)}
                 placeholder="a goal to write a tool for" />
          <button type="submit" disabled={working}>forge</button>
        </form>
        <button onClick={plant} disabled={working}>plant the starter kit</button>
        <button onClick={run} disabled={busy}>refresh</button>
      </div>

      {error && <div className="error">{error}</div>}
      {forged && (
        <div className={forged.error ? 'error' : forged.registered ? 'insight' : 'warn-box'}>
          {forged.error
            ? forged.error
            : forged.registered
              ? <>forged and verified <strong>{forged.name}</strong> — <GradeStages grade={forged.grade} compact /></>
              : <>wrote <strong>{forged.name}</strong> but it was not kept: {forged.grade.diagnostic}</>}
        </div>
      )}

      {tools.length === 0 && !busy && (
        <p className="empty">no tools yet — plant the starter kit, or forge one.</p>
      )}

      <div className="tool-grid">
        {tools.map((t) => (
          <article key={t.name} className={`tool ${t.transport}`}>
            <header onClick={() => setOpen(open === t.name ? null : t.name)}>
              <h4>{t.name}</h4>
              <span className={`transport ${t.transport}`}>{t.transport}</span>
            </header>
            <code className="sig">{t.signature || '—'}</code>
            <p className="purpose">{t.purpose}</p>
            <GradeStages grade={t.grade} compact />
            {t.deps.length > 0 && (
              <p className="deps">builds on {t.deps.join(', ')}</p>
            )}
            {t.solved.length > 0 && (
              <ul className="solved">
                {t.solved.slice(0, 3).map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            )}
            {open === t.name && (
              <div className="tool-open">
                <label className="invoke">
                  <span>arguments (JSON array, or object for named)</span>
                  <input
                    placeholder='[[5, 3, 1, 4]]'
                    onKeyDown={(e) => { if (e.key === 'Enter') invoke(t.name, e.target.value) }}
                  />
                </label>
                {invoked[t.name] && (
                  <pre className={invoked[t.name].ok ? 'result ok' : 'result no'}>
                    {JSON.stringify(invoked[t.name], null, 2)}
                  </pre>
                )}
                {t.source && <pre className="source">{t.source}</pre>}
                {t.tests && <pre className="source tests">{t.tests}</pre>}
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  )
}
