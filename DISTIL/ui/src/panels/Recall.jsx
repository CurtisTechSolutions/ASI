import React, { useEffect, useRef, useState } from 'react'
import { get, colourOf } from '../api.js'
import ScoreBars from '../viz/ScoreBars.jsx'

// The claim the memory layer rests on, made inspectable: recall ranks by
// similarity × credibility × recency, not similarity. A hit that is LESS similar
// and still ranks first is the whole design working, and this is where you can
// see it happen.
export default function Recall() {
  const [q, setQ] = useState('merge two dicts')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const run = async (e) => {
    if (e) e.preventDefault()
    setBusy(true); setError(null)
    try {
      setData(await get('recall', { q, k: 8 }))
    } catch (err) {
      setError(err.message); setData(null)
    } finally {
      setBusy(false)
    }
  }

  // Run once on mount so the panel opens showing a ranking rather than an empty
  // page: the point of this view is the ordering, and you cannot see an ordering
  // that has not been computed yet.
  const first = useRef(true)
  useEffect(() => {
    if (!first.current) return
    first.current = false
    run()
  }, [])

  // Is the top hit less similar than something below it? If so, credibility did
  // the work, and that is worth pointing at rather than leaving to be noticed.
  const reordered = data && data.hits.length > 1 &&
    data.hits.slice(1).some((h) => h.similarity > data.hits[0].similarity)

  return (
    <div className="panel recall">
      <form className="ask-form" onSubmit={run}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="query the memory" />
        <button type="submit" disabled={busy}>{busy ? 'searching…' : 'recall'}</button>
      </form>

      {error && <div className="error">{error}</div>}

      {reordered && (
        <div className="insight">
          The top hit is <strong>not</strong> the most similar one. Credibility
          outranked similarity here — which is what separates this from a vector
          store.
        </div>
      )}

      {data && data.hits.map((h, i) => (
        <article key={h.trace.id} className="hit">
          <header>
            <span className="rank">{i + 1}</span>
            <span className="pill" style={{ borderColor: colourOf(h.trace.kind) }}>
              {h.trace.kind}
            </span>
            <span className="hit-score">{h.score.toFixed(4)}</span>
            {h.trace.verified && <span className="verified">verified</span>}
            {h.trace.grade !== null && (
              <span className={`hit-grade ${h.trace.grade > 0 ? 'good' : 'bad'}`}>
                {h.trace.grade > 0 ? '+' : ''}{h.trace.grade.toFixed(2)}
              </span>
            )}
          </header>
          <p className="hit-text">{h.trace.text}</p>
          <ScoreBars hit={h} weights={data.weights} />
        </article>
      ))}

      {data && data.hits.length === 0 && <p className="empty">nothing recalled for that.</p>}
    </div>
  )
}
