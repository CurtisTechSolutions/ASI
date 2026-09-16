import React, { useMemo, useState } from 'react'
import { get, colourOf } from '../api.js'
import { useAsync } from '../useAsync.js'
import Scatter from '../viz/Scatter.jsx'

// Everything the system has lived through, in one space.
//
// There is no separate index per kind -- a tool and the bug it once fixed are
// the same query at recall time -- so the filter here hides points rather than
// switching store.
export default function Memory({ onGrade }) {
  const [kinds, setKinds] = useState([])
  const [selected, setSelected] = useState(null)
  const { data, error, busy, run } = useAsync(() => get('memory'), [])

  const traces = useMemo(() => {
    if (!data) return []
    return kinds.length ? data.traces.filter((t) => kinds.includes(t.kind)) : data.traces
  }, [data, kinds])

  const present = useMemo(
    () => [...new Set((data ? data.traces : []).map((t) => t.kind))].sort(),
    [data],
  )

  const toggle = (k) =>
    setKinds((ks) => (ks.includes(k) ? ks.filter((x) => x !== k) : [...ks, k]))

  const grade = async (score) => {
    if (!selected) return
    await onGrade(selected.id, score)
    await run()
    setSelected(null)
  }

  return (
    <div className="panel memory">
      <div className="toolbar">
        <div className="filters">
          {present.map((k) => (
            <button
              key={k}
              className={`chip${kinds.includes(k) ? ' on' : ''}`}
              style={{ '--chip': colourOf(k) }}
              onClick={() => toggle(k)}
            >
              {k}
            </button>
          ))}
          {kinds.length > 0 && (
            <button className="chip clear" onClick={() => setKinds([])}>clear</button>
          )}
        </div>
        <button onClick={run} disabled={busy}>{busy ? 'loading…' : 'refresh'}</button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="memory-body">
        <div>
          {traces.length > 0 ? (
            <Scatter traces={traces} selected={selected} onSelect={setSelected} />
          ) : (
            !busy && <p className="empty">nothing in memory yet — ask it something.</p>
          )}
          <p className="axis-note">
            Two principal components of the stored vectors. The axes have no
            meaning; only distance does.
            {data && data.backends.length > 1 && (
              <> Projected per embedding backend ({data.backends.join(', ')}) —
                vectors from different backends share no plane.</>
            )}
          </p>
        </div>

        <aside className="detail">
          {selected ? (
            <>
              <span className="pill" style={{ borderColor: colourOf(selected.kind) }}>
                {selected.kind}
              </span>
              <p className="detail-text">{selected.text}</p>
              <dl className="detail-grid">
                <div><dt>grade</dt><dd>{selected.grade === null ? 'ungraded' : selected.grade.toFixed(3)}</dd></div>
                <div><dt>credibility</dt><dd>{selected.credibility.toFixed(3)}</dd></div>
                <div><dt>recalled</dt><dd>{selected.hits}×</dd></div>
                <div><dt>written</dt><dd>{selected.seen}×</dd></div>
                <div><dt>verified</dt><dd>{selected.verified ? 'yes' : 'no'}</dd></div>
                <div><dt>links</dt><dd>{selected.links.length}</dd></div>
              </dl>
              <div className="grade-buttons">
                <span>grade it:</span>
                <button className="good" onClick={() => grade(1)}>+1 right</button>
                <button onClick={() => grade(0)}>0 unsure</button>
                <button className="bad" onClick={() => grade(-1)}>−1 wrong</button>
              </div>
              <p className="grade-note">
                A user grade counts double — a person bothering to grade is a
                stronger and rarer signal than a check firing.
              </p>
            </>
          ) : (
            <p className="empty">click a point.</p>
          )}
        </aside>
      </div>
    </div>
  )
}
