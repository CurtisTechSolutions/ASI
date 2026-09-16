import React, { useState } from 'react'
import { get, post, colourOf } from '../api.js'
import { useAsync } from '../useAsync.js'
import EntropyCurve from '../viz/EntropyCurve.jsx'

// Curiosity, with a ranking function.
//
// The curve is the argument: an experiment is worth running in proportion to how
// little you can call it. Ideas you are sure will work sit at the left floor —
// and ideas you are sure will FAIL sit at the right floor, which is the half
// people skip.
export default function Explore() {
  const { data, error, busy, run } = useAsync(() => get('ideas', { n: 10 }), [])
  const [runs, setRuns] = useState(null)
  const [strategy, setStrategy] = useState(null)
  const [working, setWorking] = useState(false)

  const experiment = async (steps) => {
    setWorking(true)
    try {
      const out = await post('explore', { steps })
      setRuns(out.experiments)
      setStrategy(out.strategy)
      await run()
    } catch (e) {
      setRuns([{ idea: e.message, ran: false, passed: false, detail: 'request failed' }])
    } finally {
      setWorking(false)
    }
  }

  const ideas = (data && data.ideas) || []

  return (
    <div className="panel explore">
      <div className="toolbar">
        <button onClick={() => experiment(3)} disabled={working}>
          {working ? 'running…' : 'run 3 experiments'}
        </button>
        <button onClick={run} disabled={busy}>re-brainstorm</button>
      </div>

      {error && <div className="error">{error}</div>}

      <section className="card">
        <h3>
          information gain
          <span className="sub">H(p) = −p·log₂p − (1−p)·log₂(1−p), peaking where you cannot call it</span>
        </h3>
        <EntropyCurve ideas={ideas} target={(data && data.target) || 0.5} />
        <p className="axis-note">
          An idea you are certain will work teaches nothing. An idea you are
          certain will <em>fail</em> teaches nothing either — that is the half
          that gets skipped, and why bad ideas are ranked, not avoided.
        </p>
      </section>

      <section className="card">
        <h3>candidates<span className="sub">ranked by bits per unit cost × novelty</span></h3>
        <ol className="ideas">
          {ideas.map((i, n) => (
            <li key={n}>
              <span className="origin" style={{ background: colourOf(i.origin) }}>{i.origin}</span>
              <span className="idea-text">{i.text}</span>
              <span className="idea-nums">
                <em title="predicted chance of success">p {i.p.toFixed(2)}</em>
                <em title="information gain in bits">{i.bits.toFixed(2)} bits</em>
                <em title="ranking value">v {i.value.toFixed(3)}</em>
              </span>
            </li>
          ))}
          {ideas.length === 0 && !busy && (
            <li className="empty">nothing to explore yet — the store needs some history first.</li>
          )}
        </ol>
      </section>

      {runs && (
        <section className="card">
          <h3>results<span className="sub">a refuted experiment is a stored result, not a wasted cycle</span></h3>
          <ul className="experiments">
            {runs.map((e, i) => (
              <li key={i} className={e.passed ? 'held' : e.ran ? 'refuted' : 'not-run'}>
                <span className="verdict">
                  {e.passed ? 'held' : e.ran ? 'refuted' : 'not run'}
                </span>
                <span className="idea-text">{e.idea}</span>
                <span className="detail">{e.detail}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {strategy && (
        <section className="card">
          <h3>which generator earns its keep<span className="sub">regret-matched mix over idea sources</span></h3>
          <div className="strategy">
            {Object.entries(strategy).map(([name, p]) => (
              <div key={name} className="strategy-row">
                <span className="strategy-name">{name}</span>
                <span className="bar-track">
                  <span className="bar-fill" style={{ width: `${p * 100}%`, background: colourOf(name) }} />
                </span>
                <span className="bar-value">{(p * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
          <p className="axis-note">
            Rewarded for <em>information</em>, not success — a generator rewarded
            for being right would stop proposing the ideas worth running.
          </p>
        </section>
      )}
    </div>
  )
}
