import React, { useEffect, useRef, useState } from 'react'
import { post } from '../api.js'
import ChainFlow from '../viz/ChainFlow.jsx'
import GoalTree from '../viz/GoalTree.jsx'
import PayoffMatrix from '../viz/PayoffMatrix.jsx'

// The main loop, in the order the system runs it:
//   understand the game -> ask if it is unclear -> distil -> choose -> act.
//
// The clarification questions are answered IN PLACE and resubmitted, because
// that is what the backend does with them: the objective becomes what the work
// is distilled against, so an answer changes the goal tree rather than decorating
// the page.
// A task can be supplied in the fragment -- #/ask?task=...&run=1 -- so a
// particular run is linkable and survives a reload.
const fromHash = () => {
  const q = window.location.hash.split('?')[1]
  return q ? new URLSearchParams(q) : new URLSearchParams()
}

export default function Ask() {
  const [task, setTask] = useState(
    fromHash().get('task') || 'build a csv parser that passes the test suite',
  )
  const [answers, setAnswers] = useState({})
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const autoRan = useRef(false)

  const run = async (withAnswers) => {
    setBusy(true)
    setError(null)
    try {
      const data = await post('ask', { task, answers: withAnswers || undefined })
      setResult(data)
      if (!data.needs_clarification) setAnswers({})
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (autoRan.current || fromHash().get('run') !== '1') return
    autoRan.current = true
    run()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const frame = result && result.frame
  const agenda = frame && frame.agenda

  return (
    <div className="panel ask">
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); run() }}>
        <input
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="give it a task"
          aria-label="task"
        />
        <button type="submit" disabled={busy || !task.trim()}>
          {busy ? 'thinking…' : 'ask'}
        </button>
      </form>

      <p className="hint">
        try <button className="link" onClick={() => setTask('make the thing better')}>
          make the thing better
        </button> to see it refuse to guess, or{' '}
        <button className="link" onClick={() => setTask('win a chess endgame against a stronger opponent')}>
          win a chess endgame
        </button> to see the game framed differently.
      </p>

      {error && <div className="error">{error}</div>}

      {frame && (
        <section className="card">
          <h3>the game<span className="sub">understood before anything is planned</span></h3>
          <dl className="frame-grid">
            <div><dt>players</dt><dd>{frame.players}</dd></div>
            <div><dt>payoff</dt><dd>{frame.payoff}</dd></div>
            <div><dt>horizon</dt><dd>{frame.horizon}</dd></div>
            <div><dt>information</dt><dd>{frame.information}</dd></div>
            <div><dt>referee</dt>
              <dd className={frame.referee ? '' : 'absent'}>
                {frame.referee || 'none — nothing here can be self-graded'}
              </dd>
            </div>
            <div className="wide"><dt>solution concept</dt><dd>{frame.solution}</dd></div>
            <div className="wide"><dt>objective</dt><dd>{frame.objective || '—'}</dd></div>
          </dl>
          <div className="confidence">
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${frame.confidence * 100}%` }} />
            </span>
            <span>confidence {frame.confidence.toFixed(2)}</span>
            {!frame.understood && <span className="warn">below the play threshold</span>}
          </div>
        </section>
      )}

      {result && result.needs_clarification && (
        <section className="card clarify">
          <h3>
            it will not guess
            <span className="sub">{result.reason}</span>
          </h3>
          <p className="clarify-note">
            Nothing was distilled. A well-organised plan for the wrong problem is
            the most expensive thing this system can produce, so it stops here.
          </p>
          {(result.questions || []).map((q) => (
            <label key={q.gap} className="question">
              <span className="q-gap">{q.gap}</span>
              <span className="q-text">{q.text}</span>
              <span className="q-unblocks">{q.unblocks}</span>
              <input
                value={answers[q.gap] || ''}
                onChange={(e) => setAnswers({ ...answers, [q.gap]: e.target.value })}
                placeholder={q.known ? `on record: ${q.known}` : 'your answer'}
              />
            </label>
          ))}
          <button
            className="primary"
            disabled={busy || !Object.values(answers).some((v) => v && v.trim())}
            onClick={() => run(answers)}
          >
            answer and continue
          </button>
        </section>
      )}

      {agenda && agenda.items.length > 0 && (
        <section className="card">
          <h3>agenda<span className="sub">what the game needs vs what this system can do</span></h3>
          <ul className="agenda">
            {agenda.items.map((item, i) => {
              const blocking = agenda.needs_person.includes(item)
              const advisory = agenda.advisories.includes(item)
              return (
                <li key={i} className={blocking ? 'blocking' : advisory ? 'advisory' : 'doable'}>
                  <span className="agenda-tag">
                    {blocking ? 'needs a person' : advisory ? 'note' : 'executable'}
                  </span>
                  {item}
                </li>
              )
            })}
          </ul>
          {agenda.capabilities.length > 0 && (
            <div className="caps">
              {agenda.capabilities.map((c, i) => (
                <span key={i} className={`cap ${c.gap ? 'gap' : 'covered'}`}>
                  {c.action}
                  <em>{c.covered_by || 'nothing covers this'}</em>
                </span>
              ))}
            </div>
          )}
        </section>
      )}

      {result && result.payoff && (
        <section className="card">
          <h3>the choice<span className="sub">a normal-form game against Nature</span></h3>
          <PayoffMatrix payoff={result.payoff} select={result.select} />
        </section>
      )}

      {result && result.tree && (
        <section className="card">
          <h3>goals<span className="sub">distilled until each one is checkable</span></h3>
          <GoalTree tree={result.tree} chosen={result.chosen} />
        </section>
      )}

      {result && result.chain && (
        <section className="card">
          <h3>reasoning<span className="sub">typed steps, replayable, each with its inputs</span></h3>
          <ChainFlow chain={result.chain} />
        </section>
      )}

      {result && !result.needs_clarification && (
        <section className={`card verdict ${result.solved ? 'ok' : 'no'}`}>
          <h3>{result.solved ? 'solved' : 'not solved'}<span className="sub">{result.reason}</span></h3>
          {result.attempts && result.attempts.length > 0 && (
            <ul className="attempts">
              {result.attempts.map((a, i) => (
                <li key={i} className={a.ok ? 'ok' : 'no'}>
                  <span className="reframe">{a.reframe || 'direct'}</span>
                  {a.via}
                  {a.grade !== undefined && <em>{a.grade > 0 ? '+' : ''}{a.grade}</em>}
                </li>
              ))}
            </ul>
          )}
          {result.boundary && (
            <pre className="boundary">{result.boundary}</pre>
          )}
        </section>
      )}
    </div>
  )
}
