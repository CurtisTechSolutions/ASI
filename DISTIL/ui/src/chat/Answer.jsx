import React, { useState } from 'react'
import Detail from './Detail.jsx'
import Recalled from './Recalled.jsx'
import PayoffMatrix from '../viz/PayoffMatrix.jsx'
import GoalTree from '../viz/GoalTree.jsx'
import ChainFlow from '../viz/ChainFlow.jsx'

// An answer, in the order a person reads: what happened, then why.
//
// The verdict is first. Everything that produced it is underneath, collapsed,
// with a summary on each header saying whether it is worth opening. Grading is
// here rather than somewhere else because here is where you have just formed an
// opinion -- before this, grading the answer you were looking at meant copying a
// trace id, opening a different tab, finding the point in a scatter plot and
// clicking it, and a feedback loop with four steps of friction in it is a
// feedback loop nobody closes.
export default function Answer({ message, onGrade, onAnswer, onTrace }) {
  const { data, steps, streaming, task, error } = message

  if (error) return <p className="error">{error}</p>

  if (streaming && !data) {
    return (
      <div className="thinking">
        <p className="thinking-head">thinking about <em>{task}</em></p>
        <ol className="live-steps">
          {steps.map((s, i) => (
            <li key={i}><span className="kind">{s.kind}</span>{s.text}</li>
          ))}
          <li className="pending"><span className="kind">…</span></li>
        </ol>
      </div>
    )
  }
  if (!data) return null

  if (data.needs_clarification) {
    return <Clarify data={data} onAnswer={onAnswer} onTrace={onTrace} />
  }

  const frame = data.frame
  const chainLength = (data.chain || []).length

  return (
    <div className="answer">
      <p className={`verdict ${data.solved ? 'ok' : 'no'}`}>
        <strong>{data.solved ? 'Solved' : 'Not solved'}</strong>
        <span>{data.reason}</span>
        {/* The task, not the last goal. "Solved" with a goal still open was
            the old behaviour, and it was a lie by omission. */}
        {(data.goals_met > 0 || data.goals_open > 0) && (
          <em className="goal-count">
            {data.goals_met} met{data.goals_open > 0 ? `, ${data.goals_open} open` : ''}
          </em>
        )}
      </p>

      {data.attempts && data.attempts.length > 0 && (
        <ul className="attempts">
          {data.attempts.map((a, i) => (
            <li key={i} className={a.ok ? 'ok' : 'no'}>
              <span className="reframe">{a.reframe || 'direct'}</span>
              {a.via}
              {a.grade !== undefined && a.grade !== null &&
                <em>{a.grade > 0 ? '+' : ''}{a.grade}</em>}
            </li>
          ))}
        </ul>
      )}

      {data.boundary && <pre className="boundary">{data.boundary}</pre>}

      {data.trace_id && <GradeBar traceId={data.trace_id} onGrade={onGrade} />}

      <Detail title="what I already knew"
              summary={`${data.recall.hits.length} from memory, ranked`}>
        <Recalled recall={data.recall} onTrace={onTrace} />
      </Detail>

      {frame && (
        <Detail title="the game"
                summary={`${frame.players}, ${frame.payoff}, ${frame.horizon} · confidence ${frame.confidence.toFixed(2)}`}>
          <dl className="frame">
            <div><dt>players</dt><dd>{frame.players}</dd></div>
            <div><dt>payoff</dt><dd>{frame.payoff}</dd></div>
            <div><dt>horizon</dt><dd>{frame.horizon}</dd></div>
            <div><dt>information</dt><dd>{frame.information}</dd></div>
            <div><dt>referee</dt><dd>{frame.referee || 'none — nothing here can be self-graded'}</dd></div>
            <div><dt>solution concept</dt><dd>{frame.solution}</dd></div>
            <div className="wide"><dt>objective</dt><dd>{frame.objective}</dd></div>
          </dl>
          {frame.agenda && frame.agenda.items.length > 0 && (
            <ul className="agenda">
              {frame.agenda.items.map((item, i) => {
                const needsPerson = frame.agenda.needs_person.includes(item)
                return (
                  <li key={i} className={needsPerson ? 'person' : 'doable'}>
                    <span className="tag">{needsPerson ? 'needs a person' : 'executable'}</span>
                    {item}
                  </li>
                )
              })}
            </ul>
          )}
        </Detail>
      )}

      {data.payoff && (
        <Detail title="the choice"
                summary={`${data.payoff.matrix.length}×${data.payoff.states.length} against Nature · chose “${data.chosen}”`}>
          <PayoffMatrix payoff={data.payoff} select={data.select} />
        </Detail>
      )}

      {data.tree && (
        <Detail title="goals"
                summary={`${data.tree.goals.length - 1} distilled, stopping where each becomes checkable`}>
          <GoalTree tree={data.tree} chosen={data.chosen} />
        </Detail>
      )}

      {chainLength > 0 && (
        <Detail title="reasoning" summary={`${chainLength} typed steps, each with its inputs`}>
          <ChainFlow chain={data.chain} />
        </Detail>
      )}
    </div>
  )
}

// Clarification as a conversational turn: it asked, you answer, it continues.
function Clarify({ data, onAnswer, onTrace }) {
  const [answers, setAnswers] = useState({})
  const filled = Object.values(answers).filter((v) => v && v.trim()).length

  const submit = () => {
    const given = Object.fromEntries(
      Object.entries(answers).filter(([, v]) => v && v.trim()))
    const summary = data.questions
      .filter((q) => given[q.gap])
      .map((q) => given[q.gap])
      .join(' · ')
    onAnswer(data.task, given, summary || '(skipped)')
  }

  return (
    <div className="clarify">
      <p className="clarify-lead">
        I will not guess at this one. {data.reason}
      </p>
      <ul className="questions">
        {data.questions.map((q) => (
          <li key={q.gap}>
            <label htmlFor={`q-${q.gap}`}>
              <span className="gap">{q.gap}</span>
              {q.text}
            </label>
            {q.known
              ? <p className="known">memory already says: {q.known}</p>
              : <p className="unblocks">{q.unblocks}</p>}
            <input
              id={`q-${q.gap}`}
              value={answers[q.gap] || ''}
              placeholder={q.known || 'your answer'}
              onChange={(e) => setAnswers({ ...answers, [q.gap]: e.target.value })}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
            />
          </li>
        ))}
      </ul>
      <button className="primary" onClick={submit}>
        {filled ? `answer ${filled} and continue` : 'continue without answering'}
      </button>

      <Detail title="what I already knew"
              summary={`${data.recall.hits.length} from memory — I searched before asking`}>
        <Recalled recall={data.recall} onTrace={onTrace} />
      </Detail>
    </div>
  )
}

function GradeBar({ traceId, onGrade }) {
  const [given, setGiven] = useState(null)
  const [failed, setFailed] = useState(null)

  const send = async (score) => {
    try { await onGrade(traceId, score); setGiven(score) }
    catch (err) { setFailed(err.message) }
  }

  if (failed) return <p className="error">could not record that: {failed}</p>
  if (given !== null) {
    return (
      <p className="graded">
        recorded {given > 0 ? '+1' : given < 0 ? '−1' : '0'} — this changes what gets
        recalled next time, and a user grade counts double.
      </p>
    )
  }
  return (
    <div className="grade-bar">
      <span>was this right?</span>
      <button className="good" onClick={() => send(1)}>+1 right</button>
      <button onClick={() => send(0)}>0 unsure</button>
      <button className="bad" onClick={() => send(-1)}>−1 wrong</button>
    </div>
  )
}
