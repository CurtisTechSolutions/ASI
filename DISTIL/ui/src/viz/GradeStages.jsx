import React from 'react'

// The five stages a tool passes to be believed (distil/grade.py §5).
// Shown as a row of pills because the ORDER matters: a later stage marked
// "not reached" is a different fact from one that ran and failed.
export default function GradeStages({ grade, compact = false }) {
  if (!grade) return <span className="stage-row"><span className="stage none">ungraded</span></span>
  return (
    <span className={`stage-row${compact ? ' compact' : ''}`}>
      {grade.stages.map((s, i) => (
        <span
          key={i}
          className={`stage ${s.passed ? 'pass' : s.detail.startsWith('not reached') ? 'skip' : 'fail'}`}
          title={`${s.name}: ${s.detail}`}
        >
          {s.name}
        </span>
      ))}
      <span className={`stage score ${grade.score > 0 ? 'pass' : 'fail'}`}>
        {grade.score > 0 ? '+' : ''}{grade.score.toFixed(2)}
      </span>
    </span>
  )
}
