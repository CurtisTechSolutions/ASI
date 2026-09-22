import React, { useState } from 'react'

// A collapsed section of an answer.
//
// The old Ask view rendered six expanded cards at once -- about two thousand
// pixels for one question, with the verdict at the very bottom, under everything
// that led to it. The reasoning still has to be *there*, because a chain nobody
// can inspect is just an assertion; it does not have to be in the way. So the
// summary carries the number that tells you whether to open it.
export default function Detail({ title, summary, children, open = false, tone }) {
  const [shown, setShown] = useState(open)
  return (
    <div className={`detail ${shown ? 'open' : ''} ${tone || ''}`}>
      <button className="detail-head" onClick={() => setShown(!shown)} aria-expanded={shown}>
        <span className="chevron" aria-hidden="true">{shown ? '▾' : '▸'}</span>
        <strong>{title}</strong>
        <span className="detail-summary">{summary}</span>
      </button>
      {shown && <div className="detail-body">{children}</div>}
    </div>
  )
}
