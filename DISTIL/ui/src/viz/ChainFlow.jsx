import React, { useState } from 'react'

// The typed chain of thought (distil/reason.py §6).
//
// Rendered as an ordered spine rather than prose, because that is what it is:
// each step is an object with its inputs attached, and the payload is worth
// opening. FRAME first is the design's claim made visible -- which solution
// concept is valid is decided before anything is planned.
const COLOUR = {
  FRAME: '#c0caf5', AGENDA: '#7dcfff', PRECEDENT: '#41a6b5', WHY: '#bb9af7',
  CHALLENGE: '#ff9e64', RETRIEVE: '#7aa2f7', DISTILL: '#e0af68', PAYOFF: '#2ac3de',
  SELECT: '#9ece6a', ACT: '#73daca', VERIFY: '#f7768e', CREDIT: '#a9b1d6', NOTE: '#565f89',
}

export default function ChainFlow({ chain }) {
  const [open, setOpen] = useState(null)
  if (!chain || !chain.length) return null
  return (
    <ol className="chain">
      {chain.map((step, i) => {
        const hasPayload = step.payload && Object.keys(step.payload).length > 0
        return (
          <li key={i} className="chain-step">
            <span className="chain-kind" style={{ color: COLOUR[step.kind] || '#9aa5ce' }}>
              {step.kind}
            </span>
            <span className="chain-text">{step.text}</span>
            {hasPayload && (
              <button className="chain-toggle" onClick={() => setOpen(open === i ? null : i)}>
                {open === i ? 'hide' : 'payload'}
              </button>
            )}
            {open === i && (
              <pre className="chain-payload">{JSON.stringify(step.payload, null, 2)}</pre>
            )}
          </li>
        )
      })}
    </ol>
  )
}
