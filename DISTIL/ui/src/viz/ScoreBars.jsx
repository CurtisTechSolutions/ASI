import React from 'react'

// Why a recall hit ranked where it did (distil/memory.py §4.2).
//
// The score is similarity^ws · credibility^wc · recency^wr, so it is NOT the sum
// of the parts and must not be drawn as a stacked bar -- that would state an
// arithmetic the numbers do not satisfy. Three separate tracks instead, with the
// product reported beside them.
export default function ScoreBars({ hit, weights }) {
  const parts = [
    { key: 'similarity', label: 'similarity', value: hit.similarity, weight: weights.similarity },
    { key: 'credibility', label: 'credibility', value: hit.credibility, weight: weights.credibility },
    { key: 'recency', label: 'recency', value: hit.recency, weight: weights.recency },
  ]
  return (
    <div className="score-bars">
      {parts.map((p) => (
        <div key={p.key} className={`bar-row bar-${p.key}`}>
          <span className="bar-label">{p.label}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${Math.max(0, p.value) * 100}%` }} />
          </span>
          <span className="bar-value">{p.value.toFixed(3)}</span>
          <span className="bar-weight" title="policy exponent">^{p.weight}</span>
        </div>
      ))}
    </div>
  )
}
