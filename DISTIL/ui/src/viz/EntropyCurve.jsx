import React from 'react'
import { colourOf } from '../api.js'

// H(p) = -p·log₂p - (1-p)·log₂(1-p), with the candidate experiments plotted on
// it (distil/explore.py §10.2).
//
// This is the one picture that makes the design's central claim obvious: the
// value of an experiment peaks where you are least able to call it. An idea you
// are sure will work sits at the left floor; one you are sure will fail sits at
// the right floor, and people forget that half.
export default function EntropyCurve({ ideas, target = 0.5, width = 560, height = 220 }) {
  const pad = { l: 40, r: 16, t: 16, b: 34 }
  const w = width - pad.l - pad.r
  const h = height - pad.t - pad.b
  const x = (p) => pad.l + p * w
  const y = (bits) => pad.t + (1 - bits) * h

  const H = (p) => (p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)))
  const curve = Array.from({ length: 101 }, (_, i) => {
    const p = i / 100
    return `${i ? 'L' : 'M'}${x(p).toFixed(1)},${y(H(p)).toFixed(1)}`
  }).join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="entropy" role="img"
         aria-label="information gain against predicted success">
      <line x1={pad.l} y1={y(0)} x2={x(1)} y2={y(0)} className="axis" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={y(0)} className="axis" />
      <line x1={x(target)} y1={pad.t} x2={x(target)} y2={y(0)} className="target" />
      <text x={x(target)} y={pad.t - 4} className="target-label" textAnchor="middle">
        target {target}
      </text>
      <path d={curve} className="curve" />

      {(ideas || []).map((idea, i) => (
        <g key={i} className="idea-point">
          <line x1={x(idea.p)} y1={y(0)} x2={x(idea.p)} y2={y(idea.bits)} />
          <circle cx={x(idea.p)} cy={y(idea.bits)} r="5" fill={colourOf(idea.origin)}>
            <title>{`${idea.origin} · p=${idea.p} · ${idea.bits.toFixed(2)} bits\n${idea.text}`}</title>
          </circle>
        </g>
      ))}

      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <text key={p} x={x(p)} y={height - 12} className="tick" textAnchor="middle">{p}</text>
      ))}
      <text x={pad.l - 6} y={pad.t + 4} className="tick" textAnchor="end">1 bit</text>
      <text x={width / 2} y={height - 1} className="axis-label" textAnchor="middle">
        predicted chance of success
      </text>
    </svg>
  )
}
