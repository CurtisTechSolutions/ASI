import React, { useMemo, useState } from 'react'
import { colourOf } from '../api.js'

// The embedding space, projected to a plane (distil/project.py).
//
// The axes have NO meaning: they are the two directions of greatest variance.
// Distance between points is the only thing this picture is entitled to claim,
// so there are no axis ticks to invite reading a value off them, and the two
// axes share one scale so a tight cluster is not stretched to fill the frame.
export default function Scatter({ traces, selected, onSelect, size = 560 }) {
  const [hover, setHover] = useState(null)
  const pad = 28
  const place = (v) => pad + ((v + 1) / 2) * (size - pad * 2)

  // Radius carries how often a trace has been recalled, on a saturating scale --
  // a trace read a thousand times should not be a hundred times the area of one
  // read ten times.
  const radius = (t) => 3.2 + Math.min(1, Math.log1p(t.hits) / Math.log(30)) * 5.5

  const kinds = useMemo(
    () => [...new Set(traces.map((t) => t.kind))].sort(),
    [traces],
  )
  const shown = hover || selected

  return (
    <div className="scatter">
      <svg viewBox={`0 0 ${size} ${size}`} className="scatter-svg" role="img"
           aria-label="memory traces projected to two dimensions">
        <defs>
          <radialGradient id="glow">
            <stop offset="0%" stopColor="#7aa2f7" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#7aa2f7" stopOpacity="0" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width={size} height={size} className="scatter-bg" />
        <line x1={pad} y1={size / 2} x2={size - pad} y2={size / 2} className="axis" />
        <line x1={size / 2} y1={pad} x2={size / 2} y2={size - pad} className="axis" />

        {traces.map((t) => {
          const isShown = shown && shown.id === t.id
          return (
            <circle
              key={t.id}
              cx={place(t.x)}
              cy={place(-t.y)}
              r={radius(t) * (isShown ? 1.9 : 1)}
              fill={colourOf(t.kind)}
              // Ungraded traces sit at the credibility prior and are drawn
              // hollow, so "nobody has checked this" is visible at a glance
              // rather than inferred from a number in a panel.
              fillOpacity={t.grade === null ? 0.18 : 0.42 + t.credibility * 0.5}
              stroke={colourOf(t.kind)}
              strokeWidth={isShown ? 2 : 1}
              className="dot"
              onMouseEnter={() => setHover(t)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect && onSelect(t)}
            />
          )
        })}
      </svg>

      <div className="scatter-legend">
        {kinds.map((k) => (
          <span key={k} className="legend-item">
            <i style={{ background: colourOf(k) }} />
            {k}
          </span>
        ))}
        <span className="legend-note">hollow = ungraded · size = times recalled</span>
      </div>

      {shown && (
        <div className="scatter-tip">
          <span className="pill" style={{ borderColor: colourOf(shown.kind) }}>
            {shown.kind}
          </span>
          <span className="tip-grade">
            {shown.grade === null ? 'ungraded' : `grade ${shown.grade.toFixed(2)}`}
            {shown.verified && ' · verified'}
          </span>
          <p>{shown.text.slice(0, 260)}</p>
        </div>
      )}
    </div>
  )
}
