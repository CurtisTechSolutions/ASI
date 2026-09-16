import React from 'react'

// The game against Nature, as played (distil/reason.py §7).
//
// Rows are the goals on the frontier, columns are the states the world might be
// in, and the cell is the payoff. The chosen row is outlined and the most likely
// state is marked, so "why that goal?" has an answer you can point at rather
// than a number in a log.
export default function PayoffMatrix({ payoff, select }) {
  if (!payoff || !payoff.matrix || !payoff.matrix.length) return null
  const { matrix, states, goals } = payoff
  const belief = (select && select.belief) || {}
  const chosenText = select && select.goal
  const likeliest = Object.entries(belief).sort((a, b) => b[1] - a[1])[0]

  // Diverging scale: red for a payoff that costs you, green for one that pays.
  // Zero is the neutral midpoint, so the colour reads as sign-then-magnitude.
  const cell = (v) => {
    const t = Math.max(-1, Math.min(1, v))
    const hue = t >= 0 ? 145 : 353
    return `hsl(${hue} 60% ${18 + Math.abs(t) * 34}%)`
  }

  return (
    <div className="payoff">
      <table>
        <thead>
          <tr>
            <th className="corner">goal ╲ state of the world</th>
            {states.map((s) => (
              <th key={s} className={likeliest && likeliest[0] === s ? 'likely' : ''}>
                <span>{s}</span>
                {belief[s] !== undefined && <em>p {belief[s].toFixed(2)}</em>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const label = goals[i] || `row ${i + 1}`
            const role = payoff.roles && payoff.roles[i]
            const chosen = chosenText && label === chosenText
            return (
              <tr key={i} className={chosen ? 'chosen' : ''}>
                <th className="goal-label" title={label}>
                  {chosen && <span className="chosen-mark">▸</span>}
                  {label}
                  {role && <em className="role">{role}</em>}
                </th>
                {row.map((v, j) => (
                  <td key={j} style={{ background: cell(v) }}>
                    {v >= 0 ? v.toFixed(2) : v.toFixed(2)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
      {select && (
        <p className="payoff-note">
          chose <strong>{select.goal}</strong> by <strong>{select.rule}</strong>
          {likeliest && <> · world most likely <strong>{likeliest[0]}</strong></>}
        </p>
      )}
    </div>
  )
}
