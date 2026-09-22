import React from 'react'

// The distilled goals. Indentation is depth; the marker is status.
//
// The badge that matters is "needs a check": distillation stops when a goal
// becomes verifiable (§8), so a leaf without one is the specification gap made
// visible rather than a rendering detail.
const MARK = {
  met: { glyph: '✓', cls: 'met' },
  failed: { glyph: '✗', cls: 'failed' },
  active: { glyph: '▸', cls: 'active' },
  blocked: { glyph: '■', cls: 'blocked' },
  abandoned: { glyph: '–', cls: 'abandoned' },
  open: { glyph: '○', cls: 'open' },
}

export default function GoalTree({ tree, chosen }) {
  if (!tree) return null
  const byParent = new Map()
  for (const g of tree.goals) {
    if (!byParent.has(g.parent)) byParent.set(g.parent, [])
    byParent.get(g.parent).push(g)
  }
  const root = tree.goals.find((g) => g.id === 'root')

  const render = (goal) => {
    const mark = MARK[goal.status] || MARK.open
    const isChosen = chosen && goal.text === chosen
    return (
      <li key={goal.id} className={`goal ${mark.cls}${isChosen ? ' chosen' : ''}`}>
        <span className="goal-mark">{mark.glyph}</span>
        <span className="goal-text">{goal.text}</span>
        {goal.id !== 'root' && (
          <span className={`goal-badge ${goal.atomic ? 'atomic' : 'vague'}`}>
            {goal.atomic ? 'verifiable' : 'needs a check'}
          </span>
        )}
        {(byParent.get(goal.id) || []).length > 0 && (
          <ul>{(byParent.get(goal.id) || []).map(render)}</ul>
        )}
      </li>
    )
  }

  return <ul className="goal-tree">{root ? render(root) : null}</ul>
}
