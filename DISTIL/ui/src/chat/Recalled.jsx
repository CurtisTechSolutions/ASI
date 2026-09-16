import React from 'react'
import { colourOf } from '../api.js'

// What the store already knew, shown on every single reply.
//
// This used to be a tab you had to think to visit, which meant the memory layer
// -- the thing the whole system is built on -- was invisible unless you went
// looking. Worse, a question that got gated on a clarification returned no
// recall at all, so the one case where "what do I already know about this?" is
// most useful was the case that answered it least.
export default function Recalled({ recall, onTrace }) {
  if (!recall || !recall.hits || !recall.hits.length) return null
  const w = recall.weights || {}
  const reordered = recall.hits.length > 1 &&
    recall.hits.slice(1).some((h) => h.similarity > recall.hits[0].similarity)

  return (
    <div className="recalled">
      {recall.asked && (
        <p className="recall-note">
          searched for <strong>{recall.query}</strong>, not “{recall.asked}” — that is
          what the clarifying questions established you meant.
        </p>
      )}
      {reordered && (
        <p className="recall-note insight">
          the top hit is <strong>not</strong> the most similar one — credibility outranked
          similarity, which is the difference between this and a vector store.
        </p>
      )}
      <ul className="hits">
        {recall.hits.map((h) => (
          <li key={h.id}>
            <button className="hit-line" onClick={() => onTrace && onTrace(h.id)}
                    title="open this memory and everything linked to it">
              <span className="pill" style={{ borderColor: colourOf(h.kind) }}>{h.kind}</span>
              <span className="hit-text">{h.text}</span>
              {h.verified && <span className="verified">verified</span>}
              {h.grade !== null && h.grade !== undefined && (
                <span className={`hit-grade ${h.grade > 0 ? 'good' : 'bad'}`}>
                  {h.grade > 0 ? '+' : ''}{h.grade.toFixed(2)}
                </span>
              )}
            </button>
            <div className="factors" title={`similarity^${w.similarity} × credibility^${w.credibility} × recency^${w.recency}`}>
              <span>sim {h.similarity.toFixed(3)}</span>
              <span>cred {h.credibility.toFixed(3)}</span>
              <span>rec {h.recency.toFixed(3)}</span>
              <strong>{h.score.toFixed(4)}</strong>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
