import React, { useCallback, useEffect, useState } from 'react'
import { get, post } from './api.js'
import { useAsync } from './useAsync.js'
import Ask from './panels/Ask.jsx'
import Memory from './panels/Memory.jsx'
import Recall from './panels/Recall.jsx'
import Tools from './panels/Tools.jsx'
import Explore from './panels/Explore.jsx'
import System from './panels/System.jsx'

const TABS = [
  ['ask', 'Ask', 'frame the game, distil it, act'],
  ['memory', 'Memory', 'the embedding layer, projected'],
  ['recall', 'Recall', 'why a memory ranked where it did'],
  ['tools', 'Tools', 'capabilities, local and remote'],
  ['explore', 'Explore', 'curiosity, ranked in bits'],
  ['system', 'System', 'stats, policy, cases, maintenance'],
]

// The tab lives in the URL fragment. Without it a reload dropped you back on
// Ask, which is the wrong place to land when you were reading the memory plot --
// and it makes a particular view linkable.
const tabFromHash = () => {
  const key = window.location.hash.replace(/^#\/?/, '').split('?')[0]
  return TABS.some(([k]) => k === key) ? key : 'ask'
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash)
  const state = useAsync(() => get('state'), [])

  useEffect(() => {
    const sync = () => setTab(tabFromHash())
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  }, [])

  const go = (key) => {
    window.location.hash = `/${key}`
    setTab(key)
  }

  const grade = useCallback(
    async (id, score) => post('grade', { id, score }),
    [],
  )

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>DISTIL</h1>
          <span>
            an agent whose memory is an embedding layer that grades itself, and
            whose reasoning is a game played against Nature
          </span>
        </div>
        <div className="status">
          {state.error ? (
            <span className="offline" title={state.error}>server unreachable</span>
          ) : state.data ? (
            <>
              <span className="badge">{state.data.provider}</span>
              <span className="badge dim">{state.data.stats.traces} traces</span>
              <span className="badge dim">{state.data.stats.tools.length} tools</span>
            </>
          ) : (
            <span className="badge dim">connecting…</span>
          )}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map(([key, label, hint]) => (
          <button
            key={key}
            className={tab === key ? 'on' : ''}
            onClick={() => go(key)}
            title={hint}
          >
            {label}
          </button>
        ))}
      </nav>

      {state.error && (
        <div className="error global">
          {state.error}
          <button className="link" onClick={state.run}>retry</button>
        </div>
      )}

      <main>
        {tab === 'ask' && <Ask />}
        {tab === 'memory' && <Memory onGrade={grade} />}
        {tab === 'recall' && <Recall />}
        {tab === 'tools' && <Tools />}
        {tab === 'explore' && <Explore />}
        {tab === 'system' && <System state={state.data} refreshState={state.run} />}
      </main>

      <footer>
        local tool — it writes files and executes generated code in a sandbox.
        Not hardened, not authenticated, bound to loopback.
      </footer>
    </div>
  )
}
