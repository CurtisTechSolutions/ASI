import React, { useCallback, useEffect, useRef, useState } from 'react'
import { get, post } from './api.js'
import { parse } from './commands.js'
import { streamAsk, streamAuto } from './stream.js'
import Composer from './chat/Composer.jsx'
import Message from './chat/Message.jsx'

// One conversation, and nothing else.
//
// This was six tabs. Tabs are a dashboard pattern: they make you navigate to a
// system and hold the correlation between views in your head. Everything here is
// a message instead -- a question, a tool listing, an experiment run -- so the
// transcript is the record of what happened, in order, and the thing you asked
// four questions ago is still on the page underneath the answer.
//
// Capabilities that used to be tabs are slash commands (see commands.js). The
// common case, asking a question, costs no syntax at all.

let counter = 0
const nextId = () => `m${counter++}`

const GREETING = {
  id: nextId(),
  role: 'agent',
  kind: 'welcome',
  text: 'Ask me something, or speak it. Every question searches my memory first.',
}

export default function App() {
  const [messages, setMessages] = useState([GREETING])
  const [busy, setBusy] = useState(false)
  const [state, setState] = useState(null)
  const [speech, setSpeech] = useState(null)
  const bottom = useRef(null)

  const refreshState = useCallback(async () => {
    try { setState(await get('state')) } catch { /* the header degrades quietly */ }
  }, [])

  useEffect(() => {
    refreshState()
    get('speech').then(setSpeech).catch(() => setSpeech(null))
  }, [refreshState])

  // Say it once, in the transcript, rather than only in a badge nobody is
  // looking at. "Why isn't it using my model?" is the question this answers.
  const warned = useRef(false)
  useEffect(() => {
    if (!state || warned.current) return
    const health = state.provider_health
    const wanted = state.providers.find((p) => p.name === 'ollama')
    if (health.failing) {
      warned.current = true
      add({ role: 'agent', kind: 'error',
            text: `I am configured to use ${state.provider}, but all ${health.calls} `
                + `call(s) to it have failed — so I am falling back to offline rules for `
                + `everything. ${health.last_error || ''}` })
    } else if (state.provider === 'local' && wanted && !wanted.available && wanted.why) {
      warned.current = true
      add({ role: 'agent', kind: 'error',
            text: `Running on offline rules, not a model. ${wanted.why}`
                + ` Then restart me — the provider is chosen once, at startup.` })
    }
  }, [state])

  // Plant the starter toolkit on an empty memory rather than making someone
  // discover that they had to. A system that can bootstrap itself and waits to
  // be told to is just a system with a hidden first step.
  const toolCount = state ? state.stats.tools.length : null
  useEffect(() => {
    if (toolCount !== 0) return          // null = not loaded yet, >0 = already has some
    let cancelled = false
    post('seed', {}).then((out) => {
      if (cancelled || !out.planted.length) return
      add({ role: 'agent', kind: 'seed', data: out,
            text: `I had no tools, so I planted ${out.planted.length} and verified each one.` })
      refreshState()
    }).catch(() => {})
    return () => { cancelled = true }
  }, [toolCount])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const add = (message) => {
    const full = { id: nextId(), ...message }
    setMessages((prior) => [...prior, full])
    return full.id
  }
  const patch = (id, change) =>
    setMessages((prior) => prior.map((m) => (m.id === id ? { ...m, ...change } : m)))

  // --- asking ---------------------------------------------------------------

  const ask = async (task, answers) => {
    const id = add({ role: 'agent', kind: 'ask', task, steps: [], streaming: true })
    setBusy(true)
    streamAsk(task, answers, {
      onStep: (step) => setMessages((prior) => prior.map(
        (m) => (m.id === id ? { ...m, steps: [...m.steps, step] } : m))),
      onResult: (data) => patch(id, { data }),
      onDone: () => { patch(id, { streaming: false }); setBusy(false); refreshState() },
      onError: (err) => { patch(id, { streaming: false, error: err.message }); setBusy(false) },
    })
  }

  // --- commands -------------------------------------------------------------

  const runCommand = async (parsed, raw) => {
    if (parsed.unknown) {
      add({ role: 'agent', kind: 'error',
            text: `no command called /${parsed.unknown} — type /help to see them all` })
      return
    }
    const { command, rest } = parsed
    if (command.needs && !rest) {
      add({ role: 'agent', kind: 'error', text: command.needs })
      return
    }
    const id = add({ role: 'agent', kind: command.name, streaming: true })
    setBusy(true)
    try {
      const out = await command.run(rest)
      patch(id, { ...out, streaming: false })
    } catch (err) {
      patch(id, { kind: 'error', text: err.message, streaming: false })
    } finally {
      setBusy(false)
      refreshState()
    }
  }

  const send = async (text, answers) => {
    if (!text.trim() || busy) return
    add({ role: 'user', kind: 'text', text })
    const parsed = parse(text)
    if (parsed) await runCommand(parsed, text)
    else await ask(text.trim(), answers)
  }

  // Answering a clarification is a reply in the conversation, not a form: the
  // agent asked, so the reply goes back the way anything else would.
  const answer = async (task, answers, summary) => {
    add({ role: 'user', kind: 'text', text: summary })
    await ask(task, answers)
  }

  // --- running by itself ----------------------------------------------------
  //
  // Not a separate screen. The loop narrates into the same transcript as
  // everything else, so what it decided on its own sits in the same record as
  // what you asked it -- which is the only way to notice that it went somewhere
  // you did not expect.
  const stopAuto = useRef(null)
  const [auto, setAuto] = useState(false)

  const toggleAuto = () => {
    if (auto) {
      stopAuto.current?.()
      stopAuto.current = null
      setAuto(false)
      add({ role: 'agent', kind: 'text', text: 'stopped — it finishes the cycle it is in first.' })
      refreshState()
      return
    }
    const id = add({ role: 'agent', kind: 'auto', cycles: [], strategy: null, streaming: true })
    setAuto(true)
    stopAuto.current = streamAuto(0, {
      onCycle: (cycle) => setMessages((prior) => prior.map((m) => (
        m.id === id ? { ...m, cycles: [...m.cycles, cycle], strategy: cycle.strategy } : m))),
      onDone: () => { patch(id, { streaming: false }); setAuto(false); refreshState() },
      onError: (err) => { patch(id, { streaming: false, error: err.message }); setAuto(false) },
    })
  }

  useEffect(() => () => stopAuto.current?.(), [])

  const grade = useCallback(async (id, score) => {
    await post('grade', { id, score })
    refreshState()
  }, [refreshState])

  const look = useCallback(async (traceId) => {
    const id = add({ role: 'agent', kind: 'trace', streaming: true })
    try {
      patch(id, { data: await get('trace', { id: traceId }), streaming: false })
    } catch (err) {
      patch(id, { kind: 'error', text: err.message, streaming: false })
    }
  }, [])

  return (
    <div className="app">
      <header className="masthead">
        <h1>DISTIL</h1>
        <div className="status">
          {state ? (
            <>
              {/* A provider that is selected but answering nothing is the one
                  failure this header has to show: every caller degrades to a
                  heuristic on error, so the agent keeps working and the only
                  symptom is that the answers got worse. */}
              <span className={`badge ${state.provider_health.failing ? 'bad' : ''}`}
                    title={state.provider_health.failing
                      ? `${state.provider_health.failures} of ${state.provider_health.calls} calls failed — ${state.provider_health.last_error}`
                      : `${state.provider_health.calls} call(s), ${state.provider_health.failures} failed`}>
                {state.provider}
                {state.provider_health.failing && ' — not answering'}
              </span>
              <span className="badge dim">{state.stats.traces} remembered</span>
              <span className="badge dim">{state.stats.tools.length} tools</span>
            </>
          ) : (
            <span className="badge dim">connecting…</span>
          )}
          <button className={`auto-toggle ${auto ? 'on' : ''}`} onClick={toggleAuto}
                  title="question, experiment, build, consolidate and tune, choosing its own next move">
            {auto ? '■ stop' : '▸ run by itself'}
          </button>
        </div>
      </header>

      <main className="transcript">
        {messages.map((m) => (
          <Message key={m.id} message={m} onGrade={grade} onAnswer={answer} onTrace={look} />
        ))}
        <div ref={bottom} />
      </main>

      <Composer onSend={send} busy={busy} speech={speech} />
    </div>
  )
}
