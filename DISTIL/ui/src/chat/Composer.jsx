import React, { useEffect, useRef, useState } from 'react'
import { COMMANDS, suggest } from '../commands.js'
import { SpeechKind, listenInBrowser, recordForServer, route, warning } from '../speech.js'

// The one input. Type, or hold the microphone.
//
// The slash menu appears as you type and Tab or Enter completes it, which is how
// every chat client works and therefore needs no explaining. The microphone
// states what it is about to do with your voice *before* it listens rather than
// in a tooltip: on this machine, or via the browser -- which in Chrome means
// Google. A local-first tool that quietly streams your voice to a third party
// would be lying by omission.
export default function Composer({ onSend, busy, speech }) {
  const [text, setText] = useState('')
  const [listening, setListening] = useState(false)
  const [error, setError] = useState(null)
  const [transcribing, setTranscribing] = useState(false)
  const [picked, setPicked] = useState(0)
  const box = useRef(null)
  const stopper = useRef(null)

  const kind = route(speech)
  const note = warning(kind, speech)
  const options = suggest(text)

  useEffect(() => { setPicked(0) }, [text])
  useEffect(() => () => { stopper.current?.() }, [])

  const submit = (value) => {
    const out = (value ?? text).trim()
    if (!out || busy) return
    setText('')
    setError(null)
    onSend(out)
  }

  const complete = (command) => {
    setText(`/${command.name} `)
    box.current?.focus()
  }

  const onKeyDown = (e) => {
    if (options.length) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setPicked((p) => (p + 1) % options.length); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); setPicked((p) => (p - 1 + options.length) % options.length); return }
      if (e.key === 'Tab' || (e.key === 'Enter' && options.length > 1)) {
        e.preventDefault(); complete(options[picked]); return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  // --- the microphone -------------------------------------------------------

  const startListening = async () => {
    setError(null)
    if (kind === SpeechKind.BROWSER) {
      setListening(true)
      stopper.current = listenInBrowser({
        onPartial: setText,
        onFinal: (final) => { setListening(false); if (final) setText(final) },
        onError: (err) => { setListening(false); setError(err.message) },
      })
      return
    }
    const recorder = await recordForServer({ onError: (err) => setError(err.message) })
    if (!recorder) return
    setListening(true)
    stopper.current = async () => {
      setListening(false)
      const wav = await recorder.stop()
      if (!wav) return
      setTranscribing(true)
      try {
        const res = await fetch('/api/transcribe', {
          method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav,
        })
        const payload = await res.json()
        if (payload.ok === false) throw new Error(payload.error)
        setText((prior) => (prior ? `${prior} ${payload.text}` : payload.text))
      } catch (err) {
        setError(err.message)
      } finally {
        setTranscribing(false)
      }
    }
  }

  const toggleMic = () => {
    if (listening) { stopper.current?.(); stopper.current = null; setListening(false) }
    else startListening()
  }

  return (
    <div className="composer">
      {options.length > 0 && (
        <ul className="slash-menu">
          {options.map((c, i) => (
            <li key={c.name} className={i === picked ? 'on' : ''}
                onMouseDown={(e) => { e.preventDefault(); complete(c) }}>
              <code>/{c.name}</code>
              {c.args && <em>{c.args}</em>}
              <span>{c.blurb}</span>
            </li>
          ))}
        </ul>
      )}

      {error && <div className="composer-error">{error}</div>}

      <div className={`composer-box ${listening ? 'listening' : ''}`}>
        <textarea
          ref={box}
          rows={1}
          value={text}
          placeholder={listening ? 'listening…' : 'ask anything, or / for commands'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
        />
        {kind !== SpeechKind.NONE && (
          <button
            className={`mic ${listening ? 'on' : ''}`}
            onClick={toggleMic}
            disabled={busy || transcribing}
            title={note.text}
            aria-label={listening ? 'stop listening' : 'speak your question'}
          >
            {transcribing ? '…' : listening ? '■' : '●'}
          </button>
        )}
        <button className="send" onClick={() => submit()} disabled={busy || !text.trim()}>
          {busy ? 'thinking…' : 'send'}
        </button>
      </div>

      <p className={`composer-note ${note.level}`}>
        {kind === SpeechKind.NONE
          ? `${COMMANDS.length} commands — type / to see them. ${note.text}`
          : `${COMMANDS.length} commands — type / to see them. Microphone: ${note.text}`}
      </p>
    </div>
  )
}
