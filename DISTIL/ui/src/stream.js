// Watching the chain being thought, rather than receiving it.
//
// EventSource rather than fetch+ReadableStream: it reconnects, it parses the
// framing, and the server is `http.server`, so the simplest thing that works on
// both ends is the right one. The caveat is that EventSource is GET-only, which
// is why the task rides in the query string.

export function streamAsk(task, answers, handlers) {
  const params = new URLSearchParams({ task })
  if (answers && Object.keys(answers).length) params.set('answers', JSON.stringify(answers))
  const source = new EventSource(`/api/ask/stream?${params}`)

  let finished = false
  const close = () => { finished = true; source.close() }

  source.addEventListener('step', (e) => handlers.onStep(JSON.parse(e.data)))
  source.addEventListener('result', (e) => { handlers.onResult(JSON.parse(e.data)) })
  source.addEventListener('error', (e) => {
    // A named `error` event carries a payload the server chose to send.
    if (e.data) { close(); handlers.onError(new Error(JSON.parse(e.data).error)) }
  })
  source.addEventListener('done', () => { close(); handlers.onDone() })

  source.onerror = () => {
    // The transport failed. After `done` this is just the close, and reporting
    // it would turn every successful run into an error; EventSource also retries
    // on its own, which would silently re-run the whole task.
    if (finished) return
    close()
    handlers.onError(new Error('the connection to the server dropped mid-answer'))
  }
  return close
}


// The autonomous loop. Same transport, different events: one per cycle, each
// carrying what it chose, what it found and what that was worth in bits.
//
// Stopping is by closing the connection. The server notices the failed write,
// sets its stop flag and lets the cycle in flight finish rather than being
// killed mid-forge -- so memory is never left half-written.
export function streamAuto(cycles, handlers) {
  const source = new EventSource(`/api/auto/stream?cycles=${cycles || 0}`)
  let finished = false
  const close = () => { finished = true; source.close() }

  source.addEventListener('cycle', (e) => handlers.onCycle(JSON.parse(e.data)))
  source.addEventListener('error', (e) => {
    if (e.data) { close(); handlers.onError(new Error(JSON.parse(e.data).error)) }
  })
  source.addEventListener('done', () => { close(); handlers.onDone() })
  source.onerror = () => {
    if (finished) return
    close()
    handlers.onError(new Error('the connection to the autonomous loop dropped'))
  }
  return close
}
