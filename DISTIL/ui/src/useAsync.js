import { useCallback, useEffect, useState } from 'react'

// One hook for "fetch something, show a spinner, show the error if it fails".
//
// Panels were each growing their own loading/error/data triple and getting the
// error case subtly wrong -- usually by leaving stale data on screen next to a
// failure message, which reads as though the stale data is the answer.
export function useAsync(fn, deps, { immediate = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, busy: immediate })
  const run = useCallback(
    async (...args) => {
      setState((s) => ({ ...s, busy: true, error: null }))
      try {
        const data = await fn(...args)
        setState({ data, error: null, busy: false })
        return data
      } catch (error) {
        // Clear the data: a failure shown beside the previous answer reads as
        // though the previous answer is current.
        setState({ data: null, error: error.message, busy: false })
        return null
      }
    },
    deps, // eslint-disable-line react-hooks/exhaustive-deps
  )
  useEffect(() => {
    if (immediate) run()
  }, [run, immediate])
  return { ...state, run, setData: (data) => setState((s) => ({ ...s, data })) }
}
