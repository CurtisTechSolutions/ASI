import { useCallback, useEffect, useRef, useState } from "react";
import { api, unwrapJob } from "../api.js";

const POLL_MS = 1000;
const RETRY_MS = 2000;

/**
 * Lifecycle of one asynchronous API job (train / 2nrl / evolve).
 *
 * `start(submit)` calls `submit()` - which POSTs and resolves to {"job": {...}} -
 * then polls GET /api/job every second until `state` is no longer "running".
 * `stop(submitStop)` posts the stop request (default POST /api/job/stop) and keeps
 * polling until the terminal state is observed. On mount the hook adopts the
 * server's current/last job when its type matches, so reopening the page during
 * a long run keeps the live view.
 */
export function useJob(type) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);
  const mounted = useRef(false);
  const trackedId = useRef(null);

  const clearTimer = useCallback(() => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const poll = useCallback(
    async function tick() {
      let next;
      try {
        next = unwrapJob(await api.job());
      } catch (err) {
        if (!mounted.current) return;
        setError(`Polling failed: ${err.message}`);
        clearTimer();
        timer.current = setTimeout(tick, RETRY_MS);
        return;
      }
      if (!mounted.current) return;
      if (!next || (trackedId.current !== null && next.id !== undefined && next.id !== trackedId.current)) {
        // The job we were following is gone or was replaced by another one.
        trackedId.current = null;
        return;
      }
      setJob(next);
      if (next.state === "running") {
        clearTimer();
        timer.current = setTimeout(tick, POLL_MS);
      } else {
        trackedId.current = null;
      }
    },
    [clearTimer],
  );

  const follow = useCallback(
    (current) => {
      trackedId.current = current && current.id !== undefined ? current.id : null;
      clearTimer();
      timer.current = setTimeout(poll, POLL_MS);
    },
    [clearTimer, poll],
  );

  const start = useCallback(
    async (submit) => {
      setError(null);
      setBusy(true);
      try {
        const first = unwrapJob(await submit());
        if (!mounted.current) return first;
        if (first) setJob(first);
        // Poll even without a job description so the server's job is discovered.
        if (!first || first.state === "running") follow(first);
        return first;
      } catch (err) {
        if (mounted.current) setError(err.message);
        return null;
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [follow],
  );

  const stop = useCallback(
    async (submitStop) => {
      setError(null);
      try {
        const next = unwrapJob(await (submitStop ? submitStop() : api.stopJob()));
        if (!mounted.current) return;
        if (next) setJob(next);
        // The worker thread may need a moment to notice the stop event; keep polling.
        if (!next || next.state === "running") follow(next);
      } catch (err) {
        if (mounted.current) setError(err.message);
      }
    },
    [follow],
  );

  useEffect(() => {
    mounted.current = true;
    (async () => {
      try {
        const current = unwrapJob(await api.job());
        if (!mounted.current || !current || current.type !== type) return;
        setJob(current);
        if (current.state === "running") follow(current);
      } catch {
        // Nothing to adopt; the status bar reports connectivity problems.
      }
    })();
    return () => {
      mounted.current = false;
      clearTimer();
    };
  }, [type, follow, clearTimer]);

  const clearError = useCallback(() => setError(null), []);

  return {
    job,
    running: Boolean(job && job.state === "running"),
    busy,
    error,
    start,
    stop,
    clearError,
  };
}
