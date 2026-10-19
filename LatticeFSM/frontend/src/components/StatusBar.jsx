import { useEffect, useState } from "react";
import { api } from "../api.js";
import { jobLine } from "../jobs.js";
import { fmtInt, fmtNum } from "../util.js";

const POLL_MS = 2000;
const STATUS_POLL_MS = 5000;
const JOB_STATUS_POLL_MS = 1000; // while a job runs, its progress is worth following closely

/**
 * Polls /api/stats every 2 s: the matrix's shape, the clock, the stimulation,
 * how much of the matrix has been written to. `onStats` receives every
 * payload (null when offline) so the rest of the app can share it; `tick`
 * changes whenever a panel has moved the machine, and makes the bar refresh
 * at once rather than at the next poll.
 *
 * Beside it, /api/status every 5 s: the language models' brief stats (the
 * chain's and the N-gram model's), the guard's, the uploads and the checkpoints, handed
 * to `onStatus`. A server without that route (404) is asked no more, and the
 * bar shows the machine alone, as it always has.
 *
 * A background job (training, teaching, a sweep) shows first, from whichever tab started it, with a button that
 * asks it to stop; a model it holds reads "busy" until it ends.
 */
export default function StatusBar({ onStats, onStatus, tick }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [noStatus, setNoStatus] = useState(false);

  useEffect(() => {
    if (noStatus) return undefined;
    let alive = true;
    let timer = null;
    async function poll() {
      let running = false;
      try {
        const next = await api.status();
        if (!alive) return;
        const value = next && typeof next === "object" ? next : null;
        setStatus(value);
        if (onStatus) onStatus(value);
        running = Boolean(value && value.job && value.job.state === "running");
      } catch (err) {
        if (!alive) return;
        if (err.status === 404) {
          setNoStatus(true); // an older server: no language models to report
          return;
        }
      }
      if (alive) timer = setTimeout(poll, running ? JOB_STATUS_POLL_MS : STATUS_POLL_MS);
    }
    poll();
    return () => {
      alive = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [onStatus, tick, noStatus]);

  useEffect(() => {
    let alive = true;
    let timer = null;
    async function poll() {
      try {
        const next = await api.stats();
        if (!alive) return;
        setStats(next && typeof next === "object" ? next : null);
        setError(null);
        if (onStats) onStats(next && typeof next === "object" ? next : null);
      } catch (err) {
        if (!alive) return;
        setError(err.message);
        if (onStats) onStats(null);
      }
      if (alive) timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => {
      alive = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [onStats, tick]);

  const s = stats || {};
  const shape = Array.isArray(s.shape) ? s.shape.join(" × ") : "–";
  const chain = status && status.chain;
  const ngram = status && status.ngram;
  const guard = status && status.guard;
  const job = status && status.job && status.job.state === "running" ? status.job : null;
  return (
    <div className="statusbar" aria-live="polite">
      {error ? <span className="stat err">API offline: {error}</span> : null}
      {job ? (
        <span className="stat job" title="A background job: the server answers everything else meanwhile">
          ⏳ {jobLine(job)}{" "}
          <button
            type="button"
            className="link"
            disabled={Boolean(job.progress && job.progress.stopping)}
            onClick={() => api.jobStop().catch(() => {})}
          >
            stop
          </button>
        </span>
      ) : null}
      {status ? (
        <>
          <span className="stat" title="The chain: its nodes, its unit, its confidence threshold, the tokens it has read">
            chain{" "}
            {chain && chain.busy ? (
              <b>busy</b>
            ) : chain ? (
              <>
                <b>{chain.nodes}</b> nodes of {chain.unit}s · conf <b>{fmtNum(chain.confidence, 2)}</b> · <b>{fmtInt(chain.tokens)}</b> tokens
              </>
            ) : (
              <b>none</b>
            )}
          </span>
          <span className="stat" title="The N-gram model: N, its unit, the tokens it has read, its n-grams">
            ngram{" "}
            {ngram && ngram.busy ? (
              <b>busy</b>
            ) : ngram ? (
              <>
                <b>{ngram.order}</b>-grams of {ngram.unit}s · <b>{fmtInt(ngram.tokens)}</b> tokens · {fmtInt(ngram.ngrams)} n-grams
              </>
            ) : (
              <b>none</b>
            )}
          </span>
          {guard && guard.busy ? (
            <span className="stat">
              guard <b>busy</b>
            </span>
          ) : guard ? (
            <span className="stat" title="The guard: the transitions it knows as failures, the texts blamed, and the texts judged and rejected">
              guard <b>{fmtInt(guard.transitions)}</b> transitions · {fmtInt(guard.blamed)} blamed · judged{" "}
              <b>{fmtInt(guard.judgements)}</b> · rejected <b className={guard.rejected ? "bad" : undefined}>{fmtInt(guard.rejected)}</b>
            </span>
          ) : null}
          <span className="stat" title="Files in the upload directory, and saved checkpoints">
            uploads <b>{fmtInt(status.uploads)}</b> · checkpoints <b>{fmtInt(status.checkpoints)}</b>
          </span>
        </>
      ) : null}
      <span className="stat" title="states × symbols × states: the three dimensions of the matrix">
        matrix <b>{shape}</b>
      </span>
      <span className="stat" title="Edges anything was ever written to, of every edge in the matrix">
        touched <b>{fmtInt(s.touched)}</b> / {fmtInt(s.edges)}
      </span>
      <span className="stat" title="Every traversal, plus time let pass">
        clock <b>{fmtInt(s.clock)}</b>
      </span>
      <span className="stat" title="The level now; it relaxes toward the baseline on the clock">
        stimulation <b>{fmtNum(s.stimulation, 2)}</b> · baseline {fmtNum(s.baseline, 1)}
      </span>
      <span className="stat">
        state <b>{s.state === undefined ? "–" : s.state}</b> · accepting{" "}
        <b>{Array.isArray(s.accepting) && s.accepting.length ? s.accepting.join(", ") : "none"}</b>
      </span>
      <span className="stat" title="Runs from the start state, and credits that landed on one">
        runs <b>{fmtInt(s.runs)}</b> · credits <b>{fmtInt(s.credits)}</b>
      </span>
      <span className="stat" title="Reward and punishment credited to edges, ever">
        rewards <b className="ok">+{fmtNum(s.total_rewarded, 1)}</b> / <b className="bad">−{fmtNum(s.total_punished, 1)}</b>
      </span>
      <span className="stat" title="Times the matrix was folded into its central node, and the automatic schedule">
        compressed <b>{fmtInt(s.compressions)}</b>
        {s.compress_every ? <> · every {fmtInt(s.compress_every)}</> : null}
        {s.core_bytes ? <> · code {fmtInt(s.core_bytes)} B</> : null}
      </span>
      <span className="stat" title="How a row's edges are weighed: each by its own weighting, or least traversed and most rewarded first">
        traverse <b>{s.mode === "explore" ? "explore" : "own"}</b>
      </span>
      <span className="stat" title="Where runs start, how often they skipped a node, and how often nodes swapped">
        focus <b>{s.learn_focus ? "learned" : s.focus === null || s.focus === undefined ? "none" : fmtNum(s.focus, 2)}</b>
        {s.origin !== undefined && !s.learn_focus ? <> · from state {s.origin}</> : null}
        {s.skip ? <> · skips <b>{fmtInt(s.skips)}</b></> : null}
        {s.swaps ? <> · swaps <b>{fmtInt(s.swaps)}</b></> : null}
      </span>
      <span className="stat" title="The widest and narrowest channel in the matrix now">
        width <b>{fmtNum(s.narrowest, 2)}</b> – <b>{fmtNum(s.widest, 2)}</b>
      </span>
    </div>
  );
}
