import { useState } from "react";
import { api } from "../api.js";
import { fmtElapsed, jobName, jobUnits } from "../jobs.js";
import { fmtInt } from "../util.js";

/**
 * A running job's progress (the report `runJob` hands to its `onProgress`): what it is doing and on what, the
 * episodes (or games) done and the symbols (or plies) read and how fast, the tenth reached, how long it has run -
 * and a button that asks it to stop. A stopped job keeps everything it learned up to then. Renders nothing
 * without a job.
 */
export default function JobProgress({ job }) {
  const [asked, setAsked] = useState(null);
  const [error, setError] = useState(null);
  if (!job) return null;
  const p = job.progress || {};
  const stopping = Boolean(p.stopping) || asked === job.id;
  const [sequences, tokens] = jobUnits(job);

  const stop = async () => {
    setAsked(job.id);
    try {
      await api.jobStop();
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="job-progress" role="status" aria-live="polite">
      <div className="job-head">
        <span className="badge running">{stopping ? "stopping" : "running"}</span>
        <b>{jobName(job)}</b>
        <span className="muted">{[p.phase, p.label].filter(Boolean).join(" ") || "starting"}</span>
        <button type="button" className="small danger" disabled={stopping || job.state !== "running"} onClick={stop}>
          {stopping ? "Stopping…" : "Stop"}
        </button>
      </div>
      <div className="job-bar" aria-hidden="true">
        <span />
      </div>
      <div className="job-stats mono">
        <span>
          <b>{fmtInt(p.sequences || 0)}</b> {sequences}
        </span>
        <span>
          <b>{fmtInt(p.tokens || 0)}</b> {tokens}
        </span>
        <span>
          <b>{fmtInt(p.tokens_per_second || 0)}</b> {tokens}/s
        </span>
        {p.round ? (
          <span>
            tenth <b>{fmtInt(p.round)}</b>
          </span>
        ) : null}
        <span>
          <b>{fmtElapsed(job.elapsed)}</b>
        </span>
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <p className="muted small-note">
        The server keeps answering meanwhile: other tabs work, though the body being trained answers “busy” until the
        job ends. Stopping keeps what was learned so far.
      </p>
    </div>
  );
}
