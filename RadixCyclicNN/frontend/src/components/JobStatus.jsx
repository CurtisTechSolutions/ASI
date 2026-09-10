import { fmtTime } from "../util.js";

/** State badge, timestamps and error of a background job (train / 2nrl / evolve). */
export default function JobStatus({ job, emptyText = "No job yet." }) {
  if (!job) return <p className="muted">{emptyText}</p>;
  const state = typeof job.state === "string" ? job.state : "unknown";
  return (
    <div>
      <div className="job-status">
        <span className={`badge ${state}`}>{state}</span>
        {job.type ? <span className="muted">type {job.type}</span> : null}
        {job.id !== undefined && job.id !== null ? <span className="muted">id {String(job.id)}</span> : null}
        {job.started_at ? <span className="muted">started {fmtTime(job.started_at)}</span> : null}
        {job.finished_at ? <span className="muted">finished {fmtTime(job.finished_at)}</span> : null}
      </div>
      {job.error ? (
        <div className="alert error" role="alert">
          <span>Job failed: {String(job.error)}</span>
        </div>
      ) : null}
    </div>
  );
}
