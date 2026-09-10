import { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmtInt, fmtNum, yesNo } from "../util.js";

const POLL_MS = 2000;

/** One-line summary of the latest progress record of a job, if any. */
function progressSummary(job) {
  const p = job && job.progress;
  if (!p || typeof p !== "object") return "";
  if (p.kind === "attempt") {
    const outcome = p.correct ? "correct" : p.runs ? "rejected" : "error";
    return `${p.phase} · ${p.problem} attempt ${fmtInt(p.attempt)} [${p.source}] · ${outcome}`;
  }
  if (p.kind === "problem") {
    const outcome = p.correct ? `solved by ${p.solved_by}` : "failed";
    return `${p.phase} · ${p.problem} · ${outcome}${p.action ? ` · ${p.action}` : ""}`;
  }
  if (p.kind === "round") return `${p.phase} round ${fmtInt(p.round)} · ${fmtInt(p.solved)}/${fmtInt(p.problems)} solved`;
  if (p.generation !== undefined) return `gen ${fmtInt(p.generation)} · gap ${fmtNum(p.gap, 3)}`;
  if (p.epoch !== undefined) {
    const phase = p.phase ? `${p.phase} ` : "";
    return `${phase}epoch ${fmtInt(p.epoch)} · loss ${fmtNum(p.loss, 4)}`;
  }
  return "";
}

/**
 * Polls /api/status every 2 s and shows model size, inversion, backend, and
 * the background job. `onStatus` receives every payload (null when offline)
 * so the rest of the app can share it.
 */
export default function StatusBar({ onStatus }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    let timer = null;
    async function tick() {
      try {
        const next = await api.status();
        if (!alive) return;
        setStatus(next && typeof next === "object" ? next : null);
        setError(null);
        if (onStatus) onStatus(next && typeof next === "object" ? next : null);
      } catch (err) {
        if (!alive) return;
        setError(err.message);
        if (onStatus) onStatus(null);
      }
      if (alive) timer = setTimeout(tick, POLL_MS);
    }
    tick();
    return () => {
      alive = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [onStatus]);

  const s = status || {};
  const job = s.job && typeof s.job === "object" ? s.job : null;
  const backends = s.backends && typeof s.backends === "object" ? s.backends : {};
  const jobState = job && job.state ? String(job.state) : "idle";
  const progress = progressSummary(job);
  const mark = (flag) => (flag ? "✓" : "✗");

  return (
    <div className="statusbar" aria-live="polite">
      {error ? <span className="stat err">API offline: {error}</span> : null}
      <span className="stat" title="The active model kind (select it in the header)">
        model <b>{s.model_label || s.kind || "–"}</b>
      </span>
      <span className="stat">
        nodes <b>{fmtInt(s.nodes)}</b>
      </span>
      <span className="stat">
        edges <b>{fmtInt(s.edges)}</b>
      </span>
      <span className="stat">
        trigrams <b>{fmtInt(s.trigrams)}</b>
      </span>
      <span className="stat">
        compression <b>{fmtNum(s.compression_ratio, 2)}×</b>
      </span>
      <span className={`stat${s.inverted ? " warn" : ""}`}>
        inverted <b>{s.inverted === undefined ? "–" : yesNo(s.inverted)}</b>
      </span>
      <span className="stat">
        backend{" "}
        <b>
          {s.backend || "–"}/{s.device || "–"}
        </b>
      </span>
      <span className="stat" title="Optional accelerator availability on the server">
        torch {mark(backends.torch)} · cuda {mark(backends.cuda)} · mps {mark(backends.mps)}
      </span>
      <span className="stat">
        epochs <b>{fmtInt(s.epochs_total)}</b> · 2NRL runs <b>{fmtInt(s.twonrl_runs)}</b> · last loss{" "}
        <b>{fmtNum(s.last_loss, 4)}</b>
      </span>
      {s.kind === "count" ? (
        <span className="stat" title="Sum of rewards and penalties applied to edges">
          rewards <b>+{fmtNum(s.edge_reward_positive, 1)}</b> / <b>{fmtNum(s.edge_reward_negative, 1)}</b>
        </span>
      ) : null}
      <span className={`stat job ${jobState}`}>
        job{" "}
        <b>
          {job && job.type ? `${job.type} · ` : ""}
          {jobState}
        </b>
        {progress ? <span>{progress}</span> : null}
      </span>
      {s.model_path ? (
        <span className="stat path" title="Model path on the server">
          model <b>{String(s.model_path)}</b>
        </span>
      ) : null}
    </div>
  );
}
