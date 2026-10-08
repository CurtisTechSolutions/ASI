import { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmtInt, fmtNum } from "../util.js";

const POLL_MS = 2000;

/**
 * Polls /api/stats every 2 s: the matrix's shape, the clock, the stimulation,
 * how much of the matrix has been written to. `onStats` receives every
 * payload (null when offline) so the rest of the app can share it; `tick`
 * changes whenever a panel has moved the machine, and makes the bar refresh
 * at once rather than at the next poll.
 */
export default function StatusBar({ onStats, tick }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

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
  return (
    <div className="statusbar" aria-live="polite">
      {error ? <span className="stat err">API offline: {error}</span> : null}
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
      <span className="stat" title="The widest and narrowest channel in the matrix now">
        width <b>{fmtNum(s.narrowest, 2)}</b> – <b>{fmtNum(s.widest, 2)}</b>
      </span>
    </div>
  );
}
