import { useEffect, useState } from "react";
import { api } from "../api.js";
import { jobLine } from "../jobs.js";
import { fmtInt, fmtNum } from "../util.js";

const POLL_MS = 3000;
const JOB_POLL_MS = 1000; // while a job runs, its progress is worth following closely

/**
 * Polls `/api/status` every 3 s (every second while a job runs): the language and the lattice's shape, the
 * episodes taught and the last held-out accuracy, the encoder's diversity, the agreement between the lattice's
 * prediction and the next named state, the wealth's spread, the Nim body's games - and the job, first, with a
 * button that asks it to stop. A body a job holds reads "busy" until it ends. `onStatus` receives every payload
 * (null when offline); `tick` changes whenever a panel has moved a body, and makes the bar refresh at once.
 */
export default function StatusBar({ onStatus, tick }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    let timer = null;
    async function poll() {
      let running = false;
      try {
        const next = await api.status();
        if (!alive) return;
        const value = next && typeof next === "object" ? next : null;
        setStatus(value);
        setError(null);
        if (onStatus) onStatus(value);
        running = Boolean(value && value.job && value.job.state === "running");
      } catch (err) {
        if (!alive) return;
        setError(err.message);
        if (onStatus) onStatus(null);
      }
      if (alive) timer = setTimeout(poll, running ? JOB_POLL_MS : POLL_MS);
    }
    poll();
    return () => {
      alive = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [onStatus, tick]);

  const body = status && status.body;
  const nim = status && status.nim;
  const job = status && status.job && status.job.state === "running" ? status.job : null;
  const settings = (body && body.settings) || {};
  const acc = body && body.accuracy;
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
      {body && body.busy ? (
        <span className="stat">
          body <b>busy</b>
        </span>
      ) : body ? (
        <>
          <span className="stat" title="The language the body reads, and the lattice's shape: states × symbols × states">
            language <b>{settings.language}</b> · <b>{Array.isArray(body.shape) ? body.shape.join(" × ") : "–"}</b>
          </span>
          <span className="stat" title="The encoder: states it names, the walk window, the gram size, the seed">
            states <b>{settings.states}</b> · window <b>{settings.window}</b> · n {settings.n} · seed {settings.seed}
          </span>
          <span className="stat" title="Strings taught, every step credited by the oracle; and the last held-out accuracy">
            episodes <b>{fmtInt(body.episodes)}</b>
            {acc ? (
              <>
                {" "}
                · held out <b>{fmtNum(acc.encoder, 3)}</b> / {fmtNum(acc.lattice, 3)}
              </>
            ) : null}
          </span>
          <span className="stat" title="The entropy of the states named so far, in bits: 0 is collapse onto one state">
            diversity <b>{fmtNum(body.diversity, 2)}</b> bits
          </span>
          <span
            className="stat"
            title="How often the lattice's prediction was the encoder's next named state: a meta-feature, not credit (DESIGN.md §8.2)"
          >
            agreement <b>{fmtNum(body.agreement, 3)}</b>
          </span>
          <span className="stat" title="The encoder's economy: mean wealth of the nodes that played, and its Gini coefficient">
            wealth <b>{fmtNum(body.wealth && body.wealth.mean, 1)}</b> · gini {fmtNum(body.wealth && body.wealth.gini, 2)}
          </span>
          <span className="stat" title="Best-response cycles that taught THINK, and states named in all">
            thinks <b>{fmtInt(body.thinks)}</b> · namings {fmtInt(body.namings)}
          </span>
        </>
      ) : null}
      {nim && nim.busy ? (
        <span className="stat">
          nim <b>busy</b>
        </span>
      ) : nim ? (
        <span className="stat" title="The Nim body: games played or babbled, and I(named state; nim-sum is zero)">
          nim <b>{fmtInt(nim.games)}</b> games · I {fmtNum(nim.information_bits, 3)} bits
        </span>
      ) : null}
    </div>
  );
}
