import { useState } from "react";
import { api } from "../api.js";
import { fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { NumberField } from "./Fields.jsx";

export const ACTION_TEXT = {
  "2nrl": "2NRL: train on the thumbs-down texts, invert the network, fine-tune on the thumbs-up texts",
  reward: "reward: a positive-phase pass on the thumbs-up texts",
  punish: "punish: a negative-phase pass on the thumbs-down texts, then the network is inverted",
};

export function actionFor(ups, downs) {
  if (ups && downs) return "2nrl";
  if (ups) return "reward";
  if (downs) return "punish";
  return null;
}

export function clipText(text, width = 70) {
  const flat = String(text ?? "").replace(/\s+/g, " ");
  return flat.length > width ? `${flat.slice(0, width - 1)}…` : flat;
}

export const DEFAULT_MARK = 10;

/** A mark out of 10 kept inside [0, 10]; anything unusable falls back to the default. */
export function clampMark(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_MARK;
  return Math.max(0, Math.min(10, n));
}

/**
 * Thumbs up / thumbs down ratings, one per distinct text, shared by the
 * Generate and Converse tabs. Pressing the active thumb again removes the
 * rating; ratings accumulate until they are trained on or cleared. Every
 * rating also carries a mark out of 10 (10 by default: a plain thumb) that
 * says *how* good or bad the text is - the network then learns each text in
 * proportion to its mark instead of treating every thumb alike.
 */
export function useRatings() {
  const [ratings, setRatings] = useState([]);
  const ratingOf = (text) => {
    const found = ratings.find((r) => r.text === text);
    return found ? found.rating : null;
  };
  function rate(text, rating, extra = {}) {
    if (!String(text ?? "").trim()) return;
    setRatings((prev) => {
      const rest = prev.filter((r) => r.text !== text);
      const current = prev.find((r) => r.text === text);
      if (current && current.rating === rating) return rest;
      return [...rest, { text, rating, mark: current ? current.mark : DEFAULT_MARK, ...extra }];
    });
  }
  const setMark = (text, mark) =>
    setRatings((prev) => prev.map((r) => (r.text === text ? { ...r, mark } : r)));
  const remove = (text) => setRatings((prev) => prev.filter((r) => r.text !== text));
  const clear = () => setRatings([]);
  return { ratings, setRatings, rate, ratingOf, setMark, remove, clear };
}

/** The thumbs up / thumbs down pair for one text. */
export function RateButtons({ text, rating, disabled, onRate, label }) {
  return (
    <span className="rating">
      <button
        type="button"
        className={`rate up${rating === "up" ? " active" : ""}`}
        aria-pressed={rating === "up"}
        aria-label={`thumbs up ${label}`}
        title="Correct: reward (2NRL positive phase)"
        disabled={disabled}
        onClick={() => onRate(text, "up")}
      >
        👍
      </button>
      <button
        type="button"
        className={`rate down${rating === "down" ? " active" : ""}`}
        aria-pressed={rating === "down"}
        aria-label={`thumbs down ${label}`}
        title="Garbage: punish (2NRL negative phase)"
        disabled={disabled}
        onClick={() => onRate(text, "down")}
      >
        👎
      </button>
    </span>
  );
}

/**
 * "Ratings → 2NRL": the rated texts, the feedback that will run (2NRL when
 * both kinds were rated, reward-only on thumbs up alone, punish-only on thumbs
 * down alone), its epochs / learning rates, and the job's progress.
 * ``feedback`` is the panel's ``useJob("feedback")`` handle.
 */
export default function RatingsCard({
  ratings,
  onClear,
  onRemove,
  onMark,
  feedback,
  status,
  emptyText = "rate some samples first",
}) {
  const [negEpochs, setNegEpochs] = useState("2");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.5");
  const [posLr, setPosLr] = useState("0.1");
  const [strength, setStrength] = useState("1");
  const [lastAction, setLastAction] = useState(null);
  const { job, running, busy, error: jobError, start, stop, clearError } = feedback;
  const countKind = Boolean(status && status.kind === "count");
  const otherJobRunning = jobIsRunning(status) && !running;
  const ups = ratings.filter((r) => r.rating === "up");
  const downs = ratings.filter((r) => r.rating === "down");
  const action = actionFor(ups.length, downs.length);
  const history = (job && Array.isArray(job.history) ? job.history : []);

  async function handleTrain() {
    if (!action) return;
    setLastAction(null);
    const result = await start(() =>
      api.feedback({
        ...(ups.length ? { good: ups.map((r) => r.text), good_ratings: ups.map((r) => clampMark(r.mark)) } : {}),
        ...(downs.length ? { bad: downs.map((r) => r.text), bad_ratings: downs.map((r) => clampMark(r.mark)) } : {}),
        neg_epochs: parseInteger(negEpochs, 2),
        pos_epochs: parseInteger(posEpochs, 3),
        neg_lr: parseNumber(negLr, 0.5),
        pos_lr: parseNumber(posLr, 0.1),
        ...(countKind ? { strength: parseNumber(strength, 1) } : {}),
      }),
    );
    if (result) setLastAction(action);
  }

  return (
    <div className="card">
      <div className="toolbar">
        <h2>Ratings → 2NRL</h2>
        <button type="button" className="small" disabled={running || ratings.length === 0} onClick={onClear}>
          Clear ratings
        </button>
      </div>
      <div className="chips">
        <span className="chip">👍 {ups.length}</span>
        <span className="chip">👎 {downs.length}</span>
        {action ? <span className="chip">{action}</span> : <span className="muted">{emptyText}</span>}
      </div>
      {action ? <p className="muted">{ACTION_TEXT[action]}.</p> : null}
      {ratings.length > 0 ? (
        <>
          <ul className="ratings">
            {ratings.map((r) => (
              <li key={r.text}>
                <span className={`badge ${r.rating}`}>{r.rating === "up" ? "👍" : "👎"}</span>
                <code>{clipText(r.text)}</code>
                {onMark ? (
                  <label className="mark">
                    <span className="muted">{r.rating === "up" ? "how good" : "how bad"}</span>
                    <input
                      type="number"
                      min={0}
                      max={10}
                      step={1}
                      value={r.mark ?? DEFAULT_MARK}
                      disabled={running}
                      aria-label={`mark out of 10 for ${clipText(r.text, 40)}`}
                      onChange={(e) => onMark(r.text, e.target.value === "" ? "" : clampMark(e.target.value))}
                    />
                    <span className="muted">/ 10</span>
                  </label>
                ) : null}
                <button type="button" className="link" disabled={running} onClick={() => onRemove(r.text)}>
                  remove
                </button>
              </li>
            ))}
          </ul>
          {onMark ? (
            <p className="muted">
              Each text is learned in proportion to its mark: 10 out of 10 uses the full learning rate, 5 half of
              it, 0 skips the text.
            </p>
          ) : null}
        </>
      ) : null}
      <div className="row">
        <NumberField label="Negative epochs" hint="thumbs down" value={negEpochs} onChange={setNegEpochs} min={0} step={1} disabled={running} />
        <NumberField label="Positive epochs" hint="thumbs up" value={posEpochs} onChange={setPosEpochs} min={0} step={1} disabled={running} />
      </div>
      <div className="row">
        <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={running} />
        <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={running} />
        {countKind ? (
          <NumberField
            label="Strength"
            hint="count model: reward / penalty per rated path"
            value={strength}
            onChange={setStrength}
            min={0}
            disabled={running}
          />
        ) : null}
      </div>
      <div className="actions">
        <button
          type="button"
          className="primary"
          disabled={!action || running || busy || otherJobRunning}
          onClick={handleTrain}
        >
          {busy ? "Starting…" : "Train on ratings"}
        </button>
        <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
          Stop
        </button>
      </div>
      {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
      <Alert message={jobError} onDismiss={clearError} />
      <JobStatus job={job} emptyText="No feedback job yet. Rate some texts above, then press Train on ratings." />
      {lastAction && job && job.state !== "running" ? (
        <p className="muted">Last feedback: {ACTION_TEXT[lastAction]}.</p>
      ) : null}
      {history.length > 0 ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>phase</th>
                <th>epoch</th>
                <th>loss</th>
                <th>perplexity</th>
                <th>nodes</th>
                <th>edges</th>
              </tr>
            </thead>
            <tbody>
              {history.map((r, i) => (
                <tr key={i}>
                  <td>{r.phase || "–"}</td>
                  <td>{fmtInt(r.epoch)}</td>
                  <td>{fmtNum(r.loss, 4)}</td>
                  <td>{fmtNum(r.perplexity, 3)}</td>
                  <td>{fmtInt(r.nodes)}</td>
                  <td>{fmtInt(r.edges)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : running ? (
        <p className="muted">Waiting for the first epoch…</p>
      ) : null}
    </div>
  );
}
