import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { clampMark, lmLabel, splitByMark } from "../lm.js";
import { parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField } from "./Fields.jsx";

/** A text cut to one line of `width` characters, for the lists. */
export function clipText(text, width = 80) {
  const flat = String(text ?? "").replace(/\s+/g, " ");
  return flat.length > width ? `${flat.slice(0, width - 1)}…` : flat;
}

/**
 * Marks out of 10, one per distinct text, kept until they are applied or cleared. `add` puts a text on the
 * card (a second add of the same text keeps its mark); `setMark` changes one.
 */
export function useRatings() {
  const [ratings, setRatings] = useState([]);
  const add = (text, mark = 5) => {
    if (!String(text ?? "").trim()) return;
    setRatings((prev) => (prev.some((r) => r.text === text) ? prev : [...prev, { text, mark: clampMark(mark) }]));
  };
  const has = (text) => ratings.some((r) => r.text === text);
  const setMark = (text, mark) => setRatings((prev) => prev.map((r) => (r.text === text ? { ...r, mark } : r)));
  const remove = (text) => setRatings((prev) => prev.filter((r) => r.text !== text));
  const clear = () => setRatings([]);
  return { ratings, add, has, setMark, remove, clear };
}

/**
 * The 👍 / 👎 pair for one text: a press credits it at once (`onRate(+1 | -1)`), and the pressed one stays lit
 * so it is plain what was done.
 */
export function RateButtons({ rated, disabled, onRate, label }) {
  return (
    <span className="rating">
      <button
        type="button"
        className={`rate up${rated > 0 ? " active" : ""}`}
        aria-label={`thumbs up ${label}`}
        title="Good: reward it (+1 to every edge of it)"
        disabled={disabled}
        onClick={() => onRate(1)}
      >
        👍
      </button>
      <button
        type="button"
        className={`rate down${rated < 0 ? " active" : ""}`}
        aria-label={`thumbs down ${label}`}
        title="Bad: punish it (-1 to every edge of it)"
        disabled={disabled}
        onClick={() => onRate(-1)}
      >
        👎
      </button>
    </span>
  );
}

/**
 * "Marks → feedback": the marked texts, each with its mark out of 10, applied in one go through
 * `POST /api/{m}/feedback` - a mark of 6 or more makes a text a good one (rewarded), 4 or less a bad one
 * (punished), a 5 is left out. `ratings` is the panel's `useRatings()` handle.
 */
export default function RatingsCard({ lm, ratings, onMoved, namespace = "ratings", emptyText }) {
  const [amount, setAmount] = useStoredState(`${namespace}.amount`, "1");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const split = splitByMark(ratings.ratings);

  async function apply() {
    setBusy(true);
    setNote(null);
    try {
      const r = await api.lmFeedback(lm, { good: split.good, bad: split.bad, amount: parseNumber(amount, 1) });
      setNote(
        `${lmLabel(lm)}: rewarded ${r.rewarded} and punished ${r.punished}${
          split.skipped ? `; ${split.skipped} marked 5 left out` : ""
        }`,
      );
      ratings.clear();
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div className="toolbar">
        <h2>Marks → feedback</h2>
        <button type="button" className="small" disabled={busy || ratings.ratings.length === 0} onClick={ratings.clear}>
          Clear
        </button>
      </div>
      <div className="chips">
        <span className="chip">good {split.good.length}</span>
        <span className="chip">bad {split.bad.length}</span>
        {split.skipped ? <span className="chip">left out {split.skipped}</span> : null}
      </div>
      {ratings.ratings.length === 0 ? (
        <p className="muted">{emptyText || "Press ☆ beside a text to mark it out of 10 here."}</p>
      ) : (
        <ul className="ratings">
          {ratings.ratings.map((r) => {
            const mark = clampMark(r.mark);
            return (
              <li key={r.text} className="mark-row">
                <span className={`badge ${mark >= 6 ? "up" : mark <= 4 ? "down" : ""}`}>{mark >= 6 ? "good" : mark <= 4 ? "bad" : "–"}</span>
                <code>{clipText(r.text)}</code>
                <label className="mark">
                  <input
                    type="number"
                    min={0}
                    max={10}
                    step={1}
                    value={r.mark}
                    disabled={busy}
                    aria-label={`mark out of 10 for ${clipText(r.text, 40)}`}
                    onChange={(e) => ratings.setMark(r.text, e.target.value === "" ? "" : clampMark(e.target.value))}
                  />
                  <span className="muted">/ 10</span>
                </label>
                <button type="button" className="link" disabled={busy} onClick={() => ratings.remove(r.text)}>
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      )}
      <p className="muted">6 or more: a good text, rewarded. 4 or less: a bad one, punished. 5: left out.</p>
      <div className="actions">
        <NumberField label="Amount" value={amount} onChange={setAmount} min="0" step="0.1" hint="reward / punishment per text" />
      </div>
      <div className="actions">
        <button
          type="button"
          className="primary"
          disabled={busy || (split.good.length === 0 && split.bad.length === 0)}
          onClick={apply}
        >
          {busy ? "Applying…" : "Apply the marks"}
        </button>
      </div>
      {note ? <p className="note">{note}</p> : null}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}
