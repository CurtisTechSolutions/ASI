import { useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { NumberField, SelectField } from "./Fields.jsx";

const ACTION_TEXT = {
  "2nrl": "2NRL: train on the thumbs-down texts, invert the network, fine-tune on the thumbs-up texts",
  reward: "reward: a positive-phase pass on the thumbs-up texts",
  punish: "punish: a negative-phase pass on the thumbs-down texts, then the network is inverted",
};

function actionFor(ups, downs) {
  if (ups && downs) return "2nrl";
  if (ups) return "reward";
  if (downs) return "punish";
  return null;
}

function clipText(text, width = 70) {
  const flat = String(text ?? "").replace(/\s+/g, " ");
  return flat.length > width ? `${flat.slice(0, width - 1)}…` : flat;
}

/**
 * Generate texts from the START node (stochastic walks or the single cheapest
 * path to END) and rate them: thumbs up = correct (2NRL positive phase),
 * thumbs down = garbage (negative phase). "Train on ratings" starts a feedback
 * job - 2NRL when both kinds were rated, reward-only on thumbs up alone,
 * punish-only (negative phase, then invert) on thumbs down alone. Ratings
 * accumulate across generations until they are trained on or cleared.
 */
export default function GeneratePanel({ status }) {
  const [count, setCount] = useState("3");
  const [maxLength, setMaxLength] = useState("60");
  const [temperature, setTemperature] = useState("1.0");
  const [mode, setMode] = useState("sample");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState(null);
  const [ratings, setRatings] = useState([]);
  const [negEpochs, setNegEpochs] = useState("2");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.5");
  const [posLr, setPosLr] = useState("0.1");
  const [lastAction, setLastAction] = useState(null);
  const { job, running, busy, error: jobError, start, stop, clearError } = useJob("feedback");

  const otherJobRunning = jobIsRunning(status) && !running;
  const ups = ratings.filter((r) => r.rating === "up");
  const downs = ratings.filter((r) => r.rating === "down");
  const action = actionFor(ups.length, downs.length);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.generate({
        count: parseInteger(count, 1),
        max_length: parseInteger(maxLength, 60),
        temperature: parseNumber(temperature, 1),
        mode,
      });
      setSamples(asArray(data && data.samples));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const textOf = (sample) => String((sample && sample.text) ?? "");
  const ratingOf = (sample) => {
    const found = ratings.find((r) => r.text === textOf(sample));
    return found ? found.rating : null;
  };

  /** Toggle a rating: pressing the active thumb again removes it. */
  function rate(sample, rating) {
    const text = textOf(sample);
    if (!text.trim()) return;
    setRatings((prev) => {
      const rest = prev.filter((r) => r.text !== text);
      const current = prev.find((r) => r.text === text);
      if (current && current.rating === rating) return rest;
      return [...rest, { text, rating, cost: sample && sample.cost }];
    });
  }

  async function handleTrain() {
    if (!action) return;
    setLastAction(null);
    const result = await start(() =>
      api.feedback({
        ...(ups.length ? { good: ups.map((r) => r.text) } : {}),
        ...(downs.length ? { bad: downs.map((r) => r.text) } : {}),
        neg_epochs: parseInteger(negEpochs, 2),
        pos_epochs: parseInteger(posEpochs, 3),
        neg_lr: parseNumber(negLr, 0.5),
        pos_lr: parseNumber(posLr, 0.1),
      }),
    );
    if (result) setLastAction(action);
  }

  const history = asArray(job && job.history);

  return (
    <>
      <form className="card" onSubmit={handleSubmit}>
        <h2>Generate</h2>
        <div className="row">
          <NumberField label="Count" value={count} onChange={setCount} min={1} step={1} />
          <NumberField label="Max length" value={maxLength} onChange={setMaxLength} min={1} step={1} />
        </div>
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              ["sample", "sample (stochastic)"],
              ["dijkstra", "dijkstra (cheapest path to END)"],
            ]}
          />
          <NumberField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0.01}
            disabled={mode !== "sample"}
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Generating…" : "Generate"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Samples</h2>
        <p className="muted">
          Rate a sample: thumbs up marks it correct (2NRL positive phase), thumbs down marks it garbage (negative
          phase). Press the same thumb again to remove the rating.
        </p>
        {samples === null ? (
          <p className="muted">Press Generate to sample texts from the model.</p>
        ) : samples.length === 0 ? (
          <p className="muted">The model returned no samples (train it first).</p>
        ) : (
          <ol className="samples">
            {samples.map((s, i) => {
              const rating = ratingOf(s);
              const empty = !textOf(s).trim();
              return (
                <li key={i} className={rating ? `rated ${rating}` : ""}>
                  <pre className="sample">{textOf(s)}</pre>
                  <div className="meta">
                    cost {fmtNum(s && s.cost, 3)} · {fmtInt(asArray(s && s.path).length)} path nodes ·{" "}
                    {fmtInt(textOf(s).length)} chars
                    <span className="rating">
                      <button
                        type="button"
                        className={`rate up${rating === "up" ? " active" : ""}`}
                        aria-pressed={rating === "up"}
                        aria-label={`thumbs up sample ${i + 1}`}
                        title="Correct: reward (2NRL positive phase)"
                        disabled={empty || running}
                        onClick={() => rate(s, "up")}
                      >
                        👍
                      </button>
                      <button
                        type="button"
                        className={`rate down${rating === "down" ? " active" : ""}`}
                        aria-pressed={rating === "down"}
                        aria-label={`thumbs down sample ${i + 1}`}
                        title="Garbage: punish (2NRL negative phase)"
                        disabled={empty || running}
                        onClick={() => rate(s, "down")}
                      >
                        👎
                      </button>
                    </span>
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <div className="card">
        <div className="toolbar">
          <h2>Ratings → 2NRL</h2>
          <button type="button" className="small" disabled={running || ratings.length === 0} onClick={() => setRatings([])}>
            Clear ratings
          </button>
        </div>
        <div className="chips">
          <span className="chip">👍 {ups.length}</span>
          <span className="chip">👎 {downs.length}</span>
          {action ? <span className="chip">{action}</span> : <span className="muted">rate some samples first</span>}
        </div>
        {action ? <p className="muted">{ACTION_TEXT[action]}.</p> : null}
        {ratings.length > 0 ? (
          <ul className="ratings">
            {ratings.map((r) => (
              <li key={r.text}>
                <span className={`badge ${r.rating}`}>{r.rating === "up" ? "👍" : "👎"}</span>
                <code>{clipText(r.text)}</code>
                <button
                  type="button"
                  className="link"
                  disabled={running}
                  onClick={() => setRatings((prev) => prev.filter((x) => x.text !== r.text))}
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        <div className="row">
          <NumberField label="Negative epochs" hint="thumbs down" value={negEpochs} onChange={setNegEpochs} min={0} step={1} disabled={running} />
          <NumberField label="Positive epochs" hint="thumbs up" value={posEpochs} onChange={setPosEpochs} min={0} step={1} disabled={running} />
        </div>
        <div className="row">
          <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={running} />
          <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={running} />
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
        <JobStatus job={job} emptyText="No feedback job yet. Rate samples above, then press Train on ratings." />
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
    </>
  );
}
