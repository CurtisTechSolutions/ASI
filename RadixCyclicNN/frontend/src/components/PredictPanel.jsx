import { Fragment, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, showWhitespace, yesNo } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

const SENTINELS = new Set(["<s>", "</s>"]);

/** A thumbs-up that rewards one predicted text (prefix + continuation). */
function LikeButton({ text, liked, disabled, onLike, label, compact = false }) {
  const empty = !String(text ?? "").trim();
  return (
    <button
      type="button"
      className={`rate up${liked ? " active" : ""}`}
      aria-pressed={liked}
      aria-label={label}
      title={liked ? "Rewarded" : "Like: reward this text (a positive-phase pass / +reward on its path)"}
      disabled={empty || disabled || liked}
      onClick={() => onLike(text)}
    >
      👍{compact ? "" : liked ? " Liked" : " Like"}
    </button>
  );
}

/** One of the top-K / bottom-K continuations of the count / reward model. */
function PathTable({ title, hint, paths, prefix, liked, likeDisabled, onLike }) {
  return (
    <div className="paths">
      <h3>{title}</h3>
      {paths.length === 0 ? (
        <p className="muted">{hint}</p>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>#</th>
                <th>continuation</th>
                <th>probability</th>
                <th>cost</th>
                <th>END</th>
                <th>like</th>
              </tr>
            </thead>
            <tbody>
              {paths.map((p, i) => {
                const text = String(p.full_text ?? `${prefix}${p.continuation ?? ""}`);
                return (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="text">
                      <span className="prefix">{prefix}</span>
                      <span className="continuation">{String(p.continuation ?? "")}</span>
                    </td>
                    <td>{fmtNum(p.probability, 4)}</td>
                    <td>{fmtNum(p.cost, 3)}</td>
                    <td>{p.reached_end === undefined ? "–" : yesNo(p.reached_end)}</td>
                    <td>
                      <LikeButton
                        text={text}
                        liked={liked.has(text)}
                        disabled={likeDisabled}
                        onLike={onLike}
                        label={`like ${title.toLowerCase().split(" ")[0]} continuation ${i + 1}`}
                        compact
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/**
 * Continue a prefix. RadixNet: the shortest path (Dijkstra) or a sampled
 * walk. The count / reward model: a beam search that returns the K most
 * likely and the K least likely continuations in one prediction (plus
 * sampling).
 *
 * A "Like" button rewards a result: it starts a feedback job with the text
 * (prefix + continuation) as a thumbs-up - a positive-phase pass for
 * RadixNet, a traversal plus reward on the path for the count model.
 */
export default function PredictPanel({ status }) {
  const [prefix, setPrefix] = useState("");
  const [length, setLength] = useState("20");
  const [mode, setMode] = useState("dijkstra");
  const [k, setK] = useState("5");
  const [beam, setBeam] = useState("");
  const [toEnd, setToEnd] = useState(false);
  const [stepPenalty, setStepPenalty] = useState("0");
  const [temperature, setTemperature] = useState("1.0");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [liked, setLiked] = useState(() => new Set());
  const [lastLiked, setLastLiked] = useState(null);
  const { job, running, busy, error: jobError, start, clearError } = useJob("feedback");

  const countKind = Boolean(status && status.kind === "count");
  const likeDisabled = busy || running || jobIsRunning(status);

  async function like(text) {
    const trimmed = String(text ?? "");
    if (!trimmed.trim() || likeDisabled) return;
    setLastLiked(null);
    const started = await start(() => api.feedback({ good: [trimmed], ...(countKind ? { strength: 1 } : {}) }));
    if (started) {
      setLiked((prev) => new Set([...prev, trimmed]));
      setLastLiked(trimmed);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const effectiveMode = countKind && mode === "dijkstra" ? "beam" : mode;
      const body = {
        prefix,
        length: parseInteger(length, 20),
        mode: effectiveMode,
        to_end: toEnd,
        step_penalty: parseNumber(stepPenalty, 0),
        temperature: parseNumber(temperature, 1),
      };
      if (effectiveMode === "beam") {
        body.k = parseInteger(k, 5);
        const width = parseInteger(beam, 0);
        if (width > 0) body.beam = width;
      }
      const data = await api.predict(body);
      setResult(data && typeof data === "object" ? data : {});
      setLastLiked(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const path = asArray(result && result.path);
  const stepCosts = asArray(result && result.step_costs);
  const nodeIds = asArray(result && result.node_ids);
  // step_costs has one entry per transition; align each cost with the chip it leads into.
  const offset = Math.max(0, path.length - stepCosts.length);
  const top = asArray(result && result.top);
  const bottom = asArray(result && result.bottom);
  const resultIsCount = Boolean(result && Array.isArray(result.top));
  const shownPrefix = result ? String(result.prefix ?? prefix) : prefix;
  const fullText = result ? String(result.full_text ?? `${shownPrefix}${result.continuation ?? ""}`) : "";

  return (
    <>
      <form className="card" onSubmit={handleSubmit}>
        <h2>Predict</h2>
        <TextField label="Prefix" value={prefix} onChange={setPrefix} placeholder="the quick br" />
        <div className="row">
          <NumberField
            label="Length"
            hint="at least this many chars; the path runs to its natural end, no cap"
            value={length}
            onChange={setLength}
            min={0}
            step={1}
          />
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={
              countKind
                ? [
                    ["dijkstra", "beam (top K and bottom K)"],
                    ["sample", "sample (stochastic)"],
                  ]
                : [
                    ["dijkstra", "dijkstra (shortest path)"],
                    ["beam", "beam (top K and bottom K)"],
                    ["sample", "sample (stochastic)"],
                  ]
            }
          />
        </div>
        {mode === "beam" || (countKind && mode === "dijkstra") ? (
          <div className="row">
            <NumberField
              label="K"
              hint="continuations per side: the K most and the K least likely"
              value={k}
              onChange={setK}
              min={0}
              step={1}
              disabled={mode === "sample"}
            />
            <NumberField
              label="Beam width"
              hint="blank = max(4K, 16)"
              value={beam}
              onChange={setBeam}
              min={1}
              step={1}
              placeholder="auto"
              disabled={mode === "sample"}
            />
          </div>
        ) : null}
        <div className="row">
          <NumberField label="Step penalty" value={stepPenalty} onChange={setStepPenalty} min={0} />
          <NumberField
            label="Temperature"
            hint="sample mode"
            value={temperature}
            onChange={setTemperature}
            min={0.01}
            disabled={mode !== "sample"}
          />
        </div>
        <CheckField label="Run to END (cheapest complete path)" checked={toEnd} onChange={setToEnd} />
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
        {mode === "beam" || (countKind && mode === "dijkstra") ? (
          <p className="muted">
            The beam search returns the K most likely continuations and the K least likely ones of the same length
            in one prediction; the same search generates whole texts on the Generate tab.
          </p>
        ) : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Result</h2>
        {!result ? (
          <p className="muted">Enter a prefix and press Predict. The highlighted part is the predicted continuation.</p>
        ) : (
          <>
            <p className="text-display">
              <span className="prefix">{shownPrefix}</span>
              <span className="continuation">{String(result.continuation ?? "")}</span>
            </p>
            <div className="like-row">
              <LikeButton
                text={fullText}
                liked={liked.has(fullText)}
                disabled={likeDisabled}
                onLike={like}
                label="like this prediction"
              />
              <span className="muted">
                {countKind
                  ? "Rewards the shown text: one traversal and +1 reward on every edge of its path."
                  : "Rewards the shown text: a positive-phase training pass (thumbs up)."}
              </span>
            </div>
            {job ? (
              <JobStatus job={job} emptyText="" />
            ) : null}
            {lastLiked !== null && job && job.state === "done" ? (
              <p className="muted">Rewarded: {JSON.stringify(lastLiked.length > 80 ? `${lastLiked.slice(0, 80)}…` : lastLiked)}</p>
            ) : null}
            <Alert message={jobError} onDismiss={clearError} />
            <dl className="kv">
              <dt>cost</dt>
              <dd>{fmtNum(result.cost, 4)}</dd>
              <dt>probability</dt>
              <dd>{fmtNum(result.probability, 4)}</dd>
              <dt>reached END</dt>
              <dd>{result.reached_end === undefined ? "–" : yesNo(result.reached_end)}</dd>
              <dt>states expanded</dt>
              <dd>{fmtInt(result.expanded)}</dd>
              <dt>path nodes</dt>
              <dd>{fmtInt(path.length)}</dd>
              {resultIsCount ? (
                <>
                  <dt>K / beam</dt>
                  <dd>
                    {fmtInt(result.k)} / {fmtInt(result.beam)}
                  </dd>
                </>
              ) : null}
            </dl>
            {resultIsCount && result.mode !== "sample" ? (
              <>
                <PathTable
                  title={`Top ${top.length} (most likely)`}
                  hint="No complete continuation was found."
                  paths={top}
                  prefix={shownPrefix}
                  liked={liked}
                  likeDisabled={likeDisabled}
                  onLike={like}
                />
                <PathTable
                  title={`Bottom ${bottom.length} (least likely)`}
                  hint="No other continuation of that length exists (every path found is already in the top list)."
                  paths={bottom}
                  prefix={shownPrefix}
                  liked={liked}
                  likeDisabled={likeDisabled}
                  onLike={like}
                />
              </>
            ) : null}
            <h3>Path</h3>
            {path.length === 0 ? (
              <p className="muted">Empty path.</p>
            ) : (
              <div className="chips">
                {path.map((label, i) => {
                  const cost = stepCosts[i - offset];
                  const text = String(label ?? "");
                  return (
                    <Fragment key={i}>
                      {i > 0 ? <span className="chip-arrow">→</span> : null}
                      <span
                        className={`chip${SENTINELS.has(text) ? " sentinel" : ""}`}
                        title={nodeIds[i] !== undefined ? `node ${nodeIds[i]}` : undefined}
                      >
                        {showWhitespace(text)}
                        {typeof cost === "number" ? <small>{fmtNum(cost, 2)}</small> : null}
                      </span>
                    </Fragment>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
