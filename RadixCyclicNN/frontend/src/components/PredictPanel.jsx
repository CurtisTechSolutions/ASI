import { Fragment, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { useStoredState } from "../hooks/useStoredState.js";
import {
  asArray,
  countingKind,
  fmtInt,
  fmtNum,
  jobIsRunning,
  parseInteger,
  parseNumber,
  showWhitespace,
  unitName,
  unitsOf,
  yesNo,
} from "../util.js";
import Alert from "./Alert.jsx";
import BackwardsField from "./BackwardsField.jsx";
import JobStatus from "./JobStatus.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import GuardNotice from "./GuardNotice.jsx";
import SearchFields from "./SearchFields.jsx";
import TraversalFields from "./TraversalFields.jsx";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { problemText, searchProblemsFor } from "../settings.js";
import { readingOrder, reverseUnits } from "../backwards.js";

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

/**
 * A prefix and its continuation, highlighted - or, for a query asked backwards
 * (`flip` = the model's units), in reading order: what the model says came
 * before, highlighted, then the query.
 */
function Continued({ prefix, continuation, flip }) {
  if (flip) {
    const { before, gap, query } = readingOrder(prefix, continuation, flip);
    return (
      <>
        <span className="continuation">{before}</span>
        {gap}
        <span className="prefix">{query}</span>
      </>
    );
  }
  return (
    <>
      <span className="prefix">{prefix}</span>
      <span className="continuation">{String(continuation ?? "")}</span>
    </>
  );
}

/** One of the top-K / bottom-K continuations of the count / reward model. */
function PathTable({ title, hint, paths, prefix, liked, likeDisabled, onLike, flip }) {
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
                // the model's own text: what a like rewards, backwards or not
                const text = String(p.full_text ?? `${prefix}${p.continuation ?? ""}`);
                return (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="text">
                      <Continued prefix={prefix} continuation={p.continuation} flip={flip} />
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
 *
 * With Query backwards on (site-wide, `BackwardsField`) the prefix is sent
 * turned around and every answer shown turned back, for a model trained with
 * "Read every text backwards": the highlighted part is then what the model
 * says came *before* the prefix. A like rewards the model's own text, which
 * is the backwards one.
 */
export default function PredictPanel({ status }) {
  const [prefix, setPrefix] = useStoredState("predict.prefix", "");
  const [length, setLength] = useStoredState("predict.length", "20");
  const [mode, setMode] = useStoredState("predict.mode", "dijkstra");
  const [k, setK] = useStoredState("predict.k", "5");
  const [beam, setBeam] = useStoredState("predict.beam", "");
  const [toEnd, setToEnd] = useStoredState("predict.toEnd", false);
  const [stepPenalty, setStepPenalty] = useStoredState("predict.stepPenalty", "0");
  const [temperature, setTemperature] = useStoredState("predict.temperature", "1.0");
  // the traversal, the sampling filters, the diversity and the direction are site-wide (the Settings tab)
  const { network, search, backwards } = useSiteSettings();
  const [guard, setGuard] = useStoredState("predict.guard", true);
  const [provenance, setProvenance] = useStoredState("predict.provenance", true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [sent, setSent] = useState({}); // the search settings the shown result was asked with
  // the units the shown result was asked backwards in, or null when it was asked the usual way round
  const [flip, setFlip] = useState(null);
  const [liked, setLiked] = useState(() => new Set());
  const [lastLiked, setLastLiked] = useState(null);
  const { job, running, busy, error: jobError, start, clearError } = useJob("feedback");

  const countKind = countingKind(status);  // takes a strength; predicts with the beam by default
  // only the count model aliases "dijkstra" to the beam - on the resonant model it is a real, exact mode
  const beamOnly = Boolean(status && status.kind === "count");
  // the resonant model searches (node, chars, phase): k-best is exact AND runs the metacognitive layer
  const resonantKind = Boolean(status && status.kind === "resonant");
  const likeDisabled = busy || running || jobIsRunning(status);
  // the mode the search really runs in: the count model's dijkstra is its beam
  const effectiveMode = beamOnly && mode === "dijkstra" ? "beam" : mode;
  const searchProblem = problemText(searchProblemsFor(search.values, effectiveMode));

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
    if (searchProblem) {
      setError(searchProblem);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const tuning = search.body(effectiveMode);
      // backwards: the model read every text from its end, so it is asked the same way
      const units = backwards.on ? unitsOf(status) : null;
      const body = {
        prefix: units ? reverseUnits(prefix, units) : prefix,
        length: parseInteger(length, 20),
        mode: effectiveMode,
        to_end: toEnd,
        step_penalty: parseNumber(stepPenalty, 0),
        temperature: parseNumber(temperature, 1),
        guard,
        provenance,
        ...network.body,
        ...tuning,
      };
      if (effectiveMode === "beam") {
        body.k = parseInteger(k, 5);
        const width = parseInteger(beam, 0);
        if (width > 0) body.beam = width;
      }
      const data = await api.predict(body);
      setResult(data && typeof data === "object" ? data : {});
      setSent(tuning);
      setFlip(units);
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
  // the prefix as the model saw it - turned around when it was asked backwards
  const shownPrefix = result ? String(result.prefix ?? (flip ? reverseUnits(prefix, flip) : prefix)) : prefix;
  // the model's own text, which is what a like rewards: backwards when it was asked backwards
  const fullText = result ? String(result.full_text ?? `${shownPrefix}${result.continuation ?? ""}`) : "";

  return (
    <>
      <form className="card" onSubmit={handleSubmit}>
        <h2>Predict</h2>
        <TextField
          label="Prefix"
          hint={backwards.on ? "backwards: the end of a text - the model says what came before it" : undefined}
          value={prefix}
          onChange={setPrefix}
          placeholder={backwards.on ? "over the lazy dog" : "the quick br"}
        />
        <div className="row">
          <NumberField
            label="Length"
            hint={`at least this many ${unitName(status)}; the path runs to its natural end, no cap`}
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
                : resonantKind
                  ? [
                      ["kbest", "k-best (the exact K cheapest walks)"],
                      ["dijkstra", "dijkstra (the single cheapest walk)"],
                      ["beam", "beam (top K and bottom K)"],
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
        {mode === "beam" || mode === "kbest" || (beamOnly && mode === "dijkstra") ? (
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
        <SearchFields mode={effectiveMode} compact />
        <TraversalFields compact />
        <BackwardsField compact status={status} />
        <CheckField label="Run to END (cheapest complete path)" checked={toEnd} onChange={setToEnd} />
        <CheckField
          label="Filter with the negative network"
          hint="the best continuation it does not veto; none survives, none comes back"
          checked={guard}
          onChange={setGuard}
        />
        <CheckField
          label="Say why it vetoed"
          hint="the provenance of each veto: the rule, the reasons and the blamed fragments; off, the guard reports how many it stopped"
          checked={provenance}
          onChange={setProvenance}
          disabled={!guard}
        />
        <div className="actions">
          <button type="submit" className="primary" disabled={loading || Boolean(searchProblem)}>
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
        {mode === "beam" || mode === "kbest" || (beamOnly && mode === "dijkstra") ? (
          <p className="muted">
            The beam search returns the K most likely continuations and the K least likely ones of the same length
            in one prediction; the same search generates whole texts on the Generate tab.
          </p>
        ) : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Result</h2>
        <GuardNotice guard={result && result.guard} what="continuations" />
        {!result ? (
          <p className="muted">Enter a prefix and press Predict. The highlighted part is the predicted continuation.</p>
        ) : (
          <>
            <p className="text-display">
              <Continued prefix={shownPrefix} continuation={result.continuation} flip={flip} />
            </p>
            {flip ? (
              <p className="muted">
                Asked backwards: the model was sent {JSON.stringify(shownPrefix)} and its answer is shown turned back
                round - the highlighted part is what it says came <b>before</b> your text. A like rewards the text the
                way the model reads it, backwards.
              </p>
            ) : null}
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
              <p className="muted">
                Rewarded{flip ? " (as the model reads it, backwards)" : ""}:{" "}
                {JSON.stringify(lastLiked.length > 80 ? `${lastLiked.slice(0, 80)}…` : lastLiked)}
              </p>
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
              <dt>traversal</dt>
              <dd>{String(result.traversal ?? "reward")}</dd>
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
            {sent.diversity ? (
              <p className="muted">
                The top {fmtInt(top.length)} were picked for diversity {fmtNum(sent.diversity, 2)}: the best first, then
                each by its cost plus what it repeats of the ones before it - so after the first they are not in cost
                order.
              </p>
            ) : null}
            {sent.top_k || sent.top_p || sent.min_p ? (
              <p className="muted">
                Sampled through{" "}
                {[
                  sent.top_k ? `top-K ${fmtInt(sent.top_k)}` : null,
                  sent.min_p ? `min-p ${fmtNum(sent.min_p, 3)}` : null,
                  sent.top_p ? `top-p ${fmtNum(sent.top_p, 3)}` : null,
                ]
                  .filter(Boolean)
                  .join(", ")}
                .
              </p>
            ) : null}
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
                  flip={flip}
                />
                <PathTable
                  title={`Bottom ${bottom.length} (least likely)`}
                  hint="No other continuation of that length exists (every path found is already in the top list)."
                  paths={bottom}
                  prefix={shownPrefix}
                  liked={liked}
                  likeDisabled={likeDisabled}
                  onLike={like}
                  flip={flip}
                />
              </>
            ) : null}
            <h3>Path</h3>
            {flip && path.length > 0 ? (
              <p className="muted">The model&apos;s own walk, as it reads: backwards.</p>
            ) : null}
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
