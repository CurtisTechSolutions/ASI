import { Fragment, useState } from "react";
import { api } from "../api.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber, showWhitespace, yesNo } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

const SENTINELS = new Set(["<s>", "</s>"]);

/** Shortest-path (Dijkstra) or sampled continuation of a prefix. */
export default function PredictPanel() {
  const [prefix, setPrefix] = useState("");
  const [length, setLength] = useState("20");
  const [mode, setMode] = useState("dijkstra");
  const [toEnd, setToEnd] = useState(false);
  const [stepPenalty, setStepPenalty] = useState("0");
  const [temperature, setTemperature] = useState("1.0");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.predict({
        prefix,
        length: parseInteger(length, 20),
        mode,
        to_end: toEnd,
        step_penalty: parseNumber(stepPenalty, 0),
        temperature: parseNumber(temperature, 1),
      });
      setResult(data && typeof data === "object" ? data : {});
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
            options={[
              ["dijkstra", "dijkstra (shortest path)"],
              ["sample", "sample (stochastic)"],
            ]}
          />
        </div>
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
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Result</h2>
        {!result ? (
          <p className="muted">Enter a prefix and press Predict. The highlighted part is the predicted continuation.</p>
        ) : (
          <>
            <p className="text-display">
              <span className="prefix">{String(result.prefix ?? prefix)}</span>
              <span className="continuation">{String(result.continuation ?? "")}</span>
            </p>
            <dl className="kv">
              <dt>cost</dt>
              <dd>{fmtNum(result.cost, 4)}</dd>
              <dt>reached END</dt>
              <dd>{result.reached_end === undefined ? "–" : yesNo(result.reached_end)}</dd>
              <dt>states expanded</dt>
              <dd>{fmtInt(result.expanded)}</dd>
              <dt>path nodes</dt>
              <dd>{fmtInt(path.length)}</dd>
            </dl>
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
