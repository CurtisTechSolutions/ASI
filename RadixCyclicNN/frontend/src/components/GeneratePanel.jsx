import { useState } from "react";
import { api } from "../api.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField } from "./Fields.jsx";

/** Generate texts from the START node (stochastic walks or the single cheapest path to END). */
export default function GeneratePanel() {
  const [count, setCount] = useState("3");
  const [maxLength, setMaxLength] = useState("60");
  const [temperature, setTemperature] = useState("1.0");
  const [mode, setMode] = useState("sample");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState(null);

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
        {samples === null ? (
          <p className="muted">Press Generate to sample texts from the model.</p>
        ) : samples.length === 0 ? (
          <p className="muted">The model returned no samples (train it first).</p>
        ) : (
          <ol className="samples">
            {samples.map((s, i) => (
              <li key={i}>
                <pre className="sample">{String((s && s.text) ?? "")}</pre>
                <div className="meta">
                  cost {fmtNum(s && s.cost, 3)} · {fmtInt(asArray(s && s.path).length)} path nodes ·{" "}
                  {fmtInt(String((s && s.text) ?? "").length)} chars
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
    </>
  );
}
