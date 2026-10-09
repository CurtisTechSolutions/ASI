import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { curvePoints, tableRows } from "../matrix.js";
import { fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField } from "./Fields.jsx";
import LineChart from "./LineChart.jsx";

/**
 * Teach a regular language: random strings are run, rewarded when the run
 * ends in the right kind of state and punished when it does not, and the
 * greedy walk's accuracy on held-out strings is measured as it goes.
 */
export default function TrainPanel({ onMoved }) {
  const [languages, setLanguages] = useState([]);
  const [language, setLanguage] = useStoredState("train.language", "even-b");
  const [episodes, setEpisodes] = useStoredState("train.episodes", "4000");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .languages()
      .then((list) => setLanguages(Array.isArray(list) ? list : []))
      .catch((err) => setError(err.message));
  }, []);

  async function train() {
    setBusy(true);
    try {
      setResult(await api.train({ language, episodes: parseInteger(episodes, 4000) }));
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const chosen = languages.find((l) => l.name === language);
  const rows = result ? tableRows(result.table) : [];

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Teach a language</h2>
        <p className="note">
          Every episode is one random string of up to six symbols, run and credited by the language's verdict:
          reward 1 when the machine ended in an accepting state and the string is in the language, or in another
          state and it is not; punishment 1 otherwise. The language's accepting states are set on the machine first.
        </p>
        <div className="row">
          <SelectField
            label="Language"
            value={language}
            onChange={setLanguage}
            options={languages.map((l) => [l.name, `${l.name} — ${l.description} (accepts at ${l.accepting.join(", ")}, ${l.min_states} states suffice)`])}
          />
          <NumberField label="Episodes" value={episodes} onChange={setEpisodes} min="1" step="100" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy || !language} onClick={train}>
            {busy ? "Training…" : "Train"}
          </button>
          {chosen ? (
            <span className="muted">
              needs an accepting state {Math.max(...chosen.accepting)} or higher: a machine of at least{" "}
              {Math.max(...chosen.accepting) + 1} states
            </span>
          ) : null}
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {result ? (
          <>
            <p>
              <b>{result.language}</b>: accuracy {fmtNum(result.before, 3)} → <b>{fmtNum(result.after, 3)}</b> after{" "}
              {result.episodes} episodes, on 200 held-out strings by the greedy walk.
            </p>
            <LineChart
              series={[{ name: "accuracy", color: "var(--accent)", values: curvePoints(result.curve) }]}
              xLabel="episodes"
              yLabel="accuracy"
              height={220}
            />
            <h3>The greedy table</h3>
            <div className="table-wrap">
              <table>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.state}>
                      <th>
                        {r.state}
                        {r.accepting ? " ✓" : ""}
                      </th>
                      {r.cells.map((c) => (
                        <td key={c} className="mono">
                          {c}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
