import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { curvePoints, tableRows } from "../matrix.js";
import { fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";
import LineChart from "./LineChart.jsx";

/**
 * Teach a regular language: random strings are run and credited by the
 * oracle - every step by whether the state it lands in classifies the prefix
 * read so far rightly (or, "whole run", the run by its final answer) -
 * rewarded when right and, unless rewards only, punished when wrong; the
 * greedy walk's accuracy on held-out strings is measured as it goes.
 */
export default function TrainPanel({ onMoved }) {
  const [languages, setLanguages] = useState([]);
  const [language, setLanguage] = useStoredState("train.language", "even-b");
  const [episodes, setEpisodes] = useStoredState("train.episodes", "4000");
  const [oracle, setOracle] = useStoredState("train.oracle", true);
  const [punish, setPunish] = useStoredState("train.punish", true);
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
      setResult(await api.train({ language, episodes: parseInteger(episodes, 4000), whole_run: !oracle, punish }));
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
          Every episode is one random string of up to six symbols, run and credited by the language - the oracle. By
          default every step is judged alone: the edge that read a symbol is rewarded 1 when the state it lands in is
          accepting exactly when the prefix read so far is in the language, and punished 1 when not, so a run that
          goes wrong at its last step still has its right steps rewarded. "Whole run" credits the run instead by its
          final answer, the last edge in full and each earlier one 0.8 times the next. Without punishment only what
          is right is rewarded. The language's accepting states are set on the machine first.
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
        <div className="row">
          <CheckField
            label="Credit every step by the oracle"
            checked={oracle}
            onChange={setOracle}
            hint="off: the whole run by its final answer"
          />
          <CheckField label="Punish what is wrong" checked={punish} onChange={setPunish} hint="off: rewards only" />
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
              {result.episodes} episodes, on 200 held-out strings by the greedy walk
              {result.credit ? ` · credited by ${result.credit === "oracle" ? "the oracle, step by step" : "the whole run"}${result.punish === false ? ", rewards only" : ""}` : ""}.
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
