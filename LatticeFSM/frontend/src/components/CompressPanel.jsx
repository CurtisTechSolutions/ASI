import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtSize } from "../matrix.js";
import { fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

const PRECISIONS = [
  ["exact", "exact - no loss at all"],
  ["float32", "float32 - numbers to 7 digits"],
  ["float16", "float16 - numbers to 3 digits"],
  ["budget", "the least loss that fits a budget"],
];

/**
 * Fold the matrix into its central node, from the outside in; rebuild it from
 * the middle outward; walk straight from the code. The report is the code's
 * size against the dense matrix, its loss in state and in behaviour, and the
 * rebuild one shell at a time.
 */
export default function CompressPanel({ stats, tick, onMoved }) {
  const [precision, setPrecision] = useStoredState("compress.precision", "exact");
  const [budget, setBudget] = useStoredState("compress.budget", "20000");
  const [shells, setShells] = useStoredState("compress.shells", "");
  const [text, setText] = useStoredState("compress.text", "abab");
  const [fromMiddle, setFromMiddle] = useStoredState("compress.fromMiddle", true);
  const [path, setPath] = useStoredState("compress.path", "core.json.gz");
  const [report, setReport] = useState(null);
  const [walk, setWalk] = useState(null);
  const [error, setError] = useState(null);
  const [note, setNote] = useState(null);
  const [busy, setBusy] = useState(false);

  // the held code, measured against the machine as it is now (automatic compressions land here too)
  useEffect(() => {
    let alive = true;
    api
      .core()
      .then((r) => alive && setReport(r))
      .catch(() => alive && setReport(null));
    return () => {
      alive = false;
    };
  }, [tick]);

  async function act(fn, after) {
    setBusy(true);
    try {
      const r = await fn();
      if (after) after(r);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const compress = () =>
    act(
      () => api.compress(precision === "budget" ? { budget: parseInteger(budget, 20000) } : { precision }),
      (r) => {
        setReport(r);
        setNote(null);
        if (onMoved) onMoved();
      },
    );

  const s = report && report.summary;
  const f = report && report.fidelity;
  const exp = report && Array.isArray(report.expansion) ? report.expansion : [];
  const table = s && Array.isArray(s.shell_table) ? s.shell_table : [];
  const maxTouched = Math.max(1, ...table.map((r) => r.touched));
  const label = s ? `(${s.center_label.join(", ")})` : stats && stats.center ? `(${stats.center.join(", ")})` : "";

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Fold the matrix into its central node</h2>
        <p className="note">
          The matrix is a cube with a centre: the cell {label}. Compression folds every edge into a code that cell
          holds, laid out from the centre outward: one bit per cell for whether the edge was ever written to - an
          untouched edge is the prototype's, and comes back exactly - then the touched edges' fields. At exact
          precision nothing is lost: every field of every edge comes back bit for bit. float32 and float16 pack the
          numbers smaller, and the report says what that costs.
        </p>
        <div className="row">
          <SelectField label="Precision" value={precision} onChange={setPrecision} options={PRECISIONS} />
          {precision === "budget" ? (
            <NumberField label="Budget" value={budget} onChange={setBudget} min="0" step="1000" hint="bytes" />
          ) : null}
        </div>
        <div className="actions">
          <button type="button" disabled={busy} onClick={compress}>
            Compress into the central node
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {s ? (
          <>
            <div className="chips compress-stats">
              <span className="stat">
                code <b>{fmtSize(s.bytes)}</b> · {s.precision}
              </span>
              <span className="stat">
                dense matrix <b>{fmtSize(s.dense_bytes)}</b>
              </span>
              <span className="stat">
                <b>{fmtNum(s.ratio, 1)}×</b> smaller
              </span>
              <span className="stat">
                touched <b>{s.touched}</b> / {s.cells}
              </span>
              {f ? (
                <span className={`stat ${f.lossless ? "" : "warn"}`}>
                  {f.lossless ? (
                    <b>lossless</b>
                  ) : (
                    <>
                      max relative error <b>{f.max_relative_error.toExponential(1)}</b> · max KL{" "}
                      <b>{f.max_kl.toExponential(1)}</b> · greedy changed <b>{f.greedy_changed}</b>
                    </>
                  )}
                </span>
              ) : null}
              {report.compressed_at !== undefined && report.compressed_at >= 0 ? (
                <span className="stat">
                  compressed at clock <b>{report.compressed_at}</b>
                  {stats && stats.clock !== report.compressed_at ? " · the machine has moved since" : ""}
                </span>
              ) : null}
            </div>
            <h3>From the central node outward</h3>
            <div className="table-wrap matrix-wrap">
              <table className="shells">
                <thead>
                  <tr>
                    <th>shell</th>
                    <th>cells</th>
                    <th>touched</th>
                    <th>bytes</th>
                    <th>rebuilt this far: greedy choices changed</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {table.map((row, i) => (
                    <tr key={row.shell}>
                      <th>{row.shell === 0 ? "0 · the central node" : row.shell}</th>
                      <td>{row.cells}</td>
                      <td>
                        <span className="shell-bar" style={{ width: `${(row.touched / maxTouched) * 100}%` }} />
                        {row.touched}
                      </td>
                      <td>{fmtSize(row.bytes)}</td>
                      <td>{exp[i] ? exp[i].greedy_changed : "–"}</td>
                      <td>
                        <button
                          type="button"
                          className="link"
                          disabled={busy}
                          onClick={() =>
                            act(
                              () => api.expand(row.shell + 1),
                              (r) => {
                                setNote(`rebuilt ${row.shell + 1} shell${row.shell ? "s" : ""} from the central node outward: ${r.stats.touched} edges written`);
                                if (onMoved) onMoved();
                              },
                            )
                          }
                        >
                          rebuild to here
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note">
              The rebuild runs from the centre out: stopping at a shell gives back every cell within it exactly and
              the rest as the prototype. A machine taught over a and b keeps everything on the a and b slices, which
              are the outermost layers of the symbol axis, so its edges come back only in the last shells.
            </p>
          </>
        ) : (
          <p className="muted">Nothing has been compressed yet.</p>
        )}
      </div>
      {s ? (
        <div className="card wide">
          <h2>Use the code</h2>
          <div className="row">
            <NumberField label="Rebuild shells" value={shells} onChange={setShells} min="0" step="1" hint="blank: all" />
          </div>
          <div className="actions">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                act(
                  () => api.expand(shells === "" ? null : parseInteger(shells, 0)),
                  (r) => {
                    setNote(`the machine rebuilt from the code: ${r.stats.touched} edges written, clock ${r.stats.clock}`);
                    if (onMoved) onMoved();
                  },
                )
              }
            >
              Rebuild the machine from the code
            </button>
          </div>
          <div className="row">
            <TextField label="Walk a string straight from the code" value={text} onChange={setText} />
          </div>
          <CheckField label="from the middle state, outward" checked={fromMiddle} onChange={setFromMiddle} />
          <div className="actions">
            <button type="button" disabled={busy || !text} onClick={() => act(() => api.coreRun(text, fromMiddle), setWalk)}>
              Walk
            </button>
          </div>
          {walk ? (
            <div className="text-display mono">
              {walk.from_middle ? <span className="muted">from the middle: </span> : null}
              {walk.states.join(" → ")}{"  "}
              <span className={`badge ${walk.accepted ? "pass" : "fail"}`}>{walk.accepted ? "accepted" : "rejected"}</span>
              <span className="muted"> · only the cells the walk reached were decoded</span>
            </div>
          ) : null}
          <div className="row">
            <TextField label="Code file" value={path} onChange={setPath} hint="on the server" />
          </div>
          <div className="actions">
            <button type="button" disabled={busy} onClick={() => act(() => api.coreSave(path), (r) => setNote(`saved ${r.saved}`))}>
              Save the code
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                act(
                  () => api.coreLoad(path),
                  () => {
                    setNote(`loaded ${path}`);
                    if (onMoved) onMoved();
                  },
                )
              }
            >
              Load a code
            </button>
          </div>
          {note ? <p className="note">{note}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
