import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { E2_COLUMNS, e1ByLanguage, e1Marks, e1Summary, e2ByWindow } from "../results.js";
import { fmtNum } from "../util.js";
import Alert from "./Alert.jsx";

/**
 * The Results tab: what `make e1` and `make e2` wrote to `results/`, read as they are through `GET /api/results`.
 * E1 - does the stage game earn its cost? - as the summary per arm against LatticeFSM's marks, then per arm and
 * language with each seed; E2 - is the walk window a memory? - by window width. A missing file says so.
 */
export default function ResultsPanel() {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setResults(await api.results());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const e1 = results && results.e1;
  const e2 = results && results.e2;
  const marks = e1Marks(e1);
  const missing = results && Array.isArray(results.missing) ? results.missing : [];
  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>E1: does the stage game earn its cost?</h2>
          <button type="button" className="link" onClick={refresh}>
            reload
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {missing.length ? (
          <Alert
            kind="info"
            message={`Not found: ${missing.join(", ")}. Run make e1 and make e2 (from GameLattice/) to write them.`}
          />
        ) : null}
        {e1 ? (
          <>
            <p className="note">
              The four languages, {Array.isArray(e1.seeds) ? e1.seeds.length : "–"} seeds, {e1.episodes} episodes, {e1.states} states, window{" "}
              {e1.window}: held-out accuracy of the encoder&apos;s verdict and of the lattice&apos;s own, the mean over
              languages and seeds. The numbers to beat are a 13-state LatticeFSM lattice alone:{" "}
              {marks.map(([name, v], k) => (
                <span key={name}>
                  {k ? ", " : ""}
                  {name} <b>{fmtNum(v, 3)}</b>
                </span>
              ))}
              .
            </p>
            <div className="table-wrap">
              <table className="data results">
                <thead>
                  <tr>
                    <th>arm</th>
                    <th>encoder</th>
                    <th>lattice</th>
                    {marks.map(([name, v]) => (
                      <th key={name} title={`beats ${name} (${v}): encoder / lattice`}>
                        &gt; {fmtNum(v, 3)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {e1Summary(e1).map((r) => (
                    <tr key={r.arm}>
                      <td className="mono">{r.arm}</td>
                      <td className="mono">
                        <b>{fmtNum(r.encoder, 3)}</b>
                      </td>
                      <td className="mono">{fmtNum(r.lattice, 3)}</td>
                      {marks.map(([name]) => (
                        <td key={name} className="mono">
                          <span className={r.beats[name].encoder ? "ok" : "bad"}>{r.beats[name].encoder ? "✓" : "·"}</span> /{" "}
                          <span className={r.beats[name].lattice ? "ok" : "bad"}>{r.beats[name].lattice ? "✓" : "·"}</span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3>Per arm and language</h3>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>arm</th>
                    <th>language</th>
                    <th>encoder</th>
                    <th>lattice</th>
                    <th>agreement</th>
                    <th>diversity (bits)</th>
                    <th>per seed: encoder / lattice</th>
                  </tr>
                </thead>
                <tbody>
                  {e1ByLanguage(e1).map((g) => (
                    <tr key={`${g.arm}-${g.language}`}>
                      <td className="mono">{g.arm}</td>
                      <td className="mono">{g.language}</td>
                      <td className="mono">{fmtNum(g.encoder, 3)}</td>
                      <td className="mono">{fmtNum(g.lattice, 3)}</td>
                      <td className="mono">{fmtNum(g.agreement, 2)}</td>
                      <td className="mono">{fmtNum(g.diversity, 2)}</td>
                      <td className="mono small-note">
                        {g.seeds.map((r) => `${r.seed}: ${fmtNum(r.encoder_accuracy, 3)} / ${fmtNum(r.lattice_accuracy, 3)}`).join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="muted">No E1 results to show.</p>
        )}
      </div>
      <div className="card wide">
        <h2>E2: is the walk window a memory?</h2>
        {e2 ? (
          <>
            <p className="note">
              Nim by window width, the mean over seeds: babbling first, then play against a random opponent. To beat:
              wins {fmtNum(e2.random_opponent_win_rate_to_beat, 2)} (chance), legal first try{" "}
              {fmtNum(e2.random_legal_first_try, 2)}; the information the named state carries about the position is the
              memory claim.
            </p>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>window</th>
                    <th>seeds</th>
                    {E2_COLUMNS.map(([key, label]) => (
                      <th key={key}>{label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {e2ByWindow(e2).map((w) => (
                    <tr key={w.window}>
                      <td className="mono">
                        <b>{w.window}</b>
                      </td>
                      <td className="mono">{w.seeds}</td>
                      {E2_COLUMNS.map(([key]) => (
                        <td key={key} className="mono">
                          {fmtNum(w[key], key === "states_used" ? 1 : 3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="muted">No E2 results to show.</p>
        )}
      </div>
    </div>
  );
}
