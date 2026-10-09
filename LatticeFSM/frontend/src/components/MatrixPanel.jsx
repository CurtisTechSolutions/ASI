import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { cellStyle, isCenter } from "../matrix.js";
import { fmtNum } from "../util.js";
import Alert from "./Alert.jsx";
import EdgeCard from "./EdgeCard.jsx";

/**
 * The matrix, one symbol-slice at a time: each row a state, each cell the
 * probability of moving to that column's state on the chosen symbol under
 * the stimulation now, shaded by it, with the channel's width and `seen`
 * beneath. A click on a cell opens the edge's record.
 */
export default function MatrixPanel({ stats, tick }) {
  const alphabet = stats && Array.isArray(stats.alphabet) ? stats.alphabet : [];
  const [symbol, setSymbol] = useState(null);
  const [slice, setSlice] = useState(null);
  const [edge, setEdge] = useState(null);
  const [error, setError] = useState(null);
  const active = symbol !== null && alphabet.includes(symbol) ? symbol : alphabet[0];

  const refresh = useCallback(async () => {
    if (active === undefined) return;
    try {
      setSlice(await api.matrix(active));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, [active]);

  useEffect(() => {
    refresh();
  }, [refresh, tick]);

  async function showEdge(source, target) {
    try {
      setEdge({ source, target, record: await api.edge(source, active, target) });
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  const rows = slice && Array.isArray(slice.rows) ? slice.rows : [];
  const accepting = slice && Array.isArray(slice.accepting) ? slice.accepting : [];
  const S = rows.length;

  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>The matrix</h2>
          <div className="tabs sub symbols" role="tablist" aria-label="Symbol">
            <span className="muted">symbol</span>
            {alphabet.map((a) => (
              <button
                key={a}
                type="button"
                role="tab"
                aria-selected={a === active}
                title={`the slice for symbol ${a}`}
                className={a === active ? "active" : ""}
                onClick={() => setSymbol(a)}
              >
                {a}
              </button>
            ))}
          </div>
        </div>
        <p className="note">
          The slice of the matrix for one symbol. From the state on the left, reading this symbol, the machine moves
          to the state above with the probability in the cell, under stimulation{" "}
          <b>{slice ? fmtNum(slice.stimulation, 2) : "–"}</b>. Beneath: the channel's width and how often the edge was
          traversed. ✓ marks an accepting state, ← the state the machine is in, and the ringed cell on symbol{" "}
          {stats && stats.center && alphabet[stats.center[1]]} is the matrix's central node, the one the Compress tab folds
          the matrix into.
        </p>
        <Alert message={error} onDismiss={() => setError(null)} />
        <div className="table-wrap matrix-wrap">
          <table className="matrix">
            <thead>
              <tr>
                <th>from \ to</th>
                {Array.from({ length: S }, (_, t) => (
                  <th key={t} className={accepting.includes(t) ? "accepting" : ""}>
                    {t}
                    {accepting.includes(t) ? " ✓" : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.source}>
                  <th className={slice && row.source === slice.state ? "here" : ""}>
                    {row.source}
                    {slice && row.source === slice.state ? " ←" : ""}
                  </th>
                  {row.edges.map((e) => (
                    <td
                      key={e.target}
                      className={`cell${isCenter(stats && stats.center, row.source, slice.symbol_index, e.target) ? " center" : ""}`}
                      style={cellStyle(e.probability)}
                      title={`seen ${e.seen}, width ${fmtNum(e.width, 2)}, net ${fmtNum(e.net, 2)}, log-weight ${fmtNum(e.log_weight, 2)}`}
                      onClick={() => showEdge(row.source, e.target)}
                    >
                      {fmtNum(e.probability, 2)}
                      <small>
                        w {fmtNum(e.width, 1)} · {e.seen}
                      </small>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {edge ? <EdgeCard edge={edge.record} symbol={active} onClose={() => setEdge(null)} /> : null}
    </div>
  );
}
