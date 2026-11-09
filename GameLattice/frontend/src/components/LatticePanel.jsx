import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { cellStyle, tableRows } from "../matrix.js";
import { fmtNum } from "../util.js";
import Alert from "./Alert.jsx";
import EdgeCard from "./EdgeCard.jsx";

/**
 * The Lattice tab: the body's `latticefsm::Machine`, as LatticeFSM's Matrix tab shows one - the `S × S` slice for
 * one symbol, each cell the probability of the row's move under the stimulation now, shaded by it, with the
 * channel's width and how often it was traversed beneath; a click opens the edge's record - and the greedy
 * transition table with the accepting states marked. The encoder names where a step starts; the lattice
 * predicts where it goes.
 */
export default function LatticePanel({ tick }) {
  const [table, setTable] = useState(null);
  const [symbol, setSymbol] = useState(null);
  const [slice, setSlice] = useState(null);
  const [edge, setEdge] = useState(null);
  const [error, setError] = useState(null);
  const alphabet = table && Array.isArray(table.alphabet) ? table.alphabet : [];
  const active = symbol !== null && alphabet.includes(symbol) ? symbol : alphabet[0];

  const refresh = useCallback(async () => {
    try {
      const t = await api.table();
      setTable(t);
      const a = symbol !== null && t.alphabet.includes(symbol) ? symbol : t.alphabet[0];
      if (a !== undefined) setSlice(await api.matrix(a));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, [symbol]);

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
  const greedy = tableRows(table);

  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>The lattice</h2>
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
            <button type="button" className="link" onClick={refresh}>
              refresh
            </button>
          </div>
        </div>
        <p className="note">
          The encoder names where we are; the lattice says where this takes us. From the state on the left, reading
          this symbol, the lattice moves to the state above with the probability in the cell, under stimulation{" "}
          <b>{slice ? fmtNum(slice.stimulation, 2) : "–"}</b>. Beneath: the channel&apos;s width and how often the edge
          was traversed. ✓ marks an accepting state. When the encoder sets the state, the next step starts from the
          state the encoder named, not from the lattice&apos;s own target - their agreement is a meta-feature, never
          credit.
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
                  <th className={accepting.includes(row.source) ? "accepting" : ""}>{row.source}</th>
                  {row.edges.map((e) => (
                    <td
                      key={e.target}
                      className="cell"
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
      <div className="card">
        <h3>The greedy transition table</h3>
        <p className="note">
          Each state&apos;s likeliest target per symbol. ✓ accepting, ▸ the start state. A walk of this table from the
          start is the lattice&apos;s own verdict on a string.
        </p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>state</th>
                {alphabet.map((a) => (
                  <th key={a}>{a}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {greedy.map((r) => (
                <tr key={r.state} className={r.accepting ? "accepting-row" : ""}>
                  <td className="mono">
                    {r.start ? "▸ " : ""}
                    {r.state}
                    {r.accepting ? " ✓" : ""}
                  </td>
                  {r.cells.map((c) => (
                    <td key={c.symbol} className="mono">
                      {c.target}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
