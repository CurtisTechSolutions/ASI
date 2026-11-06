import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { fmtInt, fmtNum } from "../util.js";
import Alert from "./Alert.jsx";
import Strip from "./Strip.jsx";

/**
 * The Encoder tab: the GameRadix encoder's view of itself - the gram graph (nodes, edges, how much the radix
 * compression saved), how often each state was named (the collapse guard: diversity in bits against log2 S), the
 * economy (mean wealth and its Gini coefficient over the nodes that ever played), the cycles that taught THINK,
 * and the richest nodes with their reputation and plays.
 */
export default function EncoderPanel({ tick }) {
  const [view, setView] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setView(await api.encoder(25));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, tick]);

  const named = view && Array.isArray(view.named) ? view.named : [];
  const total = named.reduce((a, b) => a + b, 0);
  const top = view && Array.isArray(view.top) ? view.top : [];
  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>The encoder</h2>
          <button type="button" className="link" onClick={refresh}>
            refresh
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {view ? (
          <div className="tiles">
            <span className="tile" title="alive nodes of the gram graph, sentinels included">
              <small>graph nodes</small>
              <b>{fmtInt(view.graph.nodes)}</b>
            </span>
            <span className="tile">
              <small>graph edges</small>
              <b>{fmtInt(view.graph.edges)}</b>
            </span>
            <span className="tile" title="grams stored per real node: above 1, compressed nodes hold whole chunks">
              <small>compression</small>
              <b>{fmtNum(view.graph.compression_ratio, 2)}</b>
            </span>
            <span className="tile" title="entropy of the states named so far; log2 S is every state equally">
              <small>diversity</small>
              <b>{fmtNum(view.diversity, 2)}</b>
              <small>of {fmtNum(view.max_diversity, 2)} bits</small>
            </span>
            <span className="tile" title="payoff earned less clearing prices paid, over the nodes that ever played">
              <small>mean wealth</small>
              <b>{fmtNum(view.wealth.mean, 1)}</b>
            </span>
            <span className="tile" title="0: every player equally rich; near 1: one node holds it all (specialisation)">
              <small>wealth gini</small>
              <b>{fmtNum(view.wealth.gini, 3)}</b>
            </span>
            <span className="tile" title="best-response cycles that taught THINK at a node">
              <small>thinks</small>
              <b>{fmtInt(view.thinks)}</b>
            </span>
            <span className="tile">
              <small>namings</small>
              <b>{fmtInt(view.namings)}</b>
            </span>
          </div>
        ) : null}
      </div>
      <div className="card">
        <h3>What it has named</h3>
        <p className="note">How often each of the {named.length} states was named, learning only (a quiet read counts nothing): a strip that is all one bar is a collapse.</p>
        <Strip values={named} large label="named-state histogram" />
        <table className="data histogram">
          <tbody>
            {named.map((c, s) => (
              <tr key={s}>
                <td className="mono">{s}</td>
                <td className="mono">{fmtInt(c)}</td>
                <td>
                  <span className="hbar" style={{ width: `${total ? (c / Math.max(...named)) * 100 : 0}%` }} />
                </td>
                <td className="mono muted">{total ? `${((c / total) * 100).toFixed(1)}%` : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3>The richest nodes</h3>
        <p className="note">
          {view ? `${fmtInt(view.players)} nodes have played. ` : ""}A node&apos;s reputation is (hits + 1) / (plays + 2),
          discounted; its wealth what it earned less the prices it paid.
        </p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>node</th>
                <th>gram</th>
                <th>wealth</th>
                <th>reputation</th>
                <th>plays</th>
              </tr>
            </thead>
            <tbody>
              {top.map((t) => (
                <tr key={t.node}>
                  <td className="mono">{t.node}</td>
                  <td className="mono">
                    <code>{t.label}</code>
                  </td>
                  <td className="mono">{fmtNum(t.wealth, 1)}</td>
                  <td className="mono">{fmtNum(t.reputation, 3)}</td>
                  <td className="mono">{fmtNum(t.plays, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
