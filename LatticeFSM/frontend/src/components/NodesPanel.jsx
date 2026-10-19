import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { contextText, pct, ringLayout } from "../lm.js";
import { asArray, fmtInt, fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField } from "./Fields.jsx";

const SIZE = 520;

/**
 * The chain's nodes, explored: the most-seen input-node contexts (GET /api/chain/graph) as rings - one ring per
 * depth, depth 1 (the input node next to the output) innermost - each circle sized by how often its context
 * was seen and filled by its confidence, labelled with its guess; the same as a table; and how accurate the
 * nodes of each depth are (the chain's `depths`). It always shows the chain, whichever model is selected.
 */
export default function NodesPanel({ tick }) {
  const [limit, setLimit] = useStoredState("nodes.limit", "40");
  const [info, setInfo] = useState(null);
  const [graph, setGraph] = useState([]);
  const [picked, setPicked] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [i, g] = await Promise.all([api.lm("chain"), api.chainGraph(Math.max(1, parseInteger(limit, 40)))]);
      setInfo(i);
      setGraph(asArray(g && g.nodes));
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    refresh();
  }, [refresh, tick]);

  const unit = info ? info.unit : "char";
  const placed = ringLayout(graph, SIZE);
  const rings = [...new Set(placed.map((p) => p.ring))];
  const depths = asArray(info && info.depths);
  const ringStep = (SIZE / 2 - 36) / Math.max(1, rings.length);
  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>The chain's nodes</h2>
          <div className="inline-form">
            <NumberField label="Contexts" value={limit} onChange={setLimit} min="1" max="500" step="1" hint="the most seen" />
            <button type="button" disabled={loading} onClick={refresh}>
              {loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>
        <p className="note">
          Every input node, in each context it has been in (what it and the nodes after it held), keeps what came next,
          its guess, and its record: confidence = (right + rewarded) / (tested + rewarded + punished + 2). A prediction
          walks out from the centre (the output node) ring by ring and stops once it is confident enough.
          {info ? ` This chain: ${info.nodes} nodes of ${info.unit}s, confidence ${fmtNum(info.confidence, 2)}, ${fmtInt(info.contexts)} contexts.` : ""}
        </p>
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <div className="card">
        <h2>Rings</h2>
        {placed.length === 0 ? (
          <p className="muted">{loading ? "Loading…" : "No contexts yet: train the chain first."}</p>
        ) : (
          <svg className="graph-svg" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="the chain's contexts, a ring per depth">
            {rings.map((ring) => (
              <circle key={ring} className="ring" cx={SIZE / 2} cy={SIZE / 2} r={ringStep * (ring + 1)} />
            ))}
            <circle className="graph-node sentinel" cx={SIZE / 2} cy={SIZE / 2} r={10}>
              <title>the output node</title>
            </circle>
            {placed.map((n, i) => (
              <g key={i} onClick={() => setPicked(n)}>
                <circle
                  className={`graph-node${picked === n ? " hl" : ""}`}
                  cx={n.x}
                  cy={n.y}
                  r={n.r}
                  style={{ fillOpacity: 0.15 + 0.85 * Math.max(0, Math.min(1, Number(n.confidence) || 0)) }}
                >
                  <title>{`depth ${n.depth}: "${contextText(n.context, unit)}" → ${n.guess ?? "?"} · confidence ${fmtNum(n.confidence, 2)} · seen ${n.seen}`}</title>
                </circle>
                {n.r > 7 ? (
                  <text className="graph-label" x={n.x} y={n.y + n.r + 10} textAnchor="middle">
                    {contextText([n.guess], unit)}
                  </text>
                ) : null}
              </g>
            ))}
          </svg>
        )}
        {picked ? (
          <p className="note">
            depth {picked.depth}: <code>{contextText(picked.context, unit)}</code> → guesses <code>{contextText([picked.guess], unit)}</code>,
            confidence {fmtNum(picked.confidence, 3)}, seen {picked.seen}, right {picked.right ?? 0} of {picked.tested ?? 0} tested
          </p>
        ) : (
          <p className="muted">Click a circle. The fuller, the more confident; the bigger, the more often seen.</p>
        )}
      </div>

      <div className="card">
        <h2>Accuracy by depth</h2>
        {depths.length === 0 ? (
          <p className="muted">No depths measured yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>depth</th>
                  <th>contexts</th>
                  <th>tested</th>
                  <th>accuracy</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {depths.map((d) => (
                  <tr key={d.depth}>
                    <td>{d.depth === 0 ? "0 (output)" : d.depth}</td>
                    <td>{fmtInt(d.contexts)}</td>
                    <td>{fmtInt(d.tested)}</td>
                    <td>{pct(d.accuracy)}</td>
                    <td className="bar-cell">
                      <span className="rating-bar pass">
                        <i style={{ width: `${Math.round((Number(d.accuracy) || 0) * 100)}%` }} />
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <h3>The most-seen contexts</h3>
        {graph.length === 0 ? (
          <p className="muted">None yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>context</th>
                  <th>depth</th>
                  <th>guess</th>
                  <th>confidence</th>
                  <th>seen</th>
                  <th>right / tested</th>
                </tr>
              </thead>
              <tbody>
                {graph.map((n, i) => (
                  <tr key={i} className={picked && picked.context === n.context ? "picked" : ""}>
                    <td className="text mono">{contextText(n.context, unit)}</td>
                    <td>{n.depth}</td>
                    <td className="mono">{contextText([n.guess], unit)}</td>
                    <td>{fmtNum(n.confidence, 3)}</td>
                    <td>{fmtInt(n.seen)}</td>
                    <td>
                      {n.right ?? 0} / {n.tested ?? 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
