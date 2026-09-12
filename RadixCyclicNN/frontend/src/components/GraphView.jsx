import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { asArray, fmtInt, fmtNum, parseInteger, showWhitespace } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField } from "./Fields.jsx";

const SIZE = 820;
const RING_RADIUS = SIZE / 2 - 90;
const MIN_NODE_RADIUS = 4;
const MAX_NODE_RADIUS = 18;
const LABEL_LIMIT = 80;
const LABEL_CHARS = 14;
const START_ID = 0;
const END_ID = 1;
const DEFAULT_LIMIT = 150;
const EMPTY_SET = new Set();

/** Place nodes evenly on a circle (ordered by id); radius grows with visit count. */
function layoutNodes(nodes) {
  const sorted = nodes.filter((n) => n && n.id !== null && n.id !== undefined).sort((a, b) => a.id - b.id);
  let maxCount = 1;
  for (const n of sorted) maxCount = Math.max(maxCount, Number(n.count) || 0);
  const centre = SIZE / 2;
  const placed = new Map();
  sorted.forEach((node, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(1, sorted.length);
    const visits = Math.max(0, Number(node.count) || 0);
    placed.set(node.id, {
      node,
      angle,
      x: centre + RING_RADIUS * Math.cos(angle),
      y: centre + RING_RADIUS * Math.sin(angle),
      r: MIN_NODE_RADIUS + (MAX_NODE_RADIUS - MIN_NODE_RADIUS) * Math.sqrt(visits / maxCount),
    });
  });
  return placed;
}

/** Quadratic curve from s to t, bent to the right so opposite edges are distinguishable; ends trimmed to node borders. */
function curvePath(s, t) {
  const dx = t.x - s.x;
  const dy = t.y - s.y;
  const dist = Math.hypot(dx, dy) || 1;
  const bend = Math.min(40, dist * 0.18);
  const cx = (s.x + t.x) / 2 - (dy / dist) * bend;
  const cy = (s.y + t.y) / 2 + (dx / dist) * bend;
  const sd = Math.hypot(cx - s.x, cy - s.y) || 1;
  const td = Math.hypot(cx - t.x, cy - t.y) || 1;
  const x1 = s.x + ((cx - s.x) / sd) * s.r;
  const y1 = s.y + ((cy - s.y) / sd) * s.r;
  const x2 = t.x + ((cx - t.x) / td) * (t.r + 3);
  const y2 = t.y + ((cy - t.y) / td) * (t.r + 3);
  return `M${x1.toFixed(1)} ${y1.toFixed(1)} Q${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

/** Self-loop drawn as a small circle just outside the ring. */
function loopGeometry(p) {
  const offset = p.r + 9;
  return { cx: p.x + offset * Math.cos(p.angle), cy: p.y + offset * Math.sin(p.angle), r: 8 };
}

function truncate(text) {
  return text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text;
}

/** SVG view of the top-N nodes of the graph: circular layout, hover tooltips, edge highlighting. */
export default function GraphView() {
  const [limit, setLimit] = useState(String(DEFAULT_LIMIT));
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hover, setHover] = useState(null);
  const wrapRef = useRef(null);

  const load = useCallback(async (n) => {
    setLoading(true);
    try {
      const data = await api.graph(n);
      setGraph({ nodes: asArray(data && data.nodes), edges: asArray(data && data.edges) });
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(DEFAULT_LIMIT);
  }, [load]);

  const placed = useMemo(() => layoutNodes(graph.nodes), [graph.nodes]);
  const nodeList = useMemo(() => [...placed.values()], [placed]);

  const edges = useMemo(() => {
    let maxEdgeCount = 1;
    for (const e of graph.edges) if (e) maxEdgeCount = Math.max(maxEdgeCount, Number(e.count) || 0);
    const out = [];
    graph.edges.forEach((e, i) => {
      if (!e) return;
      const s = placed.get(e.source);
      const t = placed.get(e.target);
      if (!s || !t) return;
      const prob = Number.isFinite(e.prob) ? Math.min(1, Math.max(0, e.prob)) : 0.5;
      const uses = Math.max(0, Number(e.count) || 0);
      out.push({
        key: i,
        edge: e,
        s,
        t,
        prob,
        negative: Number(e.weight) < 0,
        width: 0.8 + 2.2 * (Math.log1p(uses) / Math.log1p(maxEdgeCount)),
        loop: s === t ? loopGeometry(s) : null,
        d: s === t ? null : curvePath(s, t),
      });
    });
    return out;
  }, [graph.edges, placed]);

  const hoverId = hover ? hover.id : null;
  const neighbours = useMemo(() => {
    if (hoverId === null) return EMPTY_SET;
    const set = new Set();
    for (const g of edges) {
      if (g.edge.source === hoverId) set.add(g.edge.target);
      if (g.edge.target === hoverId) set.add(g.edge.source);
    }
    return set;
  }, [hoverId, edges]);

  const hoverNode = hoverId === null ? null : placed.get(hoverId);

  function pointer(event, id) {
    const rect = wrapRef.current ? wrapRef.current.getBoundingClientRect() : { left: 0, top: 0, width: 0 };
    return { id, x: event.clientX - rect.left, y: event.clientY - rect.top, width: rect.width };
  }

  function handleSubmit(event) {
    event.preventDefault();
    load(Math.max(2, parseInteger(limit, DEFAULT_LIMIT)));
  }

  const labelOf = (id) => {
    const p = placed.get(id);
    return showWhitespace(p && p.node.label !== undefined ? p.node.label : `#${id}`);
  };

  return (
    <div className="card">
      <div className="toolbar">
        <h2>Graph</h2>
        <form className="inline-form" onSubmit={handleSubmit}>
          <NumberField label="Top nodes by visit count" value={limit} onChange={setLimit} min={2} step={1} />
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </form>
      </div>
      <Alert message={error} onDismiss={() => setError(null)} />
      <p className="muted">
        {fmtInt(nodeList.length)} nodes, {fmtInt(edges.length)} edges shown · node radius ∝ visit count · edge opacity ∝
        transition probability · red edges carry negative weights · gold nodes are START / END · hover a node for its
        activation parameters.
      </p>
      <div className="graph-wrap" ref={wrapRef}>
        <svg className="graph-svg" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Model graph">
          <defs>
            <marker
              id="arrow"
              markerUnits="userSpaceOnUse"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path className="arrow" d="M0 0 L8 4 L0 8 Z" />
            </marker>
            <marker
              id="arrow-neg"
              markerUnits="userSpaceOnUse"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path className="arrow neg" d="M0 0 L8 4 L0 8 Z" />
            </marker>
          </defs>
          {nodeList.length === 0 ? (
            <text x={SIZE / 2} y={SIZE / 2} textAnchor="middle" className="graph-label" style={{ fontSize: 16 }}>
              {loading ? "Loading graph…" : "No nodes to show. Train the model first."}
            </text>
          ) : null}
          <g>
            {edges.map((g) => {
              const connected = hoverId !== null && (g.edge.source === hoverId || g.edge.target === hoverId);
              const opacity =
                hoverId === null ? 0.15 + 0.85 * g.prob : connected ? Math.max(0.9, g.prob) : 0.04 + 0.1 * g.prob;
              const className = `graph-edge${g.negative ? " neg" : ""}${connected ? " hl" : ""}`;
              const reward = typeof g.edge.reward === "number" ? ` · reward=${fmtNum(g.edge.reward, 2)}` : "";
              const shares = typeof g.edge.share === "number" ? ` · share=${fmtNum(g.edge.share, 2)} recent=${fmtNum(g.edge.recent_share, 2)} (${fmtInt(g.edge.recent_count)} in window)` : "";
              const title = `${labelOf(g.edge.source)} → ${labelOf(g.edge.target)} · p=${fmtNum(g.edge.prob, 3)} · cost=${fmtNum(g.edge.cost, 3)} · w=${fmtNum(g.edge.weight, 3)} · n=${fmtInt(g.edge.count)}${reward}${shares}`;
              return g.loop ? (
                <circle
                  key={g.key}
                  className={className}
                  cx={g.loop.cx}
                  cy={g.loop.cy}
                  r={g.loop.r}
                  strokeWidth={g.width}
                  style={{ opacity }}
                >
                  <title>{title}</title>
                </circle>
              ) : (
                <path
                  key={g.key}
                  className={className}
                  d={g.d}
                  strokeWidth={g.width}
                  style={{ opacity }}
                  markerEnd={`url(#${g.negative ? "arrow-neg" : "arrow"})`}
                >
                  <title>{title}</title>
                </path>
              );
            })}
          </g>
          <g>
            {nodeList.map((p) => {
              const id = p.node.id;
              const sentinel = id === START_ID || id === END_ID;
              const hovered = hoverId === id;
              const dimmed = hoverId !== null && !hovered && !neighbours.has(id);
              const showLabel = sentinel || hovered || nodeList.length <= LABEL_LIMIT;
              const rightSide = Math.cos(p.angle) >= 0;
              return (
                <g key={id} opacity={dimmed ? 0.3 : 1}>
                  <circle
                    className={`graph-node${sentinel ? " sentinel" : ""}${hovered ? " hl" : ""}`}
                    cx={p.x}
                    cy={p.y}
                    r={p.r}
                    onMouseEnter={(e) => setHover(pointer(e, id))}
                    onMouseMove={(e) => setHover(pointer(e, id))}
                    onMouseLeave={() => setHover(null)}
                  />
                  {showLabel ? (
                    <text
                      className="graph-label"
                      x={p.x + (p.r + 5) * Math.cos(p.angle)}
                      y={p.y + (p.r + 5) * Math.sin(p.angle)}
                      textAnchor={rightSide ? "start" : "end"}
                      dominantBaseline="middle"
                    >
                      {truncate(showWhitespace(p.node.label ?? `#${id}`))}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        </svg>
        {hover && hoverNode ? (
          <div
            className="tooltip"
            style={{ left: Math.min(hover.x + 14, Math.max(0, hover.width - 290)), top: hover.y + 14 }}
          >
            <code>{showWhitespace(hoverNode.node.label ?? `#${hoverNode.node.id}`)}</code>
            <dl className="kv">
              <dt>id</dt>
              <dd>{String(hoverNode.node.id)}</dd>
              <dt>count</dt>
              <dd>{fmtInt(hoverNode.node.count)}</dd>
              <dt>activation</dt>
              <dd>{fmtNum(hoverNode.node.activation, 4)}</dd>
              <dt>z</dt>
              <dd>{fmtNum(hoverNode.node.z, 3)}</dd>
              <dt>a, b, h, k</dt>
              <dd>
                {fmtNum(hoverNode.node.a, 3)}, {fmtNum(hoverNode.node.b, 3)}, {fmtNum(hoverNode.node.h, 3)},{" "}
                {fmtNum(hoverNode.node.k, 3)}
              </dd>
              <dt>edges</dt>
              <dd>{fmtInt(neighbours.size)} neighbours</dd>
            </dl>
          </div>
        ) : null}
      </div>
    </div>
  );
}
