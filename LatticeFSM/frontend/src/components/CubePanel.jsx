import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { cellAt, cubeValue, project } from "../matrix.js";
import { fmtNum, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import EdgeCard from "./EdgeCard.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";

const MODES = [
  ["probability", "probability in its row"],
  ["seen", "seen (log)"],
  ["width", "width (log)"],
  ["net", "net credit per traversal"],
];

/** A CSS colour variable of the page, read once per draw so the view follows the light or dark scheme. */
function cssVar(el, name, fallback) {
  const v = el ? getComputedStyle(el).getPropertyValue(name).trim() : "";
  return v || fallback;
}

/**
 * The whole matrix in three dimensions: every edge `(source, symbol, target)` a cell, drawn by the measure chosen,
 * turned by dragging, zoomed by the wheel, read by hovering, opened by a click. The source axis runs top to bottom
 * (state 0 at the top), the symbol axis into the page, the target axis across; the central node, the central
 * vertical vector and the focus node can be picked out, and the cells shown limited to the inner shells, the
 * touched edges, or one symbol's slice. Drawn on a canvas by hand - no 3D library.
 */
export default function CubePanel({ tick }) {
  const [cube, setCube] = useState(null);
  const [mode, setMode] = useStoredState("cube.mode", "probability");
  const [minimum, setMinimum] = useStoredState("cube.minimum", "0");
  const [shells, setShells] = useStoredState("cube.shells", "");
  const [slice, setSlice] = useStoredState("cube.slice", "");
  const [touchedOnly, setTouchedOnly] = useStoredState("cube.touched", false);
  const [showCenter, setShowCenter] = useStoredState("cube.center", true);
  const [showVector, setShowVector] = useStoredState("cube.vector", true);
  const [showFocus, setShowFocus] = useStoredState("cube.focus", true);
  const [spin, setSpin] = useStoredState("cube.spin", true);
  const [hover, setHover] = useState(null);
  const [edge, setEdge] = useState(null);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const view = useRef({ yaw: -0.6, pitch: 0.35, zoom: 1, drag: null, moved: false, drawn: [] });
  const frame = useRef(0);

  useEffect(() => {
    let alive = true;
    api
      .cube()
      .then((c) => {
        if (alive) {
          setCube(c);
          setError(null);
        }
      })
      .catch((err) => alive && setError(err.message));
    return () => {
      alive = false;
    };
  }, [tick]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !cube) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, rect.width), h = Math.max(200, rect.height);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const accent = cssVar(canvas, "--accent", "#2f5fd8");
    const danger = cssVar(canvas, "--danger", "#b42318");
    const hl = cssVar(canvas, "--accent-2", "#b8862b");
    const text = cssVar(canvas, "--muted", "#667085");

    const [S, A, T] = cube.shape;
    const { yaw, pitch, zoom } = view.current;
    const span = Math.max(S, A, T);
    const distance = span * 2.4;
    const focal = (Math.min(w, h) / span) * distance * 0.42 * zoom;
    const cx = w / 2, cy = h / 2;
    const values = cube[mode] || cube.probability;
    const maxSeen = mode === "seen" ? Math.max(1, ...cube.seen) : 1;
    const floor = parseNumber(minimum, 0);
    const shellLimit = shells === "" ? Infinity : parseNumber(shells, Infinity);
    const sliceIndex = slice === "" ? null : cube.symbols.indexOf(slice);
    const [c0, c1, c2] = cube.center;
    const focusNode = cube.focus_node;
    const cells = [];
    for (let i = 0; i < values.length; i++) {
      const [s, a, t] = cellAt(cube.shape, i);
      if (sliceIndex !== null && sliceIndex >= 0 && a !== sliceIndex) continue;
      if (cube.shell[i] > shellLimit) continue;
      if (touchedOnly && !(cube.seen[i] > 0)) continue;
      const v = cubeValue(mode, values[i], maxSeen);
      const isCenter = s === c0 && a === c1 && t === c2;
      const isVector = a === c1 && t === c2;
      const isFocus = s === focusNode[0] && a === focusNode[1] && t === focusNode[2];
      const marked = (showCenter && isCenter) || (showVector && isVector) || (showFocus && isFocus);
      if (Math.abs(v) < floor && !marked) continue;
      const p = project([t - (T - 1) / 2, -(s - (S - 1) / 2), a - (A - 1) / 2], yaw, pitch, distance, focal);
      cells.push({ i, s, a, t, v, p, isCenter, isVector, isFocus, marked });
    }
    cells.sort((x, y) => x.p[2] - y.p[2]);
    const drawn = [];
    for (const c of cells) {
      const [x, y, , scale] = c.p;
      const mag = Math.abs(c.v);
      const size = Math.max(1.5, scale * (0.18 + 0.55 * mag));
      ctx.globalAlpha = Math.min(1, 0.07 + 0.9 * mag);
      ctx.fillStyle = c.v < 0 ? danger : accent;
      ctx.fillRect(cx + x - size / 2, cy + y - size / 2, size, size);
      if (c.marked) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = c.isFocus && showFocus ? danger : hl;
        ctx.lineWidth = c.isCenter && showCenter ? 2.5 : 1.5;
        const r = Math.max(size, scale * 0.6);
        ctx.strokeRect(cx + x - r / 2, cy + y - r / 2, r, r);
      }
      drawn.push({ x: cx + x, y: cy + y, r: Math.max(4, size / 2 + 2), c });
    }
    ctx.globalAlpha = 1;
    // the axes, from the corner nearest state 0, symbol 0, target 0
    const corner = [-(T - 1) / 2 - 0.8, (S - 1) / 2 + 0.8, -(A - 1) / 2 - 0.8];
    const axes = [
      [[corner[0] + T + 0.6, corner[1], corner[2]], "target →"],
      [[corner[0], corner[1] - S - 0.6, corner[2]], "source ↓"],
      [[corner[0], corner[1], corner[2] + A + 0.6], "symbol →"],
    ];
    const o = project(corner, yaw, pitch, distance, focal);
    ctx.strokeStyle = text;
    ctx.fillStyle = text;
    ctx.lineWidth = 1;
    ctx.font = "12px system-ui, sans-serif";
    for (const [end, label] of axes) {
      const e = project(end, yaw, pitch, distance, focal);
      ctx.beginPath();
      ctx.moveTo(cx + o[0], cy + o[1]);
      ctx.lineTo(cx + e[0], cy + e[1]);
      ctx.stroke();
      ctx.fillText(label, cx + e[0] + 4, cy + e[1] + 4);
    }
    view.current.drawn = drawn;
  }, [cube, mode, minimum, shells, slice, touchedOnly, showCenter, showVector, showFocus]);

  useEffect(() => {
    let last = performance.now();
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const visible = canvasRef.current && canvasRef.current.offsetParent !== null;
      if (visible) {
        if (spin && !view.current.drag) view.current.yaw += dt * 0.35;
        draw();
      }
      frame.current = requestAnimationFrame(loop);
    };
    frame.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame.current);
  }, [draw, spin]);

  function cellUnder(evt) {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = evt.clientX - rect.left, y = evt.clientY - rect.top;
    let best = null;
    for (const d of view.current.drawn) {
      const dist = Math.hypot(d.x - x, d.y - y);
      if (dist <= d.r && (!best || d.c.p[2] > best.c.p[2])) best = d;
    }
    return best ? best.c : null;
  }

  const onPointerDown = (evt) => {
    view.current.drag = { x: evt.clientX, y: evt.clientY, yaw: view.current.yaw, pitch: view.current.pitch };
    view.current.moved = false;
    evt.currentTarget.setPointerCapture(evt.pointerId);
  };
  const onPointerMove = (evt) => {
    const d = view.current.drag;
    if (d) {
      const dx = evt.clientX - d.x, dy = evt.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) view.current.moved = true;
      view.current.yaw = d.yaw + dx * 0.01;
      view.current.pitch = Math.max(-1.5, Math.min(1.5, d.pitch + dy * 0.01));
    } else {
      setHover(cellUnder(evt));
    }
  };
  const onPointerUp = async (evt) => {
    const wasDrag = view.current.moved;
    view.current.drag = null;
    if (wasDrag || !cube) return;
    const c = cellUnder(evt);
    if (!c) return;
    try {
      const symbol = cube.symbols[c.a];
      setEdge({ symbol, record: await api.edge(c.s, symbol, c.t) });
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };
  const onWheel = (evt) => {
    evt.preventDefault();
    view.current.zoom = Math.max(0.3, Math.min(4, view.current.zoom * Math.exp(-evt.deltaY * 0.001)));
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const shape = cube ? cube.shape.join(" × ") : "";
  const hoverValue = hover && cube ? (cube[mode] || cube.probability)[hover.i] : null;
  return (
    <div className="panel">
      <div className="card wide">
        <h2>The matrix in 3D</h2>
        <p className="note">
          Every edge of the {shape || "S × A × S"} matrix is a cell: source top to bottom, symbol into the page, target
          across. Drag to turn it, scroll to zoom, hover to read a cell, click to open its record. The central node (
          {cube ? cube.center.join(", ") : "6, 6, 6"}) and the central vertical vector are outlined, the focus node in
          red.
        </p>
        <div className="row">
          <SelectField label="Colour by" value={mode} onChange={setMode} options={MODES} />
          <NumberField label="Hide below" value={minimum} onChange={setMinimum} min="0" max="1" step="0.05"
            hint="0..1 of the measure; marked cells always show" />
          <NumberField label="Shells" value={shells} onChange={setShells} min="0" step="1"
            hint="show only shells up to this from the centre (blank: all)" />
          <SelectField
            label="Symbol slice"
            value={slice}
            onChange={setSlice}
            options={[["", "every symbol"], ...(cube ? cube.symbols.map((s) => [s, `only ${s}`]) : [])]}
          />
        </div>
        <div className="row">
          <CheckField label="Touched edges only" checked={touchedOnly} onChange={setTouchedOnly} />
          <CheckField label="Central node" checked={showCenter} onChange={setShowCenter} />
          <CheckField label="Central vertical vector" checked={showVector} onChange={setShowVector} />
          <CheckField label="Focus node" checked={showFocus} onChange={setShowFocus} />
          <CheckField label="Turn by itself" checked={spin} onChange={setSpin} />
        </div>
        <canvas
          ref={canvasRef}
          className="cube-canvas"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => setHover(null)}
        />
        <p className="mono cube-readout">
          {hover && cube
            ? `edge (${hover.s}, ${cube.symbols[hover.a]}, ${hover.t}) · ${mode} ${fmtNum(hoverValue, 3)} · p ${fmtNum(
                cube.probability[hover.i],
                3,
              )} · seen ${cube.seen[hover.i]} · width ${fmtNum(cube.width[hover.i], 2)} · shell ${cube.shell[hover.i]}${
                hover.isCenter ? " · the central node" : hover.isVector ? " · on the central vertical vector" : ""
              }${hover.isFocus ? " · the focus node" : ""}`
            : "hover over a cell"}
        </p>
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>
      {edge ? <EdgeCard edge={edge.record} symbol={edge.symbol} onClose={() => setEdge(null)} /> : null}
    </div>
  );
}
