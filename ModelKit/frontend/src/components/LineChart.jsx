/**
 * Dependency-free SVG line chart.
 *
 * props.series: [{ name, color, values: [{x, y}], dashed? }]
 * Non-finite points are dropped; an empty chart renders a placeholder.
 */

const WIDTH = 640;
const MARGIN = { top: 14, right: 18, bottom: 36, left: 56 };
const POINT_LIMIT = 80;

function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const exponent = Math.floor(Math.log10(raw));
  const base = raw / 10 ** exponent;
  const nice = base < 1.5 ? 1 : base < 3 ? 2 : base < 7 ? 5 : 10;
  return nice * 10 ** exponent;
}

function ticks(min, max, count, integer) {
  const step = Math.max(integer ? 1 : 0, niceStep((max - min) / Math.max(1, count)));
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

function fmtTick(v) {
  if (Number.isInteger(v)) return String(v);
  const magnitude = Math.abs(v);
  return magnitude >= 100 ? v.toFixed(0) : magnitude >= 1 ? v.toFixed(2) : v.toPrecision(2);
}

function extent(points, pick) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    const v = pick(p);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

export default function LineChart({ series, height = 260, xLabel = "", yLabel = "", emptyText = "No data yet." }) {
  const clean = (Array.isArray(series) ? series : []).map((s) => ({
    ...s,
    values: (Array.isArray(s.values) ? s.values : []).filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)),
  }));
  const all = clean.flatMap((s) => s.values);
  if (all.length === 0) return <div className="chart-empty muted">{emptyText}</div>;

  let [xmin, xmax] = extent(all, (p) => p.x);
  let [ymin, ymax] = extent(all, (p) => p.y);
  if (xmin === xmax) {
    xmin -= 1;
    xmax += 1;
  }
  if (ymin === ymax) {
    ymin -= 1;
    ymax += 1;
  } else {
    const pad = (ymax - ymin) * 0.08;
    ymin -= pad;
    ymax += pad;
  }

  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;
  const sx = (x) => MARGIN.left + ((x - xmin) / (xmax - xmin)) * innerWidth;
  const sy = (y) => MARGIN.top + innerHeight - ((y - ymin) / (ymax - ymin)) * innerHeight;
  const integerX = all.every((p) => Number.isInteger(p.x));
  const xTicks = ticks(xmin, xmax, 6, integerX);
  const yTicks = ticks(ymin, ymax, 5, false);
  const bottom = MARGIN.top + innerHeight;
  const right = WIDTH - MARGIN.right;

  return (
    <div className="chart-box">
      <div className="legend">
        {clean.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color, opacity: s.dashed ? 0.6 : 1 }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${height}`}
        role="img"
        aria-label={`${clean.map((s) => s.name).join(", ")} over ${xLabel || "x"}`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid" x1={MARGIN.left} x2={right} y1={sy(t)} y2={sy(t)} />
            <text x={MARGIN.left - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle">
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line className="grid" x1={sx(t)} x2={sx(t)} y1={MARGIN.top} y2={bottom} />
            <text x={sx(t)} y={bottom + 14} textAnchor="middle">
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {ymin < 0 && ymax > 0 ? <line className="zero" x1={MARGIN.left} x2={right} y1={sy(0)} y2={sy(0)} /> : null}
        <line className="axis" x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={bottom} />
        <line className="axis" x1={MARGIN.left} x2={right} y1={bottom} y2={bottom} />
        {xLabel ? (
          <text x={MARGIN.left + innerWidth / 2} y={height - 4} textAnchor="middle">
            {xLabel}
          </text>
        ) : null}
        {yLabel ? (
          <text transform={`translate(12 ${MARGIN.top + innerHeight / 2}) rotate(-90)`} textAnchor="middle">
            {yLabel}
          </text>
        ) : null}
        {clean.map((s) => {
          const d = s.values
            .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`)
            .join(" ");
          return (
            <g key={s.name}>
              {s.values.length > 1 ? (
                <path
                  d={d}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeDasharray={s.dashed ? "5 4" : undefined}
                />
              ) : null}
              {s.values.length <= POINT_LIMIT
                ? s.values.map((p, i) => (
                    <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="3" fill={s.color}>
                      <title>{`${s.name}: ${p.y} at ${xLabel || "x"} ${p.x}`}</title>
                    </circle>
                  ))
                : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
