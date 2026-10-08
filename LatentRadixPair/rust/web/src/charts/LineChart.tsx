import { useState } from 'react';

export interface Series {
  name: string;
  points: { x: number; y: number }[];
}

// One or two series over steps: a legend, direct labels at the line ends, a crosshair tooltip.
export default function LineChart({ series, yLabel = 'bits' }: { series: Series[]; yLabel?: string }) {
  const [hover, setHover] = useState<{ x: number; px: number; py: number } | null>(null);
  const all = series.flatMap((s) => s.points);
  if (all.length < 2) return null;
  const width = 700, height = 220, left = 46, right = 110, topPad = 12, bottom = 28;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys) * 0.95, y1 = Math.max(...ys) * 1.05;
  const sx = (x: number) => left + ((x - x0) / Math.max(x1 - x0, 1)) * (width - left - right);
  const sy = (y: number) => topPad + (1 - (y - y0) / Math.max(y1 - y0, 1e-9)) * (height - topPad - bottom);
  const ticks = [0, 1, 2, 3, 4];
  const nearest = (x: number) => {
    let best = all[0];
    for (const p of all) if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    return best.x;
  };
  return (
    <div className="chart" onMouseLeave={() => setHover(null)}>
      <div className="legend">
        {series.map((s, k) => (
          <span key={s.name}>
            <span className="swatch" style={{ background: k ? 'var(--series-2)' : 'var(--accent)' }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="training curve"
        onMouseMove={(ev) => {
          const r = ev.currentTarget.getBoundingClientRect();
          const px = ((ev.clientX - r.left) / r.width) * width;
          const x = x0 + ((px - left) / (width - left - right)) * (x1 - x0);
          const cr = ev.currentTarget.parentElement!.getBoundingClientRect();
          setHover({ x: nearest(x), px: ev.clientX - cr.left + 12, py: ev.clientY - cr.top - 40 });
        }}
      >
        {ticks.map((i) => {
          const y = y0 + ((y1 - y0) * i) / 4;
          return (
            <g key={`y${i}`}>
              <line className="axis" x1={left} x2={width - right} y1={sy(y)} y2={sy(y)} />
              <text className="tick" x={left - 6} y={sy(y) + 4} textAnchor="end">{y.toFixed(2)}</text>
            </g>
          );
        })}
        {ticks.map((i) => {
          const x = x0 + ((x1 - x0) * i) / 4;
          return <text key={`x${i}`} className="tick" x={sx(x)} y={height - 8} textAnchor="middle">{Math.round(x)}</text>;
        })}
        <text className="tick" x={left - 6} y={topPad - 2} textAnchor="end">{yLabel}</text>
        {series.map((s, k) => {
          const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.name}>
              <path className={`line s${k + 1}`} d={d} />
              <text className="bar-label" x={sx(last.x) + 6} y={sy(last.y) + 4}>{s.name} {last.y.toFixed(2)}</text>
            </g>
          );
        })}
        {hover && <line className="crosshair" x1={sx(hover.x)} x2={sx(hover.x)} y1={topPad} y2={height - bottom} />}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.px, top: hover.py, display: 'block' }}>
          step {hover.x}
          {series.map((s) => {
            const p = s.points.find((q) => q.x === hover.x);
            return p ? <div key={s.name}>{s.name}: {p.y.toFixed(3)}</div> : null;
          })}
        </div>
      )}
    </div>
  );
}
