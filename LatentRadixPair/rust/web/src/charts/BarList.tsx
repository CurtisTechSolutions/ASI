import { useState } from 'react';

export interface Bar {
  label: string;
  value: number;
  tip?: string;
  negative?: boolean;
}

interface Props {
  items: Bar[];
  max?: number;
  valueText?: (v: number) => string;
}

// A ranked list of horizontal bars: one hue for a magnitude, a label and the value on every row (the list
// is its own table), and a hover tooltip.
export default function BarList({ items, max, valueText = (v) => v.toFixed(3) }: Props) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  if (!items.length) return null;
  const top = max ?? Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);
  const rowH = 22, labelW = 120, valueW = 70, width = 700, barW = width - labelW - valueW - 8;
  const height = items.length * rowH + 4;
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="bar list">
        {items.map((d, i) => {
          const y = i * rowH + 2;
          const w = Math.max(2, (Math.abs(d.value) / top) * barW);
          const cls = d.negative ? 'bar neg' : 'bar';
          const text = d.tip ?? `${d.label}: ${valueText(d.value)}`;
          return (
            <g
              key={i}
              onMouseMove={(ev) => {
                const r = ev.currentTarget.ownerSVGElement!.parentElement!.getBoundingClientRect();
                setTip({ x: ev.clientX - r.left + 12, y: ev.clientY - r.top - 28, text });
              }}
            >
              <rect className="hit" x={0} y={y} width={width} height={rowH} />
              <text className="bar-label" x={0} y={y + 15}>{d.label}</text>
              <rect className={cls} x={labelW} y={y + 4} width={w} height={rowH - 8} rx={4} ry={4} />
              <rect className={cls} x={labelW} y={y + 4} width={Math.min(4, w)} height={rowH - 8} />
              <text className="bar-value" x={labelW + w + 6} y={y + 15}>{valueText(d.value)}</text>
            </g>
          );
        })}
      </svg>
      {tip && <div className="tooltip" style={{ left: tip.x, top: tip.y, display: 'block' }}>{tip.text}</div>}
    </div>
  );
}
