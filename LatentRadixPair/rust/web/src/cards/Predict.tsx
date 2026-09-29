import { useState } from 'react';
import { api, PathResult } from '../api';
import Card, { ErrorLine } from '../Card';
import BarList from '../charts/BarList';
import { useAction } from '../hooks';
import { fmt, unitLabel } from '../util';

export default function Predict({ onChange }: { onChange: () => void }) {
  const [prefix, setPrefix] = useState('the cat sat on the ');
  const [length, setLength] = useState(32);
  const [mode, setMode] = useState('greedy');
  const [traversal, setTraversal] = useState('reward');
  const [backoff, setBackoff] = useState('');
  const [temperature, setTemperature] = useState(1);
  const [toEnd, setToEnd] = useState(false);
  const { busy, error, result, run } = useAction<PathResult>();
  const predict = () =>
    run(async () => {
      const r = await api<PathResult>('/api/predict', { prefix, length, mode, traversal, backoff, temperature, to_end: toEnd });
      if (mode === 'sample') onChange();
      return r;
    });
  return (
    <Card title="Predict" blurb="Continue a prefix one byte at a time from the exact fold. Each bar is what a step cost in bits.">
      <div className="row">
        <label className="grow">Prefix<textarea rows={2} value={prefix} onChange={(e) => setPrefix(e.target.value)} /></label>
      </div>
      <div className="row">
        <label>Length<input type="number" min={1} max={4096} value={length} onChange={(e) => setLength(+e.target.value)} /></label>
        <label>Mode<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="greedy">greedy</option><option value="sample">sample</option></select></label>
        <label>Traversal<select value={traversal} onChange={(e) => setTraversal(e.target.value)}><option value="reward">reward</option><option value="punishment">punishment</option></select></label>
        <label>Backoff<select value={backoff} onChange={(e) => setBackoff(e.target.value)}><option value="">model's</option><option value="all">all</option><option value="deepest">deepest</option><option value="none">none</option></select></label>
        <label>Temperature<input type="number" step={0.1} min={0.01} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></label>
        <label className="check"><input type="checkbox" checked={toEnd} onChange={(e) => setToEnd(e.target.checked)} /> to end</label>
        <button className="primary" disabled={busy} onClick={predict}>Predict</button>
      </div>
      <ErrorLine error={error} />
      {result && (
        <>
          <div className="out">
            {prefix}<span className="gen">{result.text}</span>{result.reached_end ? ' ⟨end⟩' : ''}
            {'\n'}{result.units.length} units, {fmt(result.cost)} nats, {fmt(result.bits_per_unit, 2)} bits/unit, first-step peak {fmt(result.peak)}, mode {result.mode}
          </div>
          <BarList
            items={result.units.map((_, i) => ({ label: unitLabel(result.unit_names[i]), value: result.step_costs[i] / Math.LN2, tip: `${unitLabel(result.unit_names[i])}: ${(result.step_costs[i] / Math.LN2).toFixed(2)} bits` }))}
            valueText={(v) => `${v.toFixed(2)} bits`}
          />
        </>
      )}
    </Card>
  );
}
