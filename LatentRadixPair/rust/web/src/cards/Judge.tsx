import { useState } from 'react';
import { api, JudgeResult } from '../api';
import Card, { ErrorLine } from '../Card';
import BarList from '../charts/BarList';
import { useAction } from '../hooks';
import { fmt, unitLabel } from '../util';

const PRESETS: [string, number][] = [['game: 2', 2], ['phones: 69', 69], ['phonemes: 39', 39], ['bytes: 257', 257]];

export default function Judge({ defaultOutcomes, onChange }: { defaultOutcomes: number; onChange: () => void }) {
  const [prefix, setPrefix] = useState('the cat sat on the ');
  const [kind, setKind] = useState('reward');
  const [text, setText] = useState('mat');
  const [bad, setBad] = useState('log');
  const [strength, setStrength] = useState(1);
  const [outcomes, setOutcomes] = useState(0);
  const [read, setRead] = useState(true);
  const { busy, error, result, run } = useAction<JudgeResult>();
  const judge = () =>
    run(async () => {
      const r = await api<JudgeResult>('/api/judge', { kind, text, bad, prefix, strength, outcomes, read });
      onChange();
      return r;
    });
  const rec = result?.record;
  const rows = (result?.changes ?? []).flatMap((c) => [
    { label: `${unitLabel(c.unit)} before`, value: c.before, tip: `P(${unitLabel(c.unit)}) before: ${(100 * c.before).toFixed(3)}%` },
    { label: `${unitLabel(c.unit)} after`, value: c.after, tip: `P(${unitLabel(c.unit)}) after: ${(100 * c.after).toFixed(3)}%` },
  ]);
  return (
    <Card title="Judge" blurb="A verdict is worth strength ÷ outcomes per rung: the more finite the space it comes from, the more one verdict says. The prefix is context, not outcome.">
      <div className="row">
        <label className="grow">Prefix (context)<input type="text" value={prefix} onChange={(e) => setPrefix(e.target.value)} /></label>
      </div>
      <div className="row">
        <label>Kind<select value={kind} onChange={(e) => setKind(e.target.value)}><option value="reward">reward</option><option value="punish">punish</option><option value="two_nrl">two-sided</option></select></label>
        <label className="grow">{kind === 'two_nrl' ? 'Right outcome' : 'Outcome'}<input type="text" value={text} onChange={(e) => setText(e.target.value)} /></label>
        {kind === 'two_nrl' && <label className="grow">Wrong outcome<input type="text" value={bad} onChange={(e) => setBad(e.target.value)} /></label>}
      </div>
      <div className="row">
        <label>Strength<input type="number" step={0.5} min={0} value={strength} onChange={(e) => setStrength(+e.target.value)} /></label>
        <label>Outcomes<input type="number" min={0} value={outcomes} placeholder={String(defaultOutcomes)} onChange={(e) => setOutcomes(+e.target.value)} /><span className="muted">0 = model's ({defaultOutcomes})</span></label>
        <span className="presets">
          {PRESETS.map(([name, n]) => <button key={name} type="button" onClick={() => setOutcomes(n)}>{name}</button>)}
        </span>
        <label className="check"><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} /> also read (rewards)</label>
        <button className="primary" disabled={busy} onClick={judge}>Judge</button>
      </div>
      <ErrorLine error={error} />
      {rec && (
        <div className="out">
          {rec.call}: {rec.cells ?? (rec.rewarded ?? 0) + (rec.punished ?? 0)} cells, strength {rec.strength}, outcomes {rec.outcomes}, {fmt(rec.amount ?? 0, 4)} per rung
          {rec.prefix ? `, prefix ${JSON.stringify(rec.prefix)}` : ''}
        </div>
      )}
      {rows.length > 0 && <BarList items={rows} max={Math.max(...rows.map((d) => d.value), 1e-9)} valueText={(v) => `${(100 * v).toFixed(3)}%`} />}
    </Card>
  );
}
