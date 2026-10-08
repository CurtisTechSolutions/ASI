import { useEffect, useState } from 'react';
import { api, FoldResult } from '../api';
import Card, { ErrorLine } from '../Card';
import BarList from '../charts/BarList';
import { useAction } from '../hooks';
import { fmt, unitLabel } from '../util';

export default function Fold({ version }: { version: number }) {
  const [prefix, setPrefix] = useState('the cat sat on the ');
  const [traversal, setTraversal] = useState('reward');
  const [backoff, setBackoff] = useState('');
  const { busy, error, result, run } = useAction<FoldResult>();
  const fold = () => run(() => api<FoldResult>('/api/fold', { prefix, traversal, backoff, top: 12 }));
  // refold when the model changed (a verdict, a read, a new tokenizer)
  useEffect(() => {
    void run(() => api<FoldResult>('/api/fold', { prefix, traversal, backoff, top: 12 }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);
  return (
    <Card title="Fold" blurb="What the model expects after a context: its code, what each level of the code stands for, and the next-byte distribution.">
      <div className="row">
        <label className="grow">Context<input type="text" value={prefix} onChange={(e) => setPrefix(e.target.value)} /></label>
        <label>Traversal<select value={traversal} onChange={(e) => setTraversal(e.target.value)}><option value="reward">reward</option><option value="punishment">punishment</option></select></label>
        <label>Backoff<select value={backoff} onChange={(e) => setBackoff(e.target.value)}><option value="">model's</option><option value="all">all</option><option value="deepest">deepest</option><option value="none">none</option></select></label>
        <button className="primary" disabled={busy} onClick={fold}>Fold</button>
      </div>
      <ErrorLine error={error} />
      {result && (
        <>
          <div className="out">code [{result.code.join(' ')}] · entropy {fmt(result.entropy_bits, 2)} bits</div>
          <table>
            <thead><tr><th>level</th><th>node</th><th className="num">seen</th><th className="num">own</th><th>decodes to</th></tr></thead>
            <tbody>
              {result.levels.map((l) => (
                <tr key={l.level}>
                  <td>{l.level}</td><td>{l.node}</td><td className="num">{l.seen}</td><td className="num">{fmt(l.own)}</td>
                  <td>{l.level ? JSON.stringify(l.decode) : '(root: every context)'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <BarList items={result.top.map((d) => ({ label: unitLabel(d.unit), value: d.p, tip: `${unitLabel(d.unit)}: ${(100 * d.p).toFixed(2)}%` }))} max={1} valueText={(v) => `${(100 * v).toFixed(2)}%`} />
        </>
      )}
    </Card>
  );
}
