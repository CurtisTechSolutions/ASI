import { useState } from 'react';
import { api, Score as ScoreResult } from '../api';
import Card, { ErrorLine } from '../Card';
import BarList from '../charts/BarList';
import { useAction } from '../hooks';
import { fmt, unitLabel } from '../util';

export default function Score() {
  const [text, setText] = useState('the cat sat on the mat');
  const [traversal, setTraversal] = useState('reward');
  const { busy, error, result, run } = useAction<ScoreResult>();
  const score = () => run(() => api<ScoreResult>('/api/score', { text, traversal, backoff: '' }));
  return (
    <Card title="Score" blurb="Price a text: bits per unit under the model's belief, and the reward tree's readings of its steps.">
      <div className="row">
        <label className="grow">Text<input type="text" value={text} onChange={(e) => setText(e.target.value)} /></label>
        <label>Traversal<select value={traversal} onChange={(e) => setTraversal(e.target.value)}><option value="reward">reward</option><option value="punishment">punishment</option></select></label>
        <button className="primary" disabled={busy} onClick={score}>Score</button>
      </div>
      <ErrorLine error={error} />
      {result && (
        <>
          <div className="out">{fmt(result.bits)} bits/unit over {result.units} units, mean reward {fmt(result.mean_reward)}, worst penalty {fmt(result.worst_penalty)}</div>
          <BarList
            items={result.per_unit.map((u) => ({ label: unitLabel(u.unit), value: u.bits, tip: `${unitLabel(u.unit)}: ${u.bits.toFixed(2)} bits, reward ${u.reward.toFixed(3)}, penalty ${u.penalty.toFixed(3)}` }))}
            valueText={(v) => `${v.toFixed(2)} bits`}
          />
        </>
      )}
    </Card>
  );
}
