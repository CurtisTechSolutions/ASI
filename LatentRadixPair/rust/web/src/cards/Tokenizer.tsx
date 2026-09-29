import { useEffect, useRef, useState } from 'react';
import { api, Classes, Progress } from '../api';
import Card, { ErrorLine } from '../Card';
import LineChart from '../charts/LineChart';
import { useAction } from '../hooks';
import { errorText, readFiles, splitTexts } from '../util';

const fmtList = (v: number[]) => v.map((x) => x.toFixed(2)).join('/');

export default function Tokenizer({ readTexts, onChange }: { readTexts: string; onChange: () => void }) {
  const [raw, setRaw] = useState('');
  const files = useRef<HTMLInputElement>(null);
  const [steps, setSteps] = useState(400);
  const [batch, setBatch] = useState(256);
  const [lr, setLr] = useState(0.003);
  const [window, setWindow] = useState(16);
  const [levels, setLevels] = useState('4,4:4,4:4,4');
  const [recency, setRecency] = useState(0.6);
  const [predict, setPredict] = useState(4);
  const [read, setRead] = useState(true);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const classes = useAction<Classes>();

  // poll while a tokenizer trains; on the first render, pick up a run already in progress
  useEffect(() => {
    let timer: number | undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const p = await api<Progress>('/api/tokenizer/progress');
        if (stopped) return;
        setProgress(p);
        if (p.running) timer = window_.setTimeout(poll, 1000);
        else if (p.done) onChange();
      } catch (err) {
        setError(errorText(err));
      }
    };
    const window_ = globalThis.window;
    void poll();
    return () => {
      stopped = true;
      if (timer) window_.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [starting]);

  const train = async () => {
    const texts = [...splitTexts(raw), ...(await readFiles(files.current?.files ?? null))];
    if (!texts.length) {
      setError('paste some text or choose files first');
      return;
    }
    if (!confirm('Train a new tokenizer and replace the model on the server?')) return;
    setError(null);
    try {
      await api('/api/tokenizer/train', { texts, steps, batch, lr, window, levels, recency, predict, read });
      setStarting((s) => !s); // restart the poll
    } catch (err) {
      setError(errorText(err));
    }
  };
  const showClasses = () =>
    classes.run(async () => {
      const texts = [...splitTexts(raw), ...(await readFiles(files.current?.files ?? null)), ...splitTexts(readTexts)];
      return api<Classes>('/api/classes', { texts });
    });

  const running = progress?.running ?? false;
  const status = progress?.error ? `error: ${progress.error}` : running ? 'training…' : progress?.done ? 'done: the server now holds the new model' : 'idle';
  const stats = progress?.stats ?? [];
  return (
    <Card title="Tokenizer" blurb="Train a new context tokenizer on a corpus. This replaces the model on the server with a fresh one (the trees are primed again and, if asked, read the corpus).">
      <div className="row">
        <label className="grow">Corpus<textarea rows={4} placeholder="Paste text, or choose files below. Blank lines separate texts." value={raw} onChange={(e) => setRaw(e.target.value)} /></label>
      </div>
      <div className="row">
        <input type="file" multiple ref={files} />
        <label>Steps<input type="number" min={1} value={steps} onChange={(e) => setSteps(+e.target.value)} /></label>
        <label>Batch<input type="number" min={1} value={batch} onChange={(e) => setBatch(+e.target.value)} /></label>
        <label>Rate<input type="number" step={0.001} min={0.0001} value={lr} onChange={(e) => setLr(+e.target.value)} /></label>
        <label>Window<input type="number" min={1} value={window} onChange={(e) => setWindow(+e.target.value)} /></label>
        <label>Levels<input type="text" size={12} value={levels} onChange={(e) => setLevels(e.target.value)} /></label>
        <label>Recency<input type="number" step={0.1} min={0.05} max={1} value={recency} onChange={(e) => setRecency(+e.target.value)} /></label>
        <label>Predict<input type="number" step={0.5} min={0} value={predict} onChange={(e) => setPredict(+e.target.value)} /></label>
        <label className="check"><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} /> then read the corpus</label>
        <button className="primary" disabled={running} onClick={train}>Train new tokenizer</button>
        <button disabled={classes.busy} onClick={showClasses}>Classes</button>
      </div>
      <ErrorLine error={error} />
      <div className="out">
        {status}
        {stats.map((s) => (
          <div key={s.step}>
            step {String(s.step).padStart(6)}  recon {s.loss_bits.toFixed(3)} bits  next {fmtList(s.next_bits)}  exact {fmtList(s.accuracy)}  tail {fmtList(s.tail)}  {s.seconds.toFixed(0)}s
          </div>
        ))}
      </div>
      {stats.length >= 2 && (
        <LineChart
          series={[
            { name: 'reconstruction', points: stats.map((s) => ({ x: s.step, y: s.loss_bits })) },
            { name: 'next byte', points: stats.map((s) => ({ x: s.step, y: s.next_bits[s.next_bits.length - 1] })) },
          ]}
        />
      )}
      <ErrorLine error={classes.error} />
      {classes.result && (
        <div className="out">
          first symbol over {classes.result.total} positions (radix {classes.result.radix})
          <table>
            <thead><tr><th>class</th><th className="num">share</th><th>prototype</th><th>example</th></tr></thead>
            <tbody>
              {classes.result.classes.map((c) => (
                <tr key={c.symbol}><td>{c.symbol}</td><td className="num">{(100 * c.share).toFixed(2)}%</td><td>{JSON.stringify(c.prototype)}</td><td>{JSON.stringify(c.example)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
