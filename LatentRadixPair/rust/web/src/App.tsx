import { useCallback, useState } from 'react';
import { api } from './api';
import Fold from './cards/Fold';
import Judge from './cards/Judge';
import Predict from './cards/Predict';
import Read from './cards/Read';
import Score from './cards/Score';
import Tokenizer from './cards/Tokenizer';
import { useInfo } from './hooks';
import { errorText } from './util';

export default function App() {
  const { info, error, refresh } = useInfo();
  const [version, setVersion] = useState(0);
  const [note, setNote] = useState<string | null>(null);
  // anything that changes the model bumps the version so the fold refolds and the summary refreshes
  const changed = useCallback(() => {
    setVersion((v) => v + 1);
    void refresh();
  }, [refresh]);
  const save = async () => {
    try {
      const r = await api<{ saved: string }>('/api/save', {});
      setNote(`saved ${r.saved}`);
      setTimeout(() => setNote(null), 2500);
    } catch (err) {
      setNote(errorText(err));
    }
  };
  const toggleTheme = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    root.dataset.theme = dark ? 'light' : 'dark';
  };
  const t = info?.tokenizer;
  const summary = error
    ? `no model: ${error}`
    : info && t
      ? `${info.model_path} · code ${t.levels} (${t.code_bits.toFixed(0)} bits, window ${t.window}, ${t.steps} steps) · ` +
        `${info.nodes.toLocaleString()} nodes × 257 · read ${info.read.units.toLocaleString()} units in ${info.read.texts} texts · ` +
        `${info.judged.texts} verdicts · outcomes ${info.settings.outcomes}` +
        (info.training ? ` · training: ${info.training}` : '')
      : 'connecting…';
  return (
    <>
      <header className="top">
        <div>
          <h1>LatentRadixPair</h1>
          <p className="muted">{note ?? summary}</p>
        </div>
        <div className="top-actions">
          <button type="button" onClick={toggleTheme} title="Switch light and dark">Theme</button>
          <button type="button" onClick={save} title="Save the model file on the server">Save model</button>
        </div>
      </header>
      <main>
        <Predict onChange={changed} />
        <Fold version={version} />
        <Judge defaultOutcomes={info?.settings.outcomes ?? 69} onChange={changed} />
        <Read onChange={changed} />
        <Score />
        <Tokenizer readTexts="" onChange={changed} />
      </main>
      <footer className="muted">Rust port with a React frontend. Design in <code>../DESIGN.md</code>.</footer>
    </>
  );
}
