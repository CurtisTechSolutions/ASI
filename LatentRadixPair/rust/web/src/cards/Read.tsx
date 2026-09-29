import { useRef, useState } from 'react';
import { api } from '../api';
import Card, { ErrorLine } from '../Card';
import { useAction } from '../hooks';
import { readFiles, splitTexts } from '../util';

export default function Read({ onChange }: { onChange: () => void }) {
  const [raw, setRaw] = useState('the cat sat on the mat\n\nthe cat sat on the log');
  const files = useRef<HTMLInputElement>(null);
  const { busy, error, result, run } = useAction<{ texts: number; units: number }>();
  const read = () =>
    run(async () => {
      const texts = [...splitTexts(raw), ...(await readFiles(files.current?.files ?? null))];
      const r = await api<{ texts: number; units: number }>('/api/read', { texts });
      onChange();
      return r;
    });
  return (
    <Card title="Read" blurb="Read texts into the count tree. Blank lines separate texts; one file is one text.">
      <div className="row">
        <label className="grow">Texts<textarea rows={5} value={raw} onChange={(e) => setRaw(e.target.value)} /></label>
      </div>
      <div className="row">
        <input type="file" multiple ref={files} />
        <button className="primary" disabled={busy} onClick={read}>Read</button>
      </div>
      <ErrorLine error={error} />
      {result && <div className="out">read {result.texts} texts, {result.units} units</div>}
    </Card>
  );
}
