import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField } from "./Fields.jsx";

/** Let time pass, and raise or set the stimulation. */
export default function TimePanel({ stats, onMoved }) {
  const [ticks, setTicks] = useStoredState("time.ticks", "1000");
  const [surge, setSurge] = useStoredState("time.surge", "2");
  const [level, setLevel] = useStoredState("time.level", "1");
  const [error, setError] = useState(null);
  const [note, setNote] = useState(null);

  async function act(fn, say) {
    try {
      const r = await fn();
      setNote(say(r));
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Time</h2>
        <p className="note">
          The clock is the number of traversals, plus time let pass here. Traces, ages and widths halve every{" "}
          {stats ? fmtNum(stats.life, 0) : "life"} ticks; the verdicts never fade.
        </p>
        <div className="row">
          <NumberField label="Ticks" value={ticks} onChange={setTicks} min="0" step="100" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() => act(() => api.tick(parseInteger(ticks, 0)), (r) => `clock ${r.clock}, stimulation ${fmtNum(r.stimulation, 3)}`)}
          >
            Let time pass
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Stimulation</h2>
        <p className="note">
          Every weight carries stimulation × log(width): at 0 width is not consulted, at 1 the widths are the odds,
          higher and the wide channel takes over. A surge relaxes toward the baseline ({stats ? fmtNum(stats.baseline, 2) : "–"}) by
          a half every life.
        </p>
        <div className="row">
          <NumberField label="Surge" value={surge} onChange={setSurge} step="0.5" hint="added to the level now; negative lowers it" />
          <NumberField label="Level" value={level} onChange={setLevel} min="0" step="0.5" hint="set outright" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() => act(() => api.stimulate({ amount: parseNumber(surge, 0) }), (r) => `stimulation ${fmtNum(r.stimulation, 3)}`)}
          >
            Stimulate
          </button>
          <button
            type="button"
            onClick={() => act(() => api.stimulate({ level: parseNumber(level, 1) }), (r) => `stimulation ${fmtNum(r.stimulation, 3)}`)}
          >
            Set level
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {note ? <p className="note">{note}</p> : null}
      </div>
    </div>
  );
}
