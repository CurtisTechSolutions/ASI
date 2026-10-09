import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtNum, parseInteger, parseNumber, parseStateList } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

/** A fresh machine, a saved one, a loaded one, and a lesson for one edge. */
export default function MachinePanel({ stats, onMoved }) {
  const [states, setStates] = useStoredState("machine.states", "13");
  const [alphabet, setAlphabet] = useStoredState("machine.alphabet", "abcdefghijklm");
  const [accepting, setAccepting] = useStoredState("machine.accepting", "0");
  const [life, setLife] = useStoredState("machine.life", "1000");
  const [baseline, setBaseline] = useStoredState("machine.baseline", "1");
  const [seed, setSeed] = useStoredState("machine.seed", "1");
  const [path, setPath] = useStoredState("machine.path", "machine.json.gz");
  const [every, setEvery] = useStoredState("compression.every", "0");
  const [everyPrecision, setEveryPrecision] = useStoredState("compression.precision", "exact");
  const [rebuild, setRebuild] = useStoredState("compression.rebuild", false);
  const [source, setSource] = useStoredState("teach.source", "0");
  const [symbol, setSymbol] = useStoredState("teach.symbol", "a");
  const [target, setTarget] = useStoredState("teach.target", "1");
  const [amount, setAmount] = useStoredState("teach.amount", "1");
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
        <h2>A fresh machine</h2>
        <p className="note">
          Replaces the served machine with a new one: a dense matrix of states × symbols × states, every edge at the
          prototype, nothing traversed. 13 states over the thirteen symbols a to m is a 13 × 13 × 13 matrix, the
          default. The current machine is {stats && stats.shape ? stats.shape.join(" × ") : "–"}.
        </p>
        <div className="row">
          <NumberField label="States" value={states} onChange={setStates} min="1" step="1" />
          <TextField label="Alphabet" value={alphabet} onChange={setAlphabet} hint="one character per symbol" />
          <TextField label="Accepting states" value={accepting} onChange={setAccepting} hint="comma-separated" />
          <NumberField label="Life" value={life} onChange={setLife} min="1" step="100" hint="ticks to fade by a half" />
          <NumberField label="Baseline stimulation" value={baseline} onChange={setBaseline} min="0" step="0.5" />
          <NumberField label="Seed" value={seed} onChange={setSeed} step="1" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () =>
                  api.newMachine({
                    states: parseInteger(states, 13),
                    alphabet,
                    accepting: parseStateList(accepting),
                    life: parseNumber(life, 1000),
                    baseline: parseNumber(baseline, 1),
                    seed: parseInteger(seed, 1),
                  }),
                (r) => `a fresh machine of ${r.shape.join(" × ")}`,
              )
            }
          >
            New machine
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Compress every N transitions</h2>
        <p className="note">
          Fold the matrix into its central node automatically, every N transitions the machine makes (runs, training
          and lessons alike); 0 never. With rebuild, the matrix is rebuilt from the code each time, so a lossy
          precision's rounding is applied to the machine and not only recorded. Now:{" "}
          {stats && stats.compress_every
            ? `every ${stats.compress_every}, ${stats.compress_precision}${stats.compress_rebuild ? ", rebuilt" : ""}`
            : "never"}
          {stats && stats.compressions ? ` · ${stats.compressions} so far, the last at clock ${stats.last_compressed}` : ""}.
        </p>
        <div className="row">
          <NumberField label="Every N transitions" value={every} onChange={setEvery} min="0" step="100" hint="0: never" />
          <SelectField
            label="Precision"
            value={everyPrecision}
            onChange={setEveryPrecision}
            options={[
              ["exact", "exact - no loss"],
              ["float32", "float32"],
              ["float16", "float16"],
            ]}
          />
        </div>
        <CheckField label="Rebuild the matrix from the code each time" checked={rebuild} onChange={setRebuild} />
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.compression({ every: parseInteger(every, 0), precision: everyPrecision, rebuild }),
                (r) =>
                  r.compress_every
                    ? `compressing every ${r.compress_every} transitions, ${r.compress_precision}`
                    : "automatic compression off",
              )
            }
          >
            Apply to this machine
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Save and load</h2>
        <p className="note">
          A path on the server; .json or .json.gz. Only the edges anything was written to are in the file, and the
          Python package reads it too.
        </p>
        <div className="row">
          <TextField label="File" value={path} onChange={setPath} />
        </div>
        <div className="actions">
          <button type="button" onClick={() => act(() => api.save(path), (r) => `saved ${r.saved}`)}>
            Save
          </button>
          <button type="button" onClick={() => act(() => api.load(path), (r) => `loaded a machine of ${r.shape.join(" × ")}, clock ${r.clock}`)}>
            Load
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Teach one edge</h2>
        <p className="note">
          A lesson rather than an experience: the edge is traversed once, deliberately, and credited - positive
          rewards, negative punishes - without the machine moving.
        </p>
        <div className="row">
          <NumberField label="Source" value={source} onChange={setSource} min="0" step="1" />
          <TextField label="Symbol" value={symbol} onChange={setSymbol} />
          <NumberField label="Target" value={target} onChange={setTarget} min="0" step="1" />
          <NumberField label="Amount" value={amount} onChange={setAmount} step="0.5" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () =>
                  api.teach({
                    source: parseInteger(source, 0),
                    symbol,
                    target: parseInteger(target, 0),
                    amount: parseNumber(amount, 1),
                  }),
                (r) => `edge (${r.edge.source}, ${symbol}, ${r.edge.target}): seen ${r.edge.seen}, net ${fmtNum(r.edge.net, 3)}, width ${fmtNum(r.edge.width, 3)}`,
              )
            }
          >
            Teach
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {note ? <p className="note">{note}</p> : null}
      </div>
    </div>
  );
}
