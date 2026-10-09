import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { describeRun, stimulationFrom } from "../matrix.js";
import { fmtNum, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, TextField } from "./Fields.jsx";

/**
 * Read a string from the start state - traversing it, or asking quietly -
 * at a stimulation of your choosing, then reward or punish what was run.
 */
export default function RunPanel({ stats, onMoved }) {
  const [text, setText] = useStoredState("run.text", "abab");
  const [stimulation, setStimulation] = useStoredState("run.stimulation", "");
  const [temperature, setTemperature] = useStoredState("run.temperature", "1");
  const [amount, setAmount] = useStoredState("run.amount", "1");
  const [fromMiddle, setFromMiddle] = useStoredState("run.fromMiddle", false);
  const [result, setResult] = useState(null);
  const [credited, setCredited] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const level = stimulationFrom(stimulation);

  async function run(quiet) {
    setBusy(true);
    setCredited(null);
    try {
      const r = await api.run({
        text,
        stimulation: level,
        temperature: parseNumber(temperature, 1),
        quiet,
        from_middle: fromMiddle,
      });
      setResult({ ...r, quiet });
      setError(null);
      if (!quiet && onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function credit(sign) {
    setBusy(true);
    try {
      const r = await api.credit(sign * Math.abs(parseNumber(amount, 1)));
      setCredited(r);
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Run a string</h2>
        <p className="note">
          The machine starts in state {stats ? stats.start : 0} and reads the string one symbol at a time, drawing
          each next state from the row by the edges' weights. A run traverses what it draws; asked quietly, the same
          draws are made and nothing moves. Reward or punish the last run: the credit lands on its edges, the last in
          full and each earlier one discounted.
        </p>
        <div className="row">
          <TextField label="String" value={text} onChange={setText} hint={`symbols of ${stats && stats.alphabet ? stats.alphabet.join("") : "the alphabet"}`} />
          <NumberField
            label="Stimulation"
            value={stimulation}
            onChange={setStimulation}
            min="0"
            step="0.1"
            placeholder={stats ? fmtNum(stats.stimulation, 2) : "the machine's"}
            hint="blank: the machine's own level"
          />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" hint="0 is greedy" />
          <NumberField label="Credit" value={amount} onChange={setAmount} min="0" step="0.5" hint="the size of a reward or punishment" />
        </div>
        <CheckField
          label={`Start from the middle state (${stats && stats.center ? stats.center[0] : "the centre"}) and walk outward`}
          checked={fromMiddle}
          onChange={setFromMiddle}
          hint="the central node's state, instead of the start state"
        />
        <div className="actions">
          <button type="button" disabled={busy || !text} onClick={() => run(false)}>
            Run
          </button>
          <button type="button" disabled={busy || !text} onClick={() => run(true)}>
            Ask quietly
          </button>
          <button type="button" disabled={busy || !result || result.quiet} onClick={() => credit(+1)}>
            Reward last run
          </button>
          <button type="button" disabled={busy || !result || result.quiet} onClick={() => credit(-1)}>
            Punish last run
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {result ? (
          <div className="text-display mono">
            {result.quiet ? <span className="muted">asked quietly: </span> : null}
            {describeRun(result)}
            {"  "}
            <span className={`badge ${result.accepted ? "pass" : "fail"}`}>{result.accepted ? "accepted" : "rejected"}</span>
            {"  "}
            <span className="muted">
              ended in {result.final} · log p {fmtNum(result.log_probability, 3)}
            </span>
          </div>
        ) : null}
        {credited ? (
          <p className="note">
            credited {credited.credited} edges with {credited.amount > 0 ? "+" : ""}
            {fmtNum(credited.amount, 2)}
          </p>
        ) : null}
      </div>
    </div>
  );
}
