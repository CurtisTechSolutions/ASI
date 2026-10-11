import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { lmLabel } from "../lm.js";
import { parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, TextArea } from "./Fields.jsx";

/**
 * Feedback in bulk (POST /api/{m}/feedback): good texts, one a line, are all rewarded and bad ones all
 * punished, by an amount - what the chosen model would have written in them is made likelier or less likely.
 */
export default function FeedbackPanel({ lm, onMoved }) {
  const [good, setGood] = useStoredState("feedback.good", "");
  const [bad, setBad] = useStoredState("feedback.bad", "");
  const [amount, setAmount] = useStoredState("feedback.amount", "1");
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const goods = splitLines(good);
  const bads = splitLines(bad);

  async function apply(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const r = await api.lmFeedback(lm, { good: goods, bad: bads, amount: parseNumber(amount, 1) });
      setNote(`${lmLabel(lm)}: rewarded ${r.rewarded}, punished ${r.punished}`);
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel single">
      <form className="card wide" onSubmit={apply}>
        <h2>Feedback</h2>
        <p className="muted">
          Every good text is rewarded and every bad one punished: one text a line. The chosen model ({lmLabel(lm)})
          learns which continuations to trust - for the chain, a rewarded node is trusted sooner (walks stop there), a
          punished one sends them deeper.
        </p>
        <div className="row">
          <TextArea label={`Good texts (${goods.length})`} value={good} onChange={setGood} rows={10} hint="rewarded" />
          <TextArea label={`Bad texts (${bads.length})`} value={bad} onChange={setBad} rows={10} hint="punished" />
        </div>
        <div className="actions">
          <NumberField label="Amount" value={amount} onChange={setAmount} min="0" step="0.1" hint="per text" />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || (goods.length === 0 && bads.length === 0)}>
            {busy ? "Applying…" : "Apply"}
          </button>
          <button type="button" disabled={busy || (!good && !bad)} onClick={() => {
              setGood("");
              setBad("");
            }}>
            Clear
          </button>
        </div>
        {note ? <p className="note">{note}</p> : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>
    </div>
  );
}
