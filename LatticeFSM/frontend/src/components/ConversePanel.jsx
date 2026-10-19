import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { useLmInfo } from "../hooks/useLmInfo.js";
import { MAX_TEMPERATURE, autoBody, joinContinuation, lmLabel } from "../lm.js";
import { asArray, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { AutoBadge, AutoTempFields } from "./AutoTemperature.jsx";
import HearButton from "./HearButton.jsx";
import { CheckField, NumberField, TextField } from "./Fields.jsx";
import { RateButtons } from "./RatingsCard.jsx";
import { DepthNote } from "./TokenText.jsx";
import { GuardBadge } from "./Verdict.jsx";

/**
 * The model talks to itself (POST /api/converse): from an opening line, two voices A and B take turns, each
 * turn the model's continuation of the turn before it. A turn can be rewarded or punished (👍 / 👎 credit it
 * together with the turn it answered, the context it continued) and heard. With "Guard" on, every turn is judged
 * by the guard (a rejected one vetoed for another), its verdict beside it.
 */
export default function ConversePanel({ lm, tick, onMoved }) {
  const [info] = useLmInfo(lm, tick);
  const [opening, setOpening] = useStoredState("converse.opening", "the brain is");
  const [turns, setTurns] = useStoredState("converse.turns", "6");
  const [maxTokens, setMaxTokens] = useStoredState("converse.max", "60");
  const [temperature, setTemperature] = useStoredState("converse.temperature", "0");
  const [guard, setGuard] = useStoredState("converse.guard", false);
  const [auto, setAuto] = useStoredState("converse.auto", true);
  const [autoTop, setAutoTop] = useStoredState("converse.autoTop", String(MAX_TEMPERATURE));
  const [result, setResult] = useState(null); // {model, opening, turns}
  const [rated, setRated] = useState({});
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [talking, setTalking] = useState(false);

  useEffect(() => {
    setResult(null);
    setRated({});
  }, [lm]);

  async function run(event) {
    event.preventDefault();
    setBusy(true);
    setTalking(true);
    setNote(null);
    try {
      const r = await api.converse({
        model: lm,
        opening,
        turns: Math.max(1, parseInteger(turns, 6)),
        max_tokens: Math.max(1, parseInteger(maxTokens, 60)),
        temperature: parseNumber(temperature, 0),
        seed: Math.floor(Math.random() * 1e9),
        ...(guard ? { guard: true } : {}),
        ...autoBody(auto, parseNumber(autoTop, MAX_TEMPERATURE)),
      });
      setResult({ model: lm, opening, turns: asArray(r && r.turns) });
      setRated({});
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setTalking(false);
    }
  }

  const unit = info ? info.unit : "word";
  // a turn in context: the line it answered (the opening for the first) and what it said
  const inContext = (i) => {
    const before = i === 0 ? result.opening : result.turns[i - 1].text;
    return joinContinuation(before, result.turns[i].text, unit);
  };

  async function credit(i, amount) {
    setBusy(true);
    try {
      const r = await api.lmCredit(result.model, inContext(i), amount);
      setRated((prev) => ({ ...prev, [i]: amount }));
      setNote(`turn ${i + 1}: ${amount > 0 ? "rewarded" : "punished"} ${r.credited}`);
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const maxDepth = info && info.nodes ? info.nodes - 1 : 8;
  return (
    <div className="panel">
      <form className="card" onSubmit={run}>
        <h2>Converse</h2>
        <p className="muted">
          The {lmLabel(lm)} model talks to itself: A goes on from the opening line, B goes on from what A said, and
          so on. At temperature 0 it is the same conversation every time.
        </p>
        <TextField label="Opening line" value={opening} onChange={setOpening} />
        <div className="row">
          <NumberField label="Turns" value={turns} onChange={setTurns} min="1" max="40" step="1" />
          <NumberField label="Max tokens" value={maxTokens} onChange={setMaxTokens} min="1" step="1" hint="a turn" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" />
          <AutoTempFields on={auto} onChange={setAuto} top={autoTop} onTop={setAutoTop} />
        </div>
        <CheckField label="Guard" checked={guard} onChange={setGuard} hint="the guard judges every turn, and vetoes a rejected one" />
        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            {talking ? "Talking…" : "Start"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>The conversation</h2>
        {!result ? (
          <p className="muted">Press Start.</p>
        ) : (
          <ol className="dialogue">
            <li className="turn a">
              <div className="speaker">opening</div>
              <p className="bubble">
                <span className="context">{result.opening}</span>
              </p>
            </li>
            {result.turns.map((t, i) => (
              <li key={i} className={`turn ${t.speaker === "B" ? "b" : "a"}${rated[i] ? ` rated ${rated[i] > 0 ? "up" : "down"}` : ""}`}>
                <div className="speaker">
                  <b>{t.speaker || (i % 2 ? "B" : "A")}</b>
                  {t.depths ? <DepthNote depths={t.depths} maxDepth={maxDepth} /> : null}
                  <GuardBadge guard={t.guard} text={t.text} vetoed={t.vetoed} />
                  <AutoBadge auto={t.auto} />
                </div>
                <p className="bubble">{t.text || <span className="muted">(nothing follows)</span>}</p>
                <div className="meta">
                  <RateButtons rated={rated[i] || 0} disabled={busy || !t.text} onRate={(a) => credit(i, a)} label={`turn ${i + 1}`} />
                  <HearButton text={t.text} unit={unit} label={`turn ${i + 1}`} />
                </div>
              </li>
            ))}
          </ol>
        )}
        {note ? <p className="note">{note}</p> : null}
      </div>
    </div>
  );
}

