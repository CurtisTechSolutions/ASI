import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { MAX_TEMPERATURE, autoBody, lmLabel, mean } from "../lm.js";
import { asArray, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { AutoBadge, AutoTempFields } from "./AutoTemperature.jsx";
import HearButton from "./HearButton.jsx";
import { CheckField, NumberField, TextField } from "./Fields.jsx";
import LineChart from "./LineChart.jsx";
import { GuardBadge } from "./Verdict.jsx";

/** A mark out of 10 as a bar, green from the pass mark up, red below it. */
function MarkBar({ mark, pass }) {
  const ok = mark >= pass;
  return (
    <span className={`rating-bar ${ok ? "pass" : "fail"}`} title={`${mark} / 10`}>
      <i style={{ width: `${Math.max(0, Math.min(10, mark)) * 10}%` }} />
    </span>
  );
}

/**
 * The model in conversation with a local language model (Ollama) that marks it (POST /api/chat/llm). Ollama
 * says a line on the topic, the chosen model continues it as its reply, and Ollama marks the reply out of 10;
 * they take turns. With "learn", every reply is credited by how far its mark is from the pass mark
 * ((mark - pass) / 5): a good reply rewarded, a poor one punished. With "Guard", the guard judges every reply (a
 * rejected one vetoed for another), and with "teach the guard" a reply marked under the pass mark is blamed in
 * it ("marked 3/10") and one at or above it cleared.
 */
export default function ChatPanel({ lm, onMoved }) {
  const [topic, setTopic] = useStoredState("chat.topic", "the brain");
  const [turns, setTurns] = useStoredState("chat.turns", "4");
  const [maxTokens, setMaxTokens] = useStoredState("chat.max", "60");
  const [temperature, setTemperature] = useStoredState("chat.temperature", "0.7");
  const [auto, setAuto] = useStoredState("chat.auto", true);
  const [autoTop, setAutoTop] = useStoredState("chat.autoTop", String(MAX_TEMPERATURE));
  const [pass, setPass] = useStoredState("chat.pass", "5");
  const [learn, setLearn] = useStoredState("chat.learn", true);
  const [guard, setGuard] = useStoredState("chat.guard", false);
  const [teachGuard, setTeachGuard] = useStoredState("chat.teachGuard", true);
  const [ollamaModel, setOllamaModel] = useStoredState("chat.ollamaModel", "");
  const [url, setUrl] = useStoredState("chat.url", "");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]); // every mark of every chat this visit, for the chart
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function run(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const passMark = parseNumber(pass, 5);
      const r = await api.chatLlm({
        model: lm,
        topic,
        turns: Math.max(1, parseInteger(turns, 4)),
        max_tokens: Math.max(1, parseInteger(maxTokens, 60)),
        temperature: parseNumber(temperature, 0.7),
        pass_mark: passMark,
        learn,
        ...(guard ? { guard: true, teach_guard: teachGuard } : {}),
        ...autoBody(auto, parseNumber(autoTop, MAX_TEMPERATURE)),
        ...(ollamaModel.trim() ? { ollama_model: ollamaModel.trim() } : {}),
        ...(url.trim() ? { url: url.trim() } : {}),
      });
      const marks = asArray(r && r.marks).map(Number).filter(Number.isFinite);
      setResult({ ...r, model: lm, pass: passMark, marks });
      setHistory((h) => [...h, ...marks]);
      setError(null);
      if ((learn || guard) && onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const transcript = asArray(result && result.transcript);
  const overall = mean(history);
  return (
    <div className="panel">
      <form className="card" onSubmit={run}>
        <h2>Chat with Ollama</h2>
        <p className="muted">
          Ollama says a line on the topic, the {lmLabel(lm)} model replies by continuing it, and Ollama marks the
          reply out of 10. With learn on, each reply is credited (mark - pass mark) / 5. Ollama has to be running
          (check on the Ollama tab).
        </p>
        <TextField label="Topic" value={topic} onChange={setTopic} />
        <div className="row">
          <NumberField label="Turns" value={turns} onChange={setTurns} min="1" max="20" step="1" />
          <NumberField label="Max tokens" value={maxTokens} onChange={setMaxTokens} min="1" step="1" hint="a reply" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" />
          <AutoTempFields on={auto} onChange={setAuto} top={autoTop} onTop={setAutoTop} />
          <NumberField label="Pass mark" value={pass} onChange={setPass} min="0" max="10" step="1" hint="out of 10" />
        </div>
        <div className="row">
          <TextField label="Ollama model" value={ollamaModel} onChange={setOllamaModel} hint="blank: the server's default" />
          <TextField label="Ollama URL" value={url} onChange={setUrl} hint="blank: the server's default" />
        </div>
        <div className="row">
          <CheckField label="Learn" checked={learn} onChange={setLearn} hint="credit every reply by its mark" />
          <CheckField label="Guard" checked={guard} onChange={setGuard} hint="the guard judges every reply, and vetoes a rejected one" />
          <CheckField
            label="Teach the guard"
            checked={teachGuard}
            disabled={!guard}
            onChange={setTeachGuard}
            hint="a reply under the pass mark is blamed, one at or above it cleared"
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            {busy ? "Chatting…" : "Chat"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Transcript</h2>
        {!result ? (
          <p className="muted">Press Chat.</p>
        ) : (
          <>
            <p className="note">
              {result.marks.length} mark{result.marks.length === 1 ? "" : "s"}, mean <b>{fmtNum(result.mean ?? mean(result.marks), 2)}</b> / 10
              {overall !== null && history.length > result.marks.length ? ` · ${fmtNum(overall, 2)} over every chat this visit` : ""}
            </p>
            <ol className="dialogue">
              {transcript.map((t, i) => {
                const llm = t.speaker === "llm";
                return (
                  <li key={i} className={`turn ${llm ? "b" : "a"}`}>
                    <div className="speaker">
                      <span className={`badge ${llm ? "ollama" : "model"}`}>{llm ? "ollama" : lmLabel(result.model)}</span>
                      {t.mark !== undefined && t.mark !== null ? (
                        <>
                          <b>{t.mark} / 10</b>
                          <MarkBar mark={Number(t.mark)} pass={result.pass} />
                        </>
                      ) : null}
                      {t.credited ? <span className="muted">credited {fmtNum(t.credited, 2)}</span> : null}
                      {!llm ? <GuardBadge guard={t.guard} text={t.text} vetoed={t.vetoed} /> : null}
                      {!llm ? <AutoBadge auto={t.auto} /> : null}
                      {t.blamed ? (
                        <span className="badge tight fail" title={`blamed in the guard: marked ${t.mark}/10, under the pass mark`}>
                          blamed
                        </span>
                      ) : null}
                    </div>
                    <p className="bubble">{t.text || <span className="muted">(nothing)</span>}</p>
                    {!llm ? (
                      <div className="meta">
                        <HearButton text={t.text} label={`reply ${i + 1}`} />
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ol>
          </>
        )}
        {history.length > 1 ? (
          <LineChart
            series={[{ name: "mark", color: "var(--accent)", values: history.map((y, x) => ({ x: x + 1, y })) }]}
            xLabel="reply"
            yLabel="mark / 10"
            height={180}
          />
        ) : null}
      </div>
    </div>
  );
}
