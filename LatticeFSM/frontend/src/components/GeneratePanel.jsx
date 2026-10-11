import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { useLmInfo } from "../hooks/useLmInfo.js";
import { joinContinuation, lmLabel } from "../lm.js";
import { asArray, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import HearButton from "./HearButton.jsx";
import { NumberField, TextField } from "./Fields.jsx";
import RatingsCard, { RateButtons, useRatings } from "./RatingsCard.jsx";
import TokenText, { DepthNote } from "./TokenText.jsx";

/**
 * Generate texts with the chosen language model: a number of samples continuing a prompt (or from nothing),
 * up to a number of tokens, at a temperature, from a seed. Every sample can be rewarded or punished at once
 * (👍 / 👎, ±1 to what it wrote in context), marked out of 10 for the marks card (☆), and heard (🔊). For the
 * chain, each token is tinted by how deep its walk went.
 */
export default function GeneratePanel({ lm, tick, onMoved }) {
  const [info] = useLmInfo(lm, tick);
  const [prompt, setPrompt] = useStoredState("generate.prompt", "");
  const [count, setCount] = useStoredState("generate.count", "3");
  const [maxTokens, setMaxTokens] = useStoredState("generate.max", "40");
  const [temperature, setTemperature] = useStoredState("generate.temperature", "0.8");
  const [seed, setSeed] = useStoredState("generate.seed", "");
  const [result, setResult] = useState(null);
  const [rated, setRated] = useState({}); // sample index -> +1 / -1, once credited
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [generating, setGenerating] = useState(false);
  const ratings = useRatings();

  useEffect(() => {
    setResult(null);
    setRated({});
    ratings.clear();
  }, [lm]); // eslint-disable-line react-hooks/exhaustive-deps

  async function generate(event) {
    event.preventDefault();
    setBusy(true);
    setGenerating(true);
    setNote(null);
    try {
      const r = await api.lmGenerate(lm, {
        text: prompt,
        count: Math.max(1, parseInteger(count, 3)),
        max_tokens: Math.max(1, parseInteger(maxTokens, 40)),
        temperature: parseNumber(temperature, 0.8),
        seed: seed === "" ? Math.floor(Math.random() * 1e9) : parseInteger(seed, 0),
      });
      setResult({ ...r, prompt });
      setRated({});
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setGenerating(false);
    }
  }

  const written = (s) => joinContinuation(result.prompt, s.completion, result.unit);

  async function credit(i, s, amount) {
    setBusy(true);
    try {
      const r = await api.lmCredit(lm, written(s), amount);
      setRated((prev) => ({ ...prev, [i]: amount }));
      setNote(`sample ${i + 1}: ${amount > 0 ? "rewarded" : "punished"} ${r.credited}`);
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const samples = asArray(result && result.samples);
  const maxDepth = info && info.nodes ? info.nodes - 1 : 8;
  return (
    <div className="panel">
      <form className="card" onSubmit={generate}>
        <h2>Generate</h2>
        <p className="muted">
          Samples from the {lmLabel(lm)} model, each going on from the prompt token by token. Temperature 0 takes the
          likeliest token every time (every sample the same); higher draws more freely. A seed makes a draw repeatable.
        </p>
        <TextField label="Prompt" value={prompt} onChange={setPrompt} hint="optional: every sample continues it" />
        <div className="row">
          <NumberField label="Samples" value={count} onChange={setCount} min="1" max="20" step="1" />
          <NumberField label="Max tokens" value={maxTokens} onChange={setMaxTokens} min="1" step="1" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" />
          <NumberField label="Seed" value={seed} onChange={setSeed} step="1" placeholder="random" hint="blank: a new one each time" />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            {generating ? "Generating…" : "Generate"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Samples</h2>
        {result === null ? (
          <p className="muted">Press Generate.</p>
        ) : samples.length === 0 ? (
          <p className="muted">The model wrote nothing (train it first).</p>
        ) : (
          <ol className="samples">
            {samples.map((s, i) => {
              const text = written(s);
              return (
                <li key={i} className={rated[i] ? `rated ${rated[i] > 0 ? "up" : "down"}` : ""}>
                  <pre className="sample">
                    {result.unit !== "base64" && result.prompt ? <span className="muted">{result.prompt}{result.unit === "char" ? "" : " "}</span> : null}
                    {Array.isArray(s.tokens) && s.tokens.length && result.unit !== "base64" ? (
                      <TokenText tokens={s.tokens} depths={s.depths} unit={result.unit} maxDepth={maxDepth} />
                    ) : (
                      s.completion || "(nothing)"
                    )}
                  </pre>
                  <div className="meta">
                    {asArray(s.tokens).length} tokens <DepthNote depths={s.depths} maxDepth={lm === "chain" ? maxDepth : null} />
                    <RateButtons
                      rated={rated[i] || 0}
                      disabled={busy || !text.trim()}
                      onRate={(amount) => credit(i, s, amount)}
                      label={`sample ${i + 1}`}
                    />
                    <button
                      type="button"
                      className={`rate${ratings.has(text) ? " active" : ""}`}
                      title="Mark it out of 10 on the marks card"
                      disabled={!text.trim() || ratings.has(text)}
                      onClick={() => ratings.add(text)}
                    >
                      {ratings.has(text) ? "★" : "☆"}
                    </button>
                    <HearButton text={text} unit={result.unit} label={`sample ${i + 1}`} />
                  </div>
                </li>
              );
            })}
          </ol>
        )}
        {note ? <p className="note">{note}</p> : null}
      </div>

      <RatingsCard lm={lm} ratings={ratings} onMoved={onMoved} namespace="generate.ratings" />
    </div>
  );
}
