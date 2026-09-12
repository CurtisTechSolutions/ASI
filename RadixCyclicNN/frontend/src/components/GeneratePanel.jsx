import { useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField, TextField } from "./Fields.jsx";
import RatingsCard, { RateButtons, useRatings } from "./RatingsCard.jsx";

/**
 * Generate whole texts with the prediction search (beam: the K most likely
 * complete texts, from START or continuing a prefix; stochastic walks; the
 * single cheapest text) and rate them: thumbs up = correct (2NRL positive
 * phase), thumbs down = garbage (negative phase). "Train on ratings" starts a
 * feedback job - 2NRL when both kinds were rated, reward-only on thumbs up
 * alone, punish-only (negative phase, then invert) on thumbs down alone.
 * Ratings accumulate across generations until they are trained on or cleared.
 */
export default function GeneratePanel({ status }) {
  const [count, setCount] = useState("3");
  const [maxLength, setMaxLength] = useState("60");
  const [temperature, setTemperature] = useState("1.0");
  const [mode, setMode] = useState("beam");
  const [prefix, setPrefix] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState(null);
  const feedback = useJob("feedback");
  const { ratings, rate, ratingOf, remove, clear } = useRatings();

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.generate({
        count: parseInteger(count, 1),
        max_length: parseInteger(maxLength, 60),
        temperature: parseNumber(temperature, 1),
        mode,
        ...(prefix ? { prefix } : {}),
      });
      setSamples(asArray(data && data.samples));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const textOf = (sample) => String((sample && sample.text) ?? "");

  return (
    <>
      <form className="card" onSubmit={handleSubmit}>
        <h2>Generate</h2>
        <p className="muted">
          Whole texts from the prediction search: <b>beam</b> runs it to the end of a text and returns the K most
          likely complete texts (from START, or continuing a prefix); <b>sample</b> draws stochastic walks;{" "}
          <b>dijkstra</b> is the single cheapest text.
        </p>
        <TextField label="Prefix" hint="optional: every text starts with it" value={prefix} onChange={setPrefix} placeholder="the quick" />
        <div className="row">
          <NumberField label="Count" hint="beam: the K most likely" value={count} onChange={setCount} min={1} step={1} />
          <NumberField label="Max length" value={maxLength} onChange={setMaxLength} min={1} step={1} />
        </div>
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              ["beam", "beam (the K most likely texts)"],
              ["sample", "sample (stochastic)"],
              ["dijkstra", "dijkstra (the single cheapest text)"],
            ]}
          />
          <NumberField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0.01}
            disabled={mode !== "sample"}
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Generating…" : "Generate"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Samples</h2>
        <p className="muted">
          Rate a sample: thumbs up marks it correct (2NRL positive phase), thumbs down marks it garbage (negative
          phase). Press the same thumb again to remove the rating.
        </p>
        {samples === null ? (
          <p className="muted">Press Generate to sample texts from the model.</p>
        ) : samples.length === 0 ? (
          <p className="muted">The model returned no samples (train it first).</p>
        ) : (
          <ol className="samples">
            {samples.map((s, i) => {
              const text = textOf(s);
              const rating = ratingOf(text);
              return (
                <li key={i} className={rating ? `rated ${rating}` : ""}>
                  <pre className="sample">{text}</pre>
                  <div className="meta">
                    cost {fmtNum(s && s.cost, 3)} · p {fmtNum(s && s.probability, 4)} ·{" "}
                    {fmtInt(asArray(s && s.path).length)} path nodes · {fmtInt(text.length)} chars
                    <RateButtons
                      text={text}
                      rating={rating}
                      disabled={!text.trim() || feedback.running}
                      onRate={(t, r) => rate(t, r, { cost: s && s.cost })}
                      label={`sample ${i + 1}`}
                    />
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <RatingsCard ratings={ratings} onClear={clear} onRemove={remove} feedback={feedback} status={status} />
    </>
  );
}
