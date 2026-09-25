import { useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber, unitLength, unitName, unitsOf } from "../util.js";
import Alert from "./Alert.jsx";
import BackwardsField from "./BackwardsField.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import GuardNotice from "./GuardNotice.jsx";
import RatingsCard, { RateButtons, useRatings } from "./RatingsCard.jsx";
import SearchFields from "./SearchFields.jsx";
import TraversalFields from "./TraversalFields.jsx";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { problemText, searchProblemsFor } from "../settings.js";
import { reverseUnits } from "../backwards.js";

/**
 * Generate whole texts with the prediction search (beam: the K most likely
 * complete texts, from START or continuing a prefix; stochastic walks; the
 * single cheapest text) and rate them: thumbs up = correct (2NRL positive
 * phase), thumbs down = garbage (negative phase). "Train on ratings" starts a
 * feedback job - 2NRL when both kinds were rated, reward-only on thumbs up
 * alone, punish-only (negative phase, then invert) on thumbs down alone.
 * Ratings accumulate across generations until they are trained on or cleared.
 *
 * With Query backwards on (site-wide, `BackwardsField`), for a model trained
 * with "Read every text backwards", the prefix is sent turned around - every
 * text then *ends* with it - and every sample is shown turned back round. A
 * rating keeps the model's own text, the backwards one, because that is the
 * text feedback trains on; the card shows it the right way round.
 */
export default function GeneratePanel({ status }) {
  const [count, setCount] = useStoredState("generate.count", "3");
  const [maxLength, setMaxLength] = useStoredState("generate.maxLength", "60");
  const [temperature, setTemperature] = useStoredState("generate.temperature", "1.0");
  const [mode, setMode] = useStoredState("generate.mode", "beam");
  const [prefix, setPrefix] = useStoredState("generate.prefix", "");
  // the traversal, the sampling filters, the diversity and the direction are site-wide (the Settings tab)
  const { network, search, backwards } = useSiteSettings();
  const [guard, setGuard] = useStoredState("generate.guard", true);
  const [provenance, setProvenance] = useStoredState("generate.provenance", true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState(null);
  const [guarded, setGuarded] = useState(null);
  const feedback = useJob("feedback");
  const { ratings, rate, ratingOf, setMark, remove, clear } = useRatings();
  // the resonant model's k-best search returns the exact K most likely texts, and far cheaper than a beam
  const resonantKind = Boolean(status && status.kind === "resonant");
  const searchProblem = problemText(searchProblemsFor(search.values, mode));
  const [sent, setSent] = useState({}); // the search settings the shown samples were asked with
  // the units the shown samples were asked backwards in, or null when they were asked the usual way round
  const [flip, setFlip] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (searchProblem) {
      setError(searchProblem);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const tuning = search.body(mode);
      // backwards: the model read every text from its end, so a prefix is turned around - the texts end with it
      const units = backwards.on ? unitsOf(status) : null;
      const data = await api.generate({
        count: parseInteger(count, 1),
        max_length: parseInteger(maxLength, 60),
        temperature: parseNumber(temperature, 1),
        mode,
        guard,
        provenance,
        ...(prefix ? { prefix: units ? reverseUnits(prefix, units) : prefix } : {}),
        ...network.body,
        ...tuning,
      });
      setSamples(asArray(data && data.samples));
      setSent(tuning);
      setFlip(units);
      setGuarded((data && data.guard) || null);
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
          {resonantKind ? (
            <>
              {" "}
              <b>k-best</b> returns the same K most likely texts <i>exactly</i> - Dijkstra with K labels per state
              instead of one - and stops as soon as it has them.
            </>
          ) : null}
        </p>
        <TextField
          label="Prefix"
          hint={backwards.on ? "optional, backwards: every text ends with it" : "optional: every text starts with it"}
          value={prefix}
          onChange={setPrefix}
          placeholder={backwards.on ? "the lazy dog" : "the quick"}
        />
        <div className="row">
          <NumberField label="Count" hint="beam: the K most likely" value={count} onChange={setCount} min={1} step={1} />
          <NumberField label="Max length" hint={`${unitName(status)} per text`} value={maxLength}
                       onChange={setMaxLength} min={1} step={1} />
        </div>
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              ...(resonantKind ? [["kbest", "k-best (the exact K most likely texts)"]] : []),
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
        <SearchFields mode={mode} compact />
        <TraversalFields compact />
        <BackwardsField compact status={status} />
        <CheckField
          label="Filter with the negative network"
          hint="the pair: the model over-samples and the negative network vetoes what it knows to be a failure"
          checked={guard}
          onChange={setGuard}
        />
        <CheckField
          label="Say why it vetoed"
          hint="the provenance of each veto: the rule, the reasons and the blamed fragments; off, the guard reports how many it stopped"
          checked={provenance}
          onChange={setProvenance}
          disabled={!guard}
        />
        <div className="actions">
          <button type="submit" className="primary" disabled={loading || Boolean(searchProblem)}>
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
        <GuardNotice guard={guarded} what="candidates" />
        {sent.diversity && samples && samples.length > 1 ? (
          <p className="muted">
            Picked for diversity {fmtNum(sent.diversity, 2)}: the most likely text first, then each by its cost plus
            what it repeats of the ones before it, so after the first they are not in cost order.
          </p>
        ) : null}
        {flip && samples && samples.length > 0 ? (
          <p className="muted">
            Asked backwards: every text is shown turned back round. A thumb rates it the way the model reads it,
            backwards, so feedback trains the model in its own direction.
          </p>
        ) : null}
        {samples === null ? (
          <p className="muted">Press Generate to sample texts from the model.</p>
        ) : samples.length === 0 ? (
          <p className="muted">The model returned no samples (train it first).</p>
        ) : (
          <ol className="samples">
            {samples.map((s, i) => {
              // the model's own text is what a rating keeps and feedback trains on; backwards, it is shown turned back
              const text = textOf(s);
              const shown = flip ? reverseUnits(text, flip) : text;
              const rating = ratingOf(text);
              return (
                <li key={i} className={rating ? `rated ${rating}` : ""}>
                  <pre className="sample">{shown}</pre>
                  <div className="meta">
                    cost {fmtNum(s && s.cost, 3)} · p {fmtNum(s && s.probability, 4)} ·{" "}
                    {fmtInt(asArray(s && s.path).length)} path nodes · {fmtInt(unitLength(text, status))} {unitName(status)}
                    <RateButtons
                      text={text}
                      rating={rating}
                      disabled={!text.trim() || feedback.running}
                      onRate={(t, r) => rate(t, r, { cost: s && s.cost, ...(flip ? { shown } : {}) })}
                      label={`sample ${i + 1}`}
                    />
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <RatingsCard
        ratings={ratings}
        onClear={clear}
        onRemove={remove}
        onMark={setMark}
        feedback={feedback}
        status={status}
        namespace="generate.ratings"
      />
    </>
  );
}
