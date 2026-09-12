import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField, TextField } from "./Fields.jsx";
import RatingsCard, { RateButtons, useRatings } from "./RatingsCard.jsx";

/**
 * The model converses with itself. Two voices take turns; every reply is the
 * prediction search picking up the last words of the previous line and
 * continuing them to the end of a text. Beam speaks the most likely
 * continuation the conversation has not heard yet; sample draws stochastic
 * walks. When nothing follows, the context loses a word at a time and finally
 * the voice changes the subject with a fresh text. The second voice may be the
 * model of the other kind kept in memory. Turns can be rated like samples.
 */
export default function ConversePanel({ status }) {
  const [opening, setOpening] = useState("");
  const [turns, setTurns] = useState("6");
  const [mode, setMode] = useState("beam");
  const [maxLength, setMaxLength] = useState("60");
  const [context, setContext] = useState("12");
  const [temperature, setTemperature] = useState("1.0");
  const [k, setK] = useState("5");
  const [speakerA, setSpeakerA] = useState("A");
  const [speakerB, setSpeakerB] = useState("B");
  const [partner, setPartner] = useState("");
  const [inMemory, setInMemory] = useState([]);
  const [transcript, setTranscript] = useState(null);
  const [notice, setNotice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const feedback = useJob("feedback");
  const { ratings, rate, ratingOf, remove, clear } = useRatings();
  const kind = status ? status.kind : null;
  const endRef = useRef(null);

  // the kinds kept in memory decide which partner can answer
  useEffect(() => {
    let alive = true;
    api
      .model()
      .then((m) => {
        if (alive) setInMemory(asArray(m && m.in_memory));
      })
      .catch(() => {
        if (alive) setInMemory([]);
      });
    return () => {
      alive = false;
    };
  }, [kind]);

  const partners = inMemory.filter((x) => x && x !== kind);
  useEffect(() => {
    if (partner && !partners.includes(partner)) setPartner("");
  }, [partner, partners]);

  useEffect(() => {
    if (endRef.current && transcript && transcript.length) endRef.current.scrollIntoView({ block: "nearest" });
  }, [transcript]);

  const speakers = [speakerA.trim() || "A", speakerB.trim() || "B"];

  async function run(continuing) {
    setLoading(true);
    setError(null);
    setNotice(null);
    const history = continuing && transcript ? transcript.map((t) => t.text) : [];
    try {
      const data = await api.converse({
        turns: parseInteger(turns, 6),
        mode,
        max_length: parseInteger(maxLength, 60),
        context: parseInteger(context, 12),
        temperature: parseNumber(temperature, 1),
        k: parseInteger(k, 5),
        speakers,
        ...(partner ? { partner } : {}),
        ...(history.length ? { history } : opening.trim() ? { opening } : {}),
      });
      const fresh = asArray(data && data.turns);
      setTranscript((prev) => (history.length ? [...asArray(prev), ...fresh] : fresh));
      if (!fresh.length && history.length) setNotice("The model had nothing more to say.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const spoken = asArray(transcript);
  const partnerLabel = partner ? `${speakers[0]} is the ${kind} model, ${speakers[1]} the ${partner} model` : null;

  return (
    <>
      <form
        className="card"
        onSubmit={(event) => {
          event.preventDefault();
          run(false);
        }}
      >
        <h2>Converse</h2>
        <p className="muted">
          The model talks to itself. Two voices take turns; every reply is the prediction search picking up the
          last words of the previous line (the <b>context</b>) and continuing them to the end of a text.{" "}
          <b>beam</b> speaks the most likely continuation the conversation has not heard yet; <b>sample</b> draws a
          stochastic walk. When nothing follows, the context loses a word at a time and finally the voice changes
          the subject with a fresh text.
        </p>
        <TextField
          label="Opening line"
          hint="optional: the first voice says it as given"
          value={opening}
          onChange={setOpening}
          placeholder="the cat sat on the mat"
        />
        <div className="row">
          <NumberField label="Turns" hint="per run" value={turns} onChange={setTurns} min={0} step={1} />
          <NumberField label="Context" hint="characters picked up" value={context} onChange={setContext} min={0} step={1} />
          <NumberField label="Max length" hint="characters added per turn" value={maxLength} onChange={setMaxLength} min={0} step={1} />
        </div>
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              ["beam", "beam (the most likely unheard reply)"],
              ["sample", "sample (stochastic)"],
            ]}
          />
          <NumberField label="K" hint="candidates per turn" value={k} onChange={setK} min={1} step={1} />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min={0} disabled={mode !== "sample"} />
        </div>
        <div className="row">
          <TextField label="First voice" value={speakerA} onChange={setSpeakerA} placeholder="A" />
          <TextField label="Second voice" value={speakerB} onChange={setSpeakerB} placeholder="B" />
          <SelectField
            label="Second voice is"
            hint={partners.length ? "another kind kept in memory can answer" : "select the other kind once to load it"}
            value={partner}
            onChange={setPartner}
            options={[["", `the same model (${kind || "active"})`], ...partners.map((x) => [x, `the ${x} model (in memory)`])]}
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Talking…" : spoken.length ? "Start over" : "Start"}
          </button>
          <button type="button" disabled={loading || !spoken.length} onClick={() => run(true)}>
            Continue
          </button>
          <button
            type="button"
            className="small"
            disabled={loading || !spoken.length}
            onClick={() => {
              setTranscript(null);
              setNotice(null);
            }}
          >
            Clear
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Conversation</h2>
        {partnerLabel ? <p className="muted">{partnerLabel}.</p> : null}
        {transcript === null ? (
          <p className="muted">Press Start to let the model talk to itself.</p>
        ) : spoken.length === 0 ? (
          <p className="muted">The model had nothing to say (train it first).</p>
        ) : (
          <ol className="dialogue" aria-label="conversation">
            {spoken.map((t, i) => {
              const side = t.index % 2 === 0 ? "a" : "b";
              const rating = ratingOf(t.text);
              const flags = [
                t.given ? "given" : null,
                t.fresh && !t.given ? "new topic" : null,
                t.repeat ? "repeat" : null,
              ].filter(Boolean);
              return (
                <li key={`${t.index}-${i}`} className={`turn ${side}${rating ? ` rated ${rating}` : ""}`}>
                  <div className="speaker">
                    <b>{t.speaker}</b>
                    {flags.map((f) => (
                      <span key={f} className={`badge${f === "repeat" ? " down" : ""}`}>
                        {f}
                      </span>
                    ))}
                  </div>
                  <p className="bubble">
                    {t.context ? (
                      <>
                        <span className="context" title="picked up from the previous line">
                          {t.context}
                        </span>
                        {t.reply}
                      </>
                    ) : (
                      t.text
                    )}
                  </p>
                  <div className="meta">
                    cost {fmtNum(t.cost, 3)} · p {fmtNum(t.probability, 4)}
                    {t.context ? <> · picked up “{t.context}”</> : null}
                    {t.skipped ? <> · skipped {fmtInt(t.skipped)}</> : null}
                    <RateButtons
                      text={t.text}
                      rating={rating}
                      disabled={!String(t.text ?? "").trim() || feedback.running}
                      onRate={(text, r) => rate(text, r, { cost: t.cost })}
                      label={`turn ${i + 1}`}
                    />
                  </div>
                </li>
              );
            })}
            <li ref={endRef} className="end" aria-hidden="true" />
          </ol>
        )}
        {notice ? <p className="muted">{notice}</p> : null}
      </div>

      <RatingsCard
        ratings={ratings}
        onClear={clear}
        onRemove={remove}
        feedback={feedback}
        status={status}
        emptyText="rate some turns first"
      />
    </>
  );
}
