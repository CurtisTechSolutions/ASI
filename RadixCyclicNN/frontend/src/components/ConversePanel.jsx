import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { applyEvent } from "../stream.js";
import { THINK_DEPTH, thoughtNodes, thoughtsOf } from "../thinking.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber, unitName } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import GuardNotice from "./GuardNotice.jsx";
import RatingsCard, { RateButtons, useRatings } from "./RatingsCard.jsx";
import { ThoughtLine } from "./ThoughtView.jsx";

/**
 * The model converses with itself. Two voices take turns; every reply is the
 * prediction search picking up the last words of the previous line and
 * continuing them to the end of a text. Beam speaks the most likely
 * continuation the conversation has not heard yet; sample draws stochastic
 * walks. When nothing follows, the context loses a word at a time and finally
 * the voice changes the subject with a fresh text. The second voice may be the
 * model of the other kind kept in memory. Turns can be rated like samples, and
 * the duplicates the model could not avoid are marked thumbs-down for the 2NRL
 * negative phase ("Punish duplicates") - a reply that repeats its own words
 * counts as one of those while "Avoid repeated words" is on, though a voice
 * that catches itself repeating - its own words, or the conversation's - first
 * backs up to where it would have said them again and explores other ways on
 * ("Explore"), and says so. With "Think before backing up" on, it first thinks:
 * a thought from the THINK sentinel, questioning itself where the model has
 * learned to ("Think depth"), that hands over to BACK when it stops - and the
 * turn shows what it thought. New turns are appended to the top of the
 * conversation and push the older ones down, so nothing has to scroll.
 *
 * With "Stream" on (the default) the conversation arrives as it happens
 * (`POST /api/converse/stream`, JSON Lines): every turn the moment it is
 * spoken, and above it the turn being spoken - the window a backtrack may
 * still rewrite. The live bubble shows the draft the voice caught itself on,
 * strikes through what it backed out of, underlines the way on it found, and
 * a turn keeps its draft in the meta line once it is committed, so the
 * backtracking can be seen in action and read back afterwards. A server
 * without the stream route answers 404, and the panel falls back to the
 * plain route.
 */
/** What a turn's rethink record says in one line: what it caught itself doing, and how that turned out. */
function rethinkSays(turn) {
  const r = turn.rethink;
  const caught =
    r.kind === "stutter" ? `caught itself saying “${r.noticed}” twice` : `caught itself repeating “${r.noticed}”`;
  if (!r.steps) return `${caught}: the words it picked up, not its own`;
  const learned = r.taught >= 0 ? " (and learned to hand over there)" : "";
  if (r.found) return `${caught}: kept “${r.cut}”, found another way on in ${fmtInt(r.explored)} path(s)${learned}`;
  const ending = turn.repeat ? "said it anyway" : "took a lesser answer";
  return `${caught}: kept “${r.cut}”, weighed ${fmtInt(r.explored)} path(s), ${ending}${learned}`;
}

/** The turn being spoken, before it is committed: what the voice is doing right now (see `applyEvent`). */
function LiveTurn({ live, speakers }) {
  const side = live.index % 2 === 0 ? "a" : "b";
  const speaker = live.speaker || speakers[live.index % 2];
  const retracting = Boolean(live.caught);
  return (
    <li className={`turn ${side} live`} aria-live="polite">
      <div className="speaker">
        <b>{speaker}</b>
        <span className="badge running">{retracting ? "thinking again" : "speaking"}</span>
      </div>
      <p className="bubble">
        {live.draft ? (
          <>
            {live.kept}
            {live.retracted ? <s className="retracted">{live.retracted}</s> : null}
            {live.found ? <span className="found">{live.found}</span> : null}
            {!live.found && retracting ? <span className="pending" /> : null}
          </>
        ) : (
          <>
            {live.from ? <span className="context">{live.from}</span> : null}
            <span className="pending" />
          </>
        )}
      </p>
      <div className="meta">{live.note || "looking for what to say"}</div>
    </li>
  );
}

export default function ConversePanel({ status }) {
  const [opening, setOpening] = useStoredState("converse.opening", "");
  const [turns, setTurns] = useStoredState("converse.turns", "6");
  const [mode, setMode] = useStoredState("converse.mode", "beam");
  const [maxLength, setMaxLength] = useStoredState("converse.maxLength", "60");
  const [context, setContext] = useStoredState("converse.context", "12");
  const [temperature, setTemperature] = useStoredState("converse.temperature", "1.0");
  const [k, setK] = useStoredState("converse.k", "5");
  const [speakerA, setSpeakerA] = useStoredState("converse.speakerA", "A");
  const [speakerB, setSpeakerB] = useStoredState("converse.speakerB", "B");
  const [partner, setPartner] = useStoredState("converse.partner", "");
  const [punishRepeats, setPunishRepeats] = useStoredState("converse.punishRepeats", true);
  const [avoidWordRepeats, setAvoidWordRepeats] = useStoredState("converse.avoidWordRepeats", true);
  const [explore, setExplore] = useStoredState("converse.explore", "3");
  const [learn, setLearn] = useStoredState("converse.learn", true);
  const [streaming, setStreaming] = useStoredState("converse.stream", true);
  const [think, setThink] = useStoredState("converse.think", true);
  const [thinkDepth, setThinkDepth] = useStoredState("converse.thinkDepth", String(THINK_DEPTH));
  const [inMemory, setInMemory] = useState(null); // null until GET /api/model answers
  const [transcript, setTranscript] = useState(null);
  const [live, setLive] = useState(null); // the turn being spoken, while a streamed conversation runs
  const [guard, setGuard] = useStoredState("converse.guard", true);
  const [provenance, setProvenance] = useStoredState("converse.provenance", true);
  const [guarded, setGuarded] = useState(null);
  const [notice, setNotice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const feedback = useJob("feedback");
  const { ratings, rate, punish, ratingOf, setMark, remove, clear } = useRatings();
  const kind = status ? status.kind : null;

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

  const partners = asArray(inMemory).filter((x) => x && x !== kind);
  useEffect(() => {
    // a remembered partner survives until the answer says it is not in memory any more
    if (inMemory && partner && !partners.includes(partner)) setPartner("");
  }, [inMemory, partner, partners]);

  const speakers = [speakerA.trim() || "A", speakerB.trim() || "B"];

  async function run(continuing) {
    setLoading(true);
    setError(null);
    setNotice(null);
    setLive(null);
    const history = continuing && transcript ? transcript.map((t) => t.text) : [];
    const body = {
      turns: parseInteger(turns, 6),
      mode,
      max_length: parseInteger(maxLength, 60),
      context: parseInteger(context, 12),
      temperature: parseNumber(temperature, 1),
      k: parseInteger(k, 5),
      speakers,
      guard,
      provenance,
      avoid_word_repeats: avoidWordRepeats,
      explore: parseInteger(explore, 3),
      learn,
      think,
      think_depth: Math.max(0, parseInteger(thinkDepth, THINK_DEPTH)),
      ...(partner ? { partner } : {}),
      ...(history.length ? { history } : opening.trim() ? { opening } : {}),
    };
    try {
      let data;
      let fresh;
      if (streaming) {
        // the conversation as it happens: turns are committed the moment they are spoken, and the turn being
        // spoken - the window a backtrack may still rewrite - is shown live above them
        let window = null;
        const spoken = [];
        const onEvent = (event) => {
          if (event.event === "turn") {
            // a turn keeps the draft it caught itself on, so the backtracking can be read back afterwards
            const turn = window && window.draft ? { ...event.turn, draft: window.draft, kept: window.kept } : event.turn;
            spoken.push(turn);
            window = null;
            setLive(null);
            setTranscript((prev) => [...asArray(prev), turn]);
            return;
          }
          window = applyEvent(window, event);
          setLive(window);
        };
        if (!history.length) setTranscript([]);
        try {
          data = await api.converseStream(body, onEvent);
        } catch (err) {
          if (!(err.status === 404 || err.status === 405) || spoken.length) throw err;
          // an older server without the stream route: the same conversation, all at once
          data = await api.converse(body);
          setTranscript((prev) => (history.length ? [...asArray(prev), ...asArray(data.turns)] : asArray(data.turns)));
        }
        fresh = spoken.length ? spoken : asArray(data && data.turns);
      } else {
        data = await api.converse(body);
        fresh = asArray(data && data.turns);
        setTranscript((prev) => (history.length ? [...asArray(prev), ...fresh] : fresh));
      }
      setGuarded((data && data.guard) || null);
      // the duplicates the search could not avoid: thumbs down, so "Train on ratings" punishes them
      const duplicates = Array.isArray(data && data.repeats)
        ? data.repeats
        : fresh.filter((t) => t && t.repeat).map((t) => t.text);
      const punished = punishRepeats ? punish(duplicates) : 0;
      const notes = [];
      if (!fresh.length && history.length) notes.push("The model had nothing more to say.");
      const thought = thoughtsOf(fresh).length;
      if (thought) {
        const at = thoughtNodes(fresh).length;
        notes.push(
          `It thought ${thought === 1 ? "once" : `${fmtInt(thought)} times`} before backing up` +
            (at ? `, and learned to stop and think at ${fmtInt(at)} node${at === 1 ? "" : "s"}.` : "."),
        );
      }
      if (punished) {
        notes.push(
          `${punished} duplicate${punished === 1 ? "" : "s"} the model could not avoid:` +
            " marked 👎 for “Train on ratings” below (the 2NRL negative phase).",
        );
      }
      setNotice(notes.join(" ") || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLive(null);
      setLoading(false);
    }
  }

  const spoken = asArray(transcript);
  // the newest turn first: a new turn is appended to the top of the list and pushes the older ones down,
  // so the latest reply is where the eye already is and nothing has to be scrolled to
  const newestFirst = [...spoken].reverse();
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
          the subject with a fresh text. A voice that catches itself repeating backs up and looks for another way
          on; with <b>Think before backing up</b> it thinks first, and the turn shows what it thought (💭).
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
          <NumberField label="Context" hint={`${unitName(status)} picked up`} value={context} onChange={setContext} min={0} step={1} />
          <NumberField label="Max length" hint={`${unitName(status)} added per turn`} value={maxLength} onChange={setMaxLength} min={0} step={1} />
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
          <NumberField
            label="Explore"
            hint="times it may back up out of a repeat (0 = not at all)"
            value={explore}
            onChange={setExplore}
            min={0}
            step={1}
            disabled={!avoidWordRepeats}
          />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min={0} disabled={mode !== "sample"} />
          <NumberField
            label="Think depth"
            hint="how deep a thought may question itself (0 = never)"
            value={thinkDepth}
            onChange={setThinkDepth}
            min={0}
            step={1}
            disabled={!think}
          />
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
        <CheckField
          label="Filter with the negative network"
          hint="a reply it vetoes is left unsaid and the voice looks for another one"
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
        <CheckField
          label="Avoid repeated words"
          hint="skip a reply that says the same word or phrase twice in a row (“say morning morning”)"
          checked={avoidWordRepeats}
          onChange={setAvoidWordRepeats}
          disabled={loading}
        />
        <CheckField
          label="Learn where it goes round"
          hint="what a rethink finds out is taught to the graph, so the model itself hands over there next time (this changes the model)"
          checked={learn}
          onChange={setLearn}
          disabled={loading || !explore || explore === "0"}
        />
        <CheckField
          label="Think before backing up"
          hint="a voice that caught itself repeating thinks first - a thought from the THINK sentinel that hands over to BACK when it stops"
          checked={think}
          onChange={setThink}
          disabled={loading || !explore || explore === "0"}
        />
        <CheckField
          label="Punish duplicates"
          hint="the repeats the model could not avoid are marked 👎 for the 2NRL negative phase"
          checked={punishRepeats}
          onChange={setPunishRepeats}
          disabled={loading}
        />
        <CheckField
          label="Stream"
          hint="show the conversation as it happens: each turn the moment it is spoken, and above it the turn being spoken - the draft it caught itself on, what it backed out of, the way on it found"
          checked={streaming}
          onChange={setStreaming}
          disabled={loading}
        />
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
              setLive(null);
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
        {spoken.length > 1 ? (
          <p className="muted">Newest first: the latest turn is at the top and the conversation grows downwards.</p>
        ) : null}
        {notice ? <p className="muted">{notice}</p> : null}
        <GuardNotice guard={guarded} what="replies" />
        {transcript === null && !live ? (
          <p className="muted">Press Start to let the model talk to itself.</p>
        ) : spoken.length === 0 && !live && !loading ? (
          <p className="muted">The model had nothing to say (train it first).</p>
        ) : (
          <ol className="dialogue" aria-label="conversation" reversed>
            {live ? <LiveTurn live={live} speakers={speakers} /> : null}
            {newestFirst.map((t, i) => {
              const position = spoken.length - i; // where the turn stands in the conversation, counted from its start
              const side = t.index % 2 === 0 ? "a" : "b";
              const rating = ratingOf(t.text);
              const flags = [
                t.given ? "given" : null,
                t.fresh && !t.given ? "new topic" : null,
                t.repeat ? "repeat" : null,
                t.stutter ? "repeats itself" : null,
                t.rethink && t.rethink.found ? "thought again" : null,
                t.vetoed ? `${fmtInt(t.vetoed)} vetoed` : null,
              ].filter(Boolean);
              return (
                <li key={`${t.index}-${position}`} className={`turn ${side}${rating ? ` rated ${rating}` : ""}`}>
                  <div className="speaker">
                    <b>{t.speaker}</b>
                    {flags.map((f) => (
                      <span key={f} className={`badge${f === "repeat" || f === "repeats itself" ? " down" : ""}${f === "thought again" ? " up" : ""}`}>
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
                    {t.rethink ? <> · {rethinkSays(t)}</> : null}
                    {t.draft ? (
                      <>
                        {" · "}
                        <span className="draft" title="what it was about to say, and what it backed out of">
                          was about to say “{t.kept}
                          {t.draft.length > (t.kept || "").length ? <s>{t.draft.slice((t.kept || "").length)}</s> : null}”
                        </span>
                      </>
                    ) : null}
                    <RateButtons
                      text={t.text}
                      rating={rating}
                      disabled={!String(t.text ?? "").trim() || feedback.running}
                      onRate={(text, r) => rate(text, r, { cost: t.cost })}
                      label={`turn ${position}`}
                    />
                  </div>
                  <ThoughtLine thought={t.rethink ? t.rethink.thought : null} />
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
        emptyText="rate some turns first"
        namespace="converse.ratings"
      />
    </>
  );
}
