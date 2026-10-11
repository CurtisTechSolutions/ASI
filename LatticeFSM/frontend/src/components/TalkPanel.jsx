import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { useLmInfo } from "../hooks/useLmInfo.js";
import { goalShare, lmLabel, mean, untilNote } from "../lm.js";
import { askModel, turnText } from "../talk.js";
import { fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import HearButton from "./HearButton.jsx";
import { CheckField, NumberField } from "./Fields.jsx";
import { RateButtons } from "./RatingsCard.jsx";
import TokenText, { DepthNote } from "./TokenText.jsx";
import { GuardBadge } from "./Verdict.jsx";

/**
 * The conversation as it stands, oldest at the top, newest at the bottom where the next line lands: what you
 * said, and the model's reply - its continuation of it - with 👍 / 👎 (crediting the two together) and 🔊.
 * For the chain each reply token is tinted by its walk's depth (subscripted when `subscripts`), with the mean.
 * Shared by the Talk and Voice tabs.
 */
export function TalkLog({ turns, busy, onRate, onTeach, maxDepth, subscripts, empty }) {
  const logRef = useRef(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [turns]);
  return (
    <div className="voice-log" ref={logRef}>
      {turns.length === 0 ? (
        <p className="empty muted">{empty}</p>
      ) : (
        <ol className="dialogue voice">
          {turns.map((turn) => (
            <TalkTurn
              key={turn.id}
              turn={turn}
              busy={busy}
              onRate={onRate}
              onTeach={onTeach}
              maxDepth={maxDepth}
              subscripts={subscripts}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * One exchange: what you said and the model's reply, with the thumbs, the voice and - when `onTeach` is given -
 * a box to type the right reply in and teach it (until its continuation is exactly that, or the rounds run out).
 */
function TalkTurn({ turn, busy, onRate, onTeach, maxDepth, subscripts }) {
  const [correction, setCorrection] = useState("");
  const teach = () => {
    if (correction.trim() && !busy) onTeach(turn, correction.trim());
  };
  const tokenView = turn.unit !== "base64" && turn.tokens && turn.tokens.length;
  return (
    <>
      <li className="turn b">
        <div className="speaker">you{turn.learned ? ` · learned ${turn.learned} tokens` : ""}</div>
        <div className="bubble">{turn.text}</div>
      </li>
      <li className={`turn a${turn.rated ? ` rated ${turn.rated > 0 ? "up" : "down"}` : ""}`}>
        <div className="speaker">
          {turn.model}
          {turn.depths ? <DepthNote depths={turn.depths} maxDepth={maxDepth} /> : null}
          <GuardBadge guard={turn.guard} text={String(turn.reply ?? "").trim()} vetoed={turn.vetoed} />
        </div>
        <div className="bubble">
          {tokenView ? (
            <TokenText tokens={turn.tokens} depths={turn.depths} unit={turn.unit} maxDepth={maxDepth} subscripts={subscripts} />
          ) : (
            turn.reply || <span className="muted">(nothing follows)</span>
          )}
        </div>
        <div className="meta">
          <RateButtons rated={turn.rated || 0} disabled={busy} onRate={(amount) => onRate(turn, amount)} label="the reply" />
          <HearButton text={turn.reply} unit={turn.unit} label="the reply" />
          {turn.credited ? <span className="muted">{turn.credited}</span> : null}
          {turn.guarded ? <span className="muted">{turn.guarded}</span> : null}
        </div>
        {onTeach ? (
          <div className="meta">
            <span className="inline-form">
              <input
                value={correction}
                placeholder="the right reply"
                aria-label="the right reply"
                onChange={(e) => setCorrection(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") teach();
                }}
              />
              <button type="button" className="small" disabled={busy || !correction.trim()} onClick={teach}>
                Teach this reply
              </button>
            </span>
          </div>
        ) : null}
        {turn.taught ? (
          <div className="meta">
            <span className={turn.taught.reached ? "ok-text" : "error-text"}>{turn.taught.note}</span>
            {" - now: "}
            <b>{turn.taught.completion || "(nothing)"}</b>
          </div>
        ) : null}
      </li>
    </>
  );
}

/** The turns' state and the two things done to it, shared by Talk and Voice: say a line, rate a reply. */
export function useTalk(lm, onMoved) {
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const nextId = useRef(0);

  async function say(message, opts) {
    const text = String(message ?? "").trim();
    if (!text) return null;
    setBusy(true);
    try {
      const turn = await askModel(lm, text, opts);
      const full = { ...turn, id: nextId.current++, model: lmLabel(lm), lm, rated: 0 };
      setTurns((t) => [...t, full]);
      setError(null);
      if (opts && opts.learn && onMoved) onMoved();
      return full;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  /**
   * Credit the turn (👍 / 👎). With `teachGuard`, the guard learns from it too: 👎 blames the reply ("thumbs
   * down"), 👍 clears it.
   */
  async function rate(turn, amount, { teachGuard = false } = {}) {
    setBusy(true);
    try {
      const r = await api.lmCredit(turn.lm, turnText(turn), amount);
      const credited = `${amount > 0 ? "rewarded" : "punished"} ${r.credited}`;
      let guarded = null;
      const reply = String(turn.reply ?? "").trim();
      if (teachGuard && reply) {
        if (amount < 0) {
          const g = await api.guardBlame([reply], "thumbs down", 1);
          guarded = `the guard blamed ${g.blamed}`;
        } else {
          const g = await api.guardClear([reply]);
          guarded = `the guard cleared ${g.cleared}`;
        }
      }
      setTurns((t) => t.map((x) => (x.id === turn.id ? { ...x, rated: amount, credited, guarded } : x)));
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  /** Teach that `reply` follows the turn's message (POST /api/{m}/teach), round after round until N of N. */
  async function teach(turn, reply, maxRounds = 100, threshold = 0.95) {
    setBusy(true);
    try {
      const r = await api.lmTeach(turn.lm, { text: turn.text, reply, max_rounds: maxRounds, threshold });
      const taught = { reply, note: `taught "${reply}": ${untilNote(r)}`, reached: Boolean(r.reached), completion: r.completion };
      setTurns((t) => t.map((x) => (x.id === turn.id ? { ...x, taught } : x)));
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return { turns, setTurns, busy, error, setError, say, rate, teach, clear: () => setTurns([]) };
}

/**
 * Talk to the chosen language model: type a message and its reply is the model's prediction - its own
 * continuation of what you said. "Learn what I say" trains it on every message first. A reply can be
 * rewarded or punished (👍 / 👎 credit the message and the reply together), heard, or corrected: type the
 * right reply and it is taught, round after round, until the model's own continuation is exactly that (N of
 * N tokens right, and the end) or the rounds run out. For the chain every
 * reply token is tinted by how deep the walk went to predict it, with the mean depth beside it. Enter sends.
 * "Guard" puts the guard in front of the reply (a rejected one is vetoed for another; its verdict beside it),
 * and "👎 teaches the guard" makes 👎 blame the reply in the guard and 👍 clear it.
 */
export default function TalkPanel({ lm, tick, onMoved }) {
  const [info] = useLmInfo(lm, tick);
  const [draft, setDraft] = useStoredState("lmtalk.draft", "");
  const [maxTokens, setMaxTokens] = useStoredState("lmtalk.max", "40");
  const [temperature, setTemperature] = useStoredState("lmtalk.temperature", "0");
  const [learn, setLearn] = useStoredState("lmtalk.learn", false);
  const [subscripts, setSubscripts] = useStoredState("lmtalk.subscripts", false);
  const [maxRounds, setMaxRounds] = useStoredState("lmtalk.maxRounds", "100");
  const [goal, setGoal] = useStoredState("lmtalk.goal", "95");
  const [guard, setGuard] = useStoredState("lmtalk.guard", false);
  const [teachGuard, setTeachGuard] = useStoredState("lmtalk.teachGuard", false);
  const talk = useTalk(lm, onMoved);

  const send = async () => {
    if (talk.busy || !draft.trim()) return;
    const said = await talk.say(draft, {
      learn,
      maxTokens: Math.max(1, parseInteger(maxTokens, 40)),
      temperature: parseNumber(temperature, 0),
      guard,
    });
    if (said) setDraft("");
  };

  const chain = lm === "chain";
  const maxDepth = info && info.nodes ? info.nodes - 1 : 8;
  const allDepths = talk.turns.flatMap((t) => (t.depths ? t.depths : []));
  const overall = mean(allDepths);
  return (
    <div className="panel single">
      <div className="card wide voice-room">
        <div className="toolbar">
          <h2>Talk to the {lmLabel(lm)} model</h2>
          <button type="button" className="small" disabled={talk.busy || talk.turns.length === 0} onClick={talk.clear}>
            Clear
          </button>
        </div>
        <p className="muted">
          Its reply is its prediction: what it thinks comes after what you said, token by token.
          {chain ? ` Each reply token is tinted by how deep the chain walked to predict it (of ${maxDepth} input nodes): pale, the nodes nearest the output were sure; dark, it had to go far back.` : ""}
          {chain && overall !== null ? ` Mean depth so far ${fmtNum(overall, 2)}.` : ""}
        </p>
        <div className="row">
          <NumberField label="Max tokens" value={maxTokens} onChange={setMaxTokens} min="1" step="1" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" hint="0: the likeliest" />
          <NumberField label="Teach rounds" value={maxRounds} onChange={setMaxRounds} min="1" step="1" hint="at most, teaching a reply until it is all right" />
          <NumberField label="Goal %" value={goal} onChange={setGoal} min="1" max="100" step="1" hint="of the reply's tokens right; 100: every one" />
          <CheckField label="Learn what I say" checked={learn} onChange={setLearn} hint="train on every message first" />
          {chain ? <CheckField label="Depth subscripts" checked={subscripts} onChange={setSubscripts} hint="each token's depth beside it" /> : null}
          <CheckField label="Guard" checked={guard} onChange={setGuard} hint="the guard judges the reply, and vetoes a rejected one" />
          <CheckField label="👎 teaches the guard" checked={teachGuard} onChange={setTeachGuard} hint="👎 blames the reply in the guard, 👍 clears it" />
        </div>
        <TalkLog
          turns={talk.turns}
          busy={talk.busy}
          onRate={(turn, amount) => talk.rate(turn, amount, { teachGuard })}
          onTeach={(turn, reply) => talk.teach(turn, reply, Math.max(1, parseInteger(maxRounds, 100)), goalShare(goal))}
          maxDepth={chain ? maxDepth : null}
          subscripts={chain && subscripts}
          empty="Say something - a few words it might know how to go on with - and it carries on from there."
        />
        <Alert message={talk.error} onDismiss={() => talk.setError(null)} />
        <div className="voice-composer">
          <input
            value={draft}
            placeholder="say something"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button type="button" className="primary" disabled={talk.busy || !draft.trim()} onClick={send}>
            {talk.busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
