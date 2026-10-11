import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtNum, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField, TextField } from "./Fields.jsx";

const PRESETS = [
  ["enc>dec", "enc>dec - English in, English out"],
  ["enc>13>dec", "enc>13>dec - with a hidden machine"],
  ["text>13", "text>13 - a yes / no classifier of text"],
  ["enc>13", "enc>13 - a yes / no classifier of sounds"],
];

/** One layer's part of a reply, as the trace returns it. */
function TraceLine({ step }) {
  return (
    <li className="mono">
      <strong>{step.layer}</strong> read {step.read || "-"}
      {step.states && step.states.length ? <> · walked {step.states.join(" ")}</> : null} · wrote {step.wrote || "-"}
      {step.english !== null && step.english !== undefined ? <span className="muted"> = "{step.english}"</span> : null}
      {step.accepted ? <span className="muted"> (accepting)</span> : null}
    </li>
  );
}

/** The network's reply in the conversation: what it said, its sounds, its trace, and the two ways to teach it. */
function Reply({ turn, busy, onTeach }) {
  const [correction, setCorrection] = useState("");
  const [open, setOpen] = useState(false);
  const r = turn.reply;
  const said = r.translator ? r.reply || "(silence)" : r.reply;
  return (
    <li className={`turn a${turn.taught ? " rated up" : ""}`}>
      <div className="speaker">network</div>
      <div className="bubble">
        {said}
        {r.translator && r.sounds && r.sounds.length ? <div className="meta units">{r.sounds.join(" ")}</div> : null}
        {turn.taught ? <div className="meta">{turn.taught}</div> : null}
      </div>
      <div className="actions">
        <button type="button" className="link" disabled={busy} onClick={() => onTeach(turn, r.reply)}>
          right - reinforce it
        </button>
        <span className="inline-form">
          <input
            value={correction}
            placeholder={r.translator ? "the right reply, in English" : "yes or no"}
            onChange={(e) => setCorrection(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && correction.trim()) onTeach(turn, correction.trim());
            }}
          />
          <button
            type="button"
            disabled={busy || !correction.trim()}
            onClick={() => onTeach(turn, correction.trim())}
          >
            Teach this reply
          </button>
        </span>
        <button type="button" className="link" onClick={() => setOpen(!open)}>
          {open ? "hide" : "show"} how it got there
        </button>
      </div>
      {open ? (
        <ol className="lines">
          {r.trace.map((step, i) => (
            <TraceLine key={i} step={step} />
          ))}
        </ol>
      ) : null}
    </li>
  );
}

/**
 * Talk to a network of machines: type, and it replies - in English from a translator (an encoder that reads the
 * sounds of what you typed, a decoder whose states are sounds, spelled back), or yes / no from a classifier. Every
 * reply can be reinforced or corrected, and the network learns from it on the spot (POST /api/network/teach).
 */
export default function TalkPanel({ onMoved }) {
  const [info, setInfo] = useState(null);
  const [arch, setArch] = useStoredState("talk.arch", "enc>dec");
  const [credit, setCredit] = useStoredState("talk.credit", "shared");
  const [task, setTask] = useStoredState("talk.task", "plural");
  const [epochs, setEpochs] = useStoredState("talk.epochs", "20");
  const [times, setTimes] = useStoredState("talk.times", "3");
  const [language, setLanguage] = useStoredState("talk.language", "even-b");
  const [data, setData] = useStoredState("talk.data", "");
  const [path, setPath] = useStoredState("talk.path", "network.json.gz");
  const [draft, setDraft] = useStoredState("talk.draft", "cat");
  const [turns, setTurns] = useState([]);
  const [fit, setFit] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const logRef = useRef(null);

  useEffect(() => {
    api.network().then(setInfo).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [turns]);

  async function act(fn) {
    setBusy(true);
    try {
      await fn();
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const send = () =>
    act(async () => {
      const text = draft.trim();
      if (!text) return;
      const reply = await api.talk(text);
      setTurns((t) => [...t, { id: t.length, text, reply, taught: null }]);
      setDraft("");
    });

  const teach = (turn, reply) =>
    act(async () => {
      const r = await api.networkTeach({ text: turn.text, reply, times: Math.max(1, parseNumber(times, 3)) });
      const note = `taught "${reply}" ${Math.max(1, parseNumber(times, 3))}× (${r.right} of ${r.of} right) - now: ${
        r.after.translator ? r.after.reply || "(silence)" : r.after.reply
      }`;
      setTurns((t) => t.map((x) => (x.id === turn.id ? { ...x, taught: note } : x)));
      setInfo(await api.network());
      if (onMoved) onMoved();
    });

  const make = () =>
    act(async () => {
      setInfo(await api.networkNew({ arch, credit }));
      setTurns([]);
      setFit(null);
    });

  const teachTask = () =>
    act(async () => {
      const body = info && info.translator
        ? { task, epochs: parseNumber(epochs, 20) }
        : data
          ? { data, epochs: parseNumber(epochs, 20) }
          : { language, episodes: 2000, epochs: parseNumber(epochs, 1) };
      const r = await api.networkFit(body);
      setFit(r);
      setInfo(r.network);
    });

  const translator = info && info.translator;
  return (
    <div className="panel">
      <div className="card wide">
        <h2>The network</h2>
        <p className="note">
          The server holds a network of machines beside the single machine the other tabs show. A translator reads what
          you type as its sounds (the phonetic tokenizer), walks them, and spells what its decoder writes back as
          English; a classifier answers yes or no. Make one from an architecture string, teach it a task, then talk
          to it below.
        </p>
        {info ? (
          <p className="mono">
            <strong>{info.architecture || "a network"}</strong> · reads {info.reads} ·{" "}
            {translator ? "replies in English" : "replies yes / no"} · {info.layers.length} layer
            {info.layers.length === 1 ? "" : "s"} (
            {info.layers.map((l) => `${l.name} ${l.shape.join("×")}, ${l.touched} touched`).join("; ")}) ·{" "}
            {info.examples} examples taught
          </p>
        ) : null}
        <div className="row">
          <SelectField label="Preset" value={PRESETS.some(([v]) => v === arch) ? arch : ""} onChange={(v) => v && setArch(v)}
            options={[["", "(your own)"], ...PRESETS]} />
          <TextField label="Architecture" value={arch} onChange={setArch} hint="> in a row, (a|b) side by side; enc dec text audio image" />
          <SelectField label="Credit" value={credit} onChange={setCredit}
            options={[["shared", "shared"], ["own", "own (voting branches)"]]} />
        </div>
        <div className="actions">
          <button type="button" disabled={busy || !arch} onClick={make}>
            New network
          </button>
        </div>
        <div className="row">
          {translator ? (
            <SelectField label="Task" value={task} onChange={setTask}
              options={[["plural", "plural: cat → cats"], ["past", "past tense: walk → walked"]]} />
          ) : (
            <>
              <TextField label="Language" value={language} onChange={setLanguage} hint="even-b, contains-aa, ends-ab, mod3-a" />
              <TextField label="Or a data file" value={data} onChange={setData} hint="labels.tsv on the server: input<TAB>yes/no" />
            </>
          )}
          <NumberField label="Epochs" value={epochs} onChange={setEpochs} min="1" step="1" />
          <NumberField label="Teach a reply" value={times} onChange={setTimes} min="1" step="1" hint="times, when you correct it" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy || !info} onClick={teachTask}>
            {translator ? "Teach the task" : "Teach"}
          </button>
          <TextField label="File" value={path} onChange={setPath} />
          <button type="button" disabled={busy} onClick={() => act(async () => setFit({ saved: (await api.networkSave(path)).saved }))}>
            Save
          </button>
          <button type="button" disabled={busy} onClick={() => act(async () => { setInfo(await api.networkLoad(path)); setTurns([]); })}>
            Load
          </button>
        </div>
        {fit && fit.exact ? (
          <p className="note">
            after {fit.exact.length} epochs: {fmtNum(fit.exact[fit.exact.length - 1] * 100, 0)}% of the taught pairs exactly right,{" "}
            {fmtNum(fit.held_out[fit.held_out.length - 1] * 100, 0)}% of the held-out ones;{" "}
            {fit.examples.map(([x, got, want]) => `${x} → ${got || "(silence)"}${got === want ? "" : ` (${want})`}`).join(", ")}
          </p>
        ) : null}
        {fit && fit.taught ? (
          <p className="note">
            right while taught {fmtNum(fit.taught[fit.taught.length - 1] * 100, 0)}%, held out{" "}
            {fmtNum(fit.held_out[fit.held_out.length - 1] * 100, 0)}%
          </p>
        ) : null}
        {fit && fit.saved ? <p className="note">saved {fit.saved}</p> : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>
      <div className="card wide voice-room">
        <h2>Talk</h2>
        <div className="voice-log" ref={logRef}>
          {turns.length === 0 ? (
            <p className="empty muted">
              {translator
                ? "Type an English word or a few - a noun, say - and it answers in English. Untaught it is silent; teach it the plural task above, or correct it reply by reply."
                : "Type a string and it answers yes or no; correct it to teach it."}
            </p>
          ) : (
            <ol className="dialogue voice">
              {turns.map((turn) => (
                <FragmentTurn key={turn.id} turn={turn} busy={busy} onTeach={teach} />
              ))}
            </ol>
          )}
        </div>
        <div className="voice-composer">
          <input
            value={draft}
            placeholder={translator ? "say something in English" : "a string to judge"}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy) send();
            }}
          />
          <button type="button" disabled={busy || !draft.trim()} onClick={send}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function FragmentTurn({ turn, busy, onTeach }) {
  return (
    <>
      <li className="turn b">
        <div className="speaker">you</div>
        <div className="bubble">{turn.text}</div>
      </li>
      <Reply turn={turn} busy={busy} onTeach={onTeach} />
    </>
  );
}
