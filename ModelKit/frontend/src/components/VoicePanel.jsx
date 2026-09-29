import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { encodeWav, resample } from "../audio.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, unitsOf } from "../util.js";
import {
  PcmPlayer,
  bytesToBase64,
  decodePcm,
  decodeWav,
  historyOf,
  listeningSupported,
  startDictationLog,
  startListening,
  wordsOf,
} from "../voice.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import HearButton from "./HearButton.jsx";

const ANSWERS = [
  ["auto", "the model; Ollama when it has nothing to say"],
  ["model", "the model alone"],
  ["ollama", "Ollama alone (and the model learns its replies)"],
  ["none", "nobody: just learn what I say"],
];
const SEND_RATE = 16000;
const STATUS_TEXT = {
  off: "off",
  starting: "starting…",
  listening: "listening",
  hearing: "hearing you",
  thinking: "thinking",
  speaking: "speaking",
  nothing: "no words heard",
};

function seconds(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "–";
}

/**
 * Talk to the model. The microphone stays on: an endpointer cuts what you
 * say into utterances (`src/voice.js`), the Web Speech API writes the words
 * down beside it, and every utterance goes to `POST /api/voice/turn/stream`
 * as one turn - heard (the transcript and the waveform behind one token, or
 * a model of acoustic units' units), trained on at once, answered by the
 * model or by Ollama on its behalf, spoken through the model's voice - whose
 * audio arrives in chunks and plays the moment the first one lands. While the
 * reply plays the ear is closed, so the model never hears itself; then it
 * listens again. A reply Ollama wrote is taught to the model, so it learns
 * to answer by itself. Nothing is pressed between two turns; the one button
 * starts and stops the ear (browsers need a click before a page may listen
 * or speak), and a text box says something without a microphone.
 */
export default function VoicePanel({ status }) {
  const [on, setOn] = useStoredState("voice.on", false);
  const [phase, setPhase] = useState("off");
  const [level, setLevel] = useState(0);
  const [interim, setInterim] = useState("");
  const [turns, setTurns] = useState([]);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState(null);
  const [note, setNote] = useState(null);
  const [info, setInfo] = useState(null);
  const [answer, setAnswer] = useStoredState("voice.answer", "auto");
  const [ollamaModel, setOllamaModel] = useStoredState("voice.ollamaModel", "");
  const [url, setUrl] = useStoredState("voice.url", "");
  const [persona, setPersona] = useStoredState("voice.persona", "");
  const [train, setTrain] = useStoredState("voice.train", true);
  const [epochs, setEpochs] = useStoredState("voice.epochs", "2");
  const [learnReply, setLearnReply] = useStoredState("voice.learnReply", true);
  const [wordsOnly, setWordsOnly] = useStoredState("voice.wordsOnly", true);
  const [speak, setSpeak] = useStoredState("voice.speak", true);
  const [pitch, setPitch] = useStoredState("voice.pitch", "120");
  const [tempo, setTempo] = useStoredState("voice.tempo", "1");
  const [gain, setGain] = useStoredState("voice.gain", "0.5");
  const [silenceMs, setSilenceMs] = useStoredState("voice.silenceMs", "700");
  const [threshold, setThreshold] = useStoredState("voice.threshold", "0.02");
  const [mode, setMode] = useStoredState("voice.mode", "beam");
  const [maxLength, setMaxLength] = useStoredState("voice.maxLength", "60");
  const listener = useRef(null);
  const dictation = useRef(null);
  const player = useRef(null);
  const busy = useRef(false);
  const turnsRef = useRef([]);
  const onRef = useRef(false);
  const counter = useRef(0);
  const settings = useRef({});
  const [view, setView] = useStoredState("voice.view", "talk");
  const log = useRef(null);
  const stuck = useRef(true); // the log follows the newest turn until the reader scrolls up

  const canListen = listeningSupported();
  const canDictate = typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const otherJobRunning = jobIsRunning(status);
  const units = unitsOf(status);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);
  useEffect(() => {
    onRef.current = on;
  }, [on]);
  useEffect(() => {
    const box = log.current;
    if (box && stuck.current) box.scrollTop = box.scrollHeight;
  }, [turns, phase, view]);
  const onScroll = () => {
    const box = log.current;
    if (box) stuck.current = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  };

  useEffect(() => {
    settings.current = {
      answer, ollamaModel, url, persona, train, epochs, learnReply, wordsOnly, speak, pitch, tempo, gain, mode, maxLength,
    };
  });

  useEffect(() => {
    let alive = true;
    api
      .voice()
      .then((data) => {
        if (alive) setInfo(data && typeof data === "object" ? data : null);
      })
      .catch(() => {
        if (alive) setInfo(null);
      });
    return () => {
      alive = false;
    };
  }, []);

  const setStatus = useCallback((next) => setPhase(next), []);

  /** Everything one turn sends beside the utterance: the settings as they are right now. */
  function turnBody(transcript) {
    const s = settings.current;
    return {
      transcript,
      history: historyOf(turnsRef.current),
      answer: s.answer,
      train: s.train,
      epochs: Math.max(0, parseInteger(s.epochs, 2)),
      learn_reply: s.learnReply,
      speak: s.speak,
      pitch: parseNumber(s.pitch, 120),
      tempo: parseNumber(s.tempo, 1),
      gain: parseNumber(s.gain, 0.5),
      mode: s.mode,
      max_length: Math.max(0, parseInteger(s.maxLength, 60)),
      ...(s.persona.trim() ? { persona: s.persona.trim() } : {}),
      ...(s.ollamaModel.trim() ? { ollama_model: s.ollamaModel.trim() } : {}),
      ...(s.url.trim() ? { url: s.url.trim() } : {}),
    };
  }

  /** Back to listening (or off) once a turn is over, or was never one. */
  function reopen() {
    busy.current = false;
    if (listener.current && onRef.current) {
      listener.current.resume();
      setStatus("listening");
    } else {
      setStatus(onRef.current ? "listening" : "off");
    }
  }

  /**
   * One turn: what was said (an utterance, typed text, or both) goes to the server and the reply plays as it
   * arrives. An utterance's words are waited for (the recogniser commits them a moment after the speaker
   * stops); one the dictation heard no words in is not sent at all while "Send only what has words" is on.
   */
  async function runTurn({ audio, text }) {
    if (busy.current) return;
    busy.current = true;
    if (listener.current) listener.current.pause();
    setError(null);
    let transcript = String(text || "").trim();
    if (audio && dictation.current && !transcript) {
      setStatus("hearing");
      transcript = await wordsOf(dictation.current, audio.startedAt, audio.endedAt);
    }
    setInterim("");
    if (audio && !transcript && dictation.current && settings.current.wordsOnly) {
      // a sound with no words in it (a cough, a door): not a turn
      setStatus("nothing");
      await new Promise((resolve) => setTimeout(resolve, 900));
      reopen();
      return;
    }
    if (dictation.current) dictation.current.mute(true); // what the mouth says is never the person's words
    setStatus("thinking");
    counter.current += 1;
    const id = counter.current;
    const entry = {
      id,
      you: { text: transcript, heard: Boolean(transcript), seconds: audio ? audio.seconds : null, typed: !audio, token: null, texts: 0 },
      model: null,
      trained: null,
      taught: null,
      spoken: null,
      phase: "thinking",
      error: null,
    };
    setTurns((prev) => [entry, ...prev].slice(0, 200));
    const update = (patch) => setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch(t) } : t)));
    const body = turnBody(transcript);
    if (audio) {
      const wav = encodeWav(resample(audio.samples, audio.rate, SEND_RATE), SEND_RATE);
      body.name = "utterance.wav";
      body.content_base64 = bytesToBase64(wav);
    }
    if (!player.current) player.current = new PcmPlayer();
    const mouth = player.current;
    const heard = (event) =>
      update((t) => ({
        you: {
          ...t.you,
          text: event.transcript || event.units || t.you.text,
          heard: Boolean(event.transcript || event.units),
          token: event.token,
          texts: (event.texts || []).length,
          units: event.units || null,
        },
      }));
    const replied = (event) => {
      update(() => ({
        model: event.text
          ? { text: event.text, spelled: event.spelled, by: event.by, error: event.ollama_error, turn: event.turn }
          : { text: "", spelled: "", by: event.by, error: event.ollama_error, turn: null },
      }));
      if (event.text && settings.current.speak) setStatus("speaking");
    };
    const onEvent = (event) => {
      switch (event.event) {
        case "heard":
          heard(event);
          break;
        case "trained":
          update(() => ({ trained: event }));
          break;
        case "reply":
          replied(event);
          break;
        case "audio":
          if (settings.current.speak) {
            try {
              mouth.play(decodePcm(event.pcm_base64), event.rate);
            } catch (err) {
              setNote(`The reply could not be played: ${err.message}`);
            }
          }
          break;
        case "spoken":
          update(() => ({ spoken: event }));
          break;
        case "taught":
          update(() => ({ taught: event }));
          break;
        default:
          break;
      }
    };
    try {
      let done;
      try {
        done = await api.voiceTurnStream(body, onEvent);
      } catch (err) {
        if (err && err.status !== 404) throw err;
        // a server without the stream route: the same turn whole, its audio in one piece
        done = await api.voiceTurn(body);
        heard(done);
        if (done.trained) update(() => ({ trained: done.trained }));
        replied({ ...done.reply });
        if (done.spoken) update(() => ({ spoken: done.spoken }));
        if (done.taught) update(() => ({ taught: done.taught }));
        if (done.wav_base64 && settings.current.speak) {
          const decoded = decodeWav(done.wav_base64);
          mouth.play(decoded.samples, decoded.rate);
        }
      }
      update(() => ({ phase: "done", seconds: done.seconds }));
    } catch (err) {
      update(() => ({ phase: "failed", error: err.message }));
      setError(err.message);
    } finally {
      await mouth.drained();
      if (dictation.current) dictation.current.mute(false);
      reopen();
    }
  }

  const startEar = useCallback(async () => {
    if (listener.current) return;
    setError(null);
    setStatus("starting");
    try {
      listener.current = await startListening({
        onLevel: (value) => setLevel(value),
        onState: (state) => {
          if (!busy.current) setStatus(state === "hearing" ? "hearing" : "listening");
        },
        onUtterance: (utterance) => {
          runTurn({ audio: utterance });
        },
        options: {
          threshold: Math.max(0.001, parseNumber(threshold, 0.02)),
          silenceMs: Math.max(200, parseInteger(silenceMs, 700)),
        },
      });
    } catch (err) {
      listener.current = null;
      setStatus("off");
      setOn(false);
      setError(`Cannot listen: ${err.message}`);
      return;
    }
    dictation.current = startDictationLog({ onInterim: (text) => setInterim(text) });
    if (!player.current) player.current = new PcmPlayer();
    try {
      player.current.ensure(); // made on the click, so the reply may sound without another one
    } catch {
      // the note shows when a reply cannot be played
    }
    setOn(true);
    setStatus("listening");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold, silenceMs]);

  const stopEar = useCallback(() => {
    if (listener.current) {
      listener.current.stop();
      listener.current = null;
    }
    if (dictation.current) {
      dictation.current.stop();
      dictation.current = null;
    }
    if (player.current) player.current.stop();
    setLevel(0);
    setInterim("");
    setOn(false);
    setStatus("off");
  }, [setOn]);

  // the ear was on last time: try to open it again without a click (a browser that wants one refuses,
  // and the button is there)
  useEffect(() => {
    if (on && canListen && !listener.current) {
      startEar().catch(() => {});
    }
    return () => {
      if (listener.current) listener.current.stop();
      if (dictation.current) dictation.current.stop();
      if (player.current) player.current.stop();
      listener.current = null;
      dictation.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggle() {
    if (listener.current) stopEar();
    else startEar();
  }

  function sendTyped(event) {
    event.preventDefault();
    const text = typed.trim();
    if (!text || busy.current) return;
    setTyped("");
    if (!player.current) player.current = new PcmPlayer();
    runTurn({ text });
  }

  const phaseText = STATUS_TEXT[phase] || phase;
  const meter = Math.min(100, Math.round(level * 500));
  const shown = [...turns].reverse(); // oldest first, the newest at the bottom, as a conversation reads
  const who = info
    ? `${info.encoding} through its ${info.decoder}${info.ollama ? ` · Ollama: ${info.ollama.model}` : ""}`
    : "";

  return (
    <>
      <nav className="tabs sub" role="tablist" aria-label="Voice">
        {[
          ["talk", "Conversation"],
          ["settings", "Settings"],
        ].map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={view === id}
            className={view === id ? "active" : ""}
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {view === "settings" ? (
        <div className="card voice-settings">
          <h2>Settings</h2>
          <p className="muted">
            The microphone stays on: everything you say is cut into utterances, written down with the browser&apos;s
            dictation (a sound with no words in it is dropped), <b>learned</b> - the words and the waveform, or a model
            of acoustic units&apos; units - and <b>answered</b>, and the reply is <b>spoken</b> back the moment its first
            chunk of audio arrives. While it speaks it does not listen, so it never hears itself. Ollama can answer
            for a model that has nothing to say yet, and the model learns those answers too.
          </p>
          <div className="grid">
            <SelectField label="Answered by" value={answer} onChange={setAnswer} options={ANSWERS} />
            <TextField label="Ollama model" value={ollamaModel} onChange={setOllamaModel} placeholder="(the server's default)" />
            <TextField label="Ollama URL" value={url} onChange={setUrl} placeholder="(the server's default)" />
            <TextField label="Persona (for Ollama)" value={persona} onChange={setPersona} placeholder="a friendly cat" />
            <CheckField label="Learn what I say" checked={train} onChange={setTrain} hint="the words and the waveform, trained on before the reply" />
            <NumberField label="Epochs" value={epochs} onChange={setEpochs} step="1" min="0" />
            <CheckField label="Learn Ollama's replies" checked={learnReply} onChange={setLearnReply} hint="so the model learns to answer by itself" />
            <CheckField label="Send only what has words" checked={wordsOnly} onChange={setWordsOnly} hint="a sound the dictation heard no words in is dropped; off, it is sent and the sound learned unheard" />
            <CheckField label="Speak the replies" checked={speak} onChange={setSpeak} />
            <SelectField label="Reply search" value={mode} onChange={setMode} options={[["beam", "beam (the most likely)"], ["sample", "sample (a walk)"]]} />
            <NumberField label="Reply length" value={maxLength} onChange={setMaxLength} step="1" min="0" hint={`${units} a reply may add`} />
            <NumberField label="Pitch (Hz)" value={pitch} onChange={setPitch} step="1" min="1" />
            <NumberField label="Tempo" value={tempo} onChange={setTempo} step="0.1" min="0.1" />
            <NumberField label="Gain" value={gain} onChange={setGain} step="0.05" min="0" max="1" />
            <NumberField label="Silence that ends an utterance (ms)" value={silenceMs} onChange={setSilenceMs} step="50" min="200" hint="takes effect when listening is started again" />
            <NumberField label="Sensitivity (level)" value={threshold} onChange={setThreshold} step="0.005" min="0.001" max="0.5" hint="lower hears quieter speech; takes effect when listening is started again" />
          </div>
          <p className="muted">
            {info
              ? `The model speaks ${info.encoding} through its ${info.decoder}${info.ollama ? `; Ollama: ${info.ollama.model} at ${info.ollama.url}` : ""}.`
              : ""}
          </p>
        </div>
      ) : (
        <div className="card voice-room">
          <div className="voice-bar">
            <button type="button" className={`primary mic${on ? " on" : ""}`} onClick={toggle} disabled={!canListen && !on}>
              {on ? "Stop listening" : "Start listening"}
            </button>
            <span className={`badge voice-status ${phase}`}>{phaseText}</span>
            <span className="level" role="meter" aria-label="microphone level" aria-valuenow={meter} aria-valuemin={0} aria-valuemax={100}>
              <span style={{ width: `${meter}%` }} />
            </span>
            {interim ? <span className="muted interim">{interim}</span> : null}
            {who ? <span className="muted voice-who">{who}</span> : null}
          </div>
          {!canListen ? (
            <Alert
              kind="warning"
              message="This browser cannot listen (no microphone access or no Web Audio); type below instead."
            />
          ) : null}
          {canListen && !canDictate ? (
            <p className="muted">
              This browser has no dictation, so the words come from the server&apos;s own transcription when it has one
              {info && info.transcription && info.transcription.auto ? ` (${info.transcription.auto})` : ""}; the sound is
              learned either way.
            </p>
          ) : null}
          {otherJobRunning ? <p className="muted">A job is running: a turn cannot train until it has finished.</p> : null}
          <Alert message={error} onDismiss={() => setError(null)} />
          <Alert kind="warning" message={note} onDismiss={() => setNote(null)} />
          <div className="voice-log" ref={log} onScroll={onScroll}>
            {turns.length === 0 ? (
              <p className="muted empty">
                {on
                  ? "Listening - say something."
                  : "Press Start listening, then just talk - or type below. Everything you say is learned and answered out loud."}
              </p>
            ) : (
              <ol className="dialogue voice">
                {shown.map((t) => (
                  <VoiceTurn key={t.id} turn={t} />
                ))}
              </ol>
            )}
          </div>
          <form className="voice-composer" onSubmit={sendTyped}>
            <input
              type="text"
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              placeholder="Or type something and press Enter"
              aria-label="Say something by typing"
            />
            <button type="submit" className="primary" disabled={!typed.trim()}>
              Say it
            </button>
          </form>
        </div>
      )}
    </>
  );
}

/** One turn: what you said (left) and what came back (right), newest turn first. */
function VoiceTurn({ turn }) {
  const you = turn.you || {};
  const model = turn.model;
  const learned = turn.trained ? `learned ${fmtInt(turn.trained.texts)} text(s)` : null;
  const youText = you.text || (you.heard === false ? (turn.phase === "thinking" ? "…" : "(no words were heard)") : "");
  return (
    <>
      <li className={`turn a${turn.error ? " failed" : ""}`}>
        <div className="speaker">
          <b>You</b>
          {you.typed ? <span className="badge">typed</span> : null}
          {you.units ? <span className="badge">units</span> : null}
        </div>
        <p className="bubble">{youText}</p>
        {you.units ? <p className="meta units">{you.units}</p> : null}
        <div className="meta">
          {you.seconds != null ? `${seconds(you.seconds)} · ` : ""}
          {learned || (turn.phase === "thinking" ? "hearing…" : "not learned")}
          {you.token ? ` · ${you.token}` : ""}
          {turn.error ? ` · ${turn.error}` : ""}
        </div>
      </li>
      {model || turn.phase === "thinking" ? (
        <li className={`turn b${turn.phase === "thinking" && !model ? " live" : ""}`}>
          <div className="speaker">
            <b>Model</b>
            {model && model.by === "ollama" ? <span className="badge up">ollama</span> : null}
            {model && model.by === "model" && model.turn && model.turn.fresh ? <span className="badge">new topic</span> : null}
            {turn.taught ? <span className="badge">taught</span> : null}
          </div>
          <p className={`bubble${turn.phase === "thinking" && !model ? " pending" : ""}`}>
            {model ? model.spelled || model.text || (model.error ? `(nothing: ${model.error})` : "(nothing to say)") : ""}
          </p>
          {model && model.text && model.spelled && model.spelled !== model.text ? (
            <p className="meta units">{model.text}</p>
          ) : null}
          {model ? (
            <div className="meta">
              {model.by === "model" && model.turn ? `cost ${fmtNum(model.turn.cost, 3)} · p ${fmtNum(model.turn.probability, 4)} · ` : ""}
              {turn.spoken ? `spoke ${seconds(turn.spoken.seconds)} through the ${turn.spoken.decoder}` : "not spoken"}
              {model.error && model.text ? ` · Ollama: ${model.error}` : ""}
              {model.text ? <HearButton text={model.text} label="this reply" compact /> : null}
            </div>
          ) : null}
        </li>
      ) : null}
    </>
  );
}
