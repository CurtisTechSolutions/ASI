import { useEffect, useRef, useState } from "react";
import { useStoredState } from "../hooks/useStoredState.js";
import { useLmInfo } from "../hooks/useLmInfo.js";
import { MAX_TEMPERATURE, lmLabel } from "../lm.js";
import { hush, recognitionText, speak, speechSupport } from "../speech.js";
import { parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, TextField } from "./Fields.jsx";
import { AutoTempFields } from "./AutoTemperature.jsx";
import { TalkLog, useTalk } from "./TalkPanel.jsx";

/**
 * Talk to the chosen language model by voice, all in the browser: press the microphone and speak - the
 * browser's speech recognition writes down what you said, it goes to the model as in the Talk tab (its reply
 * is its continuation of your words), and the reply is read out by the browser's speech synthesis. While it
 * speaks the microphone waits, so it never hears itself. A text box does the same without a microphone; a
 * browser without recognition or synthesis is told so.
 */
export default function VoicePanel({ lm, tick, onMoved }) {
  const [info] = useLmInfo(lm, tick);
  const [lang, setLang] = useStoredState("voice.lang", "en-US");
  const [maxTokens, setMaxTokens] = useStoredState("voice.max", "30");
  const [temperature, setTemperature] = useStoredState("voice.temperature", "0");
  const [learn, setLearn] = useStoredState("voice.learn", false);
  const [auto, setAuto] = useStoredState("voice.auto", true);
  const [autoTop, setAutoTop] = useStoredState("voice.autoTop", String(MAX_TEMPERATURE));
  const [speakReplies, setSpeakReplies] = useStoredState("voice.speak", true);
  const [draft, setDraft] = useState("");
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [interim, setInterim] = useState("");
  const [micError, setMicError] = useState(null);
  const talk = useTalk(lm, onMoved);
  const recogniser = useRef(null);
  const wanted = useRef(false); // the microphone should be on (a recogniser stops itself now and then)
  const support = speechSupport();
  const sendRef = useRef(null);

  async function send(text) {
    const said = await talk.say(text, {
      learn,
      maxTokens: Math.max(1, parseInteger(maxTokens, 30)),
      temperature: parseNumber(temperature, 0),
      auto,
      maxTemperature: parseNumber(autoTop, MAX_TEMPERATURE),
    });
    if (said && speakReplies && support.synthesis && said.reply) {
      // the ear closes while the mouth speaks, and opens again after
      const resume = wanted.current;
      const ear = recogniser.current;
      recogniser.current = null; // so its end does not start it again while the reply is read out
      if (ear) ear.abort();
      setSpeaking(true);
      speak(said.reply, {
        lang,
        onEnd: () => {
          setSpeaking(false);
          if (resume && wanted.current) start();
        },
      });
    }
    return said;
  }
  sendRef.current = send;

  function start() {
    if (!support.Recognition) return;
    try {
      const r = new support.Recognition();
      r.lang = lang;
      r.continuous = true;
      r.interimResults = true;
      r.onresult = (event) => {
        const { final, interim: unsure } = recognitionText(event);
        setInterim(unsure);
        if (final) {
          setInterim("");
          sendRef.current(final);
        }
      };
      r.onerror = (event) => {
        if (event.error === "no-speech" || event.error === "aborted") return;
        setMicError(event.error === "not-allowed" ? "the microphone was not allowed" : `speech recognition: ${event.error}`);
        wanted.current = false;
      };
      r.onend = () => {
        setListening(false);
        // a recogniser ends itself after a silence: start again while the microphone is wanted and nobody speaks
        if (wanted.current && recogniser.current === r) {
          try {
            r.start();
            setListening(true);
          } catch {
            // already started again
          }
        }
      };
      recogniser.current = r;
      r.start();
      setListening(true);
      setMicError(null);
    } catch (err) {
      setMicError(err.message);
    }
  }

  function toggle() {
    if (wanted.current) {
      wanted.current = false;
      if (recogniser.current) recogniser.current.abort();
      recogniser.current = null;
      setListening(false);
      setInterim("");
      return;
    }
    wanted.current = true;
    start();
  }

  // leaving the page stops the microphone and the voice
  useEffect(
    () => () => {
      wanted.current = false;
      if (recogniser.current) recogniser.current.abort();
      hush();
    },
    [],
  );

  const status = speaking ? "speaking" : talk.busy ? "thinking" : listening ? (interim ? "hearing" : "listening") : "off";
  const maxDepth = info && info.nodes ? info.nodes - 1 : null;
  return (
    <div className="panel single">
      <div className="card wide voice-room">
        <h2>Voice</h2>
        <p className="muted">
          Speak, and the {lmLabel(lm)} model answers with its continuation of what you said, read out loud. All of it
          runs in this browser (its speech recognition and speech synthesis) - only the words go to the server.
        </p>
        {!support.Recognition ? (
          <p className="alert info">
            This browser has no speech recognition (Chrome, Edge and Safari have it; Firefox does not) - type below
            instead.
          </p>
        ) : null}
        {!support.synthesis ? <p className="alert info">This browser cannot speak: the replies are only written.</p> : null}
        <div className="row">
          <TextField label="Language" value={lang} onChange={setLang} hint="what you speak, e.g. en-GB" />
          <NumberField label="Max tokens" value={maxTokens} onChange={setMaxTokens} min="1" step="1" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" />
          <AutoTempFields on={auto} onChange={setAuto} top={autoTop} onTop={setAutoTop} />
          <CheckField label="Learn what I say" checked={learn} onChange={setLearn} />
          <CheckField label="Speak the replies" checked={speakReplies} onChange={setSpeakReplies} disabled={!support.synthesis} />
        </div>
        <div className="voice-bar">
          <button
            type="button"
            className={`mic${listening ? " on" : ""}`}
            disabled={!support.Recognition}
            onClick={toggle}
            aria-pressed={listening}
          >
            {listening ? "🎙 Listening - press to stop" : "🎙 Start listening"}
          </button>
          <span className={`voice-status ${status}`}>{status === "off" ? "microphone off" : status}</span>
          {interim ? <span className="interim muted">{interim}…</span> : null}
          {speaking ? (
            <button
              type="button"
              className="small"
              onClick={() => {
                hush();
                setSpeaking(false);
              }}
            >
              Hush
            </button>
          ) : null}
          <button type="button" className="small voice-who" disabled={talk.busy || !talk.turns.length} onClick={talk.clear}>
            Clear
          </button>
        </div>
        <Alert message={micError} onDismiss={() => setMicError(null)} />
        <TalkLog
          turns={talk.turns}
          busy={talk.busy}
          onRate={talk.rate}
          maxDepth={lm === "chain" ? maxDepth : null}
          empty={support.Recognition ? "Press Start listening and say something." : "Type something below."}
        />
        <Alert message={talk.error} onDismiss={() => talk.setError(null)} />
        <div className="voice-composer">
          <input
            value={draft}
            placeholder="or type it"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={async (e) => {
              if (e.key === "Enter" && draft.trim() && !talk.busy) {
                const text = draft;
                setDraft("");
                await send(text);
              }
            }}
          />
          <button
            type="button"
            className="primary"
            disabled={talk.busy || !draft.trim()}
            onClick={async () => {
              const text = draft;
              setDraft("");
              await send(text);
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
