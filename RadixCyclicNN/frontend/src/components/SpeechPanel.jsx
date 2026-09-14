import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { dictationSupported, micSupported, peakOf, startDictation, startRecording, toWav } from "../audio.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import RecallCard from "./RecallCard.jsx";
import { CheckField, NumberField, SelectField, TextArea } from "./Fields.jsx";

const RATES = [
  ["4000", "4 kHz (half the text, telephone-ish)"],
  ["8000", "8 kHz (default, telephony quality)"],
  ["16000", "16 kHz (twice the text, wideband)"],
];
const CODECS = [
  ["auto", "auto (mu-law)"],
  ["mu", "mu-law (256 levels, best for speech)"],
  ["pcm8", "linear 8-bit"],
];
const RECORD_RATE = 16000;

function seconds(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "–";
}

/**
 * Teach the model by talking to it.
 *
 * The browser records the microphone and dictates the words with the Web Speech
 * API; the recording is turned into a 16-bit PCM WAV here and posted to
 * POST /api/speech/teach, which transcribes it (when the browser could not) and
 * turns one utterance into two texts behind the same unique token - the words
 * and the waveform itself, quantised to one byte per sample and base64-encoded.
 * Training on both is what ties what was said to how it sounded.
 */
export default function SpeechPanel({ status }) {
  const [info, setInfo] = useState(null);
  const [infoError, setInfoError] = useState(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [clip, setClip] = useState(null); // {blob, seconds, rate, name, peak}
  const [clipUrl, setClipUrl] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [heard, setHeard] = useState(false);
  const [rate, setRate] = useState("8000");
  const [codec, setCodec] = useState("auto");
  const [pair, setPair] = useState(false);
  const [waveform, setWaveform] = useState(true);
  const [unique, setUnique] = useState(true);
  const [normalise, setNormalise] = useState(false);
  const [epochs, setEpochs] = useState("3");
  const [lr, setLr] = useState("0.5");
  const [batchSize, setBatchSize] = useState("8");
  const [saveName, setSaveName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [decodeText, setDecodeText] = useState("");
  const [decoded, setDecoded] = useState(null);
  const [decodeError, setDecodeError] = useState(null);
  const [decoding, setDecoding] = useState(false);
  const recorder = useRef(null);
  const dictation = useRef(null);
  const ticker = useRef(null);
  const fileInput = useRef(null);
  const { job, running, busy: starting, error: jobError, start, stop, clearError } = useJob("train");

  const otherJobRunning = jobIsRunning(status) && !running;
  const canRecord = micSupported();
  const canDictate = dictationSupported();

  useEffect(() => {
    let alive = true;
    api
      .speech()
      .then((data) => {
        if (!alive) return;
        setInfo(data && typeof data === "object" ? data : null);
        if (data && data.default_rate) setRate(String(data.default_rate));
      })
      .catch((err) => {
        if (alive) setInfoError(err.message);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!clip) {
      setClipUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(clip.blob);
    setClipUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [clip]);

  useEffect(
    () => () => {
      if (ticker.current) clearInterval(ticker.current);
      if (recorder.current) recorder.current.cancel();
      if (dictation.current) dictation.current.stop();
    },
    [],
  );

  const useClip = useCallback(async (blob, name) => {
    const peak = await peakOf(blob);
    setClip({ blob, name, seconds: null, rate: null, peak, recorded: false });
    setResult(null);
  }, []);

  async function record() {
    setError(null);
    setResult(null);
    try {
      recorder.current = await startRecording();
    } catch (err) {
      setError(`Cannot use the microphone: ${err.message}`);
      return;
    }
    setHeard(false);
    setTranscript("");
    dictation.current = startDictation((text) => {
      setTranscript(text);
      setHeard(true);
    });
    setRecording(true);
    setElapsed(0);
    const started = Date.now();
    ticker.current = setInterval(() => setElapsed((Date.now() - started) / 1000), 200);
  }

  async function finish() {
    if (!recorder.current) return;
    if (ticker.current) clearInterval(ticker.current);
    ticker.current = null;
    setRecording(false);
    if (dictation.current) {
      dictation.current.stop();
      dictation.current = null;
    }
    let blob;
    try {
      blob = await recorder.current.stop();
    } catch (err) {
      setError(`The recording failed: ${err.message}`);
      return;
    } finally {
      recorder.current = null;
    }
    try {
      const wav = await toWav(blob, RECORD_RATE);
      const peak = await peakOf(wav.blob);
      setClip({ blob: wav.blob, name: "utterance.wav", seconds: wav.seconds, rate: wav.rate, peak, recorded: true });
      setResult(null);
    } catch (err) {
      setError(`The recording could not be converted: ${err.message}`);
    }
  }

  /** Everything POST /api/speech/teach needs; `train` adds the training job's settings. */
  function teachOptions(train) {
    const options = {
      transcript: transcript.trim(),
      rate: parseInteger(rate, 8000),
      codec,
      pair,
      waveform,
      unique,
      normalise,
      save_as: saveName.trim() || undefined,
    };
    if (!train) return options;
    return {
      ...options,
      train: true,
      epochs: parseInteger(epochs, 3),
      lr: parseNumber(lr, 0.5),
      batch_size: parseInteger(batchSize, 8),
    };
  }

  /** Build the texts and show them, without touching the model. */
  async function preview() {
    if (!clip) return;
    setError(null);
    setBusy(true);
    try {
      setResult(await api.speechTeach(clip.blob, teachOptions(false)));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  /** Teach: the same request, with a training job on the texts it answers with. */
  async function teachAndTrain() {
    if (!clip) return;
    setError(null);
    await start(async () => {
      const data = await api.speechTeach(clip.blob, teachOptions(true));
      setResult(data);
      return data;
    });
  }

  async function decode(text) {
    if (!String(text || "").trim()) return;
    setDecoding(true);
    setDecodeError(null);
    try {
      const data = await api.speechDecode(text);
      setDecoded(data && typeof data === "object" ? data : null);
    } catch (err) {
      setDecoded(null);
      setDecodeError(err.message);
    } finally {
      setDecoding(false);
    }
  }

  const texts = asArray(result && result.texts);
  const asr = result && result.asr ? result.asr : null;
  const audio = result && result.audio ? result.audio : null;
  const quiet = clip && typeof clip.peak === "number" && clip.peak < 0.02;

  return (
    <>
      <div className="card">
        <h2>Speak to the model</h2>
        <p className="muted">
          One utterance becomes two texts that start with the <b>same unique token</b> - the words and the waveform
          itself: <code>&lt;speech:9f2a1c7d&gt; the cat sat on the mat</code> and{" "}
          <code>&lt;speech:9f2a1c7d&gt; aud:mu:8000x1:…</code> (every sample quantised to one mu-law byte, base64). The
          token is derived from the recording, so both views leave the same node of the graph and the model learns what
          was said together with how it sounded.
        </p>
        {infoError ? <Alert message={`Cannot read the speech status: ${infoError}`} /> : null}
        {info ? (
          <p className="muted">
            transcription: browser {canDictate ? "✓" : "✗"} · faster-whisper {info.faster_whisper ? "✓" : "✗"} · whisper{" "}
            {info.whisper ? "✓" : "✗"} · server {info.server_url ? <code>{info.server_url}</code> : "✗"} · server-side
            auto → <b>{info.auto || "none (the browser or the text box supplies the words)"}</b> · ffmpeg{" "}
            {info.ffmpeg ? "✓" : "✗"}
          </p>
        ) : null}
        {canRecord ? null : (
          <Alert message="This browser cannot record audio (no MediaRecorder); choose an audio file instead." />
        )}
        <div className="actions">
          {recording ? (
            <button type="button" className="danger" onClick={finish}>
              Stop recording ({seconds(elapsed)})
            </button>
          ) : (
            <button type="button" className="primary" disabled={!canRecord || busy} onClick={record}>
              {clip && clip.recorded ? "Record again" : "Record"}
            </button>
          )}
          <input
            ref={fileInput}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => {
              const chosen = Array.from(e.target.files || [])[0];
              if (chosen) useClip(chosen, chosen.name);
              e.target.value = "";
            }}
          />
          <button type="button" className="small" disabled={recording} onClick={() => fileInput.current && fileInput.current.click()}>
            Choose an audio file…
          </button>
          {clip ? (
            <button type="button" className="small" disabled={recording} onClick={() => { setClip(null); setResult(null); }}>
              Discard
            </button>
          ) : null}
        </div>
        {recording ? (
          <p className="muted">
            Listening… {canDictate ? "the browser is writing down what it hears." : "this browser does not transcribe; type what you said below (or let the server's Whisper do it)."}
          </p>
        ) : null}
        <Alert message={error} onDismiss={() => setError(null)} />
        {clip ? (
          <>
            <p className="muted">
              {clip.name} · {clip.seconds ? `${seconds(clip.seconds)} at ${fmtInt(clip.rate)} Hz mono` : "as it was uploaded"}
              {typeof clip.peak === "number" ? ` · peak ${fmtNum(clip.peak, 2)}` : ""}
            </p>
            {clipUrl ? <audio controls src={clipUrl} /> : null}
            {quiet ? (
              <p className="muted">
                That is very quiet - tick “normalise” below, or record closer to the microphone.
              </p>
            ) : null}
          </>
        ) : null}
        <TextArea
          label="What you said"
          hint={heard ? "dictated by the browser - correct it if it misheard" : "leave it empty to let the server transcribe"}
          value={transcript}
          onChange={setTranscript}
          rows={3}
          placeholder="the cat sat on the mat"
        />
      </div>

      <div className="card">
        <h2>Teach it</h2>
        <div className="row">
          <SelectField label="Waveform rate" hint="samples per second" value={rate} onChange={setRate} options={RATES} disabled={running} />
          <SelectField label="Codec" value={codec} onChange={setCodec} options={CODECS} disabled={running} />
        </div>
        <div className="row">
          <CheckField label="Train on the waveform too" checked={waveform} onChange={setWaveform} disabled={running} />
          <CheckField label="Also learn waveform → transcript" checked={pair} onChange={setPair} disabled={running} />
          <CheckField label="Unique token per utterance" checked={unique} onChange={setUnique} disabled={running} />
          <CheckField label="Normalise the volume" checked={normalise} onChange={setNormalise} disabled={running} />
        </div>
        <div className="row">
          <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} step={1} disabled={running} />
          <NumberField label="Learning rate" value={lr} onChange={setLr} min={0} disabled={running} />
          <NumberField label="Batch size" value={batchSize} onChange={setBatchSize} min={1} step={1} disabled={running} />
          <label className="field">
            <span>Save as upload</span>
            <input type="text" value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="utterance.txt" disabled={running} />
          </label>
        </div>
        <div className="actions">
          <button type="button" className="primary" disabled={!clip || running || starting || busy || otherJobRunning} onClick={teachAndTrain}>
            {starting ? "Starting…" : "Teach the model"}
          </button>
          <button type="button" className="small" disabled={!clip || busy || running} onClick={preview}>
            {busy ? "Encoding…" : "Preview the texts"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <JobStatus job={job} emptyText="Nothing taught yet. Record something, then press Teach the model." />
        <Alert message={jobError} onDismiss={clearError} />
        {result ? (
          <>
            <dl className="kv">
              <dt>token</dt>
              <dd>
                <code>{String(result.token)}</code>
              </dd>
              <dt>transcript</dt>
              <dd>{result.transcript || <em>none</em>}</dd>
              <dt>transcribed by</dt>
              <dd>
                {asr && asr.backend ? asr.backend : "–"}
                {asr && asr.model ? ` (${asr.model})` : ""}
                {asr && asr.error ? ` · ${asr.error}` : ""}
              </dd>
              <dt>waveform</dt>
              <dd>
                {audio
                  ? `${audio.codec} · ${fmtInt(audio.rate)} Hz · ${seconds(audio.seconds)} · ${fmtInt(audio.bytes)} bytes · ${fmtInt(audio.chars)} chars`
                  : "not encoded"}
              </dd>
              <dt>texts</dt>
              <dd>
                {fmtInt(texts.length)} ({fmtInt(result.chars)} chars)
              </dd>
              {result.upload ? (
                <>
                  <dt>upload</dt>
                  <dd>{String(result.upload.name)}</dd>
                </>
              ) : null}
            </dl>
            {texts.map((text, index) => (
              <TextArea
                key={index}
                label={index === 0 && result.transcript ? "Text 1: the words" : `Text ${index + 1}: the waveform${index === 2 ? " → the words" : ""}`}
                value={text.length > 4000 ? `${text.slice(0, 4000)}… (+${fmtInt(text.length - 4000)} chars)` : text}
                onChange={() => {}}
                rows={index === 0 && result.transcript ? 2 : 4}
              />
            ))}
            {texts.length > 1 ? (
              <div className="actions">
                <button type="button" className="small" onClick={() => { setDecodeText(texts[1]); decode(texts[1]); }}>
                  Listen to the encoded waveform
                </button>
              </div>
            ) : null}
          </>
        ) : null}
      </div>

      <RecallCard
        modality="speech"
        disabled={!clip}
        run={(options) =>
          api.speechTutor(clip.blob, {
            transcript: transcript.trim(),
            rate: parseInteger(rate, 8000),
            codec,
            unique,
            normalise,
            ...options,
          })
        }
      />

      <div className="card">
        <h2>Listen to a waveform text</h2>
        <p className="muted">
          Anything the model predicts that carries <code>aud:&lt;codec&gt;:&lt;rate&gt;x1:…</code> can be played back:
          paste a prediction (Predict tab, prefix the token) and hear what the graph thinks the sound is. A cut-off tail
          is padded.
        </p>
        <TextArea
          label="Encoded or predicted text"
          value={decodeText}
          onChange={setDecodeText}
          rows={4}
          placeholder="&lt;speech:9f2a1c7d&gt; aud:mu:8000x1:gOXv7NgqFBAYTePu…"
        />
        <div className="actions">
          <button type="button" className="primary" disabled={decoding || !decodeText.trim()} onClick={() => decode(decodeText)}>
            {decoding ? "Decoding…" : "Decode"}
          </button>
        </div>
        <Alert message={decodeError} onDismiss={() => setDecodeError(null)} />
        {decoded && decoded.wav_base64 ? (
          <>
            <p className="muted">
              {decoded.codec} · {fmtInt(decoded.rate)} Hz · {seconds(decoded.seconds)} · {fmtInt(decoded.samples)} samples
              {decoded.repaired ? " · the text was repaired (padded or truncated)" : ""}
            </p>
            <audio controls src={`data:audio/wav;base64,${decoded.wav_base64}`} />
          </>
        ) : null}
      </div>
    </>
  );
}
