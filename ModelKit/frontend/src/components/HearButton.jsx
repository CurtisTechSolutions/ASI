import { useState } from "react";
import { api } from "../api.js";

/**
 * 🔊 Hear: the output decoder. The text - a prediction, a sample, a turn, in
 * whatever units the model is in - is sent to `POST /api/say`, spoken through
 * the model's voice (the formant synthesizer, or the acoustic codebook's
 * vocoder) and played back; the player stays so it can be replayed. Every
 * output the frontend shows can be heard this way, after the fact, exactly as
 * `speak` hears a walk while it walks.
 */
export default function HearButton({ text, label, compact = false, disabled }) {
  const [busy, setBusy] = useState(false);
  const [spoken, setSpoken] = useState(null); // {src, seconds, decoder, vocoder, encoding}
  const [error, setError] = useState(null);
  const empty = !String(text ?? "").trim();

  async function hear() {
    setBusy(true);
    setError(null);
    try {
      const data = await api.say([text]);
      if (!data || !data.wav_base64) {
        throw new Error("the server sent no audio back");
      }
      setSpoken({
        src: `data:audio/wav;base64,${data.wav_base64}`,
        seconds: Number(data.seconds) || 0,
        decoder: data.decoder,
        vocoder: data.vocoder || null,
        encoding: data.encoding,
      });
    } catch (err) {
      setSpoken(null);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const through = spoken ? `${spoken.decoder}${spoken.vocoder ? ` (${spoken.vocoder})` : ""}` : "";
  const title = spoken
    ? `spoken through the ${through} (${spoken.encoding}): ${spoken.seconds.toFixed(2)} s`
    : "Hear: the text spoken through the model's voice (the output decoder, POST /api/say)";
  return (
    <span className="hear">
      <button
        type="button"
        className="rate hear"
        aria-label={`hear ${label}`}
        title={title}
        disabled={empty || disabled || busy}
        onClick={hear}
      >
        🔊{compact ? "" : busy ? " Speaking…" : " Hear"}
      </button>
      {spoken ? <audio controls autoPlay src={spoken.src} aria-label={`${label}, spoken`} /> : null}
      {spoken && !compact ? (
        <span className="muted">
          {spoken.seconds.toFixed(2)} s · {through}
        </span>
      ) : null}
      {error ? <span className="error-text">{error}</span> : null}
    </span>
  );
}
