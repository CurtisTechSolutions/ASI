import { useEffect, useRef, useState } from "react";
import { hush, speak, speakable, speechSupport } from "../speech.js";

/**
 * 🔊 Hear: the text - a sample, a reply, a turn - read out by the browser's own speech synthesis. Nothing goes
 * to the server. Pressed while it speaks, it stops. Base64 and texts with nothing to say leave it disabled,
 * and a browser without speech synthesis shows no button at all.
 */
export default function HearButton({ text, label = "the text", unit, compact = true, disabled }) {
  const [speaking, setSpeaking] = useState(false);
  const speakingRef = useRef(false);
  const { synthesis } = speechSupport();
  speakingRef.current = speaking;

  // leaving the page (or the tab's panel unmounting) while it speaks stops the voice
  useEffect(() => () => (speakingRef.current ? hush() : undefined), []);

  if (!synthesis) return null;
  const can = speakable(text, unit);
  return (
    <span className="hear">
      <button
        type="button"
        className="rate hear"
        aria-label={`hear ${label}`}
        title={speaking ? "Stop" : "Hear it, read out by this browser"}
        disabled={disabled || !can}
        onClick={() => {
          if (speaking) {
            hush();
            setSpeaking(false);
            return;
          }
          setSpeaking(speak(text, { onEnd: () => setSpeaking(false) }));
        }}
      >
        {speaking ? "⏹" : "🔊"}
        {compact ? "" : speaking ? " Stop" : " Hear"}
      </button>
    </span>
  );
}
