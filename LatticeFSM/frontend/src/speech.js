/**
 * The browser's own voice: speech synthesis to read a text out (the 🔊 buttons, the Voice tab's replies) and
 * speech recognition to hear what is said (the Voice tab). Nothing goes to the server. Every function takes
 * the global object it looks in, so `test/speech.test.mjs` can hand it a stand-in.
 */

/** What this browser can do: `{synthesis, Recognition}` - the recogniser's constructor, or null. */
export function speechSupport(g = globalThis) {
  const scope = g || {};
  const synthesis = Boolean(scope.speechSynthesis && typeof scope.SpeechSynthesisUtterance === "function");
  const Recognition = scope.SpeechRecognition || scope.webkitSpeechRecognition || null;
  return { synthesis, Recognition: typeof Recognition === "function" ? Recognition : null };
}

/** Whether a text is worth reading out: something besides spaces and symbols, and not base64. */
export function speakable(text, unit) {
  if (unit === "base64") return false;
  return /[\p{L}\p{N}]/u.test(String(text ?? ""));
}

/** A text as it is read out: a model of letters writes `_` for a space, and the end of a text is silent. */
export function spokenText(text) {
  return String(text ?? "")
    .replace(/<\/s>/g, " ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Read `text` out; `onEnd` is called when it has been said (or could not be). Answers whether it started.
 * Whatever was being said before is cut off, so a second press does not queue a second reading.
 */
export function speak(text, { lang, rate = 1, onEnd, g = globalThis } = {}) {
  const { synthesis } = speechSupport(g);
  const words = spokenText(text);
  if (!synthesis || !words) {
    if (onEnd) onEnd();
    return false;
  }
  const utterance = new g.SpeechSynthesisUtterance(words);
  if (lang) utterance.lang = lang;
  utterance.rate = rate;
  utterance.onend = () => onEnd && onEnd();
  utterance.onerror = () => onEnd && onEnd();
  g.speechSynthesis.cancel();
  g.speechSynthesis.speak(utterance);
  return true;
}

/** Stop reading out. */
export function hush(g = globalThis) {
  if (speechSupport(g).synthesis) g.speechSynthesis.cancel();
}

/**
 * A recogniser's results event -> `{final, interim}`: the phrases it committed in this event (from its `resultIndex` on) and the
 * one it is still unsure of.
 */
export function recognitionText(event) {
  let final = "";
  let interim = "";
  const results = event && event.results ? event.results : [];
  for (let i = Number(event && event.resultIndex) || 0; i < results.length; i += 1) {
    const result = results[i];
    const best = result && result[0] ? String(result[0].transcript ?? "") : "";
    if (result && result.isFinal) final += best;
    else interim += best;
  }
  return { final: final.trim(), interim: interim.trim() };
}
