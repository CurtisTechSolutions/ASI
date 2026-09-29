/**
 * JSON Lines as they arrive (DESIGN.md section 22: a streamed conversation).
 *
 * A streamed answer is one JSON object per line, and a network chunk ends
 * wherever it ends - mid-line as often as not. The parser is fed the chunks
 * as text and hands back every whole line, parsed; the unfinished tail waits
 * for its newline. Pure, so it is tested on its own (test/stream.test.mjs).
 */
export class LineParser {
  constructor() {
    this.rest = "";
  }

  /** Feed one chunk of text; returns the objects of every line it completed, in order. */
  feed(text) {
    const parts = (this.rest + text).split("\n");
    this.rest = parts.pop();
    return parts.filter((line) => line.trim()).map((line) => JSON.parse(line));
  }

  /** The end of the stream: whatever is left is the last line, newline or not. */
  end() {
    const last = this.rest.trim();
    this.rest = "";
    return last ? [JSON.parse(last)] : [];
  }
}

/** Every kind of event a streamed conversation emits, in the order a turn goes through them. */
export const STREAM_EVENTS = ["look", "draft", "caught", "backtrack", "found", "stuck", "turn"];

/**
 * What one turn of a streamed conversation looks like while it is being spoken: the window a backtrack may
 * still rewrite. `apply(live, event)` folds one event into it and returns the new state (never mutating the
 * old one, so React can tell); `null` comes back for a turn event, which is committed and leaves the window.
 */
export function applyEvent(live, event) {
  const kind = event.event;
  if (kind === "turn") return null;
  const fresh = !live || live.index !== event.index;
  const next = fresh
    ? { index: event.index, speaker: event.speaker, from: "", draft: "", kept: "", retracted: "", found: "", caught: null, step: 0, note: "" }
    : { ...live };
  switch (kind) {
    case "look":
      next.from = event.from;
      next.note = event.from ? `continuing “${event.from}”` : "starting a fresh text";
      break;
    case "draft":
      next.draft = event.text;
      next.kept = event.text;
      next.retracted = "";
      next.found = "";
      next.caught = null;
      next.step = 0;
      next.note = "about to say it";
      break;
    case "caught": {
      const what = event.kind === "stutter" ? `caught itself saying “${event.noticed}” twice` : `caught itself repeating “${event.noticed}”`;
      next.caught = { kind: event.kind, noticed: event.noticed };
      next.kept = event.cut;
      next.retracted = event.cut ? next.draft.slice(event.cut.length) : next.draft;
      next.note = event.cut ? what : `${what}: the words it picked up, not its own`;
      break;
    }
    case "backtrack":
      next.kept = event.cut;
      next.retracted = next.draft.slice(event.cut.length);
      next.step = event.step;
      next.note = `backs up to “${event.cut}” and weighs up to ${event.wider} paths (step ${event.step})`;
      break;
    case "found":
      next.found = event.text.slice(next.kept.length);
      next.note = `found another way on in ${event.explored} path(s)`;
      break;
    case "stuck":
      next.note = `nothing new in ${event.explored} path(s)`;
      break;
    default:
      break;
  }
  return next;
}
