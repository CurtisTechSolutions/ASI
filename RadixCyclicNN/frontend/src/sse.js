/**
 * Server-sent events, read out of a byte stream one frame at a time.
 *
 * The Talk panel streams a reply from `POST /v1/messages` (the Messages
 * dialect of today's format, `../../DESIGN.md` section 36): the thinking
 * arrives line by line while the search runs, the text one node of the walk
 * at a time. A frame is `event: name` (optional), one or more `data:` lines
 * and a blank line; frames arrive cut at arbitrary byte boundaries, so the
 * parser keeps what it has not finished and hands back only whole frames.
 * No dependencies, no `EventSource` (which cannot POST).
 */
export class EventStreamParser {
  constructor() {
    this.buffer = "";
  }

  /** Feed a chunk of text; returns the frames it completed, `{event, data}` each (`data` JSON-decoded where it is JSON). */
  push(chunk) {
    this.buffer += String(chunk ?? "").replace(/\r\n/g, "\n");
    const frames = [];
    let at;
    while ((at = this.buffer.indexOf("\n\n")) >= 0) {
      const frame = parseFrame(this.buffer.slice(0, at));
      this.buffer = this.buffer.slice(at + 2);
      if (frame) frames.push(frame);
    }
    return frames;
  }

  /** The stream ended: whatever is left is the last frame, when it holds any data. */
  end() {
    const frame = parseFrame(this.buffer);
    this.buffer = "";
    return frame ? [frame] : [];
  }
}

/** One frame's lines -> `{event, data}`, or null when it carries no data (a comment, a keep-alive). */
export function parseFrame(text) {
  let event = null;
  const data = [];
  for (const line of String(text ?? "").split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!data.length) return null;
  const raw = data.join("\n");
  let parsed = raw;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // `[DONE]` and any other bare text stay as they are
  }
  // the Messages dialect names the event twice: as the frame's event and as `type` in its data
  if (!event && parsed && typeof parsed === "object" && typeof parsed.type === "string") event = parsed.type;
  return { event, data: parsed };
}
