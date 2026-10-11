/**
 * What a model of base64 symbols said, as the bytes it holds.
 *
 * A model whose unit is base64 learns the symbols of any content's bytes - a
 * text, an image, a video - and answers in them (`dGhlIGNhdCBzYXQ`); every
 * server decodes each answer beside them (`decoded_prediction` in
 * ../../RadixCyclicNN/radixnet/encoding.py, and the Go and Rust ports):
 * `decoded` is the whole answer's bytes read as UTF-8 (U+FFFD where they are no
 * text), `decoded_continuation` the part of them the model wrote itself, and
 * `decoded_bytes` how many bytes the answer holds. Any other model's records
 * carry no such field.
 *
 * Pure - no React, no storage - so `node --test` holds it to those rules.
 */

/**
 * A record's decoded text, split where the model's own part begins: `{ whole,
 * head, tail, bytes }` - what the answer holds, what came before the part the
 * model wrote, that part, and how many bytes the whole is - or null for a
 * record that carries no decoding. A record without `decoded_continuation` is
 * all tail, and so is one whose tail does not end its whole.
 */
export function decodedParts(record) {
  if (!record || typeof record !== "object" || typeof record.decoded !== "string") return null;
  const whole = record.decoded;
  const given = typeof record.decoded_continuation === "string" ? record.decoded_continuation : whole;
  const tail = whole.endsWith(given) ? given : whole;
  const bytes = Number.isInteger(record.decoded_bytes) ? record.decoded_bytes : null;
  return { whole, head: whole.slice(0, whole.length - tail.length), tail, bytes };
}

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const VALUE = new Map(Array.from(ALPHABET, (symbol, i) => [symbol, i]));

/**
 * Base64 symbols back to bytes, whatever was written - the servers' decoder
 * (`base64_bytes`): anything outside the 64 symbols is skipped (padding,
 * whitespace, a stray character), and the bits that cannot complete a byte are
 * dropped, a dangling last symbol's with them - so a predicted text always
 * decodes, into `floor(6 * symbols / 8)` bytes.
 */
export function base64Bytes(text) {
  const values = [];
  for (const symbol of String(text ?? "")) {
    const value = VALUE.get(symbol);
    if (value !== undefined) values.push(value);
  }
  const out = new Uint8Array(Math.floor((values.length * 6) / 8));
  let held = 0; // the bits read and not yet written, at most 7 between symbols
  let count = 0;
  let n = 0;
  for (const value of values) {
    held = ((held << 6) | value) & 0x3fff;
    count += 6;
    if (count >= 8) {
      count -= 8;
      out[n++] = (held >> count) & 0xff;
    }
  }
  return out;
}

/** A file name for an answer's bytes: the stem as given, `.bin` unless it already names an extension. */
export function bytesName(stem) {
  const name = String(stem ?? "").trim().replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "") || "decoded";
  return /\.[A-Za-z0-9]{1,8}$/.test(name) ? name : `${name}.bin`;
}
