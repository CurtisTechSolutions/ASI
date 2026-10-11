/**
 * Sending files to the server a piece at a time (the Upload picker and the checkpoint upload).
 *
 * `POST /api/uploads` (and `/api/checkpoints/upload`) take a file's bytes as base64 inside the JSON, at most
 * 8 MiB of raw bytes a request: the first piece with `append: false` (it starts the file), every other with
 * `append: true`. Cut that way there is no limit to how big a file can be. Everything here is pure or takes
 * what it works on as an argument - a Blob (a File is one), the function that sends a piece - so
 * `test/upload.test.mjs` runs it under plain `node --test`.
 */

/** The most raw bytes one request carries. */
export const CHUNK_BYTES = 8 * 1024 * 1024;

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/** Bytes are turned into base64 this many at a time: a multiple of 3, so the pieces join without padding. */
const ENCODE_SLICE = 3 * 8192;

/** Base64 without `btoa` (some places have none): three bytes become four letters. */
function encodeByHand(bytes) {
  let out = "";
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out += ALPHABET[(n >> 18) & 63] + ALPHABET[(n >> 12) & 63] + ALPHABET[(n >> 6) & 63] + ALPHABET[n & 63];
  }
  const rest = bytes.length - i;
  if (rest === 1) {
    const n = bytes[i] << 16;
    out += `${ALPHABET[(n >> 18) & 63]}${ALPHABET[(n >> 12) & 63]}==`;
  } else if (rest === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += `${ALPHABET[(n >> 18) & 63]}${ALPHABET[(n >> 12) & 63]}${ALPHABET[(n >> 6) & 63]}=`;
  }
  return out;
}

/**
 * The base64 of a run of bytes (a Uint8Array, an ArrayBuffer or a plain array of 0-255).
 *
 * Encoded a slice at a time, each slice a multiple of three bytes long, so an 8 MiB chunk never becomes one
 * 8-million-character binary string on the way.
 */
export function bytesToBase64(input) {
  const bytes = input instanceof Uint8Array ? input : input instanceof ArrayBuffer ? new Uint8Array(input) : Uint8Array.from(input || []);
  const btoaFn = typeof globalThis.btoa === "function" ? globalThis.btoa : null;
  if (!btoaFn) return encodeByHand(bytes);
  const parts = [];
  for (let start = 0; start < bytes.length; start += ENCODE_SLICE) {
    const slice = bytes.subarray(start, Math.min(bytes.length, start + ENCODE_SLICE));
    let binary = "";
    for (let i = 0; i < slice.length; i += 4096) {
      binary += String.fromCharCode.apply(null, slice.subarray(i, Math.min(slice.length, i + 4096)));
    }
    parts.push(btoaFn(binary));
  }
  return parts.join("");
}

/**
 * Where a file of `size` bytes is cut: `[{start, end, append}]`, each at most `chunk` bytes, the first with
 * `append: false`. An empty file is still one (empty) piece, so the server creates it.
 */
export function chunkRanges(size, chunk = CHUNK_BYTES) {
  const total = Math.max(0, Math.floor(Number(size) || 0));
  const step = Math.max(1, Math.floor(Number(chunk) || CHUNK_BYTES));
  if (total === 0) return [{ start: 0, end: 0, append: false }];
  const ranges = [];
  for (let start = 0; start < total; start += step) {
    ranges.push({ start, end: Math.min(total, start + step), append: start > 0 });
  }
  return ranges;
}

/** The name a file is uploaded under: its path inside a picked or dropped folder when it has one, else its name. */
export function uploadName(file) {
  if (!file) return "";
  const relative = typeof file.webkitRelativePath === "string" && file.webkitRelativePath ? file.webkitRelativePath : "";
  const path = file.relativePath || relative || file.name || "";
  return String(path).replace(/^\/+/, "");
}

/** The bytes of a Blob: `Blob.arrayBuffer` where there is one, a FileReader where there is not. */
export function readBlob(blob) {
  if (blob && typeof blob.arrayBuffer === "function") return blob.arrayBuffer();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("the file could not be read"));
    reader.readAsArrayBuffer(blob);
  });
}

/**
 * Send one file a chunk at a time: `send(base64, append, range)` is awaited for every piece in order, and
 * `onProgress(sentBytes, totalBytes)` is told after each. Resolves with what the last `send` answered.
 */
export async function uploadInChunks(blob, send, { chunk = CHUNK_BYTES, onProgress } = {}) {
  const size = Number(blob && blob.size) || 0;
  let last = null;
  for (const range of chunkRanges(size, chunk)) {
    const bytes = range.end > range.start ? await readBlob(blob.slice(range.start, range.end)) : new ArrayBuffer(0);
    last = await send(bytesToBase64(bytes), range.append, range);
    if (onProgress) onProgress(range.end, size);
  }
  return last;
}

/** "42%" of a file sent; a file of no bytes is all sent at once. */
export function progressText(sent, total) {
  if (!(total > 0)) return "100%";
  return `${Math.min(100, Math.floor((100 * sent) / total))}%`;
}
