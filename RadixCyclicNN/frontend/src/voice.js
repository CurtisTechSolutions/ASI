/**
 * Talking with the model by voice (VoicePanel.jsx): the always-on ear and the mouth.
 *
 * The ear is the microphone under an endpointer: the level of every audio
 * frame decides where an utterance starts and ends, so nothing has to be
 * pressed - the tab listens all the time. The frames of an utterance (with a
 * little before its start) are written out as a 16-bit PCM WAV for the server;
 * what was said in it comes from the Web Speech API running beside the ear,
 * its final results paired with the utterance by time (and waited for: a
 * recogniser commits a phrase a moment after the speaker stops). The mouth is a gapless
 * PCM player: the reply's audio arrives in chunks (the `audio` events of
 * `POST /api/voice/turn/stream`) and each is scheduled right after the last,
 * so playback starts on the first chunk and never waits for the whole reply
 * or for anyone to press play. While the mouth speaks the ear is closed, so
 * the model never hears itself as the person.
 *
 * Everything that can be tested without a browser is a pure function or class
 * here (test/voice.test.mjs); the microphone, the recogniser and the player
 * are the three browser-facing entry points at the end.
 */

export const DEFAULTS = {
  threshold: 0.02, // the level (RMS of a frame, 0..1) that counts as speech
  minSpeechMs: 250, // shorter than this is a click, not an utterance
  silenceMs: 700, // this much quiet ends an utterance
  maxMs: 15000, // an utterance is cut here whatever the level does
  startFrames: 2, // loud frames in a row before an utterance starts
  preRollMs: 300, // audio kept from before the start, so the first sound is not clipped
};

/** Where an utterance starts and ends, from the level of each frame alone. */
export class Endpointer {
  constructor(options = {}) {
    const o = { ...DEFAULTS, ...options };
    this.threshold = o.threshold;
    this.minSpeechMs = o.minSpeechMs;
    this.silenceMs = o.silenceMs;
    this.maxMs = o.maxMs;
    this.startFrames = Math.max(1, o.startFrames);
    this.reset();
  }

  reset() {
    this.active = false;
    this.loud = 0;
    this.firstLoudAt = 0;
    this.lastLoudAt = 0;
    this.startedAt = 0;
  }

  /**
   * One frame's level at `now` (milliseconds): "start" when an utterance begins, "end" when one that was
   * long enough ends, "cancel" when a burst was too short to be one, null otherwise.
   */
  feed(level, now) {
    const loud = level >= this.threshold;
    if (!this.active) {
      if (!loud) {
        this.loud = 0;
        return null;
      }
      if (this.loud === 0) this.firstLoudAt = now;
      this.loud += 1;
      if (this.loud < this.startFrames) return null;
      this.active = true;
      this.startedAt = this.firstLoudAt;
      this.lastLoudAt = now;
      return "start";
    }
    if (loud) this.lastLoudAt = now;
    const quiet = now - this.lastLoudAt;
    const long = now - this.startedAt >= this.maxMs;
    if (quiet < this.silenceMs && !long) return null;
    const spoke = this.lastLoudAt - this.startedAt;
    this.reset();
    return spoke >= this.minSpeechMs ? "end" : "cancel";
  }
}

/** The RMS level of a frame of samples in [-1, 1]. */
export function levelOf(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i += 1) sum += frame[i] * frame[i];
  return Math.sqrt(sum / (frame.length || 1));
}

/**
 * The words of an utterance: the dictation's final results that landed inside its time window (both on the
 * ear's clock, `performance.now()`). A recogniser commits a phrase a little after the speaker stops, so `after`
 * is generous, while `before` only covers the moment the endpointer takes to notice speech; every final is used
 * once, and one older than the window is stale - an earlier sound's, or the mouth's own - and is dropped, so
 * it can never be taken for a later utterance's words.
 */
export function pairTranscript(finals, startedAt, endedAt, { before = 500, after = 3500 } = {}) {
  const parts = [];
  for (const final of finals) {
    if (final.used) continue;
    if (final.at < startedAt - before) {
      final.used = true; // stale
      continue;
    }
    if (final.at <= endedAt + after) {
      final.used = true;
      const text = String(final.text || "").trim();
      if (text) parts.push(text);
    }
  }
  return parts.join(" ").trim();
}

/**
 * The words of an utterance, waited for: a recogniser commits its final result a moment after the speaker
 * stops, so the finals of `log` (a dictation log: `finals`, `pending`, `nextFinal(ms)`) are paired again each
 * time one lands, until the utterance has words or the wait is up - `timeout` milliseconds while words are
 * being recognised (`log.pending`), `quiet` when nothing is, which is what a noise sounds like. "" when no
 * words came; "" at once without a log.
 */
export async function wordsOf(log, startedAt, endedAt, { timeout = 3000, quiet = 1200, now = () => performance.now() } = {}) {
  if (!log) return "";
  const began = now();
  let words = pairTranscript(log.finals, startedAt, endedAt);
  while (!words) {
    const left = began + (log.pending ? timeout : quiet) - now();
    if (left <= 0) break;
    await log.nextFinal(left);
    words = pairTranscript(log.finals, startedAt, endedAt);
  }
  return words;
}

/** Base64 of 16-bit little-endian PCM (an `audio` event) -> samples in [-1, 1]. */
export function decodePcm(base64) {
  const binary = atob(base64);
  const count = Math.floor(binary.length / 2);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    let value = binary.charCodeAt(2 * i) | (binary.charCodeAt(2 * i + 1) << 8);
    if (value >= 0x8000) value -= 0x10000;
    out[i] = value / 32768;
  }
  return out;
}

/** Base64 of a 16-bit PCM mono WAV (the plain route's `wav_base64`) -> `{samples, rate}`. */
export function decodeWav(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const rate = bytes.length >= 28 ? view.getUint32(24, true) : 16000;
  let offset = 12;
  let start = 44;
  let length = Math.max(0, bytes.length - 44);
  while (offset + 8 <= bytes.length) {
    const id = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const size = view.getUint32(offset + 4, true);
    if (id === "data") {
      start = offset + 8;
      length = Math.min(size, bytes.length - start);
      break;
    }
    offset += 8 + size + (size % 2);
  }
  const count = Math.floor(length / 2);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) samples[i] = view.getInt16(start + 2 * i, true) / 32768;
  return { samples, rate };
}

/** Bytes -> base64, in pieces small enough for `btoa`. */
export function bytesToBase64(bytes) {
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + step, bytes.length)));
  }
  return btoa(binary);
}

/**
 * The conversation so far as the server takes it: `[speaker, text]` pairs, oldest first, out of the panel's
 * turns (newest first). `limit` is the number of turns kept, the most recent ones.
 */
export function historyOf(turns, limit = 12) {
  const pairs = [];
  for (const turn of [...turns].reverse().slice(-limit)) {
    if (turn.you && turn.you.text && turn.you.heard !== false) pairs.push(["You", turn.you.text]);
    if (turn.model && turn.model.text) pairs.push(["Model", turn.model.text]);
  }
  return pairs;
}

// -- the browser ---------------------------------------------------------------------

function audioContextClass() {
  if (typeof window === "undefined") return null;
  return window.AudioContext || window.webkitAudioContext || null;
}

/** True when this browser can listen: microphone access and the Web Audio API. */
export function listeningSupported() {
  return Boolean(
    typeof navigator !== "undefined" &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function" &&
      audioContextClass(),
  );
}

/**
 * Start listening: the microphone under the endpointer. `onUtterance({samples, rate, startedAt, endedAt,
 * seconds, peak})` gets every utterance as Float32 samples at `rate`, `onLevel(level)` every frame's level,
 * `onState("hearing" | "quiet")` the start and end of speech. The handle's `pause()` closes the ear (while the
 * mouth speaks, what it says must not be taken for the person's), `resume()` opens it, `stop()` releases the
 * microphone. Times are `performance.now()` milliseconds.
 */
export async function startListening({ onUtterance, onLevel, onState, options = {} } = {}) {
  const Context = audioContextClass();
  if (!listeningSupported()) throw new Error("this browser cannot listen (no microphone access or no Web Audio)");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const context = new Context();
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      // it resumes on the next gesture
    }
  }
  const source = context.createMediaStreamSource(stream);
  const frameSize = 2048;
  const processor = context.createScriptProcessor(frameSize, 1, 1);
  const silence = context.createGain();
  silence.gain.value = 0; // the processor must reach the destination to run; nothing of the microphone may
  const o = { ...DEFAULTS, ...options };
  const endpointer = new Endpointer(o);
  const frameMs = (frameSize / context.sampleRate) * 1000;
  const preRollFrames = Math.max(1, Math.ceil(o.preRollMs / frameMs));
  let ring = [];
  let recording = null;
  let startedAt = 0;
  let paused = false;
  processor.onaudioprocess = (event) => {
    const frame = new Float32Array(event.inputBuffer.getChannelData(0));
    const level = levelOf(frame);
    if (onLevel) onLevel(level);
    if (paused) return;
    const now = performance.now();
    ring.push(frame);
    if (ring.length > preRollFrames) ring.shift();
    const verdict = endpointer.feed(level, now);
    if (verdict === "start") {
      recording = ring.slice(0, -1); // the pre-roll: the frames before this one
      startedAt = now - recording.length * frameMs;
      if (onState) onState("hearing");
    }
    if (recording) recording.push(frame);
    if (verdict === "end" || verdict === "cancel") {
      const frames = recording || [];
      recording = null;
      if (onState) onState("quiet");
      if (verdict === "end" && onUtterance) {
        const total = frames.reduce((n, f) => n + f.length, 0);
        const samples = new Float32Array(total);
        let at = 0;
        let peak = 0;
        for (const f of frames) {
          samples.set(f, at);
          at += f.length;
          for (let i = 0; i < f.length; i += 1) {
            const magnitude = Math.abs(f[i]);
            if (magnitude > peak) peak = magnitude;
          }
        }
        onUtterance({ samples, rate: context.sampleRate, startedAt, endedAt: now, seconds: total / context.sampleRate, peak });
      }
    }
  };
  source.connect(processor);
  processor.connect(silence);
  silence.connect(context.destination);
  return {
    sampleRate: context.sampleRate,
    pause() {
      paused = true;
      endpointer.reset();
      recording = null;
      ring = [];
    },
    resume() {
      paused = false;
      if (context.state === "suspended") context.resume().catch(() => {});
    },
    stop() {
      paused = true;
      try {
        processor.disconnect();
        source.disconnect();
        silence.disconnect();
      } catch {
        // already disconnected
      }
      stream.getTracks().forEach((track) => track.stop());
      if (typeof context.close === "function") context.close().catch(() => {});
    },
  };
}

/**
 * Dictation that keeps going: the Web Speech API's final results with the time each one landed (`finals`, on
 * the ear's clock, for `pairTranscript` and `wordsOf`), `pending` and `onInterim(text)` with the words as they
 * are being recognised, `nextFinal(ms)` resolving when the next final lands (true) or the time is up (false),
 * and `mute(on)` to drop everything heard while the mouth speaks, so the reply is never written down as the
 * person's. A recogniser stops by itself now and then (a pause, a network hiccup, a cap on how long it
 * listens): it is started again for as long as the handle is alive. Null when the browser has no recogniser.
 */
export function startDictationLog({ onInterim, lang } = {}) {
  const Recognition = typeof window === "undefined" ? null : window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;
  const finals = [];
  let alive = true;
  let recognition = null;
  let failures = 0;
  let muted = false;
  let waiters = [];
  const handle = { finals, pending: "" };
  const landed = () => {
    const woken = waiters;
    waiters = [];
    woken.forEach((w) => w(true));
  };
  const begin = () => {
    if (!alive) return;
    recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = lang || (typeof navigator !== "undefined" && navigator.language) || "en-US";
    recognition.onresult = (event) => {
      if (muted) return;
      let pending = "";
      let final = false;
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = String(result[0].transcript || "").trim();
        if (result.isFinal) {
          failures = 0;
          if (text) {
            finals.push({ text, at: performance.now(), used: false });
            final = true;
          }
        } else {
          pending = `${pending} ${text}`.trim();
        }
      }
      handle.pending = pending;
      if (onInterim) onInterim(pending);
      if (final) landed();
    };
    recognition.onerror = (event) => {
      if (event && (event.error === "not-allowed" || event.error === "service-not-allowed")) alive = false;
    };
    recognition.onend = () => {
      if (!alive) return;
      failures += 1;
      setTimeout(begin, Math.min(3000, 150 * failures));
    };
    try {
      recognition.start();
    } catch {
      // onend brings it back
    }
  };
  begin();
  handle.nextFinal = (ms) =>
    new Promise((resolve) => {
      const timer = setTimeout(() => {
        waiters = waiters.filter((w) => w !== wake);
        resolve(false);
      }, Math.max(0, ms));
      const wake = (value) => {
        clearTimeout(timer);
        resolve(value);
      };
      waiters.push(wake);
    });
  handle.mute = (on) => {
    muted = Boolean(on);
    if (muted) {
      handle.pending = "";
      if (onInterim) onInterim("");
    }
  };
  handle.stop = () => {
    alive = false;
    waiters.forEach((w) => w(false));
    waiters = [];
    if (recognition) {
      recognition.onresult = null;
      recognition.onend = null;
      try {
        recognition.stop();
      } catch {
        // already stopped
      }
    }
  };
  return handle;
}

/**
 * The mouth: PCM chunks played back to back as they arrive. `play(samples, rate)` schedules a chunk right after
 * the previous one (the first plays at once), `drained()` resolves once everything scheduled has been heard,
 * `stop()` cuts playback. The AudioContext is made on the first `play`, after the person has clicked the tab's
 * button at least once, which is what browsers ask before a page may make a sound.
 */
export class PcmPlayer {
  constructor() {
    this.context = null;
    this.next = 0;
    this.pending = 0;
    this.sources = [];
    this.waiters = [];
  }

  ensure() {
    if (!this.context) {
      const Context = audioContextClass();
      if (!Context) throw new Error("this browser cannot play audio (no AudioContext)");
      this.context = new Context();
    }
    if (this.context.state === "suspended") this.context.resume().catch(() => {});
    return this.context;
  }

  play(samples, rate) {
    if (!samples || !samples.length) return;
    const context = this.ensure();
    const buffer = context.createBuffer(1, samples.length, rate);
    if (typeof buffer.copyToChannel === "function") buffer.copyToChannel(samples, 0);
    else buffer.getChannelData(0).set(samples);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const start = Math.max(context.currentTime + 0.02, this.next);
    source.onended = () => {
      this.sources = this.sources.filter((s) => s !== source);
      this.pending = Math.max(0, this.pending - 1);
      if (this.pending === 0) this.settle();
    };
    source.start(start);
    this.next = start + buffer.duration;
    this.pending += 1;
    this.sources.push(source);
  }

  get busy() {
    return this.pending > 0;
  }

  settle() {
    const waiters = this.waiters;
    this.waiters = [];
    waiters.forEach((resolve) => resolve());
  }

  /**
   * Resolves once everything scheduled has been heard - or after it should have been, when the context never
   * ran (a browser that wanted a gesture first): the ear must open again either way.
   */
  drained() {
    if (this.pending === 0) return Promise.resolve();
    const left = this.context ? Math.max(0, this.next - this.context.currentTime) : 0;
    return new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      this.waiters.push(done);
      setTimeout(done, Math.ceil(left * 1000) + 1500);
    });
  }

  stop() {
    for (const source of this.sources) {
      try {
        source.onended = null;
        source.stop();
      } catch {
        // never started, or already over
      }
    }
    this.sources = [];
    this.pending = 0;
    this.next = 0;
    this.settle();
  }
}
