/**
 * Microphone capture, dictation and WAV encoding - everything the Speech tab
 * needs from the browser, with no dependencies.
 *
 * MediaRecorder produces WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari), which
 * the server could only read through ffmpeg. So the recording is decoded with
 * the Web Audio API, mixed down to mono, resampled and written out as a 16-bit
 * PCM WAV here: the server then reads it with the standard library alone.
 *
 * The words come from the Web Speech API (`SpeechRecognition`) while the
 * recording runs - the browser is the speech-to-text engine that is always
 * there. The transcript is sent along with the audio and the server takes it as
 * the "given" backend; when the browser has no recogniser, the server's own
 * (faster-whisper / whisper / a transcription server) takes over, and the text
 * box can always be corrected by hand before teaching.
 */

/** True when this browser can record from a microphone at all. */
export function micSupported() {
  return Boolean(
    typeof navigator !== "undefined" &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function" &&
      typeof window !== "undefined" &&
      typeof window.MediaRecorder !== "undefined",
  );
}

/** True when this browser can turn speech into text itself (Chrome, Edge, Safari). */
export function dictationSupported() {
  return typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

/**
 * Start recording. Resolves with a handle whose `stop()` gives the recorded
 * Blob and releases the microphone; `cancel()` throws the recording away.
 */
export async function startRecording() {
  if (!micSupported()) throw new Error("this browser cannot record audio (no MediaRecorder)");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const chunks = [];
  let recorder;
  try {
    recorder = new window.MediaRecorder(stream);
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    throw err;
  }
  const type = recorder.mimeType || "audio/webm";
  const stopped = new Promise((resolve) => {
    recorder.addEventListener("stop", () => resolve(new Blob(chunks, { type })), { once: true });
  });
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size) chunks.push(event.data);
  });
  recorder.start();

  const release = () => stream.getTracks().forEach((track) => track.stop());
  return {
    mimeType: type,
    async stop() {
      if (recorder.state !== "inactive") recorder.stop();
      const blob = await stopped;
      release();
      return blob;
    },
    cancel() {
      try {
        if (recorder.state !== "inactive") recorder.stop();
      } finally {
        release();
      }
    },
  };
}

/**
 * Live dictation while the microphone runs. `onUpdate(text, final)` is called
 * with everything recognised so far; returns a handle with `stop()`, or null
 * when the browser has no recogniser.
 */
export function startDictation(onUpdate, { lang } = {}) {
  const Recognition = typeof window === "undefined" ? null : window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;
  const recognition = new Recognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = lang || (typeof navigator !== "undefined" && navigator.language) || "en-US";
  let settled = "";
  recognition.onresult = (event) => {
    let pending = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (result.isFinal) settled = `${settled} ${result[0].transcript}`.trim();
      else pending = `${pending} ${result[0].transcript}`.trim();
    }
    onUpdate(`${settled} ${pending}`.trim(), settled);
  };
  try {
    recognition.start();
  } catch {
    return null; // already running, or the page is not allowed to listen
  }
  return {
    stop() {
      recognition.onresult = null;
      try {
        recognition.stop();
      } catch {
        // stopping an already stopped recogniser is not an error worth showing
      }
      return settled;
    },
  };
}

function downmix(buffer) {
  const frames = buffer.length;
  const channels = buffer.numberOfChannels;
  const mono = new Float32Array(frames);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < frames; i += 1) mono[i] += data[i] / channels;
  }
  return mono;
}

/** Linear interpolation - the server quantises to one byte per sample anyway. */
export function resample(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate || samples.length === 0) return samples;
  const count = Math.max(1, Math.round((samples.length * targetRate) / sourceRate));
  const step = sourceRate / targetRate;
  const out = new Float32Array(count);
  const last = samples.length - 1;
  for (let i = 0; i < count; i += 1) {
    const position = i * step;
    const left = Math.floor(position);
    if (left >= last) {
      out[i] = samples[last];
      continue;
    }
    const fraction = position - left;
    out[i] = samples[left] * (1 - fraction) + samples[left + 1] * fraction;
  }
  return out;
}

/** Float samples in [-1, 1] -> a 16-bit PCM mono WAV file. */
export function encodeWav(samples, rate) {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(value * 32767), true);
  }
  return bytes;
}

/**
 * A recorded Blob (WebM, MP4, ...) -> `{blob, rate, seconds, samples}` of a
 * 16-bit PCM WAV the server reads without ffmpeg.
 */
export async function toWav(blob, rate = 16000) {
  const Context = typeof window === "undefined" ? null : window.AudioContext || window.webkitAudioContext;
  if (!Context) throw new Error("this browser cannot decode the recording (no AudioContext)");
  const context = new Context();
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const samples = resample(downmix(buffer), buffer.sampleRate, rate);
    return {
      blob: new Blob([encodeWav(samples, rate)], { type: "audio/wav" }),
      rate,
      samples: samples.length,
      seconds: samples.length / rate,
      sourceRate: buffer.sampleRate,
    };
  } finally {
    if (typeof context.close === "function") context.close();
  }
}

/** Peak amplitude of a WAV blob's samples, for the "did it hear me?" meter. */
export async function peakOf(blob) {
  const Context = typeof window === "undefined" ? null : window.AudioContext || window.webkitAudioContext;
  if (!Context) return null;
  const context = new Context();
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const data = buffer.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < data.length; i += 1) {
      const magnitude = Math.abs(data[i]);
      if (magnitude > peak) peak = magnitude;
    }
    return peak;
  } catch {
    return null;
  } finally {
    if (typeof context.close === "function") context.close();
  }
}
