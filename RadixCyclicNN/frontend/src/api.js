/**
 * Thin fetch wrapper for the RadixCyclicNN JSON API (DESIGN.md section 12).
 *
 * Every call resolves with the parsed JSON body or rejects with an ApiError.
 * The server reports failures as {"error": "..."} with a 4xx/5xx status;
 * network failures and non-JSON bodies are normalised into ApiError too, so
 * panels only ever deal with one error shape (`err.message`).
 */

/** Base URL prefix; empty by default so relative /api URLs work when the Python server serves dist. */
export const API_BASE = String(import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function request(method, path, body) {
  const init = { method, headers: { Accept: "application/json" } };
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    init.body = body; // multipart/form-data: the browser sets the boundary
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(API_BASE + path, init);
  } catch (err) {
    throw new ApiError(`Network error: ${err && err.message ? err.message : "request failed"}`);
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      const detail = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
      throw new ApiError(response.ok ? `Invalid JSON in response (${detail})` : detail, response.status);
    }
  }

  const serverError = data && typeof data === "object" && typeof data.error === "string" ? data.error : null;
  if (!response.ok) {
    const detail = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
    throw new ApiError(serverError || detail, response.status, data);
  }
  if (serverError) throw new ApiError(serverError, response.status, data);
  return data;
}

const get = (path) => request("GET", path);
const post = (path, body = {}) => request("POST", path, body);

/** Blob / File -> base64, in chunks so a long recording cannot blow the argument stack. */
async function toBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/**
 * POST audio to one of the /api/speech endpoints. A transcript travels in a
 * JSON body (any length, any character); without one the bytes go up as
 * multipart with the options in the query string, so a big file is not
 * base64-inflated.
 */
async function sendAudio(path, audio, options = {}) {
  const name = audio && audio.name ? audio.name : "utterance.wav";
  const entries = Object.entries(options).filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (options.transcript) {
    const body = { name, content_base64: await toBase64(audio) };
    for (const [key, value] of entries) body[key] = value;
    return post(path, body);
  }
  const form = new FormData();
  form.append("file", audio, name);
  const params = new URLSearchParams();
  for (const [key, value] of entries) params.set(key, String(value));
  const query = params.toString();
  return post(`${path}${query ? `?${query}` : ""}`, form);
}

/**
 * Normalise a job payload. Job-starting endpoints answer {"job": {...}}, while
 * GET /api/job and the stop endpoints answer with the job status itself; both
 * may carry null when no job exists yet.
 */
export function unwrapJob(data) {
  if (!data || typeof data !== "object") return null;
  if ("job" in data) return data.job && typeof data.job === "object" ? data.job : null;
  return "state" in data ? data : null;
}

export const api = {
  health: () => get("/api/health"),
  status: () => get("/api/status"),
  /** Model kinds: the active one, every kind and their files; switching keeps the previous model in memory. */
  model: () => get("/api/model"),
  selectModel: (kind) => post("/api/model/select", { kind }),
  /** Count model: change the dual frequency weight function (scales and the sliding window). */
  modelWeights: (body) => post("/api/model/weights", body),
  train: (body) => post("/api/train", body),
  /** Learning-rate schedules: what an expression may use (presets, variables, functions) and a per-epoch preview. */
  schedule: () => get("/api/schedule"),
  schedulePreview: (body) => post("/api/schedule/preview", body),
  job: () => get("/api/job"),
  stopJob: () => post("/api/job/stop"),
  predict: (body) => post("/api/predict", body),
  generate: (body) => post("/api/generate", body),
  /** The model converses with itself (or with the other kind in memory): turns of a dialogue. */
  converse: (body) => post("/api/converse", body),
  score: (body) => post("/api/score", body),
  twoNrl: (body) => post("/api/2nrl", body),
  /** Rated texts -> 2NRL (both kinds), reward (thumbs up only) or punish (thumbs down only). */
  feedback: (body) => post("/api/feedback", body),
  invert: () => post("/api/invert"),
  compress: () => post("/api/compress"),
  evolveStart: (body) => post("/api/evolve/start", body),
  evolveStop: () => post("/api/evolve/stop"),
  evolveHistory: () => get("/api/evolve/history"),
  save: (body) => post("/api/save", body),
  load: (body) => post("/api/load", body),
  reset: (body) => post("/api/reset", body),
  checkpoints: () => get("/api/checkpoints"),
  checkpointSave: (body) => post("/api/checkpoints/save", body),
  checkpointRestore: (body) => post("/api/checkpoints/restore", body),
  graph: (limit) => get(`/api/graph?limit=${encodeURIComponent(limit)}`),
  history: () => get("/api/history"),
  uploads: () => get("/api/uploads"),
  /** Upload one text file (read in the browser); the server keeps it under its upload directory. */
  upload: (name, content) => post("/api/uploads", { name, content }),
  /** Upload a file as bytes (multipart); a ZIP archive is unpacked into text files on the server. */
  uploadFile: (file) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return post("/api/uploads", form);
  },
  deleteUpload: (name) => post("/api/uploads/delete", { name }),
  /**
   * The negative network (see NegativePanel): the failures only, blamed with the tutor's reasons, and the filter it
   * forms with the positive model.
   */
  negative: () => get("/api/negative"),
  negativeBlame: (body) => post("/api/negative/blame", body),
  negativeClear: (body) => post("/api/negative/clear", body),
  negativeJudge: (body) => post("/api/negative/judge", body),
  negativeFilter: (body) => post("/api/negative/filter", body),
  negativeForget: (body) => post("/api/negative/forget", body),
  negativeSettings: (body) => post("/api/negative/settings", body),
  negativeReset: (body = {}) => post("/api/negative/reset", body),
  negativeSave: (body = {}) => post("/api/negative/save", body),
  /** Ollama (see OllamaPanel). `url` optionally overrides the server's configured Ollama URL. */
  ollamaModels: (url) => get(`/api/ollama/models${url ? `?url=${encodeURIComponent(url)}` : ""}`),
  ollamaCorpus: (body) => post("/api/ollama/corpus", body),
  ollamaReview: (body) => post("/api/ollama/review", body),
  /** Images as text (see ImagesPanel): the Stable Diffusion VAE run backwards, quantised and base64-encoded. */
  images: () => get("/api/images"),
  imageEncode: (file, { size, encoder, train, saveAs } = {}) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const params = new URLSearchParams();
    if (size) params.set("size", String(size));
    if (encoder) params.set("encoder", String(encoder));
    if (train) params.set("train", "true");
    if (saveAs) params.set("save_as", String(saveAs));
    const query = params.toString();
    return post(`/api/images/encode${query ? `?${query}` : ""}`, form);
  },
  imageDecode: (text, encoder) => post("/api/images/decode", encoder ? { text, encoder } : { text }),
  /**
   * The recall tutor: ask the network to draw back a picture it was shown and mark what comes back.
   * Options: size, encoder, lead, length, attempts, mode, temperature, threshold, blame.
   */
  imageTutor: (file, options = {}) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    }
    const query = params.toString();
    return post(`/api/images/tutor${query ? `?${query}` : ""}`, form);
  },
  /** English lessons (see TutorPanel): the tutor writes the prefix, the network completes it, the tutor marks it. */
  tutor: () => get("/api/tutor"),
  tutorStart: (body) => post("/api/tutor/start", body),
  tutorHistory: () => get("/api/tutor/history"),
  /** One round of exercises, completions and grades without training (a dry run). */
  tutorLesson: (body) => post("/api/tutor/lesson", body),
  /** The lessons to run next, planned from a report card (the last run's own when none is sent). */
  tutorPlan: (body) => post("/api/tutor/plan", body),
  /** Is ChatGPT usable on the server (its own OPENAI_API_KEY), and which models the key has. */
  chatgptModels: () => get("/api/chatgpt/models"),
  /** Speech (see SpeechPanel): the transcript and the waveform of one utterance, behind one unique token. */
  speech: () => get("/api/speech"),
  /** Speech to text only. `audio` is a File or a Blob; `transcript` is what the browser already dictated. */
  speechTranscribe: (audio, options = {}) => sendAudio("/api/speech/transcribe", audio, options),
  /**
   * Teach one utterance: the words and the waveform. Options: transcript, rate, codec, normalise,
   * waveform, pair, token, unique, backend, language, train, epochs, lr, batch_size, save_as.
   */
  speechTeach: (audio, options = {}) => sendAudio("/api/speech/teach", audio, options),
  /**
   * The recall tutor: ask the network to say back an utterance it was taught and mark what comes back.
   * Options: transcript, rate, codec, normalise, token, unique, lead, length, attempts, mode,
   * temperature, threshold, listen_back, blame.
   */
  speechTutor: (audio, options = {}) => sendAudio("/api/speech/tutor", audio, options),
  /** An encoded - or predicted - `aud:...` text back to audio that can be played. */
  speechDecode: (text, codec) => post("/api/speech/decode", codec ? { text, codec } : { text }),
  /** Code generation (see CodeGenPanel): sandbox runs, an Ollama or ChatGPT teacher / judge and 2NRL rewards. */
  codegenStart: (body) => post("/api/codegen/start", body),
  codegenHistory: () => get("/api/codegen/history"),
  codegenSolve: (body) => post("/api/codegen/solve", body),
  codegenRun: (body) => post("/api/codegen/run", body),
};
