/* AudioImage front end.
 *
 * The server does all of the audio work - every button here is one of the
 * commands the tool already has.  What the browser adds is the part that only
 * makes sense live: the playhead running across the plane while the sound
 * plays, a readout of what is under the cursor, and a brush, so you can paint
 * on the plane and hear what you painted.
 *
 * Audio that is not already a WAV is decoded by the browser and re-encoded
 * here, so the server only ever has to understand one format.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  info: null,
  sourceWav: null,      // Blob: what was loaded, as a WAV
  sourceMeta: null,
  planeBlob: null,      // Blob: the picture exactly as the server wrote it
  planeMeta: null,      // the settings inside it
  planeImage: null,     // ImageBitmap/Image, for drawing
  decodedWav: null,
  painted: false,       // true once the brush has touched the plane
  audio: new Audio(),
  playing: false,
};

/* ---------------------------------------------------------------- helpers */

function toast(message, kind = "error") {
  const el = $("toast");
  el.textContent = message;
  el.style.borderLeftColor = kind === "error" ? "var(--bad)" : "var(--good)";
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 6000);
}

function status(id, message, kind = "") {
  const el = $(id);
  el.textContent = message;
  el.className = "status" + (kind ? " " + kind : "");
}

function clock(seconds) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Read the report the binary routes put in their header. */
function readMeta(response) {
  const raw = response.headers.get("X-AudioImage-Meta");
  if (!raw) return {};
  try {
    return JSON.parse(new TextDecoder().decode(
      Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))));
  } catch (err) {
    return {};
  }
}

/** POST a body to a route, returning [blob, report]; throws the server's message. */
async function post(route, params, body) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(route + (query ? "?" + query : ""), {
    method: "POST",
    body: body || new Blob([]),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const err = await response.json();
      if (err && err.error) message = err.error;
    } catch (_) { /* not JSON; keep the status line */ }
    throw new Error(message);
  }
  const meta = readMeta(response);
  return [await response.blob(), meta];
}

/* ------------------------------------------------------- audio conversion */

/** Any format the browser can decode -> a mono 16-bit WAV, as a Blob. */
async function toWav(arrayBuffer) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) throw new Error("this browser has no Web Audio, so it cannot convert that file");
  const ctx = new Ctx();
  let buffer;
  try {
    buffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
  } catch (err) {
    throw new Error("the browser could not decode that file");
  } finally {
    ctx.close();
  }
  // mix to mono
  const n = buffer.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += data[i];
  }
  if (buffer.numberOfChannels > 1) {
    for (let i = 0; i < n; i++) mono[i] /= buffer.numberOfChannels;
  }
  return wavBlob(mono, Math.round(buffer.sampleRate));
}

/** Float samples -> a 16-bit PCM WAV file. */
function wavBlob(samples, rate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const text = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(v * 32767), true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

/* ------------------------------------------------------------- the canvas */

const plane = $("plane");
const overlay = $("overlay");
const planeCtx = plane.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlay.getContext("2d");

/** Put a picture on the plane canvas at its own pixel size. */
async function showPlane(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("the picture could not be displayed"));
      img.src = url;
    });
    plane.width = image.width;
    plane.height = image.height;
    overlay.width = image.width;
    overlay.height = image.height;
    planeCtx.imageSmoothingEnabled = false;
    planeCtx.clearRect(0, 0, plane.width, plane.height);
    planeCtx.drawImage(image, 0, 0);
    state.planeImage = image;
    $("plane-empty").hidden = true;
    drawAxes();
    drawOverlay(0);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** A blank, silent plane to paint on. */
function blankPlane() {
  const meta = state.planeMeta || {};
  const n_fft = Number($("n_fft").value) || 1024;
  const width = plane.width && state.planeImage ? plane.width : 256;
  plane.width = width;
  plane.height = (n_fft / 2) + 1;
  overlay.width = plane.width;
  overlay.height = plane.height;
  planeCtx.fillStyle = "#fff";           // white is 0.0 - silence
  planeCtx.fillRect(0, 0, plane.width, plane.height);
  state.planeImage = null;
  state.painted = true;
  state.planeMeta = Object.assign({}, meta, {
    n_fft: n_fft,
    hop: Number($("hop").value) || n_fft / 4,
    sample_rate: Number($("rate-override").value) || meta.sample_rate || 22050,
    height: plane.height,
    width: plane.width,
  });
  $("plane-empty").hidden = true;
  $("revert").disabled = !state.planeBlob;
  $("decode").disabled = false;
  drawAxes();
  drawOverlay(0);
  status("decode-status", "A blank plane. Paint on it, then decode to hear it.");
}

/** Frequency ticks beside the plane, time ticks beneath it. */
function drawAxes() {
  const meta = state.planeMeta || {};
  const rate = Number(meta.sample_rate) || 22050;
  const hop = Number(meta.hop) || 256;
  const fMax = Number(meta.f_max) || rate / 2;
  const fMin = Number(meta.f_min) || 0;
  const stage = plane.getBoundingClientRect();

  const y = $("axis-y");
  y.height = Math.max(40, Math.round(stage.height));
  y.width = 60;
  const yc = y.getContext("2d");
  yc.clearRect(0, 0, y.width, y.height);
  yc.fillStyle = "#98a0b0";
  yc.font = "11px ui-monospace, Menlo, monospace";
  yc.textAlign = "right";
  const steps = Math.max(2, Math.min(9, Math.floor(y.height / 46)));
  for (let i = 0; i < steps; i++) {
    const frac = i / (steps - 1);                 // 0 at the bottom
    const hz = fMin + (fMax - fMin) * frac;
    const py = Math.round((1 - frac) * (y.height - 1));
    yc.fillText(hz >= 1000 ? (hz / 1000).toFixed(1) + "k" : Math.round(hz), 50, Math.min(y.height - 2, py + 4));
    yc.fillRect(53, py, 5, 1);
  }

  const x = $("axis-x");
  x.width = Math.max(40, Math.round(stage.width));
  x.height = 26;
  const xc = x.getContext("2d");
  xc.clearRect(0, 0, x.width, x.height);
  xc.fillStyle = "#98a0b0";
  xc.font = "11px ui-monospace, Menlo, monospace";
  xc.textAlign = "center";
  const duration = (plane.width * hop) / rate;
  const tsteps = Math.max(2, Math.min(10, Math.floor(x.width / 70)));
  for (let i = 0; i < tsteps; i++) {
    const frac = i / (tsteps - 1);
    const px = Math.round(frac * (x.width - 1));
    xc.fillRect(px, 0, 1, 5);
    xc.fillText((duration * frac).toFixed(2) + "s", Math.max(16, Math.min(x.width - 16, px)), 17);
  }
}

/** The playhead, drawn over the plane. */
function drawOverlay(fraction) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (fraction <= 0 && !state.playing) return;
  const x = Math.round(fraction * (overlay.width - 1));
  overlayCtx.fillStyle = "rgba(110, 168, 254, 0.9)";
  overlayCtx.fillRect(x, 0, Math.max(1, Math.round(overlay.width / 400)), overlay.height);
}

/* --------------------------------------------------------------- playback */

function currentClip() {
  return $("which-decoded").checked && state.decodedWav ? state.decodedWav : state.sourceWav;
}

function loadIntoPlayer() {
  const clip = currentClip();
  if (!clip) return;
  if (state.audio.src) URL.revokeObjectURL(state.audio.src);
  state.audio.src = URL.createObjectURL(clip);
  state.audio.loop = $("loop").checked;
  $("play").disabled = false;
  $("stop").disabled = false;
}

function tick() {
  if (!state.playing) return;
  const duration = state.audio.duration;
  if (isFinite(duration) && duration > 0) {
    drawOverlay(state.audio.currentTime / duration);
    $("time-now").textContent = clock(state.audio.currentTime);
    $("time-total").textContent = clock(duration);
  }
  requestAnimationFrame(tick);
}

/* ----------------------------------------------------------- the settings */

function encodeParams() {
  const params = {
    n_fft: $("n_fft").value,
    hop: $("hop").value,
    window: $("window").value,
    scale: $("scale").value,
    top_db: $("top_db").value,
    freq_scale: $("freq_scale").value,
    depth: $("depth").value,
    colormap: $("colormap").value,
    phase: $("phase").value,
    lossless: $("lossless").checked ? "1" : "0",
  };
  if ($("lossless").checked) params.phase = "rgb";
  // storing the phase needs the exactly-invertible layout and the grey ramp,
  // so the controls that would break it are held at their only valid value
  if (params.phase !== "none") {
    params.freq_scale = "linear";
    params.colormap = "gray";
  }
  // a warped frequency axis loses accuracy when it has fewer rows than there
  // are bins, so give it one row per bin unless the user says otherwise
  if (params.freq_scale !== "linear") {
    params.height = String(Number(params.n_fft) / 2 + 1);
  }
  return params;
}

/* ---------------------------------------------------------------- actions */

async function setSource(blob, label) {
  state.sourceWav = blob;
  state.decodedWav = null;
  state.painted = false;
  $("which-original").checked = true;
  $("which-decoded").disabled = true;
  $("measure").disabled = false;
  loadIntoPlayer();
  status("source-status", label, "ok");
  $("dl-src").href = URL.createObjectURL(blob);
  await doEncode();
}

async function doEncode() {
  if (!state.sourceWav) { toast("Load or synthesise a sound first."); return; }
  const button = $("encode");
  button.disabled = true;
  status("decode-status", "Encoding…", "busy");
  try {
    const [blob, meta] = await post("/api/encode", encodeParams(), state.sourceWav);
    state.planeBlob = blob;
    state.planeMeta = meta.meta || {};
    state.painted = false;
    await showPlane(blob);
    $("decode").disabled = false;
    $("revert").disabled = true;
    $("dl-png").href = URL.createObjectURL(blob);
    $("downloads").hidden = false;
    const m = state.planeMeta;
    status("decode-status",
      `${meta.image.width} columns × ${meta.image.height} rows, ${meta.image.depth}-bit — ` +
      `n_fft ${m.n_fft}, hop ${m.hop}, ${m.scale} over ${m.top_db} dB. ` +
      `Encoded in ${meta.seconds}s.`, "ok");
    fetchView();
  } catch (err) {
    status("decode-status", String(err.message || err), "error");
    toast(String(err.message || err));
  } finally {
    button.disabled = false;
  }
}

/** The picture as it stands: the server's bytes, unless the brush touched it. */
function planeForDecode() {
  if (!state.painted) return Promise.resolve(state.planeBlob);
  return new Promise((resolve, reject) =>
    plane.toBlob((blob) => blob ? resolve(blob) : reject(new Error("the plane could not be read")), "image/png"));
}

async function doDecode() {
  const button = $("decode");
  button.disabled = true;
  status("decode-status", "Rebuilding the phase…", "busy");
  try {
    const body = await planeForDecode();
    if (!body) throw new Error("there is no picture to decode");
    const params = Object.assign(encodeParams(), { iters: $("iters").value });
    // a painted plane carries no settings of its own, so send them alongside
    const rate = $("rate-override").value || (state.planeMeta && state.planeMeta.sample_rate) || "";
    if (rate) params.rate = String(rate);
    const [blob, meta] = await post("/api/decode", params, body);
    state.decodedWav = blob;
    $("which-decoded").disabled = false;
    $("which-decoded").checked = true;
    loadIntoPlayer();
    $("dl-wav").href = URL.createObjectURL(blob);
    $("downloads").hidden = false;
    const how = meta.lossless
      ? "exact - the correction channel restored the original samples"
      : meta.stored_phase
      ? "phase read straight from the picture - nothing guessed"
      : `${meta.iterations} griffin-lim passes, final error ${Number(meta.griffin_lim_error).toFixed(4)}`;
    status("decode-status",
      `${meta.duration.toFixed(2)}s at ${meta.sample_rate} Hz, ${how}, in ${meta.seconds}s. ` +
      (meta.had_metadata ? "" : "(the picture carried no settings, so the ones above were used)"), "ok");
  } catch (err) {
    status("decode-status", String(err.message || err), "error");
    toast(String(err.message || err));
  } finally {
    button.disabled = false;
  }
}

async function doMeasure() {
  if (!state.sourceWav) return;
  const button = $("measure");
  button.disabled = true;
  status("decode-status", "Encoding, decoding and measuring…", "busy");
  try {
    const query = new URLSearchParams(Object.assign(encodeParams(), { iters: $("iters").value }));
    const response = await fetch("/api/roundtrip?" + query, { method: "POST", body: state.sourceWav });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    const m = payload.metrics;
    $("metrics").hidden = false;
    $("metrics").innerHTML = [
      metric("spectrum kept", (100 - m.spectral_convergence * 100).toFixed(2) + "%",
             "relative error " + (m.spectral_convergence * 100).toFixed(2) + "%"),
      metric("log distance", m.log_spectral_distance.toFixed(2) + " dB", "difference where it is audible"),
      metric("waveform SNR", m.waveform_snr.toFixed(1) + " dB",
             payload.stored_phase ? "the phase was stored, so this is real" : "expected to be poor: the phase was rebuilt"),
      metric("time", payload.encode_seconds + "s / " + payload.decode_seconds + "s", "encode / decode"),
    ].join("");
    status("decode-status", `Measured over ${payload.iterations} passes.`, "ok");
  } catch (err) {
    status("decode-status", String(err.message || err), "error");
    toast(String(err.message || err));
  } finally {
    button.disabled = false;
  }
}

function metric(key, value, note) {
  return `<div class="metric"><div class="k">${key}</div><div class="v">${value}</div><div class="n">${note}</div></div>`;
}

async function fetchView() {
  if (!state.sourceWav) return;
  try {
    const params = Object.assign(encodeParams(), { view_colormap: "magma" });
    const [blob] = await post("/api/view", params, state.sourceWav);
    $("dl-view").href = URL.createObjectURL(blob);
  } catch (err) { /* the chart is a convenience; never block on it */ }
}

/* ------------------------------------------------------------- the brush */

let painting = false;

function planePoint(event) {
  const box = overlay.getBoundingClientRect();
  return {
    x: Math.floor((event.clientX - box.left) / box.width * plane.width),
    y: Math.floor((event.clientY - box.top) / box.height * plane.height),
  };
}

function paintAt(point) {
  const size = Number($("brush").value);
  planeCtx.fillStyle = $("brush-mode").value === "ink" ? "#000" : "#fff";
  planeCtx.beginPath();
  planeCtx.arc(point.x, point.y, size / 2, 0, Math.PI * 2);
  planeCtx.fill();
  state.painted = true;
  $("revert").disabled = !state.planeBlob;
  $("decode").disabled = false;
}

function describeAt(point) {
  const meta = state.planeMeta || {};
  const rate = Number(meta.sample_rate) || 22050;
  const hop = Number(meta.hop) || 256;
  const fMax = Number(meta.f_max) || rate / 2;
  const fMin = Number(meta.f_min) || 0;
  if (point.x < 0 || point.y < 0 || point.x >= plane.width || point.y >= plane.height) return "&nbsp;";
  const frac = 1 - point.y / Math.max(1, plane.height - 1);      // origin=lower
  const hz = fMin + (fMax - fMin) * frac;
  const seconds = (point.x * hop) / rate;
  const px = planeCtx.getImageData(point.x, point.y, 1, 1).data;
  const intensity = 1 - px[0] / 255;
  const topDb = Number(meta.top_db) || 80;
  const db = intensity <= 0 ? "-inf" : ((intensity - 1) * topDb).toFixed(1);
  return `${seconds.toFixed(3)} s &nbsp;·&nbsp; ${hz.toFixed(0)} Hz &nbsp;·&nbsp; ` +
         `intensity ${intensity.toFixed(3)} &nbsp;·&nbsp; ${db} dB`;
}

/* ------------------------------------------------------------------ setup */

function fill(select, options, chosen) {
  select.innerHTML = "";
  for (const option of options) {
    const el = document.createElement("option");
    el.value = String(option);
    el.textContent = String(option);
    if (String(option) === String(chosen)) el.selected = true;
    select.appendChild(el);
  }
}

async function boot() {
  try {
    const response = await fetch("/api/info");
    state.info = await response.json();
  } catch (err) {
    toast("The server is not answering. Is `audioimage serve` still running?");
    return;
  }
  const { options, defaults, backend, version } = state.info;
  $("badge-version").textContent = "audioimage " + version;
  $("badge-backend").textContent = backend.selected + (backend.numpy ? " · numpy " + backend.numpy_version : " · standard library");

  fill($("n_fft"), [256, 512, 1024, 2048, 4096], defaults.n_fft);
  fill($("window"), options.windows, defaults.window);
  fill($("scale"), options.scales, defaults.scale);
  fill($("freq_scale"), options.freq_scales, defaults.freq_scale);
  fill($("colormap"), options.colormaps, defaults.colormap);
  fill($("depth"), options.depths, defaults.depth);
  fill($("phase"), options.phase_modes || ["none", "rgb"], defaults.phase || "none");
  fill($("kind"), options.kinds, "melody");
  $("hop").value = defaults.hop;
  $("top_db").value = defaults.top_db;
  $("iters").value = defaults.iters;
  $("iters-value").textContent = defaults.iters;

  // keep the hop at a quarter of the window unless it has been set by hand
  let hopTouched = false;
  $("hop").addEventListener("input", () => { hopTouched = true; });
  $("n_fft").addEventListener("change", () => {
    if (!hopTouched) $("hop").value = Math.max(1, Number($("n_fft").value) / 4);
  });

  $("iters").addEventListener("input", () => { $("iters-value").textContent = $("iters").value; });
  // the passes and the colour map mean nothing once the phase is in the file
  const phaseChanged = () => {
    const stored = $("phase").value !== "none" || $("lossless").checked;
    $("phase").disabled = $("lossless").checked;
    for (const id of ["iters", "colormap", "freq_scale"]) $(id).disabled = stored;
    $("iters-value").textContent = stored ? "not needed" : $("iters").value;
  };
  $("phase").addEventListener("change", phaseChanged);
  $("lossless").addEventListener("change", phaseChanged);
  phaseChanged();
  $("encode").addEventListener("click", doEncode);
  $("decode").addEventListener("click", doDecode);
  $("measure").addEventListener("click", doMeasure);

  $("file").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    status("source-status", `Reading ${file.name}…`, "busy");
    try {
      const blob = file.name.toLowerCase().endsWith(".wav") && file.type !== "audio/mpeg"
        ? file : await toWav(await file.arrayBuffer());
      await setSource(blob, `${file.name} — loaded.`);
    } catch (err) {
      status("source-status", String(err.message || err), "error");
      toast(String(err.message || err));
    }
  });

  $("synth").addEventListener("click", async () => {
    status("source-status", "Synthesising…", "busy");
    try {
      const [blob, meta] = await post("/api/tone", {
        kind: $("kind").value,
        notes: $("notes").value,
        duration: $("tone-duration").value,
        rate: $("tone-rate").value,
        freq: $("tone-freq").value,
      });
      await setSource(blob, `${$("kind").value}: ${meta.duration.toFixed(2)}s at ${meta.sample_rate} Hz.`);
    } catch (err) {
      status("source-status", String(err.message || err), "error");
      toast(String(err.message || err));
    }
  });

  setupRecording();
  setupTransport();
  setupPainting();
  window.addEventListener("resize", drawAxes);
}

function setupTransport() {
  $("play").addEventListener("click", () => {
    if (state.playing) {
      state.audio.pause();
    } else {
      state.audio.play().catch((err) => toast("The browser would not play that: " + err.message));
    }
  });
  $("stop").addEventListener("click", () => {
    state.audio.pause();
    state.audio.currentTime = 0;
    drawOverlay(0);
    $("time-now").textContent = "0:00";
  });
  $("loop").addEventListener("change", () => { state.audio.loop = $("loop").checked; });
  for (const id of ["which-original", "which-decoded"]) {
    $(id).addEventListener("change", () => {
      const was = state.playing;
      loadIntoPlayer();
      if (was) state.audio.play().catch(() => {});
    });
  }
  state.audio.addEventListener("play", () => { state.playing = true; $("play").textContent = "⏸ Pause"; tick(); });
  state.audio.addEventListener("pause", () => { state.playing = false; $("play").textContent = "▶ Play"; });
  state.audio.addEventListener("ended", () => { state.playing = false; $("play").textContent = "▶ Play"; drawOverlay(0); });
  state.audio.addEventListener("loadedmetadata", () => { $("time-total").textContent = clock(state.audio.duration); });
}

function setupPainting() {
  const seek = (event) => {
    if (!state.audio.src || !isFinite(state.audio.duration)) return;
    const box = overlay.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width));
    state.audio.currentTime = frac * state.audio.duration;
    drawOverlay(frac);
  };

  overlay.addEventListener("pointerdown", (event) => {
    if ($("paint-on").checked) {
      painting = true;
      overlay.setPointerCapture(event.pointerId);
      paintAt(planePoint(event));
    } else {
      seek(event);
    }
  });
  overlay.addEventListener("pointermove", (event) => {
    const point = planePoint(event);
    $("readout").innerHTML = describeAt(point);
    if (painting) paintAt(point);
  });
  overlay.addEventListener("pointerup", (event) => {
    painting = false;
    if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
  });
  overlay.addEventListener("pointerleave", () => { $("readout").innerHTML = "&nbsp;"; });

  $("clear").addEventListener("click", blankPlane);
  $("revert").addEventListener("click", async () => {
    if (!state.planeBlob) return;
    await showPlane(state.planeBlob);
    state.painted = false;
    $("revert").disabled = true;
  });
}

function setupRecording() {
  let recorder = null;
  let chunks = [];
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    $("record").disabled = true;
    $("record-hint").textContent = "This browser cannot record.";
    return;
  }
  $("record").addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        try {
          const blob = await toWav(await new Blob(chunks).arrayBuffer());
          await setSource(blob, "Recorded.");
        } catch (err) {
          status("source-status", String(err.message || err), "error");
        }
      };
      recorder.start();
      $("record").disabled = true;
      $("record-stop").disabled = false;
      status("source-status", "Recording…", "busy");
    } catch (err) {
      toast("The microphone is not available: " + err.message);
    }
  });
  $("record-stop").addEventListener("click", () => {
    if (recorder && recorder.state !== "inactive") recorder.stop();
    $("record").disabled = false;
    $("record-stop").disabled = true;
  });
}

boot();
