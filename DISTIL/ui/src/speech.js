// Speaking instead of typing.
//
// Two routes, and which one runs decides where your voice goes -- so this
// reports honestly rather than putting a microphone icon on it and leaving the
// question open.
//
//   local     the server found a whisper binary; audio is recorded here, POSTed
//             to /api/transcribe, and never leaves the machine.
//   browser   the Web Speech API. Free and instant, but in Chrome it is NOT
//             local: the audio goes to Google for recognition. For a tool whose
//             premise is that it runs on your machine with no key and no
//             network, that is a contradiction worth stating at the point of
//             use, which is what `warning` is for.

export const SpeechKind = { LOCAL: 'local', BROWSER: 'browser', NONE: 'none' }

const Recognition =
  typeof window !== 'undefined' &&
  (window.SpeechRecognition || window.webkitSpeechRecognition)

export function browserAvailable() {
  return Boolean(Recognition)
}

// Which route to use, given what the server reported from /api/speech.
export function route(serverSpeech) {
  if (serverSpeech && serverSpeech.local) return SpeechKind.LOCAL
  if (browserAvailable()) return SpeechKind.BROWSER
  return SpeechKind.NONE
}

export function warning(kind, serverSpeech) {
  if (kind === SpeechKind.LOCAL) {
    return { level: 'ok', text: `transcribed on this machine by ${serverSpeech.backend}` }
  }
  if (kind === SpeechKind.BROWSER) {
    return {
      level: 'warn',
      text: 'your browser transcribes this — in Chrome that sends the audio to Google. ' +
            'Install whisper.cpp and set DISTIL_WHISPER_MODEL to keep it on this machine.',
    }
  }
  return { level: 'off', text: serverSpeech ? serverSpeech.why : 'no speech recognition available' }
}

// ---------------------------------------------------------------------------
// browser route
// ---------------------------------------------------------------------------

// Interim results are passed up as they arrive so the box fills while you talk.
// Waiting for the final transcript makes it feel broken for the several seconds
// a sentence takes.
export function listenInBrowser({ onPartial, onFinal, onError }) {
  if (!Recognition) {
    onError(new Error('this browser has no speech recognition'))
    return () => {}
  }
  const rec = new Recognition()
  rec.continuous = true
  rec.interimResults = true
  rec.lang = navigator.language || 'en-US'

  let settled = ''
  rec.onresult = (event) => {
    let pending = ''
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const chunk = event.results[i][0].transcript
      if (event.results[i].isFinal) settled += chunk
      else pending += chunk
    }
    onPartial((settled + pending).trim())
  }
  rec.onerror = (event) => {
    // `no-speech` and `aborted` are what happens when you stop talking or press
    // stop. Reporting them as failures would make ordinary use look broken.
    if (event.error === 'no-speech' || event.error === 'aborted') return
    onError(new Error(event.error === 'not-allowed'
      ? 'microphone permission was refused'
      : `speech recognition failed: ${event.error}`))
  }
  rec.onend = () => onFinal(settled.trim())

  try {
    rec.start()
  } catch (err) {
    onError(err)
    return () => {}
  }
  return () => { try { rec.stop() } catch { /* already stopped */ } }
}

// ---------------------------------------------------------------------------
// local route
// ---------------------------------------------------------------------------

// MediaRecorder gives webm/opus; whisper.cpp wants a 16kHz mono wav. Converting
// in the browser with an OfflineAudioContext keeps the server dependency-free,
// which is the whole reason the server can stay standard-library only.
export async function recordForServer({ onError }) {
  let stream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch (err) {
    onError(new Error('microphone permission was refused'))
    return null
  }
  const chunks = []
  const recorder = new MediaRecorder(stream)
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data)
  recorder.start()

  return {
    async stop() {
      const done = new Promise((resolve) => { recorder.onstop = resolve })
      recorder.stop()
      await done
      stream.getTracks().forEach((t) => t.stop())
      if (!chunks.length) return null
      return encodeWav(await decode(new Blob(chunks)))
    },
  }
}

async function decode(blob) {
  const Ctx = window.AudioContext || window.webkitAudioContext
  const ctx = new Ctx()
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer())
    return await resample(decoded, 16000)
  } finally {
    ctx.close()
  }
}

async function resample(buffer, rate) {
  const frames = Math.ceil((buffer.duration * rate))
  const off = new OfflineAudioContext(1, frames, rate)
  const src = off.createBufferSource()
  src.buffer = buffer
  src.connect(off.destination)
  src.start()
  return off.startRendering()
}

// 16-bit PCM wav, written by hand. A library for 44 bytes of header would be the
// only runtime dependency in the project.
function encodeWav(buffer) {
  const samples = buffer.getChannelData(0)
  const out = new DataView(new ArrayBuffer(44 + samples.length * 2))
  const ascii = (at, text) => [...text].forEach((c, i) => out.setUint8(at + i, c.charCodeAt(0)))

  ascii(0, 'RIFF')
  out.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVEfmt ')
  out.setUint32(16, 16, true)            // PCM header size
  out.setUint16(20, 1, true)             // format: uncompressed
  out.setUint16(22, 1, true)             // mono
  out.setUint32(24, buffer.sampleRate, true)
  out.setUint32(28, buffer.sampleRate * 2, true)
  out.setUint16(32, 2, true)             // block align
  out.setUint16(34, 16, true)            // bits per sample
  ascii(36, 'data')
  out.setUint32(40, samples.length * 2, true)

  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    out.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
  }
  return new Blob([out], { type: 'audio/wav' })
}
