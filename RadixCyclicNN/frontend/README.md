# RadixCyclicNN frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart libraries)
for the RadixCyclicNN HTTP API described in `../DESIGN.md` sections 12 and 13.

Panels: Train, Predict, Generate, Converse, Chat, Score, 2NRL, Negative, Evolve, Ollama, Tutor,
Code, Agent, Images, Speech, Checkpoints, Graph.
The status bar polls `/api/status` every 2 s; asynchronous jobs (train, 2NRL,
evolve, codegen) are polled via `/api/job` every second and can be stopped from the UI.

The Converse panel (`POST /api/converse`) lets the model talk to itself in a
chat view that reads newest first: a new turn is appended to the top and pushes
the older ones down, so the latest reply is where the eye already is and nothing
has to be scrolled to. The search skips the replies the conversation has already
heard and, while "Avoid repeated words" is on, the ones that say the same words
twice in a row - and a voice that catches itself repeating, its own words or
the conversation's, keeps what it said once, backs up to where it would have
said them again and explores other ways on ("Explore"), each turn saying what
it noticed and whether it found one. With "Learn where it goes round" on (the
default) what a rethink finds out is taught to the graph, so the model itself
hands over there next time - which means a conversation changes the model. The
duplicates it could not avoid come back flagged, and "Punish duplicates" marks
them 👎 so "Train on ratings" runs the 2NRL negative phase on them.

The Chat panel (`POST /api/chat/start`) has an LLM converse with the model and
mark every reply; its transcript reads newest first as well, it has the same
"Avoid repeated words" and "Explore" settings, and a reply the model could only
repeat is punished with the failures whatever the judge made of it.

The Ollama panel talks to a local Ollama server through the API
(`GET /api/ollama/models`, `POST /api/ollama/corpus`, `POST /api/ollama/review`).
It writes a training corpus from a prompt (good or garbage style) that can be
trained on, saved as an upload or held as 2NRL data, and it acts as an
adversarial reviewer that rates samples from the model so the failed ones can
be fed back through 2NRL. The Ollama URL and model default to the server's
settings (`ollama` in `/api/status`) and can be overridden per request.

The Speech panel teaches the model by talking to it (`GET /api/speech`,
`POST /api/speech/teach`, `POST /api/speech/decode`). It records the microphone
with `MediaRecorder` and writes down what it hears with the Web Speech API at
the same time; `src/audio.js` decodes the recording with the Web Audio API,
mixes it to mono, resamples it to 16 kHz and encodes a 16-bit PCM WAV, so the
server reads it with the standard library and never needs ffmpeg for a browser
recording. One utterance is posted as two texts behind the same unique token -
the transcript and the waveform quantised to one mu-law byte per sample - and
trained on; any `aud:` text, a prediction included, can be decoded back into
audio and played in the last card.

The Code panel drives the code-generation loop (`POST /api/codegen/start`,
`GET /api/codegen/history`, `POST /api/codegen/solve`, `POST /api/codegen/run`).
Problems typed one per line or read from uploaded `.txt` / `.json` / `.jsonl`
files are solved by a teacher and by the model; every program runs in the
sandbox, is style-checked and optionally judged by the same teacher, and the
outcome rewards or punishes the model through 2NRL. The **Teacher** selector
picks who tutors — a local Ollama model or ChatGPT (`teacher_provider`) — and
the model name, the URL and the notes follow it; attempt badges name the
provider that wrote each program. ChatGPT needs an `OPENAI_API_KEY` in the
server's environment (`chatgpt.configured` in `/api/status`, and
`GET /api/chatgpt/models` for the models it may use); the panel says so when
the key is missing, and a key is never sent from the browser. The panel shows
round summaries, a problem table and an attempt feed live from the `codegen`
job (and the stored history of earlier runs), solves a single problem without
training, and runs pasted code in the sandbox. Only the teacher and the judge
need an LLM.

## Development

1. Start the API from the `RadixCyclicNN/` directory:

       python -m radixnet serve            # listens on 127.0.0.1:8000

2. Start the Vite dev server:

       cd frontend
       npm install
       npm run dev                         # http://localhost:5173

   `vite.config.js` proxies every `/api/*` request to `http://127.0.0.1:8000`,
   so the app talks to the Python server without any CORS or base-URL setup.

## Production build

    cd frontend
    npm install
    npm run build                           # writes frontend/dist

`frontend/dist` is committed so the Python server can serve it with zero npm
steps. From `RadixCyclicNN/`:

    python -m radixnet serve                # serves frontend/dist at http://127.0.0.1:8000/

Rebuild and recommit `dist` whenever `src/` changes.

## Configuration

* `VITE_API_BASE` (build time, default empty string) - prefix for API calls.
  Empty means relative `/api/...` URLs, which is what the Python server needs.
  To point a build at another host (the API sends CORS headers):

      VITE_API_BASE=http://127.0.0.1:8000 npm run build

  A `.env` / `.env.local` file with `VITE_API_BASE=...` works as well.

## Layout

    index.html                  entry page
    vite.config.js              dev proxy + build output
    src/main.jsx                React root
    src/App.jsx                 header, status bar, tabbed panels
    src/api.js                  fetch wrapper (JSON + {"error": ...} handling)
    src/util.js                 parsing / formatting helpers
    src/audio.js                microphone capture, Web Speech dictation, WAV encoding (Speech panel)
    src/styles.css              all styling (responsive; single column under 800 px)
    src/hooks/useJob.js         async job lifecycle (start, poll /api/job, stop)
    src/components/*.jsx        StatusBar, panels, GraphView, LineChart, shared widgets
