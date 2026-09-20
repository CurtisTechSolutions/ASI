# RadixCyclicNN frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart libraries)
for the RadixCyclicNN HTTP API described in `../DESIGN.md` sections 12 and 13.

Panels: Train, Predict, Generate, Converse, Chat, Score, Words, 2NRL, Negative, Evolve, Ollama, Tutor,
Code, Agent, Images, Speech, Checkpoints, Network settings, Graph.
The status bar polls `/api/status` every 2 s; asynchronous jobs (train, 2NRL,
evolve, codegen) are polled via `/api/job` every second and can be stopped from the UI.

**Words** (`GET /api/words`) appears only while the active model counts in
words - the word n-gram kind, which is the same model over an alphabet whose
symbols are words (`../SPEC-WordNGrams.md`). It lists every word training has
read, in the order it first read them, with how many of the graph's three-word
windows hold it. On that model every length field says *words* rather than
*characters*, Score reports per word, and the status bar carries the size of the
vocabulary: the unit follows `units` in `/api/status`, because a number whose
unit depends on the model is a number that will be read wrong.

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

`frontend/dist` is committed so a server can serve it with zero npm steps. From
`RadixCyclicNN/`, any of the three:

    python -m radixnet serve                # serves frontend/dist at http://127.0.0.1:8000/
    make go-serve                           # the Go count / reward model
    make rust-serve                         # the Rust count / reward model

Rebuild and recommit `dist` whenever `src/` changes.

## The three servers, and what each one can fill

The same bundle runs against all three, because they answer the same JSON
contract (`../DESIGN.md` §12) and `/api/status` says which one is replying in
`engine`. What differs is how much of the contract each one has:

| engine | what it serves |
|---|---|
| `python` | everything: every panel in the list above |
| `go` | the model, the negative network, the tutor, code, the agent and the LLM clients; no scheduler and no MCP |
| `rust` | the model itself - Train, Predict, Generate, Score, Words, 2NRL, Network settings and Graph, over any encoding |

All three switch between a character and a word **encoding** from the selector
in Network settings. An encoding is fixed for a model's life - the labels, the
split rules and the file are all measured in its units - so switching is to a
different model, and the one that was running is kept as it was left: switch to
words, back, and forward again and the training is still there. Each encoding
has its own file (`model.count.json`, `model.word.json`), and saving follows
whichever is active. `POST /api/reset` is where a new encoding is chosen.

A panel whose API the running server does not have **hides itself** rather than
failing: `App.jsx` reads `engine` from `/api/status` and drops the tabs that
engine cannot answer. So the Rust badge in the status bar is also the
explanation for the shorter tab row, and nothing in the app has to be rebuilt to
point it at a different one.

## Network settings

The **Network settings** panel holds the settings of the network itself, as
opposed to the options of one run:

* the **traversal** every search uses - `reward` (the model's own distribution,
  rewards and all) or `punishment` (the rewards leave the score and the
  penalties price every step, so the cheapest path is the least punished one),
  with a penalty and a merit scale. The Predict and Generate tabs show the same
  control: it is one setting (`src/hooks/useNetworkSettings.jsx`), remembered in
  this browser, and changing it anywhere changes it everywhere;
* the **score function** of whichever kind is active - the count model's dual
  frequency scales and sliding window, the resonant model's phase and
  resonance settings (`GET /api/model` to read, `POST /api/model/weights` to
  apply). The sine model has no score function to set and says why; the
  negative network's blame function stays on the Negative tab beside the
  failures it weighs;
* the **encoder / decoder** (`GET /api/encoding`, `POST /api/encoding/preview`):
  the sliding window, its stride and the three sentinels - read-only, because
  the window is part of the model format rather than a setting - and a live
  preview that encodes a text, decodes it back and walks it through the graph's
  own node labels, where a label longer than the window is a radix chain the
  graph merged into one node.

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
    src/hooks/useNetworkSettings.jsx  the settings several panels share, held once (the traversal)
    src/components/*.jsx        StatusBar, panels, GraphView, LineChart, shared widgets
