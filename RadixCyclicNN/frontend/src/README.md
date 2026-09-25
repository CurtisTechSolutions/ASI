# src

The source of the RadixCyclicNN single-page app. **Vite + React, plain JSX, one
CSS file, no UI or chart library** — the graph view and the line chart are hand-
written SVG.

`npm run build` compiles this into `../dist/`, which is committed so the Python
and Go servers can serve the page with zero npm steps. **Rebuild and recommit
`dist` whenever anything here changes.**

## Contents

| file | what it is |
|---|---|
| `main.jsx` | the React root |
| `App.jsx` | the header, the status bar and the tabbed panels |
| `api.js` | the fetch wrapper — JSON, and `{"error": ...}` turned into a thrown error; `converseStream` reads a route that answers as JSON Lines, one event at a time; `talkStream` reads a streamed reply from `/v1/messages` |
| `sse.js` | server-sent events read out of a byte stream one frame at a time, for the Talk panel (`test/sse.test.mjs`) |
| `stream.js` | a streamed conversation, pure: `LineParser` cuts the chunks into whole lines, `applyEvent` folds the events of the turn being spoken into what the live bubble shows (`test/stream.test.mjs`) |
| `util.js` | parsing and formatting helpers (`fmtInt`, `fmtNum`, `fmtBytes`, `fmtCounter`, `asArray`, `parseInteger`, `splitLines`, …) |
| `thinking.js` | the THINK sentinel's records read for display: the one-line summary of a thought, the questions in a text, the request bodies of `/api/think` and `/api/ollama/think`, which node ids are sentinels. Pure, and tested by `../test/thinking.test.mjs` |
| `audio.js` | microphone capture, Web Speech dictation, and WAV encoding for the Speech panel — decodes with the Web Audio API, mixes to mono, resamples to 16 kHz and writes 16-bit PCM, so the server never needs ffmpeg |
| `styles.css` | all of the styling. Responsive; a single column under 800 px |
| `components/` | the panels and shared widgets (see its own README) |
| `hooks/` | `useJob`, the async job lifecycle (see its own README) |

## Two conventions worth knowing

**Numeric fields are held as strings.** A panel keeps what was typed, so a field
can be cleared mid-edit; parsing happens on submit, through `util.js`. The
`Fields.jsx` controls assume this.

**Every long-running action is a job.** Train, 2NRL, evolve, codegen, tutor,
agent and chat all start a background job on the server and are then polled
through `useJob` — never awaited on the request.

## Development

From `..`:

```bash
npm install
npm run dev          # http://localhost:5173
```

with the API running from `../..`:

```bash
python -m radixnet serve            # 127.0.0.1:8000
```

`../vite.config.js` proxies every `/api/*` request to the Python server, so
there is no CORS or base-URL setup. `VITE_API_BASE` points a *build* at another
host; empty (the default) means relative `/api/...` URLs, which is what both
servers need.

`../README.md` documents what each panel does.
