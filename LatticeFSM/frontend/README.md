# LatticeFSM frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart
libraries) for the LatticeFSM HTTP API (`latticefsm serve`; the routes are
in `../rust/README.md`). It is the ModelKit frontend's shape
(`../../ModelKit/frontend`): the same stylesheet, form fields, alert, SVG
line chart, settings store and status-bar pattern, with this model's panels.

The header's **language model** selector picks the chain (input nodes and an output node; a prediction walks
back through the input nodes only until it has gathered enough confidence) or the N-gram model, remembered in
this browser; every language-model tab talks to the one picked.

Tabs: **Train** (a new model - the chain's nodes, unit and confidence, or N and unit - trained on server files,
uploaded files of any kind (`UploadPicker`: files, folders, drag and drop, sent in 8 MiB base64 pieces) or
pasted text, predicting while learning, optionally until N of N; history and held-out perplexity charted; for
the chain a confidence sweep), **Predict** (the next tokens as you type; for the chain the input nodes and the
walk through them), **Generate** (samples, 👍 / 👎, marks out of 10 applied as feedback, 🔊 through the
browser's speech synthesis), **Talk** (a chat whose replies are the model's continuations; per-token walk depth
for the chain; teach a reply until it is exactly right), **Network** (the chat with the network of machines,
teaching until N of N), **Converse** (the model with itself), **Chat** (Ollama talks with it and marks it),
**Score** (token-by-token probability), **Nodes** (the chain's contexts as rings, accuracy by depth),
**Feedback** (good and bad texts in bulk), **Ollama** (connection, a corpus from a prompt, auto-training),
**Voice** (browser speech recognition and synthesis), **Checkpoints** (save / load every model to a path;
named checkpoints listed, restored, downloaded, deleted, uploaded), **Settings** (the model, text size, the
remembered settings), then the lattice's own: **Matrix**, **3D**, **Run**, **Walk**, **Teach** (a regular
language), **Time**, **Compress**, **Machine**. The status bar polls `/api/stats` every 2 s and `/api/status`
every 5 s (the language models, uploads, checkpoints; skipped on a server without it).

Panel settings (the string, the stimulation, episodes, the file path) are
remembered in `localStorage` under `latticefsm.v1.`; results are not. The
footer's "settings saved in this browser" resets them.

## Building and running

`dist/` is committed and `latticefsm serve` (`make serve-prod`) serves it,
so nothing here is needed to use the page. To change it, from `..`:

```bash
make frontend-install      # npm ci
make serve                 # the backend on :8000 and Vite on http://127.0.0.1:5173 (hot reload, /api proxied); Ctrl-C stops both
make serve-frontend        # Vite alone, against a backend already running on HOST:PORT
make frontend-preview      # the built dist/ through Vite, /api proxied the same way
make frontend-build        # rebuild dist/, then commit it
make frontend-test         # node --test test/*.test.mjs - no install needed
```

`VITE_PROXY_TARGET` points the dev proxy elsewhere; `VITE_API_BASE` prefixes
the API URLs in a build that is served from another origin.

## Files

| file | what |
|---|---|
| `src/App.jsx` | the shell: header, status bar, the tab strip, the panels (all mounted, the active one shown) |
| `src/api.js` | the fetch wrapper: one `ApiError` shape, one function per route |
| `src/matrix.js` | pure helpers: cell shading, a run written out, curve points, the greedy table's rows, a cube cell's index, value and projection |
| `src/components/MatrixPanel.jsx`, `EdgeCard.jsx` | the slice and the edge's record |
| `src/lm.js`, `src/upload.js`, `src/speech.js`, `src/talk.js` | pure helpers: the chain's boxes, token tints, marks → feedback, sweep and history points; chunked base64 uploads; browser speech; one Talk turn |
| `src/hooks/useLmInfo.js` | the chosen language model's info, refetched on a switch or a move |
| `src/components/LmTrainPanel.jsx`, `PredictPanel.jsx`, `GeneratePanel.jsx`, `TalkPanel.jsx`, `ConversePanel.jsx`, `ChatPanel.jsx`, `ScorePanel.jsx`, `NodesPanel.jsx`, `FeedbackPanel.jsx`, `OllamaPanel.jsx`, `VoicePanel.jsx` | the language-model tabs |
| `src/components/ModelSelector.jsx`, `UploadPicker.jsx`, `RatingsCard.jsx`, `HearButton.jsx`, `TokenText.jsx` | shared by them |
| `src/components/CheckpointPanel.jsx`, `SettingsPanel.jsx`, `NetworkPanel.jsx`, `CubePanel.jsx` | saving and restoring; this browser's settings; the conversation with a network; the 3D view of the matrix |
| `src/components/RunPanel.jsx`, `WalkPanel.jsx`, `TrainPanel.jsx`, `TimePanel.jsx`, `CompressPanel.jsx`, `MachinePanel.jsx` | the other tabs |
| `src/components/StatusBar.jsx` | the poll of `/api/stats` and `/api/status` |
| `src/components/Fields.jsx`, `Alert.jsx`, `LineChart.jsx`, `src/hooks/useStoredState.js`, `src/storage.js`, `src/util.js`, `src/styles.css` | ModelKit's, as they are (the storage prefix renamed; `util.js` cut to what this app uses; a block for the matrix added to the stylesheet) |
| `test/` | `node --test`: the settings store (ModelKit's test), the matrix helpers, the language-model, speech and formatting helpers, and the chunked upload |
