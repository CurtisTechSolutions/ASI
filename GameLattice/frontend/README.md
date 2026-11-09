# GameLattice frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart
libraries) for the GameLattice HTTP API (`gamelattice serve`; the routes are
in `../README.md` and at the top of `../rust/src/server.rs`). It is the
LatticeFSM frontend's shape (`../../LatticeFSM/frontend`, itself ModelKit's):
the same stylesheet, form fields, alert, SVG line chart, job progress,
settings store and status-bar pattern, with this model's panels.

Tabs: **Body** (make a body for one of the four languages - the encoder's
states, window, gram size and seed, and E1's arms as toggles: encoder sets the
state, forward lessons, punish, whole run, label set; train it as a background
job with the held-out accuracy of the encoder and of the lattice charted as it
goes; read a string quietly with Classify or as one learning episode with Run,
and see the **naming trace**: one row a character - the seats the auction sold
with each node's gram, bucket, bid and the slot it played (abstain shown as
such), the aggregate over the states as a bar strip with the named state
filled and the lattice's prediction marked ▾, the margin, the clearing price,
the bidders and what would have asked the meta-player), **Game** (one step of
the last read in detail: every seat as a card with its belief over the states,
its bid, reputation and wealth, the slot it played and - after a learning run -
its Shapley value φ; the auction, the equilibrium, the aggregate, the named
state against the lattice's prediction), **Lattice** (the matrix slice by
symbol, LatticeFSM's view of it with the edge record on a click, and the greedy
transition table with the start and the accepting states marked), **Encoder**
(the gram graph's size and compression, the named-state histogram, diversity
against log2 S, the wealth's mean and Gini coefficient, THINKs, the richest
nodes), **Nim** (a fresh Nim body, babbling, play against a random opponent;
the totals as tiles - wins, legal first try, refusals a move, optimal moves,
I(state; nim-sum), states used, agreement - and the games one by one),
**Results** (E1's summary per arm against LatticeFSM's 0.894 / 0.924 / 0.944
and per arm, language and seed; E2 by window; read from `../results/`, with a
note when a file is missing), **Settings** (the text size, where the API is,
the remembered settings and a reset). The status bar polls `/api/status` every
3 s (every second while a job runs): the language and the lattice's shape, the
episodes and the last held-out accuracy, diversity, agreement, the wealth, the
Nim body, and the job with a stop button.

Panel settings (the language, the states, the window, the arms, the episodes,
the text, Nim's settings) are remembered in `localStorage` under
`gamelattice.v1.`; results and traces are not. The footer's "settings saved in
this browser" (or Settings → Reset settings) forgets them.

## Building and running

`dist/` is committed and `gamelattice serve` (`make serve-prod`) serves it,
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
| `src/App.jsx` | the shell: header, status bar, the tab strip, the panels (all mounted, the active one shown); the last read, shared by Body and Game |
| `src/api.js` | the fetch wrapper: one `ApiError` shape, one function per route, `runJob` for the background jobs |
| `src/trace.js` | pure helpers of the naming trace: the buckets, the best state, an action's label, a strip's bars, a belief split from abstain, an aggregate written out, the walk in a line |
| `src/results.js` | pure helpers of the Results tab: E1's marks, summary and per-language groups; E2 by window |
| `src/matrix.js`, `src/nim.js` | cell shading and the greedy table (LatticeFSM's), the held-out curve as two series; Nim's totals as tiles, a game in a line |
| `src/components/BodyPanel.jsx`, `NamingTrace.jsx`, `Strip.jsx` | the Body tab, the trace table, the bar strip over the states |
| `src/components/GamePanel.jsx`, `LatticePanel.jsx`, `EncoderPanel.jsx`, `NimPanel.jsx`, `ResultsPanel.jsx`, `SettingsPanel.jsx` | the other tabs |
| `src/components/StatusBar.jsx` | the poll of `/api/status` |
| `src/components/Fields.jsx`, `Alert.jsx`, `LineChart.jsx`, `JobProgress.jsx`, `EdgeCard.jsx`, `src/hooks/useStoredState.js`, `src/storage.js`, `src/jobs.js`, `src/util.js`, `src/styles.css` | LatticeFSM's, as they are (the storage prefix renamed; a job counts episodes and symbols, or games and plies; `util.js` cut to what this app uses; a block for the trace, the seats and the tiles added to the stylesheet) |
| `test/` | `node --test`: the settings store (ModelKit's test), the job line, the trace helpers, the results shaping (against `../results/*.json` too), the matrix and Nim helpers |
