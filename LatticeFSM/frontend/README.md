# LatticeFSM frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart
libraries) for the LatticeFSM HTTP API (`latticefsm serve`; the routes are
in `../rust/README.md`). It is the ModelKit frontend's shape
(`../../ModelKit/frontend`): the same stylesheet, form fields, alert, SVG
line chart, settings store and status-bar pattern, with this model's panels.

Tabs: **Walk** (the focus - a slider over the central vertical vector, none,
or learned from the input, with what the learner reads off a string - skips
and their margin, and the nodes rearranging themselves or two of them
swapped), **Matrix** (one symbol-slice of the matrix at a time — on the default
13 × 13 × 13 machine, thirteen slices of 13 × 13 — each cell shaded by its
probability, a click opening the edge's record and weighting function), **Run** (a string at a stimulation of your choosing, traversed or
asked quietly; reward or punish the last run), **Train** (a language for
some episodes, with the accuracy curve and the greedy table), **Time**
(ticks; raise or set the stimulation), **Compress** (fold the matrix into
its central node at a precision or a budget; the code's size and loss, its
rebuild shell by shell from the centre outward, a rebuild to any shell, a
walk straight from the code from the start or the middle, codes saved and
loaded), **Machine** (a fresh machine, compression every N transitions, save
and load, teach one edge). The Run tab can start from the middle state; the
Matrix tab rings the central node on its slice. The status bar polls `/api/stats` every 2 s, and
every panel that moves the machine makes the others refresh at once.

Panel settings (the string, the stimulation, episodes, the file path) are
remembered in `localStorage` under `latticefsm.v1.`; results are not. The
footer's "settings saved in this browser" resets them.

## Building and running

`dist/` is committed and `latticefsm serve` serves it, so nothing here is
needed to use the page. To change it, from `..`:

```bash
make frontend-install      # npm ci
make frontend-dev          # Vite on http://localhost:5173, proxying /api to a running `latticefsm serve` on :8000
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
| `src/matrix.js` | pure helpers: cell shading, a run written out, curve points, the greedy table's rows |
| `src/components/MatrixPanel.jsx`, `EdgeCard.jsx` | the slice and the edge's record |
| `src/components/RunPanel.jsx`, `WalkPanel.jsx`, `TrainPanel.jsx`, `TimePanel.jsx`, `CompressPanel.jsx`, `MachinePanel.jsx` | the other tabs |
| `src/components/StatusBar.jsx` | the poll of `/api/stats` |
| `src/components/Fields.jsx`, `Alert.jsx`, `LineChart.jsx`, `src/hooks/useStoredState.js`, `src/storage.js`, `src/util.js`, `src/styles.css` | ModelKit's, as they are (the storage prefix renamed; `util.js` cut to what this app uses; a block for the matrix added to the stylesheet) |
| `test/` | `node --test`: the settings store (ModelKit's test) and the matrix helpers |
