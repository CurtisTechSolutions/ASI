# RadixCyclicNN frontend

Single-page React app (Vite, plain JSX, one CSS file, no UI or chart libraries)
for the RadixCyclicNN HTTP API described in `../DESIGN.md` sections 12 and 13.

Panels: Train, Predict, Generate, Score, 2NRL, Evolve, Checkpoints, Graph.
The status bar polls `/api/status` every 2 s; asynchronous jobs (train, 2NRL,
evolve) are polled via `/api/job` every second and can be stopped from the UI.

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
    src/styles.css              all styling (responsive; single column under 800 px)
    src/hooks/useJob.js         async job lifecycle (start, poll /api/job, stop)
    src/components/*.jsx        StatusBar, panels, GraphView, LineChart, shared widgets
