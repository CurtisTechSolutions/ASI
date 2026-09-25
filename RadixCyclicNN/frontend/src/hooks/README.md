# hooks

React hooks shared by the panels.

| file | what it is |
|---|---|
| `useJob.js` | the lifecycle of one asynchronous server job: start it, poll `GET /api/job` every second until it leaves the `running` state, stop it, and keep polling until the terminal state is actually observed |
| `useStoredState.js` | `useState` that remembers its value in this browser (see `../storage.js`) — one panel's own setting, under its own name |
| `useSiteSettings.jsx` | the settings **several** panels share, held once in a provider at the top of the app: the traversal (through `useNetworkSettings`), the sampling filters and the diversity, how a run walks its texts, and whether Predict and Generate ask the model backwards. Edited on the Settings tab, and read (and written) by the Predict, Generate and Train tabs, which show the same controls. The rules are pure functions in `../settings.js` and `../backwards.js` |
| `useNetworkSettings.jsx` | the traversal a search runs and its two scales — one of the site-wide settings, under its original `network.*` names |

## Why `useJob` exists

Training, 2NRL, evolve, codegen, tutor, agent and chat all run too long for a
request to wait on. The server starts a background job and returns its handle;
the browser polls. `useJob` is that loop, written once — a panel calls it and
gets back the job plus a `start` / `stop` pair, and never manages a timer itself.

```jsx
const { job, running, busy, error, start, stop, clearError } = useJob("train");

// `start` takes the function that POSTs; it resolves to {"job": {...}}
start(() => api.train(body));
stop();                                  // defaults to POST /api/job/stop
```

## Three behaviours worth knowing

* **It adopts a job in progress.** On mount it reads `GET /api/job` and, if the
  running job's `type` matches, follows it — so reopening the page during a long
  run keeps the live view instead of showing nothing.
* **It stops following a job that was replaced.** If the polled job's `id` is no
  longer the one being tracked, the hook lets go rather than reporting another
  panel's progress as its own.
* **A failed poll is not an error state.** It retries after 2 s (against 1 s
  while healthy) and says so through `error`.

`jobIsRunning(job)` in `../util.js` is the predicate panels use to disable their
controls, and `JobStatus.jsx` renders the state badge, the timestamps and the
error. The status bar polls separately, every 2 s, from `StatusBar.jsx`.

## Why the site-wide settings are a provider and not another `useStoredState`

`useStoredState` remembers a value per name, but two mounted components using
the same name share the *stored* value and not the state — and every panel here
stays mounted while it is hidden, so two copies of one setting would drift apart
within a session and only agree again after a reload. A setting that belongs to
the site rather than to a panel therefore lives **once**, in
`SiteSettingsProvider` at the top of `App.jsx`, and every panel reads it
through `useSiteSettings()`: one source of truth, several doors into it.

```jsx
const { network, search, training, backwards } = useSiteSettings();

network.set("traversal", "punishment");  // changes it everywhere at once
search.set("topP", "0.9");
await api.predict({ prefix, mode, ...network.body, ...search.body(mode) });  // only what `mode` reads, only what is on
await api.train({ texts, ...training.body() });
search.problems;                         // {topP: "Top-p must lie in (0, 1] ..."} while a field is out of range
backwards.on;                            // ask a model trained backwards: turn the query around, and the answer back
```

Outside the provider the hook returns the defaults with no-op setters, so a
component rendered on its own still works.
