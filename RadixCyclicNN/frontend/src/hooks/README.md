# hooks

React hooks shared by the panels. One file, because there is one thing every
panel needs to do the same way.

| file | what it is |
|---|---|
| `useJob.js` | the lifecycle of one asynchronous server job: start it, poll `GET /api/job` every second until it leaves the `running` state, stop it, and keep polling until the terminal state is actually observed |

## Why it exists

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
