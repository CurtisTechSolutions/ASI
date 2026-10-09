# rust

The Rust port: crate `latticefsm` — the model (`edge.rs`, `lattice.rs`,
`machine.rs`), the four languages and the three experiments, the HTTP/1.1
server that serves the React frontend (`../frontend/dist`) and the JSON API,
and the `latticefsm` binary. No dependencies: JSON, gzip, the generator and
HTTP are written out in `src/`.

`../DESIGN.md` is the specification; `../README.md` the manual for both
ports; `../tests/test_latticefsm.py` holds this crate to the Python reference.

## Building and testing

From `..`:

```bash
make build        # cargo build --release -> rust/target/release/latticefsm
make test         # cargo test, cargo clippy -D warnings, cargo fmt --check, then the Python tests with parity
make serve        # the server on http://127.0.0.1:8000/
rust/target/release/latticefsm --help
```

## The API

`latticefsm serve [--port 8000] [--host 127.0.0.1] [--frontend-dir DIR] [--load FILE | machine options]`
holds one machine behind a lock. Everything under `/api/` is JSON, one
request per connection; every other path is served from the frontend
directory (`--frontend-dir`, or `frontend/dist` under the working directory,
or `../frontend/dist` when run from `rust/`), an unknown path there getting
`index.html` as a single-page app's route; without a frontend directory the
root answers a JSON 404 that says so and the API still works:

| route | body / query | returns |
|---|---|---|
| `GET /api/health` | | the version and the machine's shape |
| `GET /api/stats` | | `Machine::stats` |
| `GET /api/matrix` | `?symbol=a` | the `S × S` slice for one symbol: per row, each target's probability, `seen`, width, net, trace and log-weight |
| `GET /api/edge` | `?source=&symbol=&target=` | one edge, every field, the fading ones read at the clock |
| `GET /api/table` | | the greedy transition table |
| `GET /api/languages` | | the languages `train` knows |
| `POST /api/run` | `{text, stimulation?, temperature?, quiet?, from_middle?}` | the run, traversed unless quiet, from the start state or the middle one, and the stats |
| `POST /api/credit` | `{amount}` | reward (positive) or punish (negative) the last run |
| `POST /api/teach` | `{source, symbol, target, amount}` | one edge, traversed and credited |
| `POST /api/train` | `{language, episodes? (4000), max_length?}` | accuracy before and after, the curve, the table |
| `POST /api/tick` | `{ticks}` | time passes |
| `POST /api/stimulate` | `{amount}` or `{level}` | raise the stimulation, or set it |
| `POST /api/new` | `{states?, alphabet?, accepting?, life?, baseline?, …}` | a fresh machine in place of the old; 13 × 13 × 13 without `states` and `alphabet` |
| `POST /api/compress` | `{precision?, budget?}` | fold the matrix into its central node; the code's summary, its `fidelity`, and its rebuild shell by shell (`expansion`) |
| `GET /api/core` | | the code the machine holds (from `/api/compress` or the every-N schedule), measured against the machine now |
| `POST /api/expand` | `{shells?}` | replace the machine with the code's rebuild, from the central node outward; all shells without `shells` |
| `POST /api/core/run` | `{text, from_middle?}` | the greedy walk read straight from the code |
| `POST /api/core/save`, `/api/core/load` | `{path}` | write the held code, or read one in |
| `POST /api/compression` | `{every?, precision?, rebuild?}` | compress automatically every N transitions (0 never) |
| `POST /api/save` | `{path}` | write the machine (`.json` or `.json.gz`) |
| `POST /api/load` | `{path}` | read a machine in place of the old |

Errors are `{"error": "..."}` with 400 (a bad request), 404 (no such route)
or 405. `tests/server.rs` exercises every route over a real socket.

## Files

| file | what |
|---|---|
| `src/edge.rs` | `Edge`, `Weighting`, the features, the weight, traversal, credit, width |
| `src/lattice.rs` | `Lattice`, `State`: the dense matrix and its JSON |
| `src/machine.rs` | `Machine`, `Run`, `Transition`, `Settings`: the walk, the clock, credit, stimulation, the file |
| `src/languages.rs` | the four regular languages and their examples |
| `src/geometry.rs` | the cube's central node and shells; `center_out`, the order the code is laid out in |
| `src/compress.rs` | `Core`, the code the central node holds; `compress`, `fidelity`, `expansion`; the half floats and base64, written out |
| `src/experiment.rs` | learning, stimulation, adaptation; the tables; `run` |
| `src/http.rs` | the HTTP/1.1 server and a test client |
| `src/server.rs` | the routes above, `Service`, `new_machine`, the static files (`safe_join`, `content_type`) |
| `src/json.rs`, `src/gzip.rs`, `src/rng.rs` | written out, no crates |
| `src/bin/latticefsm.rs` | the CLI |
| `tests/machine.rs`, `tests/compress.rs`, `tests/server.rs` | the integration tests |
