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
make serve-prod   # the server and the committed frontend on http://127.0.0.1:8000/
make serve        # the server on :8000 and the Vite dev frontend on :5173 together
rust/target/release/latticefsm --help
```

## Predicting the next thing

```bash
latticefsm ngram train --unit word --order 3 --data data/corpus --save ngram.json.gz
latticefsm ngram predict "the brain is" --load ngram.json.gz -k 5
latticefsm ngram generate "the brain" --load ngram.json.gz [--temperature 0.8]
latticefsm ngram autotrain --ollama --rounds 5 [--model llama3.2 --topics "a,b" --keep FILE] --save ngram.json.gz
latticefsm ngram train --unit base64 --order 8 --data media/ --save b.json.gz   # any files
latticefsm ngram generate --load b.json.gz --out new.bin --prefix-from start.png --prefix-bytes 64
latticefsm base64 encode photo.png --out photo.b64; latticefsm base64 decode photo.b64 --out back.png
latticefsm ngram ollama                                                          # is Ollama there
latticefsm chain train --nodes 8 --confidence 0.9 --data data/corpus --save chain.json.gz
latticefsm chain predict "the bra" --load chain.json.gz                          # input nodes, walk, next
latticefsm chain sweep --load chain.json.gz                                       # accuracy against depth
latticefsm chain teach "what is the brain" "a lattice." --load chain.json.gz     # until every input is right
latticefsm chain train --data data/corpus --until                                 # drilled to N of N
latticefsm corpus data/corpus --show 2                                            # what a corpus is read as
latticefsm guard blame "a bad reply" --reason "off topic" --save guard.json       # the guard learns a failure
latticefsm guard judge "another reply" --load guard.json                          # reject / suspect / pass, why
```

`src/ngram.rs` is the model (`../latticefsm/ngram.py` number for number),
`src/autotrain.rs` the loop and a std-only Ollama client (`OLLAMA_HOST`,
`LATTICEFSM_OLLAMA_MODEL`), `src/b64.rs` base64.

## Networks

`latticefsm network ACTION` builds, teaches and reads networks of machines
(`src/network.rs`, `../latticefsm/network.py`; `../README.md` "Networks of
machines"):

```bash
latticefsm network new --arch "13>(13|13)>13" --alphabet ab --accepting 0 --save net.json
latticefsm network fit --load net.json --language ends-ab --episodes 4000 --save net.json
latticefsm network evaluate --load net.json --language ends-ab   # every string up to --max-length 6
latticefsm network predict abab --load net.json                  # layer by layer
latticefsm network fit --arch "enc>dec" --task plural --epochs 20 --save t.json   # or --pairs FILE.tsv
latticefsm network translate cat dog bus --load t.json           # English in, English out
```

`--arch "text>13"`, `audio>3`, `image>13`, `diff>3` or `video>13` read text, WAV
files, PNG / PPM / BMP pictures, frame differences or video (a directory of
frames and `audio.wav`, or a video file through ffmpeg); `network make-data
--kind K --out DIR` writes a synthetic labelled set and `network fit --data
DIR/labels.tsv` teaches it; `network encode INPUT` prints what is read.

The encoder and decoder use the repository's phonetic tokenizer
(`../../PhoneticTokenizer/rust`), this crate's one dependency, a path
dependency on a crate of the same repository with none of its own.

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
| `POST /api/run` | `{text, stimulation?, temperature?, quiet?, from_middle?, focus?, skip?}` | the run, traversed unless quiet, from where the machine starts (or the middle, or the node `focus` picks), with skips as asked; a skip shows as one transition with `skipped` |
| `POST /api/credit` | `{amount}` | reward (positive) or punish (negative) the last run |
| `POST /api/teach` | `{source, symbol, target, amount}` | one edge, traversed and credited |
| `POST /api/train` | `{language, episodes? (4000), max_length?}` | accuracy before and after, the curve, the table |
| `POST /api/tick` | `{ticks}` | time passes |
| `POST /api/stimulate` | `{amount}` or `{level}` | raise the stimulation, or set it |
| `POST /api/new` | `{states?, alphabet?, accepting?, life?, baseline?, …}` | a fresh machine in place of the old; 13 × 13 × 13 without `states` and `alphabet` |
| `POST /api/compress` | `{precision?, budget?}` | fold the matrix into its central node; the code's summary, its `fidelity`, and its rebuild shell by shell (`expansion`) |
| `GET /api/core` | | the code the machine holds (from `/api/compress` or the every-N schedule), measured against the machine now |
| `POST /api/expand` | `{shells?}` | replace the machine with the code's rebuild, from the central node outward; all shells without `shells` |
| `POST /api/core/run` | `{text, from_middle?, focus?, skip?}` | the greedy walk read straight from the code |
| `POST /api/core/save`, `/api/core/load` | `{path}` | write the held code, or read one in |
| `POST /api/compression` | `{every?, precision?, rebuild?}` | compress automatically every N transitions (0 never) |
| `POST /api/focus` | `{focus?: number or null, learn?, text?}` | set the focus (0 the top node of the central vertical vector, 1 the bottom; null none), switch the learned focus, or ask the learner what it reads off `text` |
| `POST /api/skip` | `{skip?, margin?, rearrange_every?}` | skip a node when a two-edge path is more efficient; the automatic rearrangement schedule |
| `POST /api/rearrange` | `{full?}` or `{axis, i, j}` | let the nodes rearrange themselves toward the centre (one pass, or until settled), or swap two states or symbols |
| `POST /api/save` | `{path}` | write the machine (`.json` or `.json.gz`) |
| `POST /api/load` | `{path}` | read a machine in place of the old |
| `GET /api/cube` | | the whole matrix at once, for the 3D view: per cell (row-major) its probability, seen, width, net and shell; the shape, symbols, centre and focus node |
| `GET /api/ngram` | | the N-gram model the server holds (trigrams of words trained on `data/corpus` at start): stats, sources, history |
| `POST /api/ngram/new` | `{order?, unit?, stimulation?}` | a fresh model: `word`, `char`, `sound` or `base64` |
| `POST /api/ngram/train` | `{data?, held_out?}` | train on files or directories (default `data/corpus`); the held-out measure |
| `POST /api/ngram/text` | `{text}` | train on pasted text, a line a sequence |
| `POST /api/ngram/predict` | `{text, k?}` | the `k` likeliest next tokens and the context they follow |
| `POST /api/ngram/generate` | `{text, max_tokens?, temperature?, seed?}` | `text` continued |
| `POST /api/ngram/credit` | `{text, amount}` | reward (positive) or punish every edge the text takes |
| `GET /api/ngram/ollama` | `?url=&model=` | is Ollama there, which models, which autotrain would use |
| `POST /api/ngram/autotrain` | `{data?, ollama?, model?, url?, topics?, rounds?, lines?, continuations?, judge?, patience?, fresh?}` | the corpus, then rounds with Ollama; the log and each round's record |
| `POST /api/ngram/save`, `/api/ngram/load` | `{path}` | write or read the model |
| `GET /api/chain` | | the chain the server holds (8 nodes of letters, confidence 0.9, trained on `data/corpus` at start): stats per depth, sources, history |
| `POST /api/chain/new` | `{nodes?, unit?, confidence?, lower?}` | a fresh chain |
| `POST /api/{chain,ngram}/train` | `{data?, uploads?, held_out?, predict?, self_credit?, until?, max_rounds?, threshold?}` | train on server paths and uploads, every token predicted first; with `until`, drilled until `threshold` (default 0.95) of the most possible is right; the held-out measure |
| `POST /api/{chain,ngram}/text` | `{text, predict?, self_credit?, until?, max_rounds?}` | train on pasted text, a line a sequence |
| `POST /api/{chain,ngram}/predict` | `{text, k?}` | the next tokens; the chain's input nodes and walk too |
| `POST /api/{chain,ngram}/generate` | `{text, max_tokens?, temperature?, seed?, count?}` | `count` samples (the chain's walk depths with them) |
| `POST /api/{chain,ngram}/credit`, `/feedback` | `{text, amount}`, `{good, bad, amount?}` | reward or punish |
| `POST /api/{chain,ngram}/score` | `{text}` | every token's probability (and walk depth), perplexity, bits |
| `POST /api/{chain,ngram}/teach` | `{text, reply, max_rounds?, threshold?}` | teach `reply` after `text` until `threshold` (default 0.95) of its tokens are right; the new continuation |
| `POST /api/chain/confidence`, `/sweep` | `{confidence}`, `{data?, uploads?, held_out?, thresholds?}` | set the threshold; accuracy and depth at each threshold on held-out lines |
| `GET /api/chain/graph` | `?limit=` | the most-seen node contexts: guess, confidence |
| `POST /api/{chain,ngram}/save`, `/load` | `{path}` | write or read the model |
| `GET /api/uploads`, `POST /api/uploads`, `POST /api/uploads/delete` | `{name, content_base64 \| content, append?}`, `{name}` | the uploads (any file; big ones in pieces), written under `data/uploads` or `$LATTICEFSM_UPLOADS` |
| `GET /api/checkpoints`, `POST /api/checkpoints/{save,restore,delete,upload}`, `GET /api/checkpoints/download?name=` | `{model, tag?}`, `{name}`, `{name, content_base64, append?}` | checkpoints of any model in `checkpoints/` or `$LATTICEFSM_CHECKPOINTS` |
| `POST /api/converse` | `{model, opening, turns?, max_tokens?, temperature?}` | the model talking to itself |
| `POST /api/chat/llm` | `{model, topic, turns?, pass_mark?, learn?, ollama_model?, url?}` | Ollama talks with the model and marks its replies; 502 without Ollama |
| `POST /api/ollama/corpus` | `{prompt, lines?, ollama_model?, url?}` | sentences Ollama writes |
| `GET /api/status` | | every model in brief, uploads, checkpoints |
| `GET /api/guard`, `POST /api/guard/{new,settings,blame,clear,judge,forget,reset,save,load}` | `{texts \| text, reason?, severity?, amount?, threshold?, min_coverage?, path?}` | the guard: blame failures with a reason, clear good text, judge text (reject / suspect / pass, risk, coverage, reasons, spans) |
| `guard: true` on `/api/{chain,ngram}/generate`, `/feedback`, `/api/converse`, `/api/chat/llm`, `/api/network/talk` | `{over_sample?, teach_guard?}` | the guard in front: rejected samples vetoed (over-sampled), bad feedback and low marks blamed, replies judged |
| `GET /api/network` | | the network the server holds beside the machine (`enc>dec` until another is made): architecture, layers, summary |
| `POST /api/network/new` | `{arch?, alphabet?, seed?, accepting?, credit?, layer_discount?, temperature?, discount?, life?}` | a fresh network |
| `POST /api/network/talk` | `{text}` | its reply - English from a translator (with its sounds), yes / no from a classifier - and the trace, layer by layer |
| `POST /api/network/teach` | `{text, reply, times?}` | teach it that `reply` answers `text` (a pair, or a verdict); its reply after |
| `POST /api/network/fit` | `{task, epochs?}`, `{language, episodes?, epochs?}` or `{data, epochs?}` | teach it a task (plural, past), a language, or a labels file |
| `POST /api/network/save`, `/api/network/load` | `{path}` | write or read the network |

Errors are `{"error": "..."}` with 400 (a bad request), 404 (no such route)
or 405. `tests/server.rs` exercises every route over a real socket.

## Files

| file | what |
|---|---|
| `src/edge.rs` | `Edge`, `Weighting`, the features, the weight, traversal, credit, width |
| `src/lattice.rs` | `Lattice`, `State`: the dense matrix and its JSON |
| `src/machine.rs` | `Machine`, `Run`, `Transition`, `Settings`: the walk, the clock, credit, stimulation, the file |
| `src/network.rs`, `src/phonetic.rs` | networks of machines; English in and out through the phonetic tokenizer |
| `src/ngram.rs`, `src/autotrain.rs`, `src/b64.rs` | the N-gram model, autotrain and the Ollama client, base64 |
| `src/chain.rs` | the chain: one input a node, walks that shorten with confidence |
| `src/corpus.rs`, `src/archive.rs` | reading a corpus: text, markup, documents; zip (ZIP64), tar, gzip streamed |
| `src/routes.rs` | the language models' routes, uploads, checkpoints, conversations |
| `src/guard.rs` | the guard: the negative network that vetoes known failures |
| `src/media.rs` | text, audio, images, differences and video as symbols; the synthetic data sets |
| `src/languages.rs` | the four regular languages and their examples |
| `src/geometry.rs` | the cube's central node and shells; `center_out`, the order the code is laid out in |
| `src/focus.rs` | `FocusLearner`: the focus read off the input, learned from credit |
| `src/compress.rs` | `Core`, the code the central node holds; `compress`, `fidelity`, `expansion`; the half floats and base64, written out |
| `src/experiment.rs` | learning, stimulation, adaptation; the tables; `run` |
| `src/http.rs` | the HTTP/1.1 server and a test client |
| `src/server.rs` | the routes above, `Service`, `new_machine`, the static files (`safe_join`, `content_type`) |
| `src/json.rs`, `src/gzip.rs`, `src/rng.rs` | written out, no crates |
| `src/bin/latticefsm.rs` | the CLI |
| `tests/machine.rs`, `tests/compress.rs`, `tests/walk.rs`, `tests/server.rs` | the integration tests |
