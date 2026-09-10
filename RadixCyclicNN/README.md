# RadixCyclicNN

A custom neural network written in pure Python (standard library only, no
numpy) whose structure is a **self-compressing cyclic graph** built on the idea
of a Radix Tree. Text goes in through a sliding window of 3 characters, the
graph learns with a local rule that updates the **activation function itself**
(the custom `-sin(x / 3)` sine), prediction is a **shortest path** search with a
cost function (Dijkstra), and the system keeps upgrading itself with a
**GAN-style** generator/discriminator loop driven by **2NRL** (train on garbage,
invert the network, fine-tune on correct data).

It ships as a Python package (`radixnet`), a CLI, a JSON HTTP API, a React
frontend, a Makefile and a Docker Compose stack. Checkpointing, save/load,
and an optional GPU backend (torch) are built in.

## The ideas, in one table

| Requirement | Implementation |
|---|---|
| Encoding: sliding window of 3 characters | `Encoder.encode("hello") -> ["hel", "ell", "llo"]`, stride 1. |
| Decoding | `Decoder.decode_trigrams` / `decode_path`: first label in full, then the new characters of every following (possibly compressed) node label. |
| Self-compressing cyclic graph (Radix Tree) | Each node holds a label of 3+ characters and therefore one or more trigrams. Unary chains (`p` has one child `c`, `c` has one parent `p`) are **merged** like a radix-tree path (`"hel" + "ell" + "llo" -> "hello"`). A transition observed into or out of the *middle* of a merged node **splits** it again. Repeated trigrams create cycles; cycles are a feature. Compression runs after every epoch. |
| Accept the vanishing gradient, update the activation function instead; `N*N`; activation(child) × activation(parent) | Weights are an N×N node-to-node matrix (stored sparse). The signal on edge `p -> c` is `W[p,c] · f_c(z_c) · f_p(z_p)`, the activation of the child times the activation of the parent. Learning is a **one-hop local rule**: an observed transition only updates `W[p,·]`, the node states `z`, and the **activation-function parameters** of the parent and its children. Nothing is propagated deeper, so vanishing gradients never enter the picture; the activation functions adapt instead. |
| Custom activation `-1 * sin(x / 3.0)` | Every node owns `f(x) = a · sin(b · (x - h)) + k`, initialised to `a = -1, b = 1/3, h = 0, k = 0` (exactly `-sin(x/3)`); all four are learned per node. |
| Shortest path prediction, cost function, Dijkstra | Edge cost `-log P(c | p) + step_penalty` where `P` is a softmax over the parent's edge signals. Dijkstra runs over the graph unrolled by emitted characters and returns the cheapest path that emits the requested length, or the cheapest path to the end-of-text node. |
| Train and predict | `train`, `predict`, `generate`, `score` in the Python API, CLI, HTTP API and frontend. |
| Constantly self-upgrading system (GAN idea) | `Evolver`: the model is the generator, a second network is the discriminator. Each generation the model samples fakes, the discriminator learns real-vs-fake with 2NRL, the worst fakes become the model's own 2NRL garbage and real corpus lines its fine-tune pass. Runs forever (`--generations 0`, or the API's evolve job) and checkpoints as it goes. |
| 2NRL | `two_nrl(bad, good)`: (1) train on bad/garbage data, (2) **invert** the network (every edge weight and every activation amplitude flips sign, so what was likely becomes unlikely), (3) fine-tune on correct data with a smaller learning rate (activation parameters use a tenth of it). |
| CLI, API, React frontend | `python -m radixnet ...`, `python -m radixnet serve` (stdlib `http.server`), `frontend/` (Vite + React, prebuilt `dist` is served by the API). |
| Checkpointing, saving, loading | JSON model files (gzip with `.gz`), `CheckpointManager` with rotation, `latest` pointer, restore and resume. |
| GPU acceleration, performance | `--backend auto` uses torch on CUDA / Apple MPS when installed, else the optimised pure-Python backend (flat CSR arrays, cached costs, ~150k transitions/s on a 4-core CPU). Both backends compute identical numbers. |

`DESIGN.md` is the full specification (math, invariants, module interfaces).

## Quick start

Requirements: Python 3.11+. Nothing to install.

```bash
cd RadixCyclicNN
make demo                     # train on data/sample_corpus.txt, then predict, generate, score
```

or step by step:

```bash
python -m radixnet train --data data/sample_corpus.txt --epochs 10 --lr 0.5 --batch-size 4 --checkpoint-dir checkpoints
python -m radixnet predict --prefix "the quick brown" --length 20
python -m radixnet generate --count 3
python -m radixnet score --text "the cat sat on the mat"
python -m radixnet 2nrl --bad data/sample_garbage.txt --good data/sample_corpus.txt --neg-lr 0.5 --pos-lr 0.1 --batch-size 4
python -m radixnet evolve --data data/sample_corpus.txt --generations 3 --batch-size 4
python -m radixnet serve      # API + frontend on http://127.0.0.1:8000
```

Training is incremental: `--model` (default `model.json`) is loaded first when
it exists. Small corpora only learn visibly with small batches and a large
learning rate (`--batch-size 1..8`, `--lr 0.5..1.0`); those are the Makefile
defaults.

`pip install -e .` adds a `radixnet` console script (same commands).

## Makefile

`make help` prints every target. Variables can be overridden on the command
line, e.g. `make train EPOCHS=20 LR=0.8 MODEL=big.json.gz`.

| Target | What it does |
|---|---|
| `make test` | unit tests (`python -m unittest discover -s tests -v`) |
| `make check` | byte-compile and show which backends are available |
| `make train` / `make resume` | train on `DATA` with checkpoints in `CKPT_DIR` / continue from the latest checkpoint |
| `make predict PREFIX="..." LENGTH=20 MODE=dijkstra` | continue a prefix |
| `make generate COUNT=5` | generate texts from scratch |
| `make score TEXT="..."` | log-probability of a text |
| `make 2nrl` | 2NRL with `GARBAGE` as bad and `DATA` as good data |
| `make invert` / `make compress` | invert the network / merge unary chains |
| `make evolve GENERATIONS=3` / `make evolve-forever` | GAN-style self-upgrade loop |
| `make info` / `make checkpoints` / `make restore NAME=latest` | statistics / list checkpoints / restore one into `MODEL` |
| `make bench CHARS=50000 BACKEND=python` | throughput benchmark |
| `make serve PORT=8000` | API + prebuilt frontend |
| `make frontend-install` / `frontend-build` / `frontend-dev` | npm install / rebuild `frontend/dist` / Vite dev server with hot reload |
| `make up` / `up-auto` / `up-dev` / `up-gpu` / `down` | Docker Compose stack (see below) |
| `make docker-train` / `docker-evolve` / `docker-test` / `docker-bench` | one-shot jobs inside the image |
| `make docker-reload` / `docker-export` / `docker-clean` | reload `/data/model.json` into the running API / copy the model out / remove containers and the volume |
| `make clean` / `make clean-all` | remove caches / also models, checkpoints and `node_modules` |

Defaults: `MODEL=model.json DATA=data/sample_corpus.txt GARBAGE=data/sample_garbage.txt CKPT_DIR=checkpoints UPLOAD_DIR=uploads EPOCHS=10 LR=0.5 BATCH=4 BACKEND=auto PORT=8000`.

## Docker Compose

The image builds the frontend with Node, then ships a slim Python runtime with
zero dependencies (`WITH_TORCH=1` adds torch). The container runs the `radixnet`
CLI; the model, discriminator, checkpoints and files uploaded through the
frontend (`/data/uploads`) live in the named volume `radixnet-data` (mounted at
`/data`), corpora are bind-mounted read-only from `./data`.

```bash
make up            # docker compose up -d api        -> http://localhost:8000 (API + frontend)
make up-auto       # ... and immediately start the self-upgrade loop inside the API (profile: auto)
make docker-train  # docker compose run --rm train   (one-shot training into the volume)
make docker-evolve # headless self-upgrade loop beside the API (profile: evolve)
make up-dev        # + Vite dev server with hot reload on http://localhost:5173 (profile: dev)
make docker-test   # unit tests inside the image
make up-gpu        # build with torch and hand the NVIDIA GPUs to the API (docker-compose.gpu.yml)
make down          # stop everything, keep the volume
```

| Service | Profile | Role |
|---|---|---|
| `api` | default | `radixnet serve` on port 8000 with `/data/model.json` and `/data/checkpoints`; healthcheck on `/api/health` |
| `autoevolve` | `auto` | one-shot sidecar: waits for the API, then `POST /api/evolve/start` with the corpus so the loop runs inside the API and shows in the frontend |
| `train` | `tools` | `radixnet train` on `./data/$RADIXNET_CORPUS` into the volume (`EPOCHS`, `LR`, `BATCH`) |
| `evolve` | `evolve` | `radixnet evolve --generations 0` with its own discriminator and checkpoint directory; writes `/data/model.json` when stopped |
| `test`, `bench` | `tools` | unit tests / benchmark inside the image |
| `frontend-dev` | `dev` | Node container running `npm run dev` with `/api` proxied to the `api` service |

Settings come from the environment or a `.env` file (`cp .env.example .env`):
`RADIXNET_PORT`, `RADIXNET_BACKEND`, `WITH_TORCH`, `RADIXNET_RESUME`,
`RADIXNET_CORPUS`, `EPOCHS`, `LR`, `BATCH`, `EVOLVE_SAMPLES`, `EVOLVE_MAX_LENGTH`,
`EVOLVE_CHECKPOINT_EVERY`, `BENCH_CHARS`.

The API keeps its own in-memory model. Whatever the `train` or `evolve` service
writes to `/data/model.json` appears in the API after `make docker-reload`
(`POST /api/load`), or on the next start with `RADIXNET_RESUME=1`, which makes
the entrypoint restore the newest checkpoint before serving. The `auto` profile
avoids the hand-off entirely by evolving inside the API.

GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`
installs torch and reserves all NVIDIA GPUs (needs the NVIDIA Container Toolkit).
Behind a registry mirror, pass `--build-arg PYTHON_IMAGE=... --build-arg NODE_IMAGE=...`.

## CLI reference

Global options (before or after the command): `--model PATH` (default
`model.json`, gzip when the name ends with `.gz`), `--backend auto|python|torch`,
`--device cpu|cuda|mps`, `--seed N`, `--json` (one JSON document on stdout).

| Command | Main options |
|---|---|
| `train --data FILE [FILE...]` | `--whole-file`, `--epochs`, `--lr`, `--act-lr`, `--batch-size`, `--no-compress`, `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--resume`, `--out` |
| `predict --prefix TEXT` | `--length`, `--max-length`, `--mode dijkstra\|sample`, `--to-end`, `--step-penalty`, `--temperature` |
| `generate` | `--count`, `--max-length`, `--mode`, `--temperature` |
| `score --text TEXT` / `--data FILE` | log-probability, per-character score, unknown transitions |
| `2nrl --bad FILE --good FILE` | `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--batch-size`, `--out` |
| `invert` / `compress` | flip the network / merge unary chains, then save |
| `evolve --data FILE` | `--generations` (0 = forever, Ctrl-C saves), `--samples`, `--real-per-generation`, `--max-length`, `--temperature`, `--discriminator PATH`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--disc-neg-epochs`, `--disc-pos-epochs`, `--batch-size`, `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--out` |
| `info` | statistics and the training history tail |
| `checkpoints` | `--dir`, `--restore NAME\|latest`, `--out` |
| `bench` | `--chars`, `--epochs` |
| `serve` | `--host`, `--port`, `--frontend-dir`, `--checkpoint-dir`, `--upload-dir` (training files uploaded through the API / frontend, default `uploads`) |

Every command has `--help`. Exit code 1 with a message on stderr on errors.

## HTTP API

`python -m radixnet serve --host 127.0.0.1 --port 8000`. All `/api/*` responses
are JSON with CORS headers; errors are `{"error": "..."}` with 400/404/409/500.
Long operations (train, 2NRL, evolve) run as a background **job**; only one job
at a time, and mutating requests answer 409 while it runs.

| Method and path | Body / result |
|---|---|
| `GET /api/health` | `{"ok": true, "version"}` |
| `GET /api/status` | model statistics, current job, available backends, model path |
| `POST /api/train` | `{"texts": [...]}` or `{"text": "one per line"}` and/or `{"files": ["upload names"], "whole_file": false}` + `epochs`, `lr`, `act_lr`, `batch_size`, `auto_compress` -> `{"job": {...}}` |
| `GET /api/uploads` | uploaded training files: `{"uploads": [{"name","bytes","chars","lines","modified"}], "upload_dir"}` |
| `POST /api/uploads` | upload text files: JSON `{"name","content"}` or `{"files": [{"name","content"}, ...]}`, `multipart/form-data` (`curl -F file=@corpus.txt`), or a raw body with `?name=corpus.txt` -> `{"uploads": [...]}` (201) |
| `POST /api/uploads/delete` | `{"name"}` |
| `GET /api/job` / `POST /api/job/stop` | job status `{"id","type","state","progress","history","error",...}` / request a stop |
| `POST /api/predict` | `{"prefix","length","mode","to_end","step_penalty","temperature"}` -> `{"continuation","full_text","cost","step_costs","path","node_ids","expanded","reached_end"}` |
| `POST /api/generate` | `{"count","max_length","mode","temperature"}` -> `{"samples": [{"text","cost","path"}]}` |
| `POST /api/score` | `{"text"}` -> `{"log_prob","per_char","chars","transitions","unknown_transitions"}` |
| `POST /api/2nrl` | `{"bad": [...], "good": [...], "neg_epochs","pos_epochs","neg_lr","pos_lr"}` (or `bad_files` / `good_files` upload names) -> job |
| `POST /api/invert` / `POST /api/compress` | statistics / `{"merges", ...}` |
| `POST /api/evolve/start` / `POST /api/evolve/stop` / `GET /api/evolve/history` | `{"corpus": [...]` or `"corpus_text"` or `"corpus_files"`, `"generations"` (null = forever), `samples`, `max_length`, `temperature`, `checkpoint_every`, ...}` -> job |
| `POST /api/save` / `POST /api/load` / `POST /api/reset` | `{"path"}` / `{"path"}` / `{"seed"}` |
| `GET /api/checkpoints` / `POST /api/checkpoints/save` / `POST /api/checkpoints/restore` | list / `{"tag"}` / `{"name"}` |
| `GET /api/graph?limit=150` | top nodes by visit count with their activation parameters, and the edges between them with weight, probability and cost |
| `GET /api/history` | training history |
| `GET /` | the built frontend (`frontend/dist`), or a small page explaining how to build it |

```bash
curl -X POST localhost:8000/api/train -H 'Content-Type: application/json' \
     -d '{"texts": ["the cat sat on the mat", "the dog runs in the park"], "epochs": 5, "lr": 0.5, "batch_size": 4}'
curl localhost:8000/api/job
curl -X POST localhost:8000/api/predict -H 'Content-Type: application/json' -d '{"prefix": "the cat", "length": 15}'
curl -X POST localhost:8000/api/evolve/start -H 'Content-Type: application/json' \
     -d '{"corpus": ["the cat sat on the mat", "the dog runs in the park"], "generations": null}'
curl -X POST localhost:8000/api/evolve/stop
```

## Frontend

`frontend/` is a Vite + React app (React, ReactDOM, Vite only). The prebuilt
`frontend/dist` is committed and served by the API, so nothing needs npm to use
it. Panels: status bar (live statistics and job progress), Train (texts and/or
uploaded files), Predict (path with per-step costs), Generate, Score, 2NRL,
Evolve (live chart of the discriminator gap), Checkpoints (save / restore /
load / reset) and a Graph view of the most visited nodes.

Training files: drop text files onto the Train panel (or press "Upload
files…"); the browser reads them and sends them to `POST /api/uploads`, the
server keeps them in its `--upload-dir`, and the list lets you tick which files
to train on, one text per line or each file as one text. The same picker feeds
the 2NRL (bad / good files) and Evolve (corpus files) panels.

```bash
make frontend-install && make frontend-build   # rebuild dist
make serve                                     # then make frontend-dev in another shell for hot reload
```

## Checkpoints, saving, loading

Models are JSON: `{"format": "radixnet", "version": 1, "saved_at", "meta",
"history", "backend", "graph": {nodes, edges, rng_state, inverted, ...}}`; a
`.gz` name gzips it. Loading restores the exact graph, activation parameters,
random state and history, so predictions and further training are reproducible.

`--checkpoint-dir DIR` writes `ckpt-<tag>-<step>.json.gz` files plus
`latest.json`, keeping the newest `--keep` (default 5) and never pruning the
latest. `train --resume` continues from the latest checkpoint;
`checkpoints --restore latest` (or a name) writes it into a model file; the API
has `/api/checkpoints`, `/api/checkpoints/save` and `/api/checkpoints/restore`;
the frontend's Checkpoints panel does the same.

## GPU acceleration and performance

Backends: `python` (always available, optimised pure Python over flat CSR arrays)
and `torch` (optional: `pip install torch` or `make install-gpu`; device
`cuda` > `mps` > `cpu`). `--backend auto` picks torch only when a GPU is
available, so CPU-only machines stay dependency-free; `--backend torch
--device cpu` runs the vectorised path on the CPU. Both backends implement the
same maths and agree to 1e-6 (`tests/test_backend_torch.py`).

`make bench` (or `python -m radixnet bench --chars 50000 --backend python`)
reports training transitions/s and chars/s, predictions/s and Dijkstra
expansions/s. Dijkstra always runs on the CPU. The graph exports CSR arrays once
per epoch, caches per-node edge costs, and reuses transition arrays across
epochs while the structure is unchanged.

## Python API

```python
from radixnet import RadixNet, Evolver

net = RadixNet(seed=0, backend="python")
net.train(["the cat sat on the mat", "the dog runs in the park"], epochs=10, lr=0.5, batch_size=4)
print(net.predict("the cat", length=15).full_text)
print(net.score("the cat sat on the mat")["per_char"])
net.two_nrl(bad=["the cat sat on the sky"], good=["the cat sat on the mat"], neg_lr=0.5, pos_lr=0.1)
net.save("model.json.gz")

Evolver(RadixNet.load("model.json.gz"), corpus=["the cat sat on the mat"]).run(generations=2)
```

## Tests

```bash
make test        # python -m unittest discover -s tests -v
```

## Layout

```
RadixCyclicNN/
  radixnet/           activation, encoding, graph, backend(+torch), search, model, gan, checkpoint, bench, cli, api
  tests/              unittest suite
  frontend/           Vite + React app (dist/ is prebuilt and served by the API)
  data/               sample_corpus.txt (correct data), sample_garbage.txt (bad data)
  docker/             container entrypoint (optional checkpoint resume)
  Dockerfile, docker-compose.yml, docker-compose.gpu.yml, .env.example, Makefile
  DESIGN.md           the specification
```

## Design decisions

* **`N*N`** is read as the node-to-node weight matrix: N nodes, N×N possible edges, stored sparsely as adjacency lists and exported as CSR for the backends.
* **"activation of the child × activation of the parent"** is the edge signal `W[p,c] · f_c(z_c) · f_p(z_p)`; the softmax over a parent's signals is the next-node distribution, and `-log` of it is the Dijkstra cost.
* **"accept the vanishing gradient, update the activation function instead"** means no back-propagation through depth. Every observed transition applies a one-hop gradient to the edge weight, the two node states and the four sine parameters of the parent and children, so the activation functions carry the learning.
* **"invert the network"** (2NRL) flips the sign of every edge weight and every activation amplitude `a`, which negates every edge signal: the most likely continuation becomes the least likely. Two inversions are the identity.
* **Prediction prefers short, confident completions** because the cost is summed per edge; `--step-penalty` and `--length` / `--to-end` steer that, and `--mode sample` gives diverse output for the GAN loop.
* **Self-compression is lossy on purpose**: merging a unary chain keeps the parent's parameters; the chain was deterministic (probability 1, cost 0), so predictions are unchanged.

## License

See the `LICENSE` file at the repository root. Source-available, all rights reserved.
