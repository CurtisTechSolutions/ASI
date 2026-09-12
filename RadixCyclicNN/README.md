# RadixCyclicNN

A custom neural network written in pure Python (standard library only, no
numpy) whose structure is a **self-compressing cyclic graph** built on the idea
of a Radix Tree. Text goes in through a sliding window of 3 characters, the
graph learns with a local rule that updates the **activation function itself**
(the custom `-sin(x / 3)` sine), prediction is a **shortest path** search with a
cost function (Dijkstra), and the system keeps upgrading itself with a
**GAN-style** generator/discriminator loop driven by **2NRL** (train on garbage,
invert the network, fine-tune on correct data).  A second copy of the network
keeps only the **negative** side - the failures the tutor found, and *why* they
failed - and filters the first one's output.

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
| Automated English lessons | `tutor` / the Tutor tab / `POST /api/tutor/start` (both servers): the teacher - a local Ollama model or ChatGPT - writes sentence openings that drill a point of grammar, the network completes them with the prediction search, the same teacher marks each sentence out of 10 for grammar, spelling and fluency and writes the correction; the correction is then aligned with what the network wrote and only the trigram nodes that differ move (`correct`), and the round's mistakes become the next round's syllabus. |
| The report card plans the next lessons | `tutor --plan N` / the Tutor tab's **Lesson plan** / `POST /api/tutor/plan` (both servers): the report card at the end of a run goes back to the teacher, which answers with the syllabus that repairs it - one point of grammar per lesson, the mistake of the card it targets, a topic, a line on why, and a level that goes up when the card is strong. The marks alone already plan it (a lesson per weak point, worst first); the teacher improves on that floor and never drops a weakness from it. Each lesson carries the settings to run it, so one click starts it. |
| Rewards follow the rating | `two_nrl(good_weights=)`, `reward(weights=)` and `punish(weights=)` (both models, Python and Go) scale every pass per text: a sentence marked 9 out of 10 is learned nine tenths as hard as a perfect one, a 0 is skipped. `/api/feedback` and `/api/2nrl` take `good_ratings` / `bad_ratings` (marks out of 10), the Ratings card a mark per rated text. |
| The model converses with itself | `converse` / the Converse tab: two voices take turns, every reply is the prediction search picking up the last words of the previous line and continuing them to the end of a text; beam speaks the most likely reply the conversation has not heard yet, sample draws walks; the second voice can be the model of the other kind. |
| Teach it by talking to it | `speech`, the Speech tab and `POST /api/speech/teach`: the browser records the microphone and dictates the words (Web Speech API; faster-whisper, openai-whisper or an OpenAI-compatible transcription server do it on the server side), and **one utterance becomes two texts behind the same unique token** - `<speech:9f2a1c7d> the cat sat on the mat` and `<speech:9f2a1c7d> aud:mu:8000x1:<base64>`, the waveform itself with every sample quantised to one mu-law byte. Both are trained on, so the words and the sound leave the same node of the graph; `speech decode` plays a predicted waveform back. |
| Images as text | `image encode` / the Images tab run the Stable Diffusion VAE **backwards** (image -> compressed latent, 48x fewer numbers than the pixels), quantise it to bytes, base64-encode it and feed the text to the model; `decode` runs the forward process again so a predicted text becomes an image. Needs `pillow` (+ `torch`, `diffusers` and the VAE weights for the real encoder; a thumbnail stand-in works without them). |
| Count / reward model | a second algorithm on the same graph, selectable at the top of the frontend (`--kind count` in the CLI, `POST /api/model/select`): every edge tracks how often training traversed it and a reward / penalty number, `weight = log(1 + traversals) + reward`, and one prediction returns the **top K and bottom K** continuations (beam search). |
| Go port of the count / reward model | `go/`: the same model in Go with one goroutine per text (lines, paragraphs or pages), counters bumped without locks (racy by default, `--exact` for atomics), parallel weight and cost recomputes, the two beams of a prediction side by side, and corpora of any size streamed through in chunks (ZIP archives entry by entry); model files are interchangeable with Python (same structure, counts, sliding window and even the Mersenne Twister state). |
| Learning-rate schedules | `lr` and `act_lr` as *graph functions* of the epoch (`linear(lr0, 4 * lr0)`, `lr0 * 1.25 ** i`, `warmup(...)`, `lr / 10`), previewed as a graph in the CLI (`schedule`), the API and the Train tab. |
| Constantly self-upgrading system (GAN idea) | `Evolver`: the model is the generator, a second network is the discriminator. Each generation the model samples fakes, the discriminator learns real-vs-fake with 2NRL, the worst fakes become the model's own 2NRL garbage and real corpus lines its fine-tune pass. Runs forever (`--generations 0`, or the API's evolve job) and checkpoints as it goes. |
| The negative network | `NegativeNet` (`--kind negative`, the Negative tab): a copy of the network that keeps only its negative portions. Every node and edge in it exists because something went wrong there, every edge remembers the blame it collected and the tutor's reasons behind it, and `judge` walks a text through that structure to say how much of it is built out of known failure, which reasons those failures carried and which fragments carry them. It is trained on negative data alone; text the tutor *passed* only ever takes blame away (net evidence is `max(0, blame - clear)`). |
| The tutor supplies the negatives | `blame.py`: the **English tutor** names the mistake it marked a sentence down for (`agreement`, `tense`, `article`, ...), hands over its mark as the severity and its correction as the diff to blame (`tutor --blame`); the Ollama reviewer's critique becomes the reason and its rating the severity (`ollama review --blame`), the code sandbox / style checker / judge name why a program was rejected (`codegen --blame`), the evolve discriminator blames every fake it scores below the real texts (`evolve --blame`), and a person can blame a text by hand. The negative network never invents a failure. |
| A correction blames only what changed | `NegativeNet.correct(wrong, right)`: the sentence the network wrote and the sentence the teacher wrote instead are aligned character by character (`diff.py`, the same alignment the count model's `correct` teaches from) and only the steps that wrote a character the teacher struck out are blamed - with the tutor's error type as the reason. The correction clears blame everywhere else, and a blamed transition is never compressed away, so the fragment that went wrong stays nameable. |
| The pair as a GAN at output time | `NegativeFilter` (`negative filter`, `POST /api/negative/filter`): the positive model over-samples candidates and the negative one vetoes them - by blame (`risk` over the threshold), by the likelihood ratio `log P_negative - log P_positive` per character (the discriminator logit of the two networks), or by `peak`, the blame on a single fragment, which is how one corrected word vetoes an otherwise clean sentence. What survives comes back ranked; what does not comes back with the reason, the blamed fragment and who said so. |
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
python -m radixnet negative blame --data data/sample_garbage.txt --reason gibberish --source review
python -m radixnet negative why --text "the the the the cat"
python -m radixnet negative filter --count 3        # the positive model writes, the negative one vetoes
python -m radixnet speech listen --seconds 5 --train     # say something; it learns the words and the sound
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
| `make tutor-blame TOPIC="..."` | English lessons that also teach the negative network what the teacher marked down |
| `make negative-blame REASON=gibberish` / `negative-clear` | teach the negative network every line of `GARBAGE` as a failure / let `DATA` take blame back off what it shares |
| `make negative-why TEXT="..."` / `negative-filter COUNT=3` / `negative-reasons` / `negative-forget REASON=...` | explain a text, run the pair, list what the tutor blamed, drop a reason |
| `make invert` / `make compress` | invert the network / merge unary chains |
| `make evolve GENERATIONS=3` / `make evolve-forever` / `make evolve-blame` | GAN-style self-upgrade loop (`evolve-blame` also teaches the negative network) |
| `make info` / `make checkpoints` / `make restore NAME=latest` | statistics / list checkpoints / restore one into `MODEL` |
| `make bench CHARS=50000 BACKEND=python` | throughput benchmark |
| `make go-build` / `go-test` / `go-parity` / `go-serve PORT=8001` | build the Go count / reward model CLI, run its tests, the cross-language parity tests, or serve the frontend from the Go model |
| `make serve PORT=8000` | API + prebuilt frontend |
| `make ollama-models` / `ollama-corpus PROMPT="..."` / `ollama-garbage` / `ollama-review` / `ollama-2nrl` | Ollama: list models, prompt -> corpus (+ train), prompt -> garbage file, adversarial review of the model's samples, review + 2NRL |
| `make tutor TOPIC="..." ROUNDS=5` / `tutor-dry` / `tutor-focus FOCUS="past tense"` / `tutor-plan PLAN=3` | automated English lessons taught by `TUTOR=ollama\|chatgpt` (`TUTOR_MODEL`, `TUTOR_URL`); `tutor-plan` ends with the next lessons planned from the report card |
| `make codegen PROBLEMS=data/sample_problems.jsonl PHASE=both` / `codegen-teacher` / `codegen-model` | code generation with the sandbox, the Ollama judge (`CODEGEN_MODEL=gemma4`) and 2NRL rewards |
| `make chatgpt-models` / `chatgpt-ask PROMPT="..."` | ChatGPT (OpenAI): what `$OPENAI_API_KEY` may use, one question — the quickest check that ChatGPT can tutor |
| `make ollama-models` / `ollama-corpus PROMPT="..."` / `ollama-garbage` / `ollama-review` / `ollama-blame` / `ollama-2nrl` | Ollama: list models, prompt -> corpus (+ train), prompt -> garbage file, adversarial review of the model's samples, review + blame the negative network, review + 2NRL |
| `make speech-info` / `speech-teach AUDIO=clip.wav TRANSCRIPT="..."` / `speech-listen SECONDS=5` / `speech-decode TEXT="aud:…" AUDIO=out.wav` | speech: available backends, teach an audio file, record from the microphone and teach that, play a waveform text back |
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
| `ollama` | `ollama` | the official `ollama/ollama` image with a model volume; `make up-ollama` starts it with the API pointed at it (`OLLAMA_HOST=http://ollama:11434`), then `docker compose exec ollama ollama pull llama3.2` once |

Settings come from the environment or a `.env` file (`cp .env.example .env`):
`RADIXNET_PORT`, `RADIXNET_BACKEND`, `WITH_TORCH`, `RADIXNET_RESUME`,
`RADIXNET_CORPUS`, `EPOCHS`, `LR`, `BATCH`, `EVOLVE_SAMPLES`, `EVOLVE_MAX_LENGTH`,
`EVOLVE_CHECKPOINT_EVERY`, `BENCH_CHARS`, `OLLAMA_HOST`, `RADIXNET_OLLAMA_MODEL`
and — to let ChatGPT teach in the Tutor and Code tabs — `OPENAI_API_KEY` (or
`OPENAI_API_KEY_FILE` for a Docker secret), `RADIXNET_OPENAI_MODEL`,
`OPENAI_BASE_URL`. The key is read from the `api` container's environment
only; it never travels through the HTTP API.

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
`model.json`, gzip when the name ends with `.gz`), `--kind radix|count` (the
algorithm of a *new* model; a file's own kind wins; with `count` the default
model file is `model.count.json`), `--backend auto|python|torch`,
`--device cpu|cuda|mps`, `--seed N`, `--json` (one JSON document on stdout).

| Command | Main options |
|---|---|
| `train --data FILE [FILE...]` | `--whole-file`, `--epochs`, `--lr`, `--act-lr`, `--lr-schedule EXPR`, `--act-lr-schedule EXPR` (graph functions of the epoch, see below), `--reverse-schedule`, `--batch-size`, `--no-compress`, `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--resume`, `--out`; a `.zip` in `--data` contributes every text file inside it |
| `schedule` | preview a learning-rate schedule: `--lr-schedule EXPR`, `--act-lr-schedule EXPR`, `--reverse-schedule`, `--epochs 10`, `--lr`, `--act-lr` print the rate of every epoch with a bar graph; without expressions the presets, variables and functions are listed |
| `predict --prefix TEXT` | `--length`, `--max-length`, `--mode dijkstra\|beam\|sample`, `--to-end`, `--step-penalty`, `--temperature`; `--mode beam` (both models; the count model's default): `--k 5` (top K and bottom K continuations in one search), `--beam N` |
| `generate` | `--count`, `--max-length`, `--mode beam\|sample\|dijkstra`, `--prefix TEXT`, `--temperature`, `--step-penalty`, `--beam N`; `beam` is the prediction search run to the end of a text: the `--count` most likely complete texts, most likely first |
| `score --text TEXT` / `--data FILE` | log-probability, per-character score, unknown transitions |
| `converse` | the model talks to itself: `--opening TEXT`, `--turns 6`, `--mode beam\|sample`, `--context 12` (characters of the previous line a reply picks up), `--max-length 60`, `--k 5`, `--beam N`, `--temperature`, `--step-penalty`, `--speakers A,B`, `--partner FILE` (a second model speaks the second voice), `--allow-repeats`; prints the transcript with cost, probability and the words each reply picked up |
| `weights` | count model: show the dual frequency function and the tracked totals, or change it: `--global-scale`, `--window-scale`, `--reward-scale`, `--count-scale`, `--window N` (then every weight is recomputed and the model saved) |
| `2nrl --bad FILE --good FILE` | `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--batch-size`, `--strength` (count model), `--out` |
| `feedback` | rated texts: `--good FILE` / `--good-text TEXT` (thumbs up), `--bad FILE` / `--bad-text TEXT` (thumbs down); both -> 2NRL, thumbs up alone -> reward, thumbs down alone -> punish then invert; `--good-ratings 10,5,8` / `--bad-ratings` give a mark out of 10 per text (in the order they were collected) and every text is learned in proportion to it; `--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4`, `--out` |
| `negative <action>` | the negative network (`--negative PATH`, default `model.negative.json` beside `--model`): `blame --text/--data --reason TAG --severity N --source NAME --note TEXT` (teach it a failure), `clear --text/--data` (the tutor passed these: take blame off what they share), `why --text/--data [--threshold --min-coverage --spans]` (risk, coverage, the reasons and the blamed fragments), `filter [--count --prefix --mode --max-length --over-sample --threshold --min-coverage --ratio --no-ratio --peak --strict --learn]` or `filter --text/--data` (the pair: the positive model writes, the negative one vetoes), `reasons [--limit --log]`, `forget [--reason TAG] [--factor F]` |
| `invert` / `compress` | flip the network / merge unary chains, then save |
| `evolve --data FILE` | `--blame` / `--negative PATH` (the discriminator teaches the negative network), `--generations` (0 = forever, Ctrl-C saves), `--samples`, `--real-per-generation`, `--max-length`, `--temperature`, `--discriminator PATH`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--disc-neg-epochs`, `--disc-pos-epochs`, `--batch-size`, `--blatant-mode none\|fail_invert\|activation\|state`, `--blatant-margin`, `--blatant-boost` (failure handling, see below), `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--out` |
| `info` | statistics and the training history tail |
| `checkpoints` | `--dir`, `--restore NAME\|latest`, `--out` |
| `bench` | `--chars`, `--epochs` |
| `serve` | `--host`, `--port`, `--frontend-dir`, `--checkpoint-dir`, `--upload-dir` (training files uploaded through the API / frontend, default `uploads`), `--ollama-url`, `--ollama-model` |
| `ollama [--url] [--ollama-model] [--timeout] <action>` | `models`; `corpus --prompt TEXT [--lines 20] [--style good\|garbage] [--out FILE] [--train --epochs --lr --batch-size --model-out]`; `review [--count 8] [--prefix] [--max-length 60] [--text ... \| --data FILE] [--threshold 6] [--context] [--blame [--negative PATH]] [--2nrl --good FILE ...]` |
| `speech info` / `transcribe FILE` / `teach FILE` / `listen` / `decode` | teaching by talking. `info`: backends, recorders, codecs. `transcribe FILE [--backend auto\|given\|faster-whisper\|whisper\|server] [--text TEXT] [--language en] [--asr-model] [--asr-url] [--out]`: the words. `teach FILE`: the transcript **and** the waveform behind one unique token - `--text` (what you said, skips the ASR), `--rate 8000`, `--codec auto\|mu\|pcm8`, `--normalise`, `--no-waveform`, `--pair` (also learn waveform → transcript), `--token` / `--shared-token`, `--out FILE`, `--train --epochs 3 --lr 0.5 --batch-size 8 --model-out`. `listen --seconds 5 [--recorder arecord\|rec\|sox\|ffmpeg] [--save clip.wav]`: record from the microphone first, then the same. `decode (--text\|--data) --out out.wav [--codec]`: an encoded or *predicted* waveform as audio |
| `tutor` | automated English lessons: `--blame` / `--negative PATH` (every failed sentence also teaches the negative network what the teacher marked it down for), `--topic TEXT`, `--rounds 3`, `--exercises 5`, `--attempts 1`, `--focus TEXT` (one point of grammar), `--level`, `--words "3 to 6"`, `--tutor-provider ollama\|chatgpt`, `--tutor-model`, `--grader-provider`, `--grader-model`, `--url`, `--grader-url`, `--timeout`; completion: `--mode dijkstra\|beam\|sample`, `--length 20`, `--max-length 80`, `--temperature`, `--no-to-end`, `--beam N`; marking: `--threshold 6` (pass mark), `--grammar-weight 0.6`, `--batch 10`, `--no-adapt`, `--drills N`, `--plan N` (plan the next N lessons from the report card at the end), `--no-teach-answer`, `--dry-run`; corrections: `--keep-weight 0.25`, `--no-diff-corrections`; 2NRL: `--twonrl-per round\|lesson`, `--min-weight 0.25`, `--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4 --strength`, `--no-replay`, `--replay-limit`, checkpoint options, `--out`, `--report FILE` |
| `correct` | teach one correction: `--wrong TEXT` (what the network wrote), `--right TEXT` (what it should say), `--blame` / `--reason TAG` / `--note TEXT` / `--negative PATH` (teach the negative network from the same diff), `--strength 1`, `--weight 1` (how bad the attempt was), `--reward 1`, `--keep 0.25` (what the unchanged words still earn), `--no-count`, `--dry-run` (show the alignment only), `--out` |
| `chatgpt [--url] [--chatgpt-model] [--timeout] <action>` | `models` (what the key may use); `ask --prompt TEXT [--system TEXT] [--temperature 0.7] [--json]`. Needs `$OPENAI_API_KEY` (or `$OPENAI_API_KEY_FILE`); `$OPENAI_BASE_URL` points at any OpenAI-compatible server |
| `image info` / `image encode FILE` / `image decode` | encoders and their dependencies; `encode --size 128 --encoder auto\|sd\|tiny [--out TEXTFILE] [--train --epochs 3 --lr 0.5 --batch-size 8 --model-out]`; `decode (--text TEXT \| --data FILE) --out image.png [--encoder]` |
| `codegen --problems FILE` | `--blame` / `--negative PATH` (the sandbox and the judge teach the negative network), `--phase both\|teacher\|model`, `--rounds`, `--teacher-provider ollama\|chatgpt`, `--teacher-model gemma4`, `--judge-provider`, `--judge-model`, `--url`, `--judge-url`, `--timeout`, `--teacher-attempts 3`, `--model-attempts 4`, `--sample-first`, `--temperature`, `--max-length 800`, `--strictness strict\|lenient`, `--no-judge`, `--no-fallback-teacher`, `--twonrl-per problem\|round`, `--no-replay`, `--teacher-prompt`, `--model-prompt`, `--sandbox-timeout 10`, `--memory-mb 256`, `--no-network-isolation`, 2NRL options (`--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4`), checkpoint options, `--out`, `--report FILE` |

Every command has `--help`. Exit code 1 with a message on stderr on errors.

## HTTP API

`python -m radixnet serve --host 127.0.0.1 --port 8000`. All `/api/*` responses
are JSON with CORS headers; errors are `{"error": "..."}` with 400/404/409/500.
Long operations (train, 2NRL, evolve) run as a background **job**; only one job
at a time, and mutating requests answer 409 while it runs.

| Method and path | Body / result |
|---|---|
| `GET /api/health` | `{"ok": true, "version"}` |
| `GET /api/status` | model statistics (with the active `kind`), current job, available backends, model path |
| `GET /api/model` | `{"kind", "label", "kinds": [{"kind","label","description"}], "model_path", "paths", "in_memory"}` |
| `POST /api/model/select` | `{"kind": "radix"\|"count"}` -> the same document plus `origin` (`memory`, `file`, `new`, `active`) and `stats`; the previous model stays in memory |
| `POST /api/train` | `{"texts": [...]}` or `{"text": "one per line"}` and/or `{"files": ["upload names"], "whole_file": false}` + `epochs`, `lr`, `act_lr`, `lr_schedule`, `act_lr_schedule` (expressions of the epoch), `reverse_schedule`, `batch_size`, `auto_compress` -> `{"job": {...}}`; every epoch record carries the `lr` / `act_lr` used |
| `GET /api/schedule` | what a schedule expression may use: `{"variables", "constants", "functions", "helpers", "presets": [{"name","lr","act_lr","description"}]}` |
| `POST /api/schedule/preview` | `{"lr_schedule", "act_lr_schedule", "epochs": 5, "lr": 0.05, "act_lr": 0.005, "reverse_schedule": false}` -> `{"points": [{"epoch","lr","act_lr"}], ...}` (400 with the reason for a bad expression) |
| `GET /api/uploads` | uploaded training files: `{"uploads": [{"name","bytes","chars","lines","modified"} (+ `archive`, `files`, `skipped` for a ZIP)], "upload_dir"}` |
| `POST /api/uploads` | upload text files or ZIP archives: JSON `{"name","content"}` / `{"name","content_base64"}` or `{"files": [...]}`, `multipart/form-data` (`curl -F file=@corpus.zip`), or a raw body with `?name=corpus.zip` -> `{"uploads": [...], "archives": [{"name","entries","extracted","skipped": [{"path","reason"}]}]}` (201). A ZIP stays one upload (its record carries `archive: true`, `files`, `skipped` and the summed `lines`); whenever it is selected the server unpacks its text entries in memory. Directories, `__MACOSX` / system files, nested archives, encrypted, binary and empty entries are ignored; an archive with no text entry (or a corrupt one) is refused. There is no size limit: the upload body limit does not apply to `/api/uploads`, and an archive may hold any number of entries (`extract_texts(max_entries=, max_bytes=)` exists for callers who want a cap) |
| `POST /api/uploads/delete` | `{"name"}` |
| `GET /api/ollama/models?url=` | always 200: `{"available", "url", "model", "models": [{"name","size","modified_at","details"}], "error"}` |
| `POST /api/ollama/corpus` | `{"prompt", "lines": 20, "style": "good"\|"garbage", "model", "url", "save_as": upload name, "train": false, "epochs", "lr", "batch_size"}` -> `{"texts", "upload", "job", ...}` (202 with a train job; 502 when Ollama fails) |
| `POST /api/ollama/review` | `{"count": 8, "prefix", "max_length": 60, "temperature", "texts": [...] (review these instead of sampling), "threshold": 6, "context", "apply": "none"\|"2nrl", "blame" (teach the negative network), "good", "good_files", 2NRL settings}` -> `{"reviews": [{"index","text","rating","verdict","critique"}], "mean_rating", "pass_rate", "good", "bad", "job", ...}` |
| `GET /api/chatgpt/models?url=` | always 200: `{"available", "configured" (the server has a key), "url", "model", "models": [{"name","owned_by","created"}], "error"}`. The key is never a request field: it is the server's own `$OPENAI_API_KEY` |
| `GET /api/images` | `{"pillow","torch","diffusers","sd_model","sd_loaded","sd_error","encoders","default_size","auto","text_format"}` |
| `POST /api/images/encode` | an image as multipart (`curl -F file=@photo.png`), a raw body, or JSON `{"name","content_base64"}` + `?size=128&encoder=auto\|sd\|tiny&train=true&save_as=photo.txt` (train settings `epochs`, `lr`, `batch_size`) -> `{"text","encoder","width","height","latent_shape","bytes","chars","source_size","name","upload","job"}` (202 with a train job) |
| `POST /api/images/decode` | `{"text", "encoder"}` -> `{"png_base64","encoder","width","height","bytes","repaired"}` (a cut-off or rambling prediction is padded / truncated) |
| `GET /api/speech` | `{"backends", "faster_whisper", "whisper", "whisper_model", "server_url", "server_model", "auto", "ffmpeg", "recorders", "codecs", "default_rate", "token", "token_example", "text_format", "formats"}` |
| `POST /api/speech/transcribe` | audio as multipart (`curl -F file=@clip.wav`), a raw body, or JSON `{name, content_base64}`; options from the query string or the body (`backend`, `language`, `asr_model`, `asr_url`, `transcript`) -> `{"transcript", "backend", "model", "language", "seconds"}` |
| `POST /api/speech/teach` | the same audio forms + `transcript` (what the browser dictated), `rate`, `codec`, `normalise`, `waveform`, `pair`, `token`, `unique`, `train`, `epochs`, `lr`, `batch_size`, `save_as` -> `{"token", "transcript", "asr", "audio", "texts", "chars", "pair", "upload", "job"}` (202 with a train job on the texts) |
| `POST /api/speech/decode` | `{"text", "codec"}` -> `{"wav_base64", "codec", "rate", "samples", "seconds", "repaired"}` - an encoded or predicted waveform as playable audio |
| `POST /api/codegen/start` | `{"problems": [str or {"id","prompt","tests","expected_output"}], "problems_text", "problem_files", "phases": "both"\|"teacher"\|"model", "rounds", "teacher_provider": "ollama"\|"chatgpt", "teacher_model", "judge_provider", "judge_model", "url", "judge_url", "teacher_attempts", "model_attempts", "strictness", "judge", "fallback_teacher", "twonrl_per", "replay", "sandbox_timeout", "memory_mb", "blame" (the sandbox and the judge also teach the negative network), 2NRL settings, ...}` -> job whose records are `{"kind": "attempt"\|"problem"\|"round", ...}`; an attempt's `source` and a verdict's `judged_by` name the provider (400 when `teacher_provider` is `chatgpt` and the server has no key) |
| `GET /api/codegen/history` | `{"history": [records of all codegen runs]}` |
| `POST /api/codegen/solve` | `{"problem", "source": "model"\|"teacher", "attempts", "judge", "teacher_provider", ...}` -> `{"attempts": [{"code","run","style","verdict","correct"}], "correct"}` (no training) |
| `POST /api/codegen/run` | `{"code", "tests", "expected_output", "sandbox_timeout", "memory_mb"}` -> `{"run", "style", "verdict"}` |
| `GET /api/tutor` | the English tutor: `{"url", "model", "env_model", "providers": {"ollama": {...}, "chatgpt": {"url","model","configured"}}, "error_types", "modes", "twonrl_per", "levels", "plan_lessons", "defaults": {every setting}}` |
| `POST /api/tutor/start` | `{"blame" (teach the negative network why each sentence failed), "topic", "rounds": 3, "exercises": 5, "attempts", "focus", "level", "words", "tutor_provider": "ollama"\|"chatgpt", "tutor_model", "grader_provider", "grader_model", "url", "grader_url", "timeout", "mode", "length", "max_length", "temperature", "to_end", "threshold": 6, "grammar_weight": 0.6, "batch", "adapt", "drills", "teach_answer", "learn", "twonrl_per": "round"\|"lesson", "diff_corrections", "keep_weight", "min_weight", 2NRL settings, "plan" (lessons to plan from the final report card, 0 = none), "checkpoint_every"}` -> a job whose records are `{"kind": "lesson"\|"round"\|"report"\|"plan"\|"note", ...}`; a lesson carries `score`, `grammar`, `spelling`, `fluency`, `passed`, `error`, `sentence`, `correction`, `changes` (what the teacher changed, span by span), `comment`, a round the report card and what it taught (`corrections`, `edits`, `penalised`, `rewarded`), and the final `plan` record the lesson plan (see `POST /api/tutor/plan`) |
| `GET /api/tutor/history` | `{"history": [lesson / round / report records of all tutor runs]}` |
| `POST /api/tutor/lesson` | one round without training: the same settings plus `{"prefixes": [...]}` (skip the exercise writer and complete these) -> `{"source": "ollama"\|"chatgpt"\|"given", "exercises", "lessons": [{"exercise","continuation","sentence","grade"}], "report": report card}` (400 when `tutor_provider` is `chatgpt` and the server has no key, 502 when the teacher fails) |
| `POST /api/tutor/plan` | the lessons a report card calls for: `{"report": {report card}` (default: the card at the end of the last run), `"count": 3, "topic", "level", "exercises", "drills", "tutor_provider", "tutor_model", "url"}` -> `{"plan": {"summary", "level", "topic", "source": "ollama"\|"chatgpt"\|"report card", "weak": [{"error","count","share","focus"}], "targets", "lessons": [{"focus","targets","topic","why","exercises","drills","prefixes"}]}, "source", "provider", "model", "report"}` (400 without a card, 502 when the teacher fails) |
| `GET /api/job` / `POST /api/job/stop` | job status `{"id","type","state","progress","history","error",...}` / request a stop |
| `POST /api/predict` | `{"prefix","length","mode","to_end","step_penalty","temperature"}` -> `{"kind","continuation","full_text","cost","probability","step_costs","path","node_ids","expanded","reached_end"}`; `mode: "beam"` (both models), `k`, `beam` -> plus `top` / `bottom` (K entries each with `continuation`, `full_text`, `cost`, `probability`, `path`, `reached_end`) |
| `POST /api/generate` | `{"count","max_length","mode": "beam"\|"sample"\|"dijkstra","prefix","temperature","step_penalty","beam","seed"}` -> `{"samples": [{"text","full_text","cost","probability","path","node_ids","step_costs","reached_end"}]}`; `beam` returns the `count` most likely complete texts (the prediction search run to END), every `text` is the whole text, prefix included |
| `POST /api/converse` | `{"opening","turns": 6,"mode": "beam"\|"sample","context": 12,"max_length": 60,"k": 5,"beam","temperature","step_penalty","seed","speakers": ["A","B"],"history": [utterances so far],"partner": kind in memory,"avoid_repeats": true}` -> `{"kind","partner","speakers","count","turns": [{"index","speaker","text","context","reply","cost","probability","reached_end","fresh","given","repeat","candidates","skipped","labels","node_ids","step_costs"}]}`; `history` continues a conversation (only the new turns come back) |
| `POST /api/score` | `{"text"}` -> `{"log_prob","per_char","chars","transitions","unknown_transitions"}` |
| `POST /api/2nrl` | `{"bad": [...], "good": [...], "neg_epochs","pos_epochs","neg_lr","pos_lr", "bad_weights" \| "bad_ratings", "good_weights" \| "good_ratings"}` (or `bad_files` / `good_files` upload names) -> job; the weights (0..1 shares) or ratings (marks out of 10) scale each phase per text |
| `POST /api/feedback` | rated texts: `{"good": [thumbs up], "bad": [thumbs down], "good_ratings": [10, 5], "bad_ratings": [...] (or "good_weights" / "bad_weights" as 0..1 shares), "neg_epochs": 2, "pos_epochs": 3, "neg_lr": 0.5, "pos_lr": 0.1}` (also `*_text`, `*_files`) -> `{"job", "action": "2nrl"\|"reward"\|"punish", "good", "bad", "good_weights", "bad_weights"}`: 2NRL when both kinds are given, reward-only on thumbs up alone, punish (negative phase, then invert) on thumbs down alone. A rating is more than a like: each text is learned in proportion to its mark (10 = the full rate, 0 skips it) |
| `GET /api/negative` | the negative network: `{"path","active","stats","reasons": [{"reason","blame","fails","edges","share"}],"journal": [{"at","text","reason","severity","source","note"}],"weights","settings"}` |
| `POST /api/negative/blame` | teach it a failure: `{"texts"\|"text","reason","severity": 1,"source","note","epochs": 1}` -> `{"records","reasons","stats", ...}`; the only call that adds structure to the negative network |
| `POST /api/negative/clear` | the tutor passed these: `{"texts"\|"text","weight": 1,"epochs": 1}` -> `{"matched","unmatched","records","stats"}`; nothing is created |
| `POST /api/negative/judge` | `{"texts"\|"text","threshold","min_coverage","spans": 5}` -> `{"verdicts": [{"verdict": "reject"\|"suspect"\|"pass","risk","coverage","blame","reasons","spans": [{"start","end","fragment","blame","fails","reason"}],"why"}]}` |
| `POST /api/negative/filter` | the pair: `{"count": 3,"prefix","mode","max_length","temperature","over_sample": 3,"threshold","min_coverage","ratio": 0,"no_ratio","peak","strict","learn"}` (or `{"texts"}` to judge given texts) -> `{"texts" (the cleanest survivors),"kept","rejected": [verdicts],"verdicts","candidates","asked","rate","pair"}` |
| `POST /api/negative/forget` / `POST /api/negative/settings` / `POST /api/negative/reset` / `POST /api/negative/save` | drop or fade a reason `{"reason","factor"}` / `{"threshold","min_coverage","share_scale","blame_scale","clear_scale"}` / a fresh negative network `{"seed"}` / write it `{"path"}` |
| `POST /api/invert` / `POST /api/compress` | statistics / `{"merges", ...}` |
| `POST /api/evolve/start` / `POST /api/evolve/stop` / `GET /api/evolve/history` | `{"corpus": [...]` or `"corpus_text"` or `"corpus_files"`, `"generations"` (null = forever), `samples`, `max_length`, `temperature`, `checkpoint_every`, `blatant_mode`, `blatant_margin`, `blatant_boost`, `blame`, ...}` -> job; generation records carry `failures`, `blatant`, `boost_mean`, `boost_max`, `flipped`, `twonrl`, `mode` (and `negative_blamed` / `negative_reasons` with `blame`) |
| `POST /api/save` / `POST /api/load` / `POST /api/reset` | `{"path"}` (default: the active kind's file; the negative network is written beside it when it holds failures) / `{"path"}` (any kind; switches to it) / `{"seed", "kind"}` (+ `count_scale`, `global_scale`, `window_scale`, `reward_scale`, `window` for a fresh count model) |
| `POST /api/model/weights` | count model: `{"count_scale", "global_scale", "window_scale", "reward_scale", "window"}` -> `{"weights", "stats"}`; every edge weight is recomputed |
| `GET /api/checkpoints` / `POST /api/checkpoints/save` / `POST /api/checkpoints/restore` | list / `{"tag"}` / `{"name"}` |
| `GET /api/graph?limit=150` | top nodes by visit count with their activation parameters, and the edges between them with weight, count, probability, cost (count model: also `reward`, `share`, `recent_share`, `recent_count`, plus `total_traversals`, `window_traversals`, `window`) |
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
uploaded files), Predict (path with per-step costs and a Like button that
rewards the shown text - a thumbs-up feedback job), Generate (whole texts from
the prediction search - beam: the K most likely complete texts, optionally
continuing a prefix; sample; dijkstra - with thumbs up / thumbs down ratings:
"Train on ratings" runs 2NRL on them, thumbs down as the negative phase, thumbs
up as the positive phase), Converse (the model talks to itself in a chat
view: an opening line, turns, context, beam / sample, the two voices' names,
the other kind in memory as the second voice; Continue extends the
conversation, and turns are rated like samples; every rating carries a mark out
of 10 - "how good" / "how bad" - and the network learns each text in proportion
to it), Score, 2NRL, Negative (the
failure network: run the pair and see what was vetoed and why, judge a text
with its blamed fragments marked, blame or clear texts by hand, and the table
of everything the tutor has blamed with the journal of what it said),
Evolve (live chart of the discriminator gap), Ollama (corpus from a prompt,
adversarial review), Tutor (automated English lessons: the settings, a dry run
that marks without training, a chart of the marks per round, the report card
with the mistakes, every lesson with what the network wrote, the correction
and the teacher's line, and the lesson plan the teacher writes from the report
card - each lesson loadable into the settings with one click), Code (code
generation with the sandbox and the judge),
Speech (record the microphone, the browser writes down what it hears, teach
the words and the waveform),
Checkpoints (save / restore / load / reset) and a Graph view of the most
visited nodes.

Training files: drop text files onto the Train panel (or press "Upload
files…"); the browser reads them and sends them to `POST /api/uploads` (a
`.zip` goes up as bytes and stays one entry in the list; the server unpacks the
text files inside it behind the scenes whenever it is used), the server keeps
them in its `--upload-dir`, and the list lets you tick which files
to train on, one text per line or each file as one text. The same picker feeds
the 2NRL (bad / good files) and Evolve (corpus files) panels.

```bash
make frontend-install && make frontend-build   # rebuild dist
make serve                                     # then make frontend-dev in another shell for hot reload
```

## Ollama: corpus from a prompt, adversarial review

The network can be hooked into a local LLM served by [Ollama](https://ollama.com)
(standard library only; nothing to install on the Python side). Two uses:

* **Corpus from a prompt** — Ollama writes *N* lines about a prompt, either
  correct (`--style good`) or deliberately wrong (`--style garbage`): the two
  halves of 2NRL. The lines can be trained on directly, written to a file, or
  kept as an upload.
* **Adversarial review / rating** — Ollama plays the harsh critic: every sample
  the network generates (or any text you give it) gets a rating from 0 to 10,
  a pass/fail verdict against a threshold and a one-sentence critique. With
  `--2nrl` the failed samples become the negative phase and the passed ones
  (plus a corpus) the positive phase, so an external LLM discriminator drives
  the self-upgrade.

```bash
ollama pull llama3.2                                    # once, on the machine running Ollama
python -m radixnet ollama models
python -m radixnet ollama corpus --prompt "short true sentences about the sea" --lines 30 --train
python -m radixnet ollama corpus --prompt "short true sentences about the sea" --style garbage --out garbage.txt
python -m radixnet ollama review --count 8 --threshold 6
python -m radixnet ollama review --count 8 --2nrl --good data/sample_corpus.txt
python -m radixnet ollama review --text "the cat sat on the mat" --text "mat the on sat cat the"
```

`--url` / `--ollama-model` (or `OLLAMA_HOST` / `RADIXNET_OLLAMA_MODEL` in the
environment) select the server (default `http://127.0.0.1:11434`) and model
(default `llama3.2`). `make ollama-models`, `ollama-corpus`, `ollama-garbage`,
`ollama-review` and `ollama-2nrl` wrap the same commands (`PROMPT`, `LINES`,
`STYLE`, `COUNT`, `THRESHOLD`, `OLLAMA_URL`, `OLLAMA_MODEL`).

The API exposes the same through `GET /api/ollama/models`, `POST /api/ollama/corpus`
and `POST /api/ollama/review` (see the table above), and the frontend's Ollama
tab wraps them: generate a corpus and train on it / save it as an upload, or
review the model's samples and apply the verdicts as a 2NRL job. In Docker the
API reaches an Ollama on the host through `host.docker.internal`; `make up-ollama`
starts an Ollama container next to the API instead (`OLLAMA_HOST=http://ollama:11434`).

## Tutor: automated English lessons

`radixnet tutor` (and the Tutor tab, and `POST /api/tutor/start`) is the
prediction process with nobody at the keyboard: the LLM sets the exercise, the
network answers it, the LLM marks the answer, and the marks drive the learning.

```
topic -> prefix (LLM) -> completion (the prediction search) -> grade (LLM) -> 2NRL
```

One **round** is:

1. **The exercise.** The teacher writes `--exercises` sentence openings about
   `--topic`, each drilling one point of English grammar (`focus`) and each
   with its own model answer, so a lesson can teach even when the network says
   nothing. `--focus "past tense"` pins every exercise to one point.
2. **The completion.** The network continues each prefix with the ordinary
   prediction search (`--mode dijkstra` - the cheapest path - `beam` or
   `sample`; `--attempts N` asks for more than one answer, the extra ones
   sampled). Exactly what the Predict tab does with a human-typed prefix.
3. **The grade.** The same LLM marks every finished sentence as an English
   teacher: grammar, spelling and fluency out of 10, the single worst mistake
   named from a fixed list (`agreement`, `tense`, `article`, `preposition`,
   `plural`, `pronoun`, `word-order`, `spelling`, `punctuation`, `vocabulary`,
   `fragment`, `nonsense`, or `none`), one sentence of teaching, and the
   **correction**: the same sentence written out in correct English, keeping
   the prefix word for word. Grammar is what is being taught, so grammar is
   most of the mark: `score = grammar_weight * grammar + (1 - grammar_weight) *
   mean(spelling, fluency)`, `--grammar-weight 0.6` by default. A sentence
   passes at `--threshold` (6 out of 10).
4. **The lesson learned.** A correction is taught *as a correction*. The
   sentence the network wrote and the sentence the teacher wrote instead are
   aligned character by character, and only the trigram nodes they disagree on
   move: the step that wrote the struck-out character is penalised, the step
   that writes the teacher's version is rewarded, and the words both sentences
   share keep what they earned. Punishing a whole sentence for one wrong
   plural taxed the trigrams that were right; this does not.

   The rest is unchanged. The whole corrected sentence is still traversed -
   it is correct English whatever the mistake was - and `--keep-weight` (0.25)
   gives its unchanged words a smaller share of the reward (0 teaches the fix
   alone, 1 rewards the whole sentence as before). A failure the teacher left
   uncorrected is still 2NRL garbage weighted by how bad the mark was
   (`--min-weight` for a near miss, 1 for a hopeless answer); the model
   answers and the sentences that passed are still the fine-tune pass,
   weighted by how good the mark was - a sentence marked 9 gets nine tenths of
   the learning rate, the teacher's own English the full rate. Nothing is
   punished when the network wrote nothing: the prefix itself is correct
   English. `--no-diff-corrections` goes back to the whole-sentence way.

   The same alignment is a command of its own, for a correction typed by hand:

   ```bash
   python -m radixnet correct --wrong "the cat sit on the mat" --right "the cat sits on the mat"
   python -m radixnet correct --wrong "he go to school" --right "he goes to school" --dry-run
   go/bin/radixnet-count correct --wrong "a apple a day" --right "an apple a day" --keep 0
   ```

The mistakes of a round add up to a **report card** (marks, pass rate, an error
histogram and the weakest points). With `--adapt` (on by default) the weakest
points become the next round's syllabus - the teacher notices that the class
keeps failing plurals and sets plural exercises - and `--drills N` asks for N
extra correct example sentences about them, which join the fine-tune pass.

5. **The next lesson plan.** `--plan N` hands the report card at the end of the
   run back to the teacher, which writes the syllabus of the lessons that
   follow: N lessons, each drilling one point of grammar, each naming the
   mistake of the card it repairs, with a topic and a line on why it is being
   taught, and a `level` that goes up when the card is strong (80% passed at 8
   out of 10). The marks alone already imply a plan - one lesson per weak
   point, worst first - and that is the floor: a weakness the teacher's plan
   skips takes the place of a lesson that drills nothing the card marked down,
   and an answer that cannot be read leaves the card's own plan standing
   (`source` says which wrote it). Every lesson carries the settings to run it
   with, so the Tutor tab can load one into the form and start it, and the CLI
   prints the command for the first one.

```bash
ollama pull llama3.2
python -m radixnet tutor --topic "everyday life" --rounds 5 --exercises 5
python -m radixnet tutor --topic "the sea" --focus "past tense" --drills 5 --threshold 7
python -m radixnet tutor --topic animals --dry-run            # set and mark, train nothing
python -m radixnet tutor --topic animals --rounds 3 --plan 3  # ... and plan the next three lessons
python -m radixnet tutor --topic animals --rounds 3 --report lessons.json
make tutor TOPIC="everyday life" ROUNDS=5
make tutor-dry TOPIC="everyday life"
make tutor-plan TOPIC="everyday life" PLAN=3
```

Cost, per round: one call for the exercises, one per `--batch` marked
sentences, and one more with `--drills`; `--plan` adds one for the whole run.
`--tutor-model` (or `RADIXNET_TUTOR_MODEL`, else `RADIXNET_OLLAMA_MODEL`, else
`llama3.2`) is the teacher, `--grader-model` lets a second model do the marking,
and `--url` / `OLLAMA_HOST` picks the server. Ctrl-C stops after the current
round and saves.

### Who teaches: a local model or ChatGPT

Every lesson is set and marked by an LLM, and `--tutor-provider` chooses which
one — the same choice the code generator makes with `--teacher-provider`:

| | `ollama` (default) | `chatgpt` |
|---|---|---|
| where | a local [Ollama](https://ollama.com) server | OpenAI's hosted API |
| model | lessons: `--tutor-model`, default `$RADIXNET_TUTOR_MODEL` or `llama3.2`; codegen: `--teacher-model`, default `$RADIXNET_CODEGEN_MODEL` or `gemma4` | `--tutor-model` / `--teacher-model`, default `$RADIXNET_OPENAI_MODEL` or `gpt-4o-mini` |
| endpoint | `--url`, else `$OLLAMA_HOST` or `http://127.0.0.1:11434` | `--url`, else `$OPENAI_BASE_URL` or `https://api.openai.com/v1` |
| credentials | none | `$OPENAI_API_KEY`, or `$OPENAI_API_KEY_FILE` holding it |
| privacy / cost | nothing leaves the machine, no cost | every exercise, every sentence the network writes and every generated program is sent to OpenAI and billed |

```bash
export OPENAI_API_KEY=sk-...
python -m radixnet chatgpt models                                    # does the key work?
python -m radixnet chatgpt ask --prompt "write one simple English sentence"
python -m radixnet tutor --topic "everyday life" --tutor-provider chatgpt
python -m radixnet tutor --topic animals --tutor-provider chatgpt --grader-provider ollama
make tutor TOPIC="everyday life" TUTOR=chatgpt
```

The marking follows the teacher unless `--grader-provider` (and optionally
`--grader-url` / `--grader-model`) names the other one, so ChatGPT can set the
exercises and a local model mark them, or the other way round; every grade
records which one gave it (`graded_by`). The key is only ever read from the
environment of the process talking to OpenAI — it is never a request field, and
never lands in a config, a record or a report.

`$OPENAI_BASE_URL` also points the `chatgpt` provider at any OpenAI-compatible
server (llama.cpp, vLLM, LM Studio, a gateway). Ollama's `options` are
translated to the chat-completions fields, and a model that rejects one (the
reasoning models refuse `temperature`, older ones `response_format`) is retried
without it. A key is never sent unencrypted to a remote host: use `https://`, a
server on this machine, or set `RADIXNET_OPENAI_ALLOW_INSECURE=1` deliberately.

The API adds `GET /api/tutor` (the teachers on offer, defaults and the marking vocabulary),
`POST /api/tutor/start` (the job), `GET /api/tutor/history`,
`POST /api/tutor/lesson` (one round of exercises, completions and grades
without training - give it `prefixes` to skip the exercise writer and mark your
own) and `POST /api/tutor/plan` (a report card in, the next lessons out; with
no `report` the card at the end of the last run is used). The Tutor tab drives
all of it and shows the marks per round, the report card and every lesson next
to its correction, with the changed words struck out against what replaced
them, then the lesson plan under it - "Use this lesson" loads one into the
settings, ready to start; the **Teacher** selector switches between Ollama
and ChatGPT (the URL, the model and the notes follow it). **Both servers run
the lessons**: the Go server has the same endpoints, the same two teachers and
`radixnet-count tutor` the same command (`-tutor-provider chatgpt`, `-plan N`),
with the count / reward model answering the exercises.

## Code generation: sandbox, LLM tutor and judge, 2NRL rewards

`radixnet codegen` turns the network into a code generator trained by
reinforcement: programs are run in a sandbox, judged, and every attempt feeds
2NRL, wrong answers as the negative phase before the correct answer as the
positive phase.

```
problem -> Python program -> sandbox run -> judge (PEP 8, naming, runs, task) -> punish / reward (2NRL)
```

Two semi-supervised phases over the same problem list (`--phase both`, the default):

1. **teacher** — the tutor writes a solution; the sandbox runs it; on an error
   or a rejected verdict the tutor is asked to fix it (up to
   `--teacher-attempts`); the judge confirms the result. The network then
   learns the concatenated question + answer, with every wrong attempt as 2NRL
   garbage first. The tutor is a local Ollama model (`--teacher-model`,
   default `gemma4`) or ChatGPT — see below.
2. **model** — the network itself continues each problem prompt into code
   (first the cheapest path, then samples, up to `--model-attempts`). A program
   that errors or is judged wrong is punished (negative phase) and the loop
   tries again; a correct one is rewarded (positive phase). When the network
   never succeeds the teacher supplies the answer for the reward
   (`--no-fallback-teacher` disables that).

Correctness needs all of: the program runs in the sandbox (exit 0, no timeout),
its stdout matches `expected_output` and its appended `tests` pass when the
problem has them, the judge says the task is accomplished, and (`--strictness
strict`) both the objective PEP 8 / naming checker (indentation, line length,
whitespace, blank lines, snake_case / CapWords / UPPER_CASE) and the judge accept
the formatting and naming. `--strictness lenient` needs only "runs" and "task".

```bash
ollama pull gemma4
python -m radixnet codegen --problems data/sample_problems.jsonl                   # teacher, then model
python -m radixnet codegen --problems data/sample_problems.txt --phase model --rounds 3
python -m radixnet codegen --problems problems.jsonl --twonrl-per round --report report.json
make codegen PROBLEMS=data/sample_problems.jsonl PHASE=both
```

### Who tutors here

`--teacher-provider ollama|chatgpt` picks the LLM that writes, fixes and judges
the solutions, exactly as `--tutor-provider` does for the lessons
([who teaches](#who-teaches-a-local-model-or-chatgpt)):

```bash
python -m radixnet codegen --problems data/sample_problems.jsonl --teacher-provider chatgpt
python -m radixnet codegen --problems problems.jsonl --teacher-provider chatgpt --judge-provider ollama
make codegen PROBLEMS=data/sample_problems.jsonl TUTOR=chatgpt
```

The judge follows the teacher unless `--judge-provider` (and optionally
`--judge-url` / `--judge-model`) names the other one, so a ChatGPT teacher can
be reviewed by a local model, or the other way round. Attempt records carry the
provider that wrote them (`source`) and the one that judged them (`judged_by`),
and `--report` / `/api/codegen/history` keep both.

Problem files: one prompt per line (`.txt`, `#` comments), or `.json` / `.jsonl`
objects `{"id", "prompt", "tests", "expected_output"}` (`data/sample_problems.*`).
`--twonrl-per problem` (default) applies 2NRL after every problem, `round` once
per pass over all attempts; `--no-replay` stops earlier correct solutions from
being added to every positive phase; `--report FILE` writes all records and the
solutions. Ctrl-C stops after the current problem and saves.

The sandbox runs each program with `python -I` in a scratch directory with an
empty environment, memory (`--memory-mb`), CPU and file-size limits, a wall-clock
`--sandbox-timeout`, and, where `unshare` can create a network namespace, no
network. That contains accidents, not a hostile program; run generated code
inside the Docker image when the problems or the models are untrusted. Every
result reports `network_isolated`; inside a Docker container the default
seccomp policy prevents the per-program namespace, so there the container's
own network is the boundary (give the `api` service no network it should not
have).

API: `POST /api/codegen/start` (job), `GET /api/codegen/history`,
`POST /api/codegen/solve` (solve one problem with the model or the teacher,
no training) and `POST /api/codegen/run` (sandbox only); `teacher_provider`
(and `judge_provider`) pick the tutor there too, and `GET /api/chatgpt/models`
reports whether the server has a usable key. The server only ever uses **its
own** `$OPENAI_API_KEY` — a key is never accepted as a request field — so set it
in the environment of `radixnet serve` (or the `api` container) and restart it.
The frontend's Code tab drives all of it: the tutor (Ollama or ChatGPT),
problems (typed or uploaded), live attempt / problem / round records, a "try a
problem" box and a sandbox runner.

## Two models: RadixNet and the count / reward model

The selector at the top of the frontend (and `--kind` in the CLI, `POST
/api/model/select` in the API) chooses the algorithm.  Both live on the same
self-compressing cyclic graph and share encoding, prefix location, sampling,
scoring, compression, checkpoints and persistence; a model file records its
kind, so `load` always restores the right one.

| | RadixNet (`radix`) | Count / reward (`count`) |
|---|---|---|
| edge weight | learned by the one-hop rule together with the per-node sine activations | a **dual frequency function**: the edge's share of its node's traversals, all time (`R_all`) and inside a sliding window of the last N traversals (`R_recent`), plus rewards - `global_scale · log R_all + window_scale · log R_recent + reward_scale · reward` (+ an optional `count_scale · log(1 + traversals)`); no gradient, no learning rate |
| training | epochs over mini-batches with `lr` / `act_lr` (and their schedules) | every epoch counts one more traversal of each text's path (all time, in the sliding window and in the global total) |
| feedback (thumbs, 2NRL, codegen judge, adversarial review) | train on the bad texts, invert, fine-tune on the good ones | `punish`: reward −= `strength` on every edge of a bad path; `reward`: a traversal plus reward += `strength`; nothing is inverted |
| `invert` | flips every weight and activation amplitude | flips the sign of every reward |
| prediction | Dijkstra's cheapest path (or sampling) | a beam search that returns the **top K** (most likely) and **bottom K** (least likely) continuations of one prefix in one call; the best one is the prediction |

```bash
python -m radixnet --kind count train --data data/sample_corpus.txt --epochs 3     # -> model.count.json
python -m radixnet --model model.count.json predict --prefix 'the quick' --length 10 --k 5
python -m radixnet --model model.count.json feedback --good-text 'the quick brown fox' --bad-text 'zzz qqq' --strength 2
```

**The count model's weight function** keeps several numbers per edge - its
all-time traversals, its traversals inside a sliding window of the last
`window` traversals seen anywhere in the graph (default 10 000), and its
reward - together with the global totals.  Each count is compared against
the node the edge leaves (the sum over the node's children, smoothed by 0.5),
giving two ratios, and the weight is

```
R_all    = (count + 0.5) / (node traversals + 0.5 · children)      # what the node did, all time
R_recent = (window count + 0.5) / (node window traversals + 0.5 · children)   # what it did recently
weight   = global_scale · log R_all + window_scale · log R_recent + reward_scale · reward
           (+ count_scale · log(1 + count), off by default)
```

so `P(child | node) ∝ R_all^global_scale · R_recent^window_scale · e^reward`;
the default scales are 0.5 and 0.5 (the geometric mean of the two shares:
when history and the window agree the probability is the share itself),
so a text seen a thousand times last year and one seen ten times today can
both win, and raising one scale trusts history or recency more.  The Train tab (count model) shows the scales and the window with an
"Apply" button, `radixnet weights` does the same from the shell, and the
status bar shows the traversal totals; the Graph tab's edge tooltips show
each edge's all-time and recent share.

The server keeps the model of each kind in memory: switching kinds parks the
active model (unsaved work included) and brings the other one back, loading
its file (`model.json` / `model.count.json`) or creating a fresh one the
first time.

## Learning-rate schedules (graph functions)

The learning rate and the activation learning rate can grow (or shrink) from
epoch to epoch.  A schedule is a small expression of the epoch that the model
evaluates once per epoch - a *graph function* - given with `--lr-schedule` /
`--act-lr-schedule` (CLI), `lr_schedule` / `act_lr_schedule` (API) or the
schedule block of the Train tab, which draws the curve while you type.

| Expression | Meaning |
|---|---|
| `linear(lr0, 4 * lr0)` | ramp linearly from the base rate to four times it |
| `geometric(lr0, 4 * lr0)` | same, by a constant factor per epoch |
| `cosine(lr0, 4 * lr0)` | a smooth S-shaped rise |
| `step(lr0, 1.5, 2)` | multiply by 1.5 every two epochs |
| `lr0 * 1.25 ** i` | compound growth of 25 % per epoch |
| `warmup(lr0 / 10, lr0, 3)` | warm up over three epochs, then hold |
| `0.1 if epoch < 3 else 0.5` | a conditional |
| `lr / 10` (activation schedule) | a tenth of whatever the learning rate is that epoch |

Variables: `epoch` (1-based), `i` (0-based), `epochs`, `t` (0 at the first
epoch, 1 at the last), `lr0` (the base rate: `--lr` or `--act-lr`), `act_lr0`
and, for the activation schedule, `lr` (the epoch's learning rate).  Functions:
`sin cos tan exp log log2 log10 sqrt pow abs floor ceil round min max tanh
clamp`, constants `pi`, `e`.  The "Reverse the schedule" checkbox
(`--reverse-schedule`, `reverse_schedule`) plays a schedule backwards - the
last epoch's rates come first, so a ramp up becomes a ramp down and a warm-up
a cool-down.  Expressions are validated against a whitelist
(no names, attributes, strings or calls outside that list), and every value
must be finite and non-negative - a bad expression is rejected before training
starts.  `python -m radixnet schedule --epochs 6 --lr-schedule 'linear(lr0, 4 * lr0)' --act-lr-schedule 'lr / 10'`
prints the rates with a bar graph; `python -m radixnet schedule` lists the presets.

## Images: the Stable Diffusion encoder, base64, the model

Stable Diffusion generates a picture by *decoding* a latent - a `4 x H/8 x
W/8` block of numbers - with its VAE.  The Images tab (and `radixnet image`,
`POST /api/images/encode`) runs that process backwards: the same VAE's
encoder turns an image into its compressed latent (the deterministic latent
mean, 48 times fewer numbers than the RGB pixels), every number becomes one
signed byte and the bytes become base64.  The result is a text,

```
img:sd:128x128:AAECAwQFBgcICQoLDA0ODxAREhMUFRYX…
```

which the trigram network trains on, scores, continues and generates like any
other text.  `decode` runs the forward process again (base64 -> latent -> VAE
decoder -> PNG), so an encoded image or a **predicted** text can be looked at;
a prediction whose base64 tail is cut off or garbled is repaired (padded or
truncated) first.

```bash
pip install pillow torch diffusers            # the real encoder (weights: $RADIXNET_SD_VAE, default stabilityai/sd-vae-ft-mse)
python -m radixnet image encode photo.jpg --size 128 --train --epochs 3      # text on stdout, model trained on it
python -m radixnet predict --prefix 'img:sd:128x128:' --length 800 --json | python -c 'import json,sys; print(json.load(sys.stdin)["full_text"])' > guess.txt
python -m radixnet image decode --data guess.txt --out guess.png
```

Without the diffusion weights (or `torch` / `diffusers`) the `tiny` stand-in
- an RGB thumbnail at 1/8 of the size, the same reduction - keeps the format,
the API, the CLI and the tab working; `auto` (the default) picks `sd` when it
loads.  Sizes are squares that are multiples of 8 (64 .. 512); 128 gives a
4 x 16 x 16 latent, 1 024 bytes, about 1 400 characters of text.

## Speech: teach it by talking to it

Say something and the network learns **two texts that start with the same
unique token** - what you said and how it sounded:

```
<speech:9f2a1c7d> the cat sat on the mat
<speech:9f2a1c7d> aud:mu:8000x1:gOXv7NgqFBAYTePu7dwvFRAXPuHu7d82FhAWNt/t7uE+FxAV…
```

The token is `<speech>` with a short digest of the recording folded in, so it
is unique to that utterance, the same every time that recording comes back, and
identical in both texts: in the cyclic graph the words and the waveform leave
the **same node**, which is what ties them together. The waveform text is the
recording itself - mixed to mono, resampled to 8 kHz and quantised to one
mu-law byte per sample (G.711 / WaveNet's 256 levels, which keep ~2 % relative
error from a shout down to a whisper where linear 8-bit is already at 38 %) -
base64-encoded, so the trigram network trains on it, scores it, continues it
and generates it like any other text. `speech decode` turns an encoded - or
**predicted** - waveform back into a WAV file, so you can listen to what the
graph thinks the sound is.

**In the browser** (the Speech tab): press Record, talk, press Stop. The page
records with `MediaRecorder`, writes down what it hears with the Web Speech API
while you speak (Chrome, Edge, Safari), converts the recording to a 16-bit PCM
WAV itself, and "Teach the model" posts both texts and trains on them. Correct
the transcript by hand before teaching if it misheard; leave it empty and the
server transcribes instead. "Preview the texts" shows what would be learned
without touching the model, and the last card plays any `aud:` text back.

**From the shell:**

```bash
python -m radixnet speech info                         # which backends, recorders and codecs are there
python -m radixnet speech listen --seconds 5 --train   # record, transcribe, learn the words and the sound
python -m radixnet speech teach clip.wav --text "the cat sat on the mat" --train --pair
python -m radixnet speech transcribe clip.wav          # speech to text only
python -m radixnet predict --prefix '<speech:9f2a1c7d> ' --length 40
python -m radixnet speech decode --text 'aud:mu:8000x1:gOXv7Ngq…' --out heard.wav
```

Speech to text uses the first backend that is available:

| backend | what it is |
|---|---|
| `given` | the transcript comes from the browser's Web Speech API, `--text`, or the API's `transcript` field - always available, and why the feature needs nothing installed |
| `faster-whisper` | `pip install radixnet[speech]` (CTranslate2 Whisper, fast on the CPU) |
| `whisper` | `pip install radixnet[whisper]` (openai-whisper) |
| `server` | any OpenAI-compatible `/v1/audio/transcriptions` endpoint (whisper.cpp's server, Speaches, ...) - set `RADIXNET_ASR_URL` |

`RADIXNET_WHISPER_MODEL` (default `base`), `RADIXNET_ASR_URL`,
`RADIXNET_ASR_MODEL` and `RADIXNET_ASR_KEY` configure them. A failing
transcription is *reported, not fatal*: the waveform alone is still learned.

WAV files are read by the standard library (PCM 8/16/24/32, IEEE float, A-law,
mu-law); MP3, M4A, WebM, Ogg and FLAC need `ffmpeg` on the PATH - the browser
never does, because it converts its recording before uploading.
`--rate 4000` halves the text an utterance produces, `--rate 16000` doubles it;
one second at the default 8 kHz is about 10 700 characters, so short
utterances are the ones to teach. `--pair` adds a third text - the waveform
followed by its transcript - so the prediction search can run from the sound
straight into the words; `--shared-token` puts every utterance behind the plain
`<speech>` instead of a unique token.

## Evolve: train on failures, blatantly fail on purpose, then invert

The evolve loop (Evolve tab, `evolve`, `POST /api/evolve/start`) can treat the
generator's failures dynamically instead of feeding the worst half of its
samples to a uniform 2NRL pass.  A *failure* is a fake the discriminator
scores below the real texts; `g`, how far below (per-char log-prob), is how
bad it is.

| `blatant_mode` | What happens each generation |
|---|---|
| `none` (default) | the worst half of the fakes is 2NRL garbage with the plain `neg_lr` |
| `fail_invert` | **train on every failure with learning rates multiplied by `1 + g / margin`** (weights, node states and the activation parameters alike, capped at `blatant_boost`) - the worse the response, the more the activation functions update, so the model *blatantly fails on purpose* - **then invert the model**, turning what it now does confidently into what it confidently avoids, and fine-tune on real texts. Nothing failed: only the fine-tune pass, no inversion. The count model applies the same multipliers to its penalties instead. |
| `activation` / `state` | the local variant: no negative pass; every other node on a failed path has its activation amplitude (or trained node value `z`) moved toward its negation by `g / (2 · margin)` - a slight attenuation for a slightly worse fake, a neutralised path at the margin, a full sign flip at twice it - so only that path's transitions turn unlikely. Fakes beyond the margin are *blatant* and leave the 2NRL garbage set; when every bad fake was blatant the generation skips the negative pass and the global inversion. |

`blatant_margin` (default 1.0 nats per character) is where a failure counts
as blatant; `blatant_boost` (default 4) caps the multiplier.  Generation
records and the Evolve tab's table show `failures`, `blatant`, the mean
boost / amount and whether a 2NRL pass ran.  Outside the loop the same
primitives are available directly: `RadixNet.two_nrl(bad, good,
bad_weights=[...])` and `model.invert_paths(texts, mode, amounts)`.

## The negative network: what went wrong, and why

The positive model learns what text looks like.  The **negative network**
(`radixnet/negative.py`, `--kind negative`, the Negative tab) is a second copy
of the same machinery - the same self-compressing cyclic graph, the same
trigram window, the same Dijkstra / beam searches - that keeps only the
negative portions: **every node and edge in it exists because something went
wrong there**.

What an edge remembers:

| Number | Meaning |
|---|---|
| `blame` | the summed severity of the failures that ran through it |
| `fails` | how many failing texts ran through it |
| `clear` | how much text the tutor *passed* ran through it |
| `reasons` | `{reason: blame}` - the tutor's reasons, split by how much blame each contributed |

The **net evidence** against an edge is `max(0, blame - clear)`: blame and
clearing cancel, so an edge the tutor's passes cross as often as its failures
do carries no verdict at all - which is how a common fragment like `" the "`
stays out of the judgement.  There is no learning rate and no gradient; like
the count / reward model every activation is the constant 1, so an edge's
score *is* its weight, and the weight is its share of the failure mass leaving
its parent:

```
net    = max(0, blame - clear_scale * clear)
R_bad  = (net + 0.5) / (net leaving the parent + 0.5 * children)
weight = share_scale * log(R_bad) + blame_scale * log(1 + net)
```

A softmax over that is `P(child | parent)` **under the failure distribution**:
the network models how text goes wrong.  `negative predict` therefore returns
the most likely ways to *fail* from a prefix (and the least likely ones as the
bottom-K) - a warning, not a suggestion.

### The negatives come from the tutor

Nothing is invented.  Every failure arrives from something outside the network
that looked at an output and said it was wrong, and why (`radixnet/blame.py`
turns each verdict into a **fault**: a reason tag, a severity, the tutor's own
sentence for the journal and, where there is one, the correction to diff
against):

| Tutor | How it blames | Reasons it gives |
|---|---|---|
| the **English tutor** (`tutor --blame`, the Tutor tab's checkbox, `POST /api/tutor/start {"blame": true}`) | the mistake it named marks the sentence, its mark out of 10 is the severity, its sentence of teaching is the note, and its correction is diffed so **only the characters it changed** are blamed | `agreement`, `tense`, `article`, `preposition`, `plural`, `pronoun`, `word-order`, `spelling`, `punctuation`, `vocabulary`, `fragment`, `nonsense` |
| the Ollama reviewer (`ollama review --blame`, the Ollama tab's checkbox, `"blame": true`) | its critique picks the reason, its rating the severity (0 -> 2.0, the pass threshold -> 0.25); the texts it passed clear blame | `gibberish`, `repetition`, `truncated`, `grammar`, `spelling`, `contradiction`, `false`, `incoherent`, `off-topic`, `empty`, `unrated`, `other` |
| the code sandbox, the style checker and the judge (`codegen --blame`) | every rejected program is blamed for what they found, with the teacher's feedback as the note | `timeout`, `crash`, `wrong-output`, `task-not-done`, `style`, `naming` |
| the evolve discriminator (`evolve --blame`) | every fake it scores below the real texts, by how far below | `discriminator`, `blatant` |
| a person | `negative blame --text ... --reason ... --note ...`, the Negative tab, a thumbs down | anything you type |

A correction is the sharpest lesson of all.  When the teacher writes the
sentence out in correct English, `NegativeNet.correct(wrong, right)` aligns the
two character by character (`radixnet/diff.py`) and blames only the steps that
wrote something the teacher struck out - the rest of the sentence was right and
keeps no verdict - while the correction itself clears blame wherever the
failure structure already knows it:

```
$ python -m radixnet negative why --text "the cat sit on the mat"     # after one tutor round
verdict     suspect
peak        1.5000
why         1 of 21 transitions are known failures (risk 0.07), mostly 'agreement', worst at 'sit '; below the threshold, kept

at      fragment   blame  fails  reason
------  --------  ------  -----  ---------
7..12   "sit "    1.5000      1  agreement
```

A blamed transition is never compressed away (`NegativeGraph.merge_child`
refuses to merge across it), so the fragment that went wrong stays an edge and
stays nameable however much the rest of the graph is folded up.

`negative reasons` (and `GET /api/negative`) prints the table of everything
blamed so far and the journal of what the tutor said, entry by entry.  The
tutor can be wrong too: `negative forget --reason TAG [--factor 0.5]` drops
that reason's blame, or fades it.

### Why a text is a failure

`negative why --text "..."` (`judge` / `POST /api/negative/judge`) walks the
text through the failure structure and reports

* `risk` - the net evidence per transition (repeating a failure blamed once
  scores about 1, sharing a third of one's transitions with it about 0.33),
* `coverage` - the share of its transitions that are known failures,
* `reasons` - the tutor's reasons behind that blame, heaviest first,
* `spans` - the worst fragments with their character range (the Negative tab
  marks them inside the text), and
* `verdict` - `reject` when coverage reaches `min_coverage` (0.5) and `risk`
  the `threshold` (1.0), `suspect` when something failed but not enough,
  `pass` when nothing here has ever failed.

```
$ python -m radixnet negative why --text "the the the the cat"
verdict     reject
risk        2.3333
coverage    1.0000
why         9 of 9 transitions are known failures (risk 2.33), mostly 'repetition', worst at 'he t'; rejected

at     fragment   blame  fails  reason
-----  --------  ------  -----  ----------
1..5   "he t"    3.0000      3  repetition
3..7   " the"    3.0000      3  repetition
```

### The pair: a GAN at output time

`radixnet/duo.py` puts the two networks on one output path.  In the evolve
loop the generator and the discriminator take turns improving each other; here
the finished pair works together on every answer: the positive model
over-samples candidates (it is the only one that can write), the negative one
judges each of them (it is the only one that knows what going wrong looks
like), what survives comes back ranked and what does not comes back with the
reason it was dropped.

Two signals reject, either one is enough, and both sit behind the coverage
gate so text the tutor has never failed is never vetoed on a hunch:

* **blame** - `risk` at or above the threshold,
* **ratio** - `log P_negative(text) - log P_positive(text)` per character, the
  classic discriminator logit of two generative models: how much more the
  candidate reads like known failure than like the text the positive model was
  trained on, and
* **peak** (`--peak N`, off by default) - the blame on a *single* fragment,
  which is how one word the tutor has already corrected vetoes a sentence that
  is otherwise perfectly good.

```bash
python -m radixnet negative filter --count 3 --max-length 60      # write, then veto
python -m radixnet negative filter --text "the the the the cat"   # judge given texts
```

```
decision  rule    risk    peak     ratio  reason      text
--------  -----  ------  ------  --------  ----------  --------------------------
reject    blame  2.3333  3.0000    1.5157  repetition  "the the the the cat"
suspect   -      0.0625  1.0000   -9.5765  repetition  "the rain in autumn"

2 candidates: 1 passed the filter, 1 vetoed (acceptance 0.5000)
  vetoed: "the the the the cat" - 9 of 9 transitions are known failures ...
```

`--strict` also drops the suspects, `--no-ratio` judges by blame alone, and
`--learn` blames what the filter itself rejected (off by default: the tutor
supplies the negatives, the filter only applies them).  `negative predict`
also comes back from `POST /api/negative/filter` as a `warning` - what the
negative network expects to go wrong from that prefix - whether or not
anything was actually vetoed.

The negative model is an ordinary model file (`model.negative.json` beside the
model, `--negative PATH` to move it) and an ordinary model kind, so
`--kind negative train` blames, `info`, `checkpoints`, `save` / `load` and the
model selector at the top of the frontend all work on it as usual.

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

## Go implementation of the count / reward model

`go/` holds a Go port of the count / reward model (`CountRewardNet`), a
standalone module with a library (`go/radixnet`) and a CLI
(`go/cmd/radixnet-count`); the Python implementation stays as it is. Model
files are interchangeable: both sides read and write the `radixnet-count`
JSON format, including the Mersenne Twister state, so a model trained on one
side continues on the other with identical numbers (`tests/test_go_parity.py`
trains the same corpus on both, compares structure, counts, rewards, window,
RNG state, predictions, generated texts, scores and conversations, and lets
each side continue the other's file).

```bash
make go-build                                   # -> go/bin/radixnet-count (needs Go 1.24+)
go/bin/radixnet-count --model model.count.json train --data data/sample_corpus.txt --epochs 5
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --k 5
go/bin/radixnet-count --model model.count.json generate --mode beam --count 5
go/bin/radixnet-count --model model.count.json converse --opening "the cat sat on the mat"
go/bin/radixnet-count --model model.count.json tutor --topic "everyday life" --rounds 3   # Ollama teaches it English
go/bin/radixnet-count --model model.count.json train --data book.txt --split paragraphs --workers 8
python -m radixnet --model model.count.json info    # the Python side reads the same file
```

Commands: `train`, `predict`, `generate`, `score`, `feedback`, `2nrl`, `invert`,
`weights`, `info`, `converse`, `serve`, `version`; global options `--model`,
`--json`, `--seed`, `--workers N` (a cap on the goroutines; 0, the default, is
none), `--exact` (atomic counting), `--out`, `--memlimit SIZE` (soft heap
limit, 80 % of the machine or container by default), `--memprofile PATH`.
Where the goroutines go:

| phase | concurrency |
|---|---|
| reading a corpus | `--split lines\|paragraphs\|pages\|file` decides what one text is (`--page-lines` cuts pages when a file has no form feeds); every text is one unit of work. Files and ZIP archives are **streamed**, never loaded whole: `--chunk N` (default 8192) texts at a time, at most `--inflight N` chunks (default two per CPU) in flight, so the reader runs ahead but memory does not grow with the corpus; `--parallel-parts` reads every archive entry at once instead of in order |
| encoding, tracing texts through the structure, counting | **one goroutine per text** of a chunk (no pool unless `--workers N`); the counters are bumped with plain increments from all of them at once - racy by design, a collision loses an update. `--exact` uses atomic increments instead: no lost updates, the result identical to the sequential run and to Python |
| building the structure | the one sequential phase: node splits reshape a shared radix index, and Go aborts the process on concurrent map writes, so this is not a race that can be ignored; texts that already walk through the graph are detected in parallel and skipped |
| the sliding window | applied in corpus order after each chunk's parallel pass (its semantics are the order of traversals); exact in both modes |
| weights and edge costs | recomputed lazily, only the touched rows after feedback; a full recompute after structural changes runs on a goroutine per 64 nodes |
| loss, scoring many texts | parallel reductions / one goroutine per text; the loss is the traversal-weighted mean edge cost, so it needs no list of transitions |
| prediction | the top and the bottom beam run side by side |

Measured on this 4-core machine (2 epochs over a 39 MB corpus: 1,000,000
lines in a ZIP of 10 entries):

| mode | time | peak RSS |
|---|---|---|
| one goroutine per text, racy (the default) | 12.0 s | 393 MB |
| `--exact --workers 4` | 6.6 s | 172 MB |
| `--workers 1` | 14.2 s | 79 MB |
| `--inflight 1` (one chunk at a time) | 11.2 s | 74 MB |
| `--chunk 1024` | 10.2 s | 68 MB |
| `--parallel-parts` (all 10 entries at once) | 11.6 s | 414 MB |

Racy counting lost 0.7 % of the traversals on that corpus and was not faster
than a small exact pool: spawning a goroutine per text and the cache-line
contention on shared counters cost more than they save. It is the default
because it was asked for; `--exact --workers 4` is the reproducible choice and
the faster one on this hardware. The structure, the window and the weights'
consistency with whatever was counted are exact in both modes.

### Massive ZIP archives: streaming, chunking and memory

Nothing in the Go path holds a corpus in memory. The CLI streams `--data`
files - a ZIP archive entry by entry (directories, macOS metadata, system
files, nested archives, encrypted, binary and empty entries skipped like the
Python `archive` module), a text file line by line, any line length - and the
model consumes the stream in chunks of `--chunk` texts (default 8192): the
structure pass walks the chunks once (novel texts observed in corpus order,
walkable ones skipped in parallel), every epoch re-streams the corpus chunk by
chunk (one goroutine per text, the window in order, rewards per chunk) and the
loss is computed from per-edge traversal counts, so memory is the graph plus
the chunks in flight whatever the archive's size.

Uncapped goroutines need one bound to stay alive: how many chunks may be in
flight at once. The reader spawns a goroutine per chunk and never waits for
it, but it does wait for a free slot - `--inflight N`, two per CPU by default -
and a chunk holds its slot from the moment it is read until the sequencer has
applied it, because a finished chunk waiting for its turn occupies memory like
any other. Without that bound the archive is read far faster than it is
counted: earlier builds reached 1.7 GB on the 39 MB corpus, and 3.2 GB with
`--parallel-parts` (a reader goroutine per archive entry, off by default,
where each open part now gets its own slot budget and only a window of parts
is open at a time). With the bound, memory is flat in the corpus size and set
by `--chunk` x `--inflight`: the same corpus trains in 393 MB, or 74 MB with
`--inflight 1`.

The other half of the problem is the garbage collector, and it is what kills a
server on a 240 MB archive. Go collects when the heap has grown to about twice
what is live, so a run whose graph is a gigabyte asks the operating system for
two - on a container with a hard limit that is an OOM kill rather than a
collection. Every process therefore sets a **soft memory limit** at 80 % of
its cgroup limit (or of the memory the machine has available) at startup,
which makes the collector work harder as the heap approaches it instead of
growing past it. `--memlimit 2GiB` sets it explicitly, `--memlimit off` turns
it off, and `GOMEMLIMIT` in the environment wins over both. The server reports
its heap and limit in `/api/status`, and the status bar shows `heap ... / ...`.

Measured on a 113 MB corpus (10,703 files of the Go source tree, one epoch,
421 K nodes and 2.34 M edges, 426 MB of live graph):

| run | time | peak RSS |
|---|---|---|
| no limit | 31.1 s | 1452 MB |
| `--memlimit 1GiB` | 30.3 s | 1027 MB |
| `--memlimit 512MiB` (below what the graph needs) | 50.2 s | 767 MB |
| **in a 1 GiB container**, `--memlimit off` | killed after 26.9 s | - |
| **in a 1 GiB container**, the new default | 32.5 s | 908 MiB of the 1 GiB |

The limit costs nothing until the heap approaches it, and only then trades
speed for staying inside the box. Peak RSS runs above the limit itself because
it also counts the runtime's unreturned pages and the page cache of the files
being read and written; the anonymous memory stays inside it. As a rule of
thumb the graph needs about three times the corpus text, so a 240 MB archive
of compressed text wants several gigabytes: with the limit the process slows
down and survives instead of being killed, but a corpus whose graph cannot fit
has to be trained in parts.

Saving and loading stream too: a model is encoded straight into its file
(gzip when the name ends with `.gz`) and decoded straight out of it, so a
100 MB model file never doubles in memory. If the graph itself does not fit in
the limit, the collector will run continuously rather than grow - train a
smaller corpus, or raise the limit.

The server does the same everywhere: `POST /api/uploads` streams multipart
parts and raw bodies straight into the upload directory (an archive is
validated by streaming its entries; the JSON forms are capped at 512 MB), the
listing inspects archives by streaming, and `POST /api/train` with `files`
(plus the optional `chunk_size`, `inflight`, `parallel_parts`) streams them
through the job; the frontend needs no change.

### The Go HTTP server and the frontend

`radixnet-count serve` is an HTTP server speaking the Python API's JSON
contract for everything the count model supports, and it serves the same
prebuilt frontend:

```bash
make go-serve PORT=8001           # go/bin/radixnet-count --model model.count.json serve --port 8001 \
                                  #   --frontend-dir frontend/dist --upload-dir uploads --checkpoint-dir checkpoints
open http://localhost:8001        # the React app, now backed by the Go model
```

The frontend detects the engine (`GET /api/health` and `/api/status` carry
`engine: "go"`, the worker count and the live goroutine count): it shows a
**Go engine** badge in the header and `engine go · workers · goroutines` in the
status bar, hides the tabs that need the Python server (Evolve, Ollama, Code,
Images, Speech - the Tutor tab stays, both servers run the lessons), locks the model
selector to the count model, and the Train tab gains a
**Texts are** selector (`lines | paragraphs | pages`) so the pasted text and the
uploaded files are cut into the units the goroutines fan out over. Train,
Predict (with the Like button), Generate (with ratings), Converse, Score, 2NRL,
Tutor, Checkpoints and Graph work unchanged.

| endpoint | Go server |
|---|---|
| `GET /api/health`, `GET /api/status`, `GET /api/model`, `POST /api/model/select` (count only), `POST /api/model/weights` | as the Python server, plus `engine`, `workers` (0 = one goroutine per text), `goroutines`, `counting` (`racy` \| `exact`), `heap_bytes`, `memory_limit_bytes` |
| `POST /api/train` | `{texts \| text \| files, whole_file, split: lines \| paragraphs \| pages \| file, page_lines, epochs, auto_compress, chunk_size, inflight, parallel_parts}` -> a job; uploads stream through in chunks whatever their size; learning rates are accepted and ignored |
| `GET /api/job`, `POST /api/job/stop` | one job at a time (409 while it runs); a job holds the model between epochs only, so predictions and the status poll keep answering |
| `POST /api/predict`, `/api/generate`, `/api/converse`, `/api/score` | same bodies and results as the Python count model |
| `POST /api/2nrl`, `POST /api/feedback` | jobs with `strength` (penalties, then traversal + reward); `good_ratings` / `bad_ratings` (marks out of 10) or `good_weights` / `bad_weights` scale the reward and the penalty per text |
| `GET /api/tutor`, `POST /api/tutor/start`, `GET /api/tutor/history`, `POST /api/tutor/lesson`, `GET /api/chatgpt/models` | the English lessons, same bodies and records as the Python server: the teacher (`tutor_provider`: a local Ollama model or ChatGPT) sets and marks the exercises, the count / reward model answers them (`serve --ollama-url / --ollama-model / --chatgpt-url / --chatgpt-model` set the defaults, the key is the server's own `$OPENAI_API_KEY`) |
| `POST /api/invert`, `/api/compress`, `/api/save`, `/api/load`, `/api/reset` | as the Python server (reset / load of another kind is refused) |
| `GET /api/checkpoints`, `POST /api/checkpoints/save`, `POST /api/checkpoints/restore` | the Python `CheckpointManager` layout (`ckpt-<tag>-<step>.json.gz`, `latest.json`, `index.json`), so both servers can share a directory |
| `GET /api/uploads`, `POST /api/uploads` (JSON, multipart, raw), `POST /api/uploads/delete` | text files and ZIP archives of any size: multipart and raw bodies stream to disk, archives are inspected and read entry by entry with the same rules as the Python module |
| `GET /api/graph`, `GET /api/history` | as the Python server (edges carry `reward`, `share`, `recent_share`, `recent_count`) |
| `/api/evolve/*`, `/api/ollama/*` (corpus / review), `/api/images/*`, `/api/speech/*`, `/api/codegen/*`, `/api/schedule/preview` | 404 with a message naming the Python server |

`tests/test_go_parity.py::TestGoTutorParity` points both tutors at one fake
Ollama and asserts that they send the teacher the same prompts, get the same
marks and leave the model in the same state, so the two implementations of the
lessons cannot drift apart.

`tests/test_go_parity.py` also starts the Go server and checks its answers
against the key sets the Python API tests assert on, loads the model it saves
in Python, trains from a ZIP upload with `split: paragraphs`, and reads its
checkpoints with the Python `CheckpointManager`.

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

The negative half, and the two of them as one output path:

```python
from radixnet import NegativeFilter, NegativeNet, RadixNet, blame

negative = NegativeNet(seed=1)
negative.blame(["the the the the cat"], reason="repetition", source="review", note="it repeats the same word")
negative.correct("the cat sit on the mat", "the cat sits on the mat", reason="agreement")   # only "sit " is blamed
blame.teach_lessons(negative, tutor_round_lessons)            # or let the tutor do the blaming
blame.teach_reviews(negative, ollama_review["reviews"])

print(negative.judge("the the the the cat")["why"])
# 9 of 9 transitions are known failures (risk 1.00), mostly 'repetition', worst at 'he t'; rejected

pair = NegativeFilter(RadixNet.load("model.json.gz"), negative)
out = pair.generate(count=3, max_length=60)                   # over-sample, then veto
print(out["texts"], [(v["text"], v["why"]) for v in out["rejected"]])
```

```python
from radixnet import teach_by_speech          # one utterance -> the words and the waveform

spoken = teach_by_speech(open("clip.wav", "rb").read(), transcript="the cat sat on the mat")
print(spoken["token"], spoken["texts"][0])     # <speech:9f2a1c7d> <speech:9f2a1c7d> the cat sat on the mat
net.train(spoken["texts"], epochs=3, lr=0.5, batch_size=8)
```

## Tests

```bash
make test        # python -m unittest discover -s tests -v (includes the Go parity test when `go` is on PATH)
make go-test     # cd go && go test -race ./...
```

## Layout

```
RadixCyclicNN/
  radixnet/           activation, encoding, graph, backend(+torch), search, beam, model, countnet, negative,
                      blame, duo, diff, schedule, gan, checkpoint, bench, cli, api, llm, ollama, chatgpt,
                      tutor, codegen, vision, speech, dialogue
  tests/              unittest suite
  frontend/           Vite + React app (dist/ is prebuilt and served by the API)
  go/                 Go port of the count / reward model: radixnet/ (library), cmd/radixnet-count (CLI)
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
