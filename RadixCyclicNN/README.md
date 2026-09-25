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
| A second way through: the least punished | `--traversal least-punished` (Python, Go and Rust, `predict` / `generate` / `bench`, `traversal` in the HTTP API and a selector on the Predict and Generate tabs): the walk is ranked by the **blame** on its worst step first and by the cost only between steps nothing is held against, and at every node it may only take the children the model has the least against. A step's punishment is the penalty side of its reward plus `log(1 + incorrect)` of the judged path context - the failures counted **against nothing**, so a reward cannot buy blame off the way it nets it off the edge. On a graph where nothing was ever punished it is the ordinary search, to the bit. `SPEC-LeastPunished.md` is the specification. |
| Train and predict | `train`, `predict`, `generate`, `score` in the Python API, CLI, HTTP API and frontend. |
| Traverse by the punishments, not the rewards | `--traversal punishment` (`traversal` in the HTTP API, a selector on the Predict and Generate tabs, all three languages): the rewards leave the score altogether and the **penalties** price every step, so the cheapest path is the one that accumulated the **least punishment**. It is *what* a search looks for, as opposed to `--mode`, which is how it looks - every mode of every kind can run either traversal. See below. |
| Automated English lessons | `tutor` / the Tutor tab / `POST /api/tutor/start` (both servers): the teacher - a local Ollama model or ChatGPT - writes sentence openings that drill a point of grammar, the network completes them with the prediction search, the same teacher marks each sentence out of 10 for grammar, spelling and fluency and writes the correction; the correction is then aligned with what the network wrote and only the trigram nodes that differ move (`correct`), the failures are asked about (*why* is this wrong, and what else is wrong the same way - see below), and the round's mistakes become the next round's syllabus. |
| The report card plans the next lessons | `tutor --plan N` / the Tutor tab's **Lesson plan** / `POST /api/tutor/plan` (both servers): the report card at the end of a run goes back to the teacher, which answers with the syllabus that repairs it - one point of grammar per lesson, the mistake of the card it targets, a topic and a line on why. The marks alone already plan it (a lesson per weak point, worst first); the teacher improves on that floor and never drops a weakness from it. |
| Auto run: the lessons teach themselves | `tutor --batches N` / the Tutor tab's **Batches** / `{"batches": N}` (both servers): a batch is `rounds` rounds and the report card over them. With more than one batch the run closes its own loop - card -> plan -> the plan applied to itself (brief, level, openings, pass mark, drills) -> the next batch taught to it - and `--batches 0` keeps going until Ctrl-C or **Stop**. The Tutor tab fills the brief and the step up into the form as each batch starts, so the settings always show what is being taught. |
| ... and writes the prompt that teaches them | The plan ends in a **brief** - two or three sentences telling the next batch's exercise writer what to drill, how the difficulty steps up and what the sentences are about - and the step up itself is the marks' decision, not the LLM's: **advance** (a level up, longer openings, a higher pass mark) when 80% passed at 8/10, **stretch** (longer openings) at 50%, **hold** (nothing harder, plus correct sentences to imitate) below that. `--brief TEXT` / `{"brief": ...}` / the Tutor tab's **Teach the next batch** hands that prompt to the run that follows, where every round is written to it. |
| Rewards follow the rating | `two_nrl(good_weights=)`, `reward(weights=)` and `punish(weights=)` (every kind, Python and Go) scale every pass per text: a sentence marked 9 out of 10 is learned nine tenths as hard as a perfect one, a 0 is skipped. `/api/feedback` and `/api/2nrl` take `good_ratings` / `bad_ratings` (marks out of 10), the Ratings card a mark per rated text. |
| The model converses with itself | `converse` / the Converse tab: two voices take turns, every reply is the prediction search picking up the last words of the previous line and continuing them to the end of a text; beam speaks the most likely reply the conversation has not heard yet, sample draws walks; the second voice can be the model of the other kind. |
| Where it goes round becomes part of the graph | A third sentinel, **BACK**, beside START and END. An edge into it means *walks that get here go round*, and it is an ordinary edge - a weight, a counter, a share of the node's probability - taught by the rethinks rather than by a corpus, because no text says where a walk loops. Every hand-over teaches three edges at once: the hand-over itself, a penalty on the step it was about to loop through, a reward on the step it took instead. After a few of them BACK is the node's most likely next step, and the **search itself** stops walking through it - in `predict` and `generate` as much as in a conversation. A conversation therefore *changes the model* (`--no-learn` keeps it read-only, and the CLI's `--save` writes what it learned back). |
| Where it stops to think is part of the graph too - and so are its thoughts | A fourth sentinel, **THINK**, that faces both ways. An edge *into* it is BACK's twin - *something here made me stop and think* - taught by experience whenever an event called for a thought: a voice catching itself repeating, a question asked about a text, a thought questioning itself. The edges *out* of it are how thoughts begin: a thought is a text trained from THINK instead of START, so the model learns how its thoughts open without a word of them leaking into what it says. `think` is one thought - it teaches where it had to think, thinks (the search run from THINK), questions itself wherever its own path crosses a node it has learned to think at, and when it stops triggers the sentinel the event calls for: a conversation's thought hands over to BACK, a question returns to the thought that asked, a request ends. The thoughts come from a thinking LLM: `ollama think --train` teaches Ollama's reasoning about a topic as thoughts, and the questions it asked itself as places to stop and think. |
| It notices a repeat and explores out of it | A repeat is not just skipped: the words *before* it were said once and were the most likely thing to say, so the voice keeps exactly those, backs up to where it would have started repeating, and searches again from there - which forces the walk to leave the line at that point rather than ranking the same answers again. It works on both kinds: its own words twice in a row are cut where the walk went round, and a whole utterance the conversation has already heard is cut at its last word, the latest place a retread can still differ. Nothing new? It backs up another word and looks wider, `--explore` times over (3 by default). What it caught itself doing, what it kept, how many paths it weighed and whether it found a way on ride on the turn as its `rethink` record, and the CLI and both tabs say it in a line: *caught itself saying "ha" twice; kept "ha " and found another way on in 3 path(s)*. |
| Duplicates are avoided, and punished | A reply that was **said** before, that **adds** what an earlier reply added, that merely **echoes** a line already spoken, or that **repeats its own words** - "say morning morning", a run of up to four words twice in a row - is skipped, in the Converse tab and in the Chat tab, where both sides of the conversation count as heard. Two settings decide which of those count (`--allow-repeats` / `--allow-word-repeats`, `avoid_repeats` / `avoid_word_repeats`, a checkbox each), both on by default; English that repeats a word and means it ("where there is a will there is a way") is never touched. When every candidate is a duplicate the best one is spoken and flagged `repeat`; saying that same duplicate again would only go round in circles, so the conversation ends there. The flagged utterances come back as `repeats`: the Converse tab marks them 👎 so "Train on ratings" punishes them (the 2NRL negative phase), and the Chat loop punishes them with the failures whatever its judge made of them - the model is taught out of the duplicates it cannot avoid by itself. |
| Teach it by talking to it | `speech`, the Speech tab and `POST /api/speech/teach`: the browser records the microphone and dictates the words (Web Speech API; faster-whisper, openai-whisper or an OpenAI-compatible transcription server do it on the server side), and **one utterance becomes two texts behind the same unique token** - `<speech:9f2a1c7d> the cat sat on the mat` and `<speech:9f2a1c7d> aud:mu:8000x1:<base64>`, the waveform itself with every sample quantised to one mu-law byte. Both are trained on, so the words and the sound leave the same node of the graph; `speech decode` plays a predicted waveform back. |
| Images as text | `image encode` / the Images tab run the Stable Diffusion VAE **backwards** (image -> compressed latent, 48x fewer numbers than the pixels), quantise it to bytes, base64-encode it and feed the text to the model; `decode` runs the forward process again so a predicted text becomes an image. Needs `pillow` (+ `torch`, `diffusers` and the VAE weights for the real encoder; a thumbnail stand-in works without them). |
| External tools, browsing, exploring on its own | `agent` / `explore` and the Agent tab: the network calls tools by *writing* them (`<tool>web_fetch {"url": "..."}</tool>`) and reads the answer back as `<result>...</result>`, so a whole attempt is one training text. Ollama writes the acceptance criteria before anything is attempted, repairs the calls the network cannot write yet, judges the answer against those criteria and demonstrates with the same real tools when it failed; then 2NRL trains on the failures — the harder the worse they were — inverts, and fine-tunes on what was right. `explore` lets the network choose every task itself and follow what it finds. |
| A real browser, and MCP | `--browser` draws each page in a headless **Chrome** over the WebDriver protocol (no driver library: `chromedriver` is started and spoken to with the standard library), so a page that renders itself with JavaScript is readable. `radixnet mcp` serves the tools **and** the network — predict, generate, score, judge against the negative network, solve a task through the agent loop — over the Model Context Protocol, so any MCP client can use this instance. |
| Count / reward model | a second algorithm on the same graph, selectable at the top of the frontend (`--kind count` in the CLI, `POST /api/model/select`): every edge tracks how often training traversed it and a reward / penalty number, `weight = log(1 + traversals) + reward`, and one prediction returns the **top K and bottom K** continuations (beam search). |
| Word n-grams | the same count / reward model over an alphabet of **words** (`--encoding word:3:1`, the Words tab): the encoding's `unit` dial, not a second model, so a gram is three words and compression turns a repeated phrase into one node. There is no vocabulary to freeze or learn - a gram is text, so the alphabet is whatever the grams are made of; lengths, counts and scores are per word. |
| Resonant model | a fourth algorithm on the same graph (`--kind resonant`): a walk carries an analog **phase** advanced by every trigram (a position clock plus a hash kick), edges learn the phase at which they fire and how **coherently**, and the score adds `resonance_scale · coherence · cos(phase − mu)` to the edge's share of its node. Prediction searches `(node, chars, phase)`. A **phase-locked** cycle - back to the same node at the same phase - hands the decision to a metacognitive layer that learned from the corpus whether to ride the loop, escape it or stop. |
| Go port of the count / reward model and the negative network | `go/`: the same model in Go with one goroutine per text (lines, paragraphs or pages), counters bumped without locks (racy by default, `--exact` for atomics), parallel weight and cost recomputes, the two beams of a prediction side by side, and corpora of any size streamed through in chunks (ZIP archives entry by entry); model files are interchangeable with Python (same structure, counts, sliding window and even the Mersenne Twister state). The negative network is ported too: blame, corrections from a diff, verdicts, the filter, the `negative` command group and the `/api/negative/*` endpoints, with model files interchangeable both ways. The punishment traversal is ported as well, and the parity suite requires both sides to walk the same least-punished paths at the same costs. |
| Rust port of the whole package | `rust/`: everything the Python package does, again - every model kind (`count`, the sine-activation `radix`, the phase model `resonant`, the negative network), the encoding dial, all three traversals, **the model files** (byte for byte what Python writes, but for the `version` cache stamp), the negative network and the guard, every teaching loop (tutor, chat, critic, evolve, the recall tutor), the Ollama and ChatGPT clients, the tools, the sandbox, code generation, the agent, images and speech, MCP, the CLI and **the HTTP server the frontend runs against** - with no dependencies (HTTPS goes through the system `curl`, D-076), atomic counting and a thread pool in place of a goroutine per text. Deliberately not ported: the torch backend, the Stable Diffusion encoder and local Whisper (`rust/README.md` says why). `tests/test_rust_parity*.py` hold it to Python, one suite per area, and it is 2.2-2.8x faster than Go at counting and 3.8-6.2x at predicting on the same corpus (`bench/RESULTS.md`, `make bench-compare`, which refuses to report a timing until the two ports agree on the graph, the loss and the prediction). |
| Judgements follow the path, not the edge | An edge is right in one sentence and wrong in the next, so a verdict is not filed against the edge but against the **caller that reached it**: the key is the node *before* the edge's parent, so `the cat -> sat` and `a cat -> sat` are counted apart (`paths`, `GET /api/paths`). A correction only rewards a path when the whole answer was right - one wrong word and nothing on that walk is rewarded - and each context keeps `correct`, `incorrect` and how often it has been walked since (`seen`). The search pays for what it learns there: `path_scale · log((correct + ½) / (incorrect + ½))` joins the edge weight before the softmax, so a step that was right *from here* is cheaper here and nowhere else. |
| A node sees itself from where it stands | An edge's counters say what that step did, not what it did *here*, among the other ways out of the same node. `nodes` / `GET /api/nodes` / clicking a node in the Graph tab shares a node out both ways: a row per previous node and a row per next node, each with its share of that side's traffic, its **signed** share of that side's reward - a penalty reads as a negative share of the pressure on the node - and what the judged paths on it came to. The denominators are the side's own, not the node's visits: a node is entered without an in-edge whenever a text starts on it. |
| Learning-rate schedules | `lr` and `act_lr` as *graph functions* of the epoch (`linear(lr0, 4 * lr0)`, `lr0 * 1.25 ** i`, `warmup(...)`, `lr / 10`), previewed as a graph in the CLI (`schedule`), the API and the Train tab. |
| Constantly self-upgrading system (GAN idea) | `Evolver`: the model is the generator, a second network is the discriminator. Each generation the model samples fakes, the discriminator learns real-vs-fake with 2NRL, the worst fakes become the model's own 2NRL garbage and real corpus lines its fine-tune pass. Runs forever (`--generations 0`, or the API's evolve job) and checkpoints as it goes. |
| The negative network | `NegativeNet` (`--kind negative`, the Negative tab, and `radixnet-count negative` in Go): a copy of the network that keeps only its negative portions. Every node and edge in it exists because something went wrong there, every edge remembers the blame it collected and the tutor's reasons behind it, and `judge` walks a text through that structure to say how much of it is built out of known failure, which reasons those failures carried and which fragments carry them. It is trained on negative data alone; text the tutor *passed* only ever takes blame away (net evidence is `max(0, blame - clear)`). |
| The tutor supplies the negatives | `blame.py`: the **English tutor** names the mistake it marked a sentence down for (`agreement`, `tense`, `article`, ...), hands over its mark as the severity and its correction as the diff to blame (`tutor --blame`); the Ollama reviewer's critique becomes the reason and its rating the severity (`ollama review --blame`), the code sandbox / style checker / judge name why a program was rejected (`codegen --blame`), the **speech and image tutors** compare what the network remembers of a recording or a picture with the original (`speech tutor --blame`, `image tutor --blame`), the evolve discriminator blames every fake it scores below the real texts (`evolve --blame`), and a person can blame a text by hand. The negative network never invents a failure. |
| A tutor that needs no teacher | `recall.py`: an utterance and a picture were *encoded* into text before being trained on, so the right answer is on file and marking needs no LLM. The network is given the opening of a text it was taught - the utterance's own token, or an image header and a few characters - and asked to write the rest; what comes back is run back through the codec and compared with the original. The agreement over the payload is the mark out of 10, the single worst thing wrong with it is named (`silence`, `clipping`, `mishearing`, `blank`, `noise`, `truncated`, ...), and the original is the correction the negative network blames from. |
| An LLM on the other side of the line | `chat` (the **Chat** tab): a local **Ollama** model - or ChatGPT - holds an actual conversation with the network. It says a short line, the network replies by continuing it (the same search `converse` uses, so a reply is a real walk of the graph), they take turns, and then the LLM marks every reply out of 10 *against the line it answered* and the conversation as a whole. The failures blame the negative network, the passes clear it, and 2NRL trains the model on both - with the partner's own lines joining the positive phase, because they are what a good reply there would have looked like, and a reply the model could only repeat punished whatever the judge made of it. It is the one thing a language model is for, and the first teacher here that answers back. |
| The negative network feeds itself | `negative auto` (the Negative tab's *Automatic* card): the model writes texts of its own, a local **Ollama** model (or ChatGPT) marks each one out of 10 and says what is wrong with it, and everything below the pass mark blames the negative network - round after round, with nobody typing a failure in by hand. The positive model is only read from, so the loop can run beside whatever else is teaching it. |
| Ask *why*, and see the mistake again | A mark says *that* a sentence is wrong. When the tutor is teaching a negative network (`tutor --blame`), every failed sentence goes back to the teacher one more time: it explains **why** it is wrong - the rule that was broken and the pattern behind it - and writes `--variants` more short sentences that make the **same** mistake, each with its own correct form. Those sentences are blamed under the same reason at `--variant-weight` of its severity, so the negative network learns the *error* instead of the one sentence it appeared in - and a sentence the network never wrote is already known to be wrong. Nothing synthetic reaches the model being taught. |
| A correction blames only what changed | `NegativeNet.correct(wrong, right)`: the sentence the network wrote and the sentence the teacher wrote instead are aligned character by character (`diff.py`, the same alignment the count model's `correct` teaches from) and only the steps that wrote a character the teacher struck out are blamed - with the tutor's error type as the reason. The correction clears blame everywhere else, and a blamed transition is never compressed away, so the fragment that went wrong stays nameable. |
| The pair as a GAN at output time | `NegativeFilter` (`negative filter`, `POST /api/negative/filter`): the positive model over-samples candidates and the negative one vetoes them - by blame (`risk` over the threshold), by the likelihood ratio `log P_negative - log P_positive` per character (the discriminator logit of the two networks), or by `peak`, the blame on a single fragment, which is how one corrected word vetoes an otherwise clean sentence. What survives comes back ranked; what does not comes back with the reason, the blamed fragment and who said so. |
| Both networks on every answer | The pair is not something you have to ask for: `generate`, `predict` and `converse` run it by default, so nothing the tutor has already corrected goes out again. The model over-samples and the negative network vetoes; a vetoed continuation is dropped from `top`, a vetoed reply is left unsaid and the voice looks for another one. Every answer carries what was stopped and why, and `--no-guard` / `{"guard": false}` hands out what the positive model wrote. With no negative network, or one that has never been taught a failure, nothing is filtered and nothing is paid. |
| 2NRL | `two_nrl(bad, good)`: (1) train on bad/garbage data, (2) **invert** the network (every edge weight, and every node's activation - amplitude `a` and offset `k` together - flips sign, so what was likely becomes unlikely), (3) fine-tune on correct data with a smaller learning rate (activation parameters use a tenth of it). |
| CLI, API, React frontend | `python -m radixnet ...`, `python -m radixnet serve` (stdlib `http.server`), `frontend/` (Vite + React, prebuilt `dist` is served by the API). |
| Checkpointing, saving, loading | JSON model files (gzip with `.gz`), `CheckpointManager` with rotation, `latest` pointer, restore and resume. |
| GPU acceleration, performance | `--backend auto` uses torch on CUDA / Apple MPS when installed, else the optimised pure-Python backend (flat CSR arrays, cached costs, ~150k transitions/s on a 4-core CPU). Both backends compute identical numbers. |
| Counters that never overflow | Every growing integer (traversals, visit counts, epochs, trained characters, version stamps) is a **cyclic counter**: at `10^15` it goes back to 0 and the reset is counted, so the exact total is `resets × 10^15 + value` and nothing ever outgrows a 64-bit integer or a JSON number. The weights are computed from the exact totals, so a wrapped model behaves exactly as one that counted forever. |

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
| `make predict PREFIX="..." LENGTH=20 MODE=dijkstra TRAVERSAL=reward` | continue a prefix (`TRAVERSAL=punishment` walks the least punished way on) |
| `make generate COUNT=5 TRAVERSAL=reward` | generate texts from scratch |
| `make score TEXT="..."` | log-probability of a text |
| `make 2nrl` | 2NRL with `GARBAGE` as bad and `DATA` as good data |
| `make tutor-blame TOPIC="..." VARIANTS=3` | English lessons that also teach the negative network what the teacher marked down - and, for every failure, why it is wrong plus `VARIANTS` more sentences with the same mistake |
| `make negative-blame REASON=gibberish` / `negative-clear` | teach the negative network every line of `GARBAGE` as a failure / let `DATA` take blame back off what it shares |
| `make negative-why TEXT="..."` / `negative-filter COUNT=3` / `negative-reasons` / `negative-forget REASON=...` | explain a text, run the pair, list what the tutor blamed, drop a reason |
| `make invert` / `make compress` | invert the network / merge unary chains |
| `make evolve GENERATIONS=3` / `make evolve-forever` / `make evolve-blame` | GAN-style self-upgrade loop (`evolve-blame` also teaches the negative network) |
| `make info` / `make checkpoints` / `make restore NAME=latest` | statistics / list checkpoints / restore one into `MODEL` |
| `make bench CHARS=50000 BACKEND=python` | throughput benchmark |
| `make rust-build` / `rust-test` / `rust-parity` | build the Rust port's binaries / run its tests, clippy and the formatter check / hold it to Python, every `tests/test_rust_parity*.py` |
| `make rust-train` / `rust-predict` / `rust-serve` / ... | the Rust CLI, one target per subcommand (`make help` lists them; `RUST_KIND=count\|word`, `MODEL_RUST` is its file) |
| `make bench-compare` | the Go port and the Rust port over one corpus, checked against each other, into `bench/RESULTS.md` (`BENCH_CHARS`, `BENCH_EPOCHS`, `BENCH_REPEAT`, `PUNISH_EVERY`) |
| `make go-build` / `go-test` / `go-parity` / `go-negative` / `go-serve PORT=8001` | build the Go count / reward model CLI, run its tests, the cross-language parity tests, or serve the frontend from the Go model, blame a garbage file into the Go negative network and judge a text through it |
| `make serve PORT=8000` | API + prebuilt frontend |
| `make ollama-models` / `ollama-corpus PROMPT="..."` / `ollama-garbage` / `ollama-review` / `ollama-2nrl` | Ollama: list models, prompt -> corpus (+ train), prompt -> garbage file, adversarial review of the model's samples, review + 2NRL |
| `make tutor TOPIC="..." ROUNDS=5` / `tutor-dry` / `tutor-focus FOCUS="past tense"` / `tutor-plan PLAN=3` | automated English lessons taught by `TUTOR=ollama\|chatgpt` (`TUTOR_MODEL`, `TUTOR_URL`); `tutor-plan` ends with the next lessons planned from the report card, and the brief that teaches them (`BRIEF="..."` runs a batch to one) |
| `make codegen PROBLEMS=data/sample_problems.jsonl PHASE=both` / `codegen-teacher` / `codegen-model` | code generation with the sandbox, the Ollama judge (`CODEGEN_MODEL=gemma4`) and 2NRL rewards |
| `make chatgpt-models` / `chatgpt-ask PROMPT="..."` | ChatGPT (OpenAI): what `$OPENAI_API_KEY` may use, one question — the quickest check that ChatGPT can tutor |
| `make ollama-models` / `ollama-corpus PROMPT="..."` / `ollama-garbage` / `ollama-review` / `ollama-blame` / `ollama-2nrl` | Ollama: list models, prompt -> corpus (+ train), prompt -> garbage file, adversarial review of the model's samples, review + blame the negative network, review + 2NRL |
| `make speech-info` / `speech-teach AUDIO=clip.wav TRANSCRIPT="..."` / `speech-listen SECONDS=5` / `speech-decode TEXT="aud:…" AUDIO=out.wav` | speech: available backends, teach an audio file, record from the microphone and teach that, play a waveform text back |
| `make tools` / `agent TASKS=data/sample_tasks.txt` / `explore STEPS=10` / `explore-forever` | tool use: list the tools, solve a task list with Ollama writing the criteria, judging and teaching, or let the network choose its own tasks and browse |
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
`model.json`, gzip when the name ends with `.gz`), `--kind radix|count|word|resonant`
(the algorithm of a *new* model; a file's own kind wins; the default model file
follows the kind - `model.count.json`, `model.word.json`, `model.resonant.json`),
`--backend auto|python|torch`,
`--device cpu|cuda|mps`, `--seed N`, `--json` (one JSON document on stdout).

| Command | Main options |
|---|---|
| `train --data FILE [FILE...]` | `--whole-file`, `--epochs`, `--lr`, `--act-lr`, `--lr-schedule EXPR`, `--act-lr-schedule EXPR` (graph functions of the epoch, see below), `--reverse-schedule`, `--batch-size`, `--no-compress`, `--order corpus\|shortest-first\|longest-first\|shuffle`, `--curriculum C`, `--replay R`, `--replay-size N`, `--patience N`, `--min-delta X` (how the run walks its texts, see [Search and training methods](#search-and-training-methods)), `--reverse` (read every text backwards, so the model learns what came before - see [Training in reverse](#training-in-reverse)), `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--resume`, `--out`; a `.zip` in `--data` contributes every text file inside it |
| `schedule` | preview a learning-rate schedule: `--lr-schedule EXPR`, `--act-lr-schedule EXPR`, `--reverse-schedule`, `--epochs 10`, `--lr`, `--act-lr` print the rate of every epoch with a bar graph; without expressions the presets, variables and functions are listed |
| `predict --prefix TEXT` | `--length`, `--max-length`, `--mode dijkstra\|kbest\|beam\|sample`, `--to-end`, `--step-penalty`, `--temperature`, `--traversal reward\|punishment` with `--penalty-scale` / `--merit-scale` (what the search looks for, see below); `--top-k`, `--top-p`, `--min-p` (what a sampled step draws from) and `--diversity` (how far apart the beam's K are picked), all off by default; `--mode beam` (every kind; the count model's default): `--k 5` (top K and bottom K continuations in one search), `--beam N`; the guard flags below. `--mode kbest` is the resonant model's default: the exact K cheapest walks over `(node, chars, phase)`, metacognitive layer included; its `dijkstra` is the same search with one label per state, and so cycle-blind |
| `generate` | `--count`, `--max-length`, `--mode beam\|sample\|dijkstra\|kbest`, `--prefix TEXT`, `--temperature`, `--step-penalty`, `--beam N`, `--traversal reward\|punishment` with `--penalty-scale` / `--merit-scale`, `--top-k` / `--top-p` / `--min-p` (sample) and `--diversity` (beam); `beam` is the prediction search run to the end of a text: the `--count` most likely complete texts, most likely first; `kbest` (the resonant model's default) returns the same list *exactly* and stops as soon as it has it; the guard flags below |
| `score --text TEXT` / `--data FILE` | log-probability, per-character score, unknown transitions |
| `converse` | the model talks to itself: `--opening TEXT`, `--turns 6`, `--mode beam\|sample`, `--context 12` (characters of the previous line a reply picks up), `--max-length 60`, `--k 5`, `--beam N`, `--temperature`, `--step-penalty`, `--speakers A,B`, `--partner FILE` (a second model speaks the second voice), `--allow-repeats`, `--allow-word-repeats`, `--explore 3` (times a reply that caught itself repeating - its own words, or the conversation's - may back up and look for another way on), `--no-learn` (do not teach the graph where it goes round), `--no-think` (do not think before backing up out of a repeat), `--think-depth 2` (how deep such a thought may question itself), `--save` / `--out` (write what it learned back); prints the transcript with cost, probability and the words each reply picked up, each rethink's thought under it, then the `radixnet feedback --bad-text …` command that punishes the duplicates it could not avoid; the guard flags below |
| `think` | the model thinks: one thought from the THINK sentinel, questioning itself where it has learned to: `--about TEXT` (think at the node where that text ends, and teach the model to stop and think there), `--mode beam\|sample`, `--k 5`, `--beam N`, `--max-length 60`, `--temperature`, `--step-penalty`, `--depth 2` (how deep it may question itself), `--questions 1` (per thought), `--no-learn`, `--save` / `--out`; prints the thought and its questions, and what it triggered when it stopped |
| `chat` | an LLM converses with the model and marks every reply (a reply it could only repeat is punished whatever the judge said): `--conversations 1` (0 = until Ctrl-C), `--turns 4` (replies per conversation), `--topic TEXT`, `--opening TEXT`, `--persona TEXT`, `--context 12`, `--max-length 60`, `--mode beam\|sample`, `--k 5`, `--temperature`, `--partner-temperature`, `--threshold 6` (pass mark), `--provider ollama\|chatgpt`, `--partner-model`, `--judge-model`, `--url`, `--judge-url`, `--timeout`, `--no-guard` (do not veto a reply before it is spoken), `--no-blame`, `--no-clear`, `--no-learn` (mark it but do not train), `--no-teach-partner`, `--allow-repeats`, `--allow-word-repeats`, `--explore 3`, `--negative PATH`, `--epochs`, 2NRL options, `--out` |
| the guard (on `predict`, `generate`, `converse`) | the negative network filters what the model writes, by default: `--no-guard` (print it unfiltered), `--negative PATH` (default `model.negative.json` beside `--model`), `--threshold RISK`, `--min-coverage SHARE`, `--over-sample N`. It stands aside when there is no negative model file, or when the one there has never been taught a failure |
| `weights` | show or change the score function of the kind that has one, then recompute every weight and save. Count model: `--global-scale`, `--window-scale`, `--reward-scale`, `--count-scale`, `--path-scale` (how loudly the judged paths speak), `--window N`. Resonant model: `--buckets`, `--period`, `--kick-scale`, `--resonance-scale`, `--amp-scale`, `--reward-scale`, `--concentration`. Another kind's options are rejected by name |
| `2nrl --bad FILE --good FILE` | `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--batch-size`, `--strength` (count and resonant models), `--out` |
| `feedback` | rated texts: `--good FILE` / `--good-text TEXT` (thumbs up), `--bad FILE` / `--bad-text TEXT` (thumbs down); both -> 2NRL, thumbs up alone -> reward, thumbs down alone -> punish then invert; `--good-ratings 10,5,8` / `--bad-ratings` give a mark out of 10 per text (in the order they were collected) and every text is learned in proportion to it; `--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4`, `--out` |
| `negative <action>` | the negative network (`--negative PATH`, default `model.negative.json` beside `--model`): `blame --text/--data --reason TAG --severity N --source NAME --note TEXT` (teach it a failure), `clear --text/--data` (the tutor passed these: take blame off what they share), `why --text/--data [--threshold --min-coverage --spans]` (risk, coverage, the reasons and the blamed fragments), `filter [--count --prefix --mode --max-length --over-sample --threshold --min-coverage --ratio --no-ratio --peak --strict --learn]` or `filter --text/--data` (the pair: the positive model writes, the negative one vetoes), `reasons [--limit --log]`, `forget [--reason TAG] [--factor F]`, `auto` (teach it automatically: the model writes, an LLM reviews, the failures are blamed - `--rounds 3` (0 = until Ctrl-C), `--count 8`, `--prefix`, `--max-length 60`, `--temperature`, `--threshold 6`, `--context TEXT` (the reviewer's yardstick), `--provider ollama\|chatgpt`, `--reviewer-model`, `--url`, `--timeout`, `--epochs`, `--no-clear`, `--out`) |
| `invert` / `compress` | flip the network / merge unary chains, then save |
| `evolve --data FILE` | `--blame` / `--negative PATH` (the discriminator teaches the negative network), `--generations` (0 = forever, Ctrl-C saves), `--samples`, `--real-per-generation`, `--max-length`, `--temperature`, `--discriminator PATH`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--disc-neg-epochs`, `--disc-pos-epochs`, `--batch-size`, `--blatant-mode none\|fail_invert\|activation\|state`, `--blatant-margin`, `--blatant-boost` (failure handling, see below), `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--out` |
| `info` | statistics and the training history tail |
| `checkpoints` | `--dir`, `--restore NAME\|latest`, `--out` |
| `bench` | `--chars`, `--epochs` |
| `serve` | `--host`, `--port`, `--frontend-dir`, `--checkpoint-dir`, `--upload-dir` (training files uploaded through the API / frontend, default `uploads`), `--ollama-url`, `--ollama-model`, the tool options below |
| `ollama [--url] [--ollama-model] [--timeout] <action>` | `models`; `corpus --prompt TEXT [--lines 20] [--style good\|garbage] [--out FILE] [--train --epochs --lr --batch-size --model-out]`; `review [--count 8] [--prefix] [--max-length 60] [--text ... \| --data FILE] [--threshold 6] [--context] [--blame [--negative PATH]] [--2nrl --good FILE ...]`; `think --prompt TEXT [--lines 5] [--think true\|false\|low\|medium\|high] [--temperature 0.7] [--out FILE] [--train [--with-answers] [--no-questions] --epochs --lr --batch-size --model-out]` (a thinking model's reasoning about the prompt, taught as thoughts) |
| `speech info` / `transcribe FILE` / `teach FILE` / `listen` / `tutor FILE...` / `decode` | teaching by talking. `info`: backends, recorders, codecs. `transcribe FILE [--backend auto\|given\|faster-whisper\|whisper\|server] [--text TEXT] [--language en] [--asr-model] [--asr-url] [--out]`: the words. `teach FILE`: the transcript **and** the waveform behind one unique token - `--text` (what you said, skips the ASR), `--rate 8000`, `--codec auto\|mu\|pcm8`, `--normalise`, `--no-waveform`, `--pair` (also learn waveform → transcript), `--token` / `--shared-token`, `--out FILE`, `--train --epochs 3 --lr 0.5 --batch-size 8 --model-out`. `listen --seconds 5 [--recorder arecord\|rec\|sox\|ffmpeg] [--save clip.wav]`: record from the microphone first, then the same. `tutor FILE...`: the recall tutor - ask it to say back what it was taught and mark what comes back, `--length 400` (payload characters asked for, and what the marking compares against), `--lead`, `--attempts`, `--mode beam\|sample`, `--threshold 6`, `--listen-back` (transcribe what it said and compare the words), `--train` (teach it first), `--blame` / `--negative PATH`. `decode (--text\|--data) --out out.wav [--codec]`: an encoded or *predicted* waveform as audio |
| `tutor` | automated English lessons: `--blame` / `--negative PATH` (every failed sentence also teaches the negative network what the teacher marked it down for), `--variants 3` / `--variant-weight 0.5` (with `--blame`: the teacher explains why each failure is wrong and writes that many more sentences with the same mistake, blamed at that share of its severity), `--topic TEXT`, `--rounds 3`, `--batches 1` (auto run: batches of `--rounds` rounds, each planned from the one before; 0 = until Ctrl-C), `--exercises 5`, `--attempts 1`, `--focus TEXT` (one point of grammar), `--level`, `--words "3 to 6"`, `--brief TEXT` (what this batch is being taught to: the prompt the last report card led to), `--tutor-provider ollama\|chatgpt`, `--tutor-model`, `--grader-provider`, `--grader-model`, `--url`, `--grader-url`, `--timeout`; completion: `--mode dijkstra\|beam\|sample`, `--length 20`, `--max-length 80`, `--temperature`, `--no-to-end`, `--beam N`; marking: `--threshold 6` (pass mark), `--grammar-weight 0.6`, `--batch 10`, `--no-adapt`, `--drills N`, `--plan N` (plan the next N lessons from the report card at the end), `--no-teach-answer`, `--dry-run`; corrections: `--keep-weight 0`, `--no-diff-corrections`; 2NRL: `--twonrl-per round\|lesson`, `--min-weight 0.25`, `--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4 --strength`, `--no-replay`, `--replay-limit`, checkpoint options, `--out`, `--report FILE` |
| `correct` | teach one correction: `--wrong TEXT` (what the network wrote), `--right TEXT` (what it should say), `--blame` / `--reason TAG` / `--note TEXT` / `--negative PATH` (teach the negative network from the same diff), `--strength 1`, `--weight 1` (how bad the attempt was), `--reward 1`, `--keep 0` (what the unchanged words still earn; a whole path is only rewarded when the answer was right), `--no-count`, `--dry-run` (show the alignment only), `--out` |
| `paths` | count model: the judged paths - `--limit 20`, `--node LABEL` (only the paths leaving one node). Each line is `prev -> parent -> child`, its correct / incorrect counter, how often it has been walked since (`seen`) and what that says about the edge (`seen ratio`, `correct ratio`) |
| `words` | word model: its alphabet - `--limit 20` (0 = all). Every word it has read, with how many of the graph's three-word windows hold it; a word graph is addressed in words everywhere else too (`nodes --node "sat on the mat"`) |
| `nodes` | count model: each node against the nodes around it - `--limit 10`, `--node LABEL`. A row per previous node and a row per next node, each with its share of that side's traffic (`seen %`) and of that side's reward (`reward %`, signed), how much of the edge a judged context has been watching, and what those contexts made of it |
| `chatgpt [--url] [--chatgpt-model] [--timeout] <action>` | `models` (what the key may use); `ask --prompt TEXT [--system TEXT] [--temperature 0.7] [--json]`. Needs `$OPENAI_API_KEY` (or `$OPENAI_API_KEY_FILE`); `$OPENAI_BASE_URL` points at any OpenAI-compatible server |
| `image info` / `image encode FILE` / `image tutor FILE...` / `image decode` | encoders and their dependencies; `encode --size 128 --encoder auto\|sd\|tiny [--out TEXTFILE] [--train --epochs 3 --lr 0.5 --batch-size 8 --model-out]`; `tutor FILE...`: the recall tutor - ask it to draw back what it was shown and mark what comes back, `--size`, `--encoder`, `--lead 16` (payload characters the opening gives away, so it knows which picture), `--length`, `--attempts`, `--mode`, `--threshold 6`, `--train`, `--blame` / `--negative PATH`; `decode (--text TEXT \| --data FILE) --out image.png [--encoder]` |
| `codegen --problems FILE` | `--phase both\|teacher\|model`, `--rounds`, `--teacher-model gemma4`, `--judge-model`, `--url`, `--timeout`, `--teacher-attempts 3`, `--model-attempts 4`, `--sample-first`, `--temperature`, `--max-length 800`, `--strictness strict\|lenient`, `--no-judge`, `--no-fallback-teacher`, `--twonrl-per problem\|round`, `--no-replay`, `--teacher-prompt`, `--model-prompt`, `--sandbox-timeout 10`, `--memory-mb 256`, `--no-network-isolation`, 2NRL options (`--neg-epochs 2 --pos-epochs 3 --neg-lr 0.5 --pos-lr 0.1 --batch-size 4`), checkpoint options, `--out`, `--report FILE` |
| `tools list \| describe \| call \| browser` | the external tools the network can call: `list`, `describe --tool NAME` (with its JSON schema), `call --tool NAME --arg k=v ...` or `call --call 'web_fetch {"url": "..."}'`, `browser` (what `--browser` would drive); tool options below. The Go CLI has the same three actions and the same flags |
| `agent --tasks FILE` | `--phase model\|teacher\|both`, `--rounds`, `--twonrl-per task\|round`, `--agent-model`, `--judge-model`, `--url`, `--timeout`, `--criteria 4`, `--lenient`, `--no-judge`, `--mediation repair\|always\|never`, `--no-teach`, `--max-steps 6`, `--model-attempts 2`, `--teacher-attempts 1`, `--sample-first`, `--temperature`, `--max-length 200`, `--observation-chars 600`, `--read-reward`, `--no-replay`, `--blatant-mode fail_invert\|activation\|state\|none`, `--blatant-margin 0.5`, `--blatant-boost 4`, `--blame` (teach the negative network from every failure), `--no-avoid`, `--negative PATH`, 2NRL options, checkpoint options, tool options, `--out`, `--report FILE` |
| `explore` | the network picks its own tasks: `--steps 10` (0 = until Ctrl-C), `--seed-url URL` (repeatable), and every `agent` option. `agent` and `explore` are in the Go CLI too (`--strength` in place of the learning rates) |
| `mcp` | serve the tools and the network over the Model Context Protocol (stdio): `--no-model`, `--no-solve`, `--blame`, `--negative PATH`, `--agent-model`, `--url`, tool options |
| tool options (`tools`, `agent`, `explore`, `mcp`, `serve`) | `--offline` (no browsing), `--allow-private` (allow loopback / private addresses), `--search-url URL` (`{query}` is substituted), `--web-timeout 20`, `--max-bytes 2000000`, `--browser` (draw pages in a real headless Chrome), `--no-headless`, `--page-timeout 30`, `--python-tool` (offer the sandboxed `python` tool), `--sandbox-timeout`, `--no-network-isolation`, `--upload-dir DIR` (offer `read_file` over it) |

Every command has `--help`. Exit code 1 with a message on stderr on errors.

## HTTP API

`python -m radixnet serve --host 127.0.0.1 --port 8000`. All `/api/*` responses
are JSON with CORS headers; errors are `{"error": "..."}` with 400/404/409/500.
Long operations (train, 2NRL, evolve) run as a background **job**; only one job
at a time, and mutating requests answer 409 while it runs.

| Method and path | Body / result |
|---|---|
| `GET /api/health` | `{"ok": true, "version"}` |
| `GET /api/status` | model statistics (with the active `kind` and its `encoding`), current job, available backends, model path, `replay` (the replay buffer the model keeps: `{"size","texts","seen"}`, or null); the resonant model adds `buckets`, `coherence_mean` / `coherence_max`, `cycles_seen` and `meta` (the metacognitive layer) |
| `GET /api/model` | `{"kind", "label", "units", "kinds": [{"kind","label","description","units"}], "model_path", "paths", "in_memory"}` |
| `GET /api/words` | word model: `?limit=50` -> `{"vocabulary", "units", "words": [{"word","id","trigrams"}]}`, most read first (400 on a model that counts in characters) |
| `POST /api/model/select` | `{"kind": "radix"\|"count"\|"word"\|"resonant"}` -> the same document plus `origin` (`memory`, `file`, `new`, `active`) and `stats`; the previous model stays in memory |
| `GET /api/encoding` | how the active model reads text: `{"encoding": "char:3:1", "unit", "ngram", "stride", "overlap", "start_label", "end_label", "back_label", "configurable": true, "note"}` (`window` is `ngram` under its old name). The encoding is chosen when a model is made - `POST /api/reset` with an `encoding` - and fixed for its life |
| `POST /api/encoding/preview` | `{"text"}` -> the same document plus `{"chars", "windows", "count", "decoded", "round_trip", "unknown_windows", "kind", "path": {"known", "reason", "labels", "node_ids", "decoded", "nodes", "compressed"}}`: one text through the encoder, back through the decoder, and through the graph's own (possibly merged) node labels. `path.known` is false with the reason - a window never seen, or a text that cannot be walked from START to END as it stands |
| `POST /api/train` | `{"texts": [...]}` or `{"text": "one per line"}` and/or `{"files": ["upload names"], "whole_file": false}` + `epochs`, `lr`, `act_lr`, `lr_schedule`, `act_lr_schedule` (expressions of the epoch), `reverse_schedule`, `batch_size`, `auto_compress`, and how the run walks its texts: `order`, `curriculum`, `replay`, `replay_size`, `patience`, `min_delta` (each off when left out; a value out of range is a 400), and `reverse` (read every text backwards, in the model's units - [Training in reverse](#training-in-reverse)) -> `{"job": {...}}`; every epoch record carries the `lr` / `act_lr` used, and the one that stopped the run early `"early_stop": true` |
| `GET /api/schedule` | what a schedule expression may use: `{"variables", "constants", "functions", "helpers", "presets": [{"name","lr","act_lr","description"}]}` |
| `POST /api/schedule/preview` | `{"lr_schedule", "act_lr_schedule", "epochs": 5, "lr": 0.05, "act_lr": 0.005, "reverse_schedule": false}` -> `{"points": [{"epoch","lr","act_lr"}], ...}` (400 with the reason for a bad expression) |
| `GET /api/uploads` | uploaded training files: `{"uploads": [{"name","bytes","chars","lines","modified"} (+ `archive`, `files`, `skipped` for a ZIP)], "upload_dir"}` |
| `POST /api/uploads` | upload text files or ZIP archives: JSON `{"name","content"}` / `{"name","content_base64"}` or `{"files": [...]}`, `multipart/form-data` (`curl -F file=@corpus.zip`), or a raw body with `?name=corpus.zip` -> `{"uploads": [...], "archives": [{"name","entries","extracted","skipped": [{"path","reason"}]}]}` (201). A ZIP stays one upload (its record carries `archive: true`, `files`, `skipped` and the summed `lines`); whenever it is selected the server unpacks its text entries in memory. Directories, `__MACOSX` / system files, nested archives, encrypted, binary and empty entries are ignored; an archive with no text entry (or a corrupt one) is refused. There is no size limit: the upload body limit does not apply to `/api/uploads`, and an archive may hold any number of entries (`extract_texts(max_entries=, max_bytes=)` exists for callers who want a cap) |
| `POST /api/uploads/delete` | `{"name"}` |
| `GET /api/ollama/models?url=` | always 200: `{"available", "url", "model", "models": [{"name","size","modified_at","details"}], "error"}` |
| `POST /api/ollama/corpus` | `{"prompt", "lines": 20, "style": "good"\|"garbage", "model", "url", "save_as": upload name, "train": false, "epochs", "lr", "batch_size"}` -> `{"texts", "upload", "job", ...}` (202 with a train job; 502 when Ollama fails) |
| `POST /api/ollama/review` | `{"count": 8, "prefix", "max_length": 60, "temperature", "texts": [...] (review these instead of sampling), "threshold": 6, "context", "apply": "none"\|"2nrl", "blame" (teach the negative network), "good", "good_files", 2NRL settings}` -> `{"reviews": [{"index","text","rating","verdict","critique"}], "mean_rating", "pass_rate", "good", "bad", "job", ...}` |
| `POST /api/ollama/think` | `{"prompt", "lines": 5 (questions to think about), "think": true\|false\|"low"\|"medium"\|"high", "temperature": 0.7, "model", "url", "save_as": upload name, "train": false (teach the thinking as thoughts that begin at the THINK sentinel, and the questions it asked itself as places to stop and think), "with_answers": false, "questions": true, "epochs", "lr", "batch_size"}` -> `{"prompt", "model", "url", "think", "count", "thinking", "thoughts": [{"question","thinking","answer"}], "upload", "job"}` (202 with a train job; 502 when Ollama fails or a model that does not think is asked to train) |
| `GET /api/chatgpt/models?url=` | always 200: `{"available", "configured" (the server has a key), "url", "model", "models": [{"name","owned_by","created"}], "error"}`. The key is never a request field: it is the server's own `$OPENAI_API_KEY` |
| `GET /api/images` | `{"pillow","torch","diffusers","sd_model","sd_loaded","sd_error","encoders","default_size","auto","text_format"}` |
| `POST /api/images/encode` | an image as multipart (`curl -F file=@photo.png`), a raw body, or JSON `{"name","content_base64"}` + `?size=128&encoder=auto\|sd\|tiny&train=true&save_as=photo.txt` (train settings `epochs`, `lr`, `batch_size`) -> `{"text","encoder","width","height","latent_shape","bytes","chars","source_size","name","upload","job"}` (202 with a train job) |
| `POST /api/images/tutor` | the recall tutor: the same image forms, or `{"texts": ["img:…"]}` for pictures already encoded, + `size`, `encoder`, `lead`, `length`, `attempts`, `mode`, `temperature`, `threshold`, `blame` -> `{"modality","lessons","report","negative"}`; every lesson carries the mark out of 10, the agreement, the reason it failed and the facts behind it |
| `POST /api/images/decode` | `{"text", "encoder"}` -> `{"png_base64","encoder","width","height","bytes","repaired"}` (a cut-off or rambling prediction is padded / truncated) |
| `GET /api/speech` | `{"backends", "faster_whisper", "whisper", "whisper_model", "server_url", "server_model", "auto", "ffmpeg", "recorders", "codecs", "default_rate", "token", "token_example", "text_format", "formats"}` |
| `POST /api/speech/transcribe` | audio as multipart (`curl -F file=@clip.wav`), a raw body, or JSON `{name, content_base64}`; options from the query string or the body (`backend`, `language`, `asr_model`, `asr_url`, `transcript`) -> `{"transcript", "backend", "model", "language", "seconds"}` |
| `POST /api/speech/teach` | the same audio forms + `transcript` (what the browser dictated), `rate`, `codec`, `normalise`, `waveform`, `pair`, `token`, `unique`, `train`, `epochs`, `lr`, `batch_size`, `save_as` -> `{"token", "transcript", "asr", "audio", "texts", "chars", "pair", "upload", "job"}` (202 with a train job on the texts) |
| `POST /api/speech/tutor` | the recall tutor: the same audio forms, or `{"texts": ["<speech:…> aud:…"]}` for utterances already encoded, + `transcript`, `rate`, `codec`, `normalise`, `token`, `unique`, `lead`, `length`, `attempts`, `mode`, `temperature`, `threshold`, `listen_back`, `blame` -> `{"modality","lessons","report","negative"}` |
| `POST /api/speech/decode` | `{"text", "codec"}` -> `{"wav_base64", "codec", "rate", "samples", "seconds", "repaired"}` - an encoded or predicted waveform as playable audio |
| `POST /api/codegen/start` | `{"problems": [str or {"id","prompt","tests","expected_output"}], "problems_text", "problem_files", "phases": "both"\|"teacher"\|"model", "rounds", "teacher_provider": "ollama"\|"chatgpt", "teacher_model", "judge_provider", "judge_model", "url", "judge_url", "teacher_attempts", "model_attempts", "strictness", "judge", "fallback_teacher", "twonrl_per", "replay", "sandbox_timeout", "memory_mb", "blame" (the sandbox and the judge also teach the negative network), 2NRL settings, ...}` -> job whose records are `{"kind": "attempt"\|"problem"\|"round", ...}`; an attempt's `source` and a verdict's `judged_by` name the provider (400 when `teacher_provider` is `chatgpt` and the server has no key) |
| `GET /api/codegen/history` | `{"history": [records of all codegen runs]}` |
| `POST /api/codegen/solve` | `{"problem", "source": "model"\|"teacher", "attempts", "judge", "teacher_provider", ...}` -> `{"attempts": [{"code","run","style","verdict","correct"}], "correct"}` (no training) |
| `POST /api/codegen/run` | `{"code", "tests", "expected_output", "sandbox_timeout", "memory_mb"}` -> `{"run", "style", "verdict"}` |
| `GET /api/tutor` | the English tutor: `{"url", "model", "env_model", "providers": {"ollama": {...}, "chatgpt": {"url","model","configured"}}, "error_types", "modes", "twonrl_per", "levels", "words_ladder", "upgrade_steps", "plan_lessons", "defaults": {every setting}}` |
| `POST /api/tutor/start` | `{"blame" (teach the negative network why each sentence failed), "variants": 3, "variant_weight": 0.5 (with blame: the teacher explains why each failure is wrong and writes that many more sentences with the same mistake, blamed at that share of its severity; a lesson record carries them as `why` and `variants`, a round record as `explained` and `similar`), "topic", "rounds": 3, "exercises": 5, "attempts", "focus", "level", "words", "brief" (the last plan's prompt: handed to the teacher with every set of exercises), "tutor_provider": "ollama"\|"chatgpt", "tutor_model", "grader_provider", "grader_model", "url", "grader_url", "timeout", "mode", "length", "max_length", "temperature", "to_end", "threshold": 6, "grammar_weight": 0.6, "batch", "adapt", "drills", "teach_answer", "learn", "twonrl_per": "round"\|"lesson", "diff_corrections", "keep_weight", "min_weight", 2NRL settings, "plan" (lessons to plan from the final report card, 0 = none), "batches": 1 (auto run: each batch planned from the one before, 0 = until stopped), "checkpoint_every"}` -> a job whose records are `{"kind": "lesson"\|"round"\|"report"\|"plan"\|"batch"\|"note", ...}`, each carrying the `batch` it belongs to; a lesson carries `score`, `grammar`, `spelling`, `fluency`, `passed`, `error`, `sentence`, `correction`, `changes` (what the teacher changed, span by span), `comment`, a round the report card and what it taught (`corrections`, `edits`, `penalised`, `rewarded`), a `plan` record the lesson plan (see `POST /api/tutor/plan`) and a `batch` record what the next batch of an auto run is being taught to (`step`, `brief`, `topic`, `level`, `words`, `threshold`, `drills`, `note`) |
| `GET /api/tutor/history` | `{"history": [lesson / round / report records of all tutor runs]}` |
| `POST /api/tutor/lesson` | one round without training: the same settings plus `{"prefixes": [...]}` (skip the exercise writer and complete these) -> `{"source": "ollama"\|"chatgpt"\|"given", "exercises", "lessons": [{"exercise","continuation","sentence","grade"}], "report": report card}` (400 when `tutor_provider` is `chatgpt` and the server has no key, 502 when the teacher fails) |
| `POST /api/tutor/plan` | the lessons a report card calls for: `{"report": {report card}` (default: the card at the end of the last run), `"count": 3, "topic", "level", "words", "threshold", "exercises", "drills", "tutor_provider", "tutor_model", "url"}` -> `{"plan": {"summary", "prompt"` (the brief: start the next run with it as `"brief"`)`, "upgrade": {"step": "hold"\|"stretch"\|"advance", "level", "words", "threshold", "drills", "note"}, "level", "topic", "source": "ollama"\|"chatgpt"\|"report card", "weak": [{"error","count","share","focus"}], "targets", "lessons": [{"focus","targets","topic","why","exercises","drills","prefixes"}]}, "source", "provider", "model", "report"}` (400 without a card, 502 when the teacher fails) |
| `GET /api/job` / `POST /api/job/stop` | job status `{"id","type","state","progress","history","error",...}` / request a stop |
| `POST /api/predict` | `{"prefix","length","mode","to_end","step_penalty","temperature","traversal": "reward"\|"punishment","penalty_scale","merit_scale","top_k","top_p","min_p","diversity","guard": true}` -> `{"kind","continuation","full_text","cost","probability","step_costs","path","node_ids","expanded","reached_end","guard"}` (plus `traversal` on the kinds that answer with `top` / `bottom`); `mode: "beam"` (both models), `k`, `beam` -> plus `top` / `bottom` (K entries each with `continuation`, `full_text`, `cost`, `probability`, `path`, `reached_end`). The guard keeps the survivors in `top` and the best of them is the continuation; when it vetoes every one of them the continuation is empty and `full_text` is the prefix |
| `POST /api/generate` | `{"count","max_length","mode": "beam"\|"sample"\|"dijkstra","prefix","temperature","step_penalty","beam","seed","traversal","penalty_scale","merit_scale","top_k","top_p","min_p","diversity","guard": true}` -> `{"samples": [{"text","full_text","cost","probability","path","node_ids","step_costs","reached_end"}],"guard"}`; `beam` returns the `count` most likely complete texts (the prediction search run to END), every `text` is the whole text, prefix included. With the guard on, the model is asked for `count * over_sample` and the survivors come back (fewer than `count` when it vetoed too much) |
| `POST /api/converse` | `{"opening","turns": 6,"mode": "beam"\|"sample","context": 12,"max_length": 60,"k": 5,"beam","temperature","step_penalty","seed","speakers": ["A","B"],"history": [utterances so far],"partner": kind in memory,"avoid_repeats": true,"avoid_word_repeats": true,"explore": 3,"learn": true,"think": true,"think_depth": 2,"guard": true}` -> `{"kind","partner","speakers","count","guard","turns": [{"index","speaker","text","context","reply","cost","probability","reached_end","fresh","given","repeat","stutter","rethink" (what it caught itself saying twice, what it kept, paths explored, whether it found a way on, and "thought": what it thought before backing up - the record `POST /api/think` returns),"candidates","skipped","vetoed","labels","node_ids","step_costs"}],"repeats": [the duplicates spoken anyway, to punish]}`; `history` continues a conversation (only the new turns come back) |
| `POST /api/think` | `{"about","mode": "beam"\|"sample","k": 5,"beam","max_length": 60,"temperature","step_penalty","seed","depth": 2,"questions": 1,"learn": true}` -> `{"kind","trigger","at","about","text","depth","stopped": "end"\|"length"\|"nothing","then": "end"\|"back"\|"think","taught","handed_over","cost","probability","expanded","questioned","questions": [the same records],"labels","node_ids","step_costs"}` - one thought from the THINK sentinel, in the language of the thoughts it was taught; `about` thinks at the node where that text ends and teaches the model to stop and think there (`learn`) |
| the `guard` of those three | `null` when nothing filtered the answer (no negative network, or one that has never been taught a failure), else `{"on": true,"vetoed","rejected": [verdicts],"verdicts","negative" (its stats),"config"}` plus `candidates` / `kept` / `asked` / `rate` / `refusals`. Send `"guard": false` to get what the positive model wrote |
| `POST /api/score` | `{"text"}` -> `{"log_prob","per_char","chars","transitions","unknown_transitions"}` |
| `POST /api/2nrl` | `{"bad": [...], "good": [...], "neg_epochs","pos_epochs","neg_lr","pos_lr", "bad_weights" \| "bad_ratings", "good_weights" \| "good_ratings"}` (or `bad_files` / `good_files` upload names) -> job; the weights (0..1 shares) or ratings (marks out of 10) scale each phase per text |
| `POST /api/feedback` | rated texts: `{"good": [thumbs up], "bad": [thumbs down], "good_ratings": [10, 5], "bad_ratings": [...] (or "good_weights" / "bad_weights" as 0..1 shares), "neg_epochs": 2, "pos_epochs": 3, "neg_lr": 0.5, "pos_lr": 0.1}` (also `*_text`, `*_files`) -> `{"job", "action": "2nrl"\|"reward"\|"punish", "good", "bad", "good_weights", "bad_weights"}`: 2NRL when both kinds are given, reward-only on thumbs up alone, punish (negative phase, then invert) on thumbs down alone. A rating is more than a like: each text is learned in proportion to its mark (10 = the full rate, 0 skips it) |
| `GET /api/negative` | the negative network: `{"path","active","stats","reasons": [{"reason","blame","fails","edges","share"}],"journal": [{"at","text","reason","severity","source","note"}],"weights","settings"}` |
| `POST /api/negative/blame` | teach it a failure: `{"texts"\|"text","reason","severity": 1,"source","note","epochs": 1}` -> `{"records","reasons","stats", ...}`; the only call that adds structure to the negative network |
| `POST /api/negative/clear` | the tutor passed these: `{"texts"\|"text","weight": 1,"epochs": 1}` -> `{"matched","unmatched","records","stats"}`; nothing is created |
| `POST /api/negative/judge` | `{"texts"\|"text","threshold","min_coverage","spans": 5}` -> `{"verdicts": [{"verdict": "reject"\|"suspect"\|"pass","risk","coverage","blame","reasons","spans": [{"start","end","fragment","blame","fails","reason"}],"why"}]}` |
| `POST /api/negative/filter` | the pair: `{"count": 3,"prefix","mode","max_length","temperature","over_sample": 3,"threshold","min_coverage","ratio": 0,"no_ratio","peak","strict","learn"}` (or `{"texts"}` to judge given texts) -> `{"texts" (the cleanest survivors),"kept","rejected": [verdicts],"verdicts","candidates","asked","rate","pair"}` |
| `POST /api/negative/forget` / `POST /api/negative/settings` / `POST /api/negative/reset` / `POST /api/negative/save` | drop or fade a reason `{"reason","factor"}` / `{"threshold","min_coverage","share_scale","blame_scale","clear_scale"}` / a fresh negative network `{"seed"}` / write it `{"path"}` |
| `POST /api/chat/start` | start a chat job - an LLM converses with the model and marks every reply: `{"conversations": 1 (0 = until stopped),"turns": 4,"topic","opening","persona","context": 12,"max_length": 60,"mode": "beam"\|"sample","k": 5,"temperature","partner_temperature","threshold": 6,"provider": "ollama"\|"chatgpt","partner_model","judge_model","url","judge_url","timeout","guard": true,"blame": true,"clear_passes": true,"learn": true,"teach_partner": true,"avoid_repeats": true,"neg_epochs","pos_epochs","neg_lr","pos_lr","batch_size","strength","epochs","seed"}` -> 202 `{"job","config","url","partner","judge","speakers"}` |
| `GET /api/chat/history` | the `exchange` records as they are spoken, then one `conversation` record each (its transcript, every review with the line it answered, the marks, what was blamed and what was learned) and a `report` at the end of a run |
| `POST /api/negative/auto` / `GET /api/negative/auto/history` | the Negative tab, automatic: start a job that has the model write texts, an LLM reviewer mark them and every failure blame the negative network - `{rounds (0 = until stopped), count, prefix, max_length, temperature, threshold, context, provider: ollama\|chatgpt, reviewer_model, url, timeout, clear_passes, epochs, seed}` -> 202 `{"job","config","url","reviewer"}`; the history is its round / report records. The positive model is only read from |
| `POST /api/invert` / `POST /api/compress` | statistics / `{"merges", ...}` |
| `POST /api/evolve/start` / `POST /api/evolve/stop` / `GET /api/evolve/history` | `{"corpus": [...]` or `"corpus_text"` or `"corpus_files"`, `"generations"` (null = forever), `samples`, `max_length`, `temperature`, `checkpoint_every`, `blatant_mode`, `blatant_margin`, `blatant_boost`, `blame`, ...}` -> job; generation records carry `failures`, `blatant`, `boost_mean`, `boost_max`, `flipped`, `twonrl`, `mode` (and `negative_blamed` / `negative_reasons` with `blame`) |
| `POST /api/save` / `POST /api/load` / `POST /api/reset` | `{"path"}` (default: the active kind's file; the negative network is written beside it when it holds failures) / `{"path"}` (any kind; switches to it) / `{"seed", "kind", "encoding"}` (the encoding as `unit:n:stride`, or `unit` + `ngram` + `stride`; + the kind's score-function settings for a fresh count or resonant model) |
| `POST /api/model/weights` | the active model's score function - count: `{"count_scale", "global_scale", "window_scale", "reward_scale", "path_scale", "window"}`; resonant: `{"buckets", "period", "kick_scale", "resonance_scale", "amp_scale", "reward_scale", "concentration"}` -> `{"weights", "stats"}`; every edge weight is recomputed (400 for RadixNet, and for an option of another kind) |
| `GET /api/checkpoints` / `POST /api/checkpoints/save` / `POST /api/checkpoints/restore` | list / `{"tag"}` / `{"name"}` |
| `GET /api/tools` | the external tools the network can call: names, arguments, JSON schemas, the call format |
| `POST /api/tools/call` | `{"tool", "arguments"}` or `{"call": "web_fetch {\"url\": \"...\"}"}` (+ `offline`, `allow_private`, `search_url`, `web_timeout`, `max_bytes`, `python_tool`, `browser`) -> the tool result; a failing tool is 200 with `ok: false` |
| `POST /api/agent/start` | `{"tasks" \| "tasks_text" \| "task_files", "phase", "rounds", "max_steps", "mediation", "criteria", "judge", "teach", "blatant_mode", "blatant_margin", "blatant_boost", "blame" (teach the negative network from the failures), 2NRL options}` -> job |
| `POST /api/agent/explore` | the network chooses every task: `{"steps"` (0 = until stopped)`, "seed_urls", ...}` -> job |
| `GET /api/agent/history` | criteria / step / attempt / task records of all agent and explore runs |
| `POST /api/agent/criteria` | `{"tasks"}` -> the acceptance criteria the LLM writes, nothing attempted |
| `POST /api/agent/solve` | `{"task", "source": "model"\|"teacher"}` -> the transcript, the verdict and how badly it failed; no training |
| `GET /api/graph?limit=150` | top nodes by visit count with their activation parameters, and the edges between them with weight, count, probability, cost (count model: also `reward`, `share`, `recent_share`, `recent_count`, plus `total_traversals`, `window_traversals`, `window`). Every count comes with its `count_resets` / `total_traversals_resets`: counters are cyclic, so the exact number of events is `resets × 10^15 + count` |
| `GET /api/history` | training history |
| `GET /api/paths?limit=50` | count model: the judged paths, most walked first - `prev` / `parent` / `child`, `correct`, `incorrect`, `seen`, `seen_ratio`, `correct_ratio`, and the totals over the whole graph |
| `GET /api/nodes?limit=20` | count model: each node against the nodes around it (`?node=LABEL` for one) - `{nodes: [{node, label, visits, from: [...], to: [...], in_totals, out_totals}]}`, a row per previous and per next node with `seen_ratio`, `reward_ratio`, `path_ratio` and `correct_ratio` |
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
it. Two tabs hold the settings. **Settings** keeps the ones of this browser -
what every search and every run starts from, whichever model is loaded: the
**traversal** (follow the rewards, or avoid the punishments), the **sampling
filters** (top-K, top-p, min-p) and the beam's **diversity**, and **how a run
walks its texts** (the order, the curriculum, the replay and the early stop).
The Predict, Generate and Train tabs show the same controls, and each is one
setting: changing it on any tab changes it on all of them. **Model settings**
holds what belongs to the model and is saved with it: this model's kind,
encoding, size and replay buffer; a form that makes a **new model** in any kind
and encoding (`POST /api/reset`); the **score function** of whichever kind is
active (the count model's dual frequency scales and sliding window, the
resonant model's phase and resonance settings; the sine model has none and says
why, and the negative network's blame function stays on the Negative tab); and
the **encoder / decoder** - the unit, the n-gram and the stride, with a live
preview that encodes a text, decodes it back and walks it through the graph's
own node labels so the radix compression is visible.

Panels: status bar (live statistics and job progress), Train (texts and/or
uploaded files), Predict (path with per-step costs and a Like button that
rewards the shown text - a thumbs-up feedback job), Generate (whole texts from
the prediction search - beam: the K most likely complete texts, optionally
continuing a prefix; sample; dijkstra - with thumbs up / thumbs down ratings:
"Train on ratings" runs 2NRL on them, thumbs down as the negative phase, thumbs
up as the positive phase), Converse (the model talks to itself in a chat
view that reads newest first - a new turn is appended to the top and pushes the
older ones down, so nothing has to be scrolled to: an opening line, turns,
context, beam / sample, the two voices' names, the other kind in memory as the
second voice; Continue extends the conversation, and turns are rated like
samples; every rating carries a mark out of 10 - "how good" / "how bad" - and
the network learns each text in proportion to it; "Avoid repeated words" keeps
a reply from saying the same words twice in a row, "Explore" is how many times
a voice that catches itself repeating - its own words, or the conversation's -
may back up and look for another way on -
each turn saying what it noticed and found - "Learn where it goes round" teaches
the graph what each rethink found out, "Think before backing up" has a voice
think first and shows the thought under its turn (💭), and "Punish duplicates" marks
the repeats the model could not avoid 👎 for the 2NRL negative phase), Think (one
thought from the THINK sentinel per press, about a text or nothing in particular,
with the questions it asked itself nested under it, what it triggered when it
stopped and what it taught), Chat (an
LLM converses with the model and marks every reply; its transcript reads newest
first too, it has the same "Avoid repeated words" and "Explore" settings, and
the replies the model could only repeat are punished with the failures), Score,
2NRL, Negative (the
failure network: **Automatic** - press Start and an Ollama reviewer marks the
model's own output round after round, blaming what fails, while the tables
below fill in by themselves - plus run the pair and see what was vetoed and
why, judge a text with its blamed fragments marked, blame or clear texts by
hand, and the table of everything the tutor has blamed with the journal of what
it said),
Evolve (live chart of the discriminator gap), Ollama (corpus from a prompt,
adversarial review, and a thinking model's thinking about a prompt - the
questions it asked itself marked - taught to the network as thoughts), Tutor (automated English lessons: the settings, a dry run
that marks without training, a chart of the marks per round, the report card
with the mistakes, every lesson with what the network wrote, the correction
and the teacher's line, and the lesson plan the teacher writes from the report
card - the brief for the next batch and each lesson loadable into the settings
with one click, or **Batches** above 1 to let the run plan and teach itself,
filling the brief and the step up into the form as it goes), Code (code
generation with the sandbox and the judge),
Speech (record the microphone, the browser writes down what it hears, teach
the words and the waveform),
Checkpoints (save / restore / load / reset), Model settings and Settings (see
above) and a Graph view of the most visited nodes.  The Images and Speech tabs each end with a **What does it
remember?** card - the recall tutor: ask the network for the picture or the
utterance back, see the mark out of 10, the agreement and the reason each
failure failed, and (with "blame it" ticked) hand those failures to the
negative network. The **Agent** tab is tool use: solve a list of tasks or let
the network explore on its own, with a live log of the acceptance criteria, the
tasks it chose, every tool call tagged by who wrote it, and the verdicts.

**Settings are remembered in the browser.** Every settings field of every
panel - epochs, learning rates, modes, thresholds, prefixes, prompts, the topic
of a tutor run, the text in the boxes - is written to `localStorage` under the
`radixnet.v1.` prefix as you change it, so a reload (or coming back tomorrow)
finds the forms as you left them. Nothing is stored until you actually change a
field, so untouched defaults leave no trace, and results, transcripts, ratings,
job state and uploaded-file selections are not settings and are never stored. A
value too large to keep (a pasted corpus over 256 KB) is skipped rather than
filling the quota, and where the browser has no usable storage (a private
window, blocked site data) the panels simply work as before. The footer's
*settings saved in this browser* button forgets all of it - click it twice -
and reloads with the defaults; nothing is ever sent to the server.

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
make frontend-test                             # unit tests of the settings store and the site-wide settings (node --test, no install needed)
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
python -m radixnet ollama think --prompt "the sea" --lines 5 --train      # a thinking model's reasoning, taught as thoughts
python -m radixnet think --about "the sea"                                # the network thinks, in the words it was taught
```

`--url` / `--ollama-model` (or `OLLAMA_HOST` / `RADIXNET_OLLAMA_MODEL` in the
environment) select the server (default `http://127.0.0.1:11434`) and model
(default `llama3.2`). `make ollama-models`, `ollama-corpus`, `ollama-garbage`,
`ollama-review` and `ollama-2nrl` wrap the same commands (`PROMPT`, `LINES`,
`STYLE`, `COUNT`, `THRESHOLD`, `OLLAMA_URL`, `OLLAMA_MODEL`).

The API exposes the same through `GET /api/ollama/models`, `POST /api/ollama/corpus`,
`POST /api/ollama/review` and `POST /api/ollama/think` (see the table above), and the frontend's Ollama
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

   **A whole path is rewarded only when the output is correct.** A sentence
   that passed is rewarded end to end and every step of it counted as a
   correct path; a sentence that had to be corrected earns its fix, not its
   sentence - `--keep-weight` is 0 by default, and 1 rewards the whole
   corrected sentence as an earlier version did. The correction is still
   traversed - it is correct English whatever the mistake was - but traversal
   is counting, not reward.

   A failure the teacher left
   uncorrected is still 2NRL garbage weighted by how bad the mark was
   (`--min-weight` for a near miss, 1 for a hopeless answer); the model
   answers and the sentences that passed are still the fine-tune pass,
   weighted by how good the mark was - a sentence marked 9 gets nine tenths of
   the learning rate, the teacher's own English the full rate. Nothing is
   punished when the network wrote nothing: the prefix itself is correct
   English. `--no-diff-corrections` goes back to the whole-sentence way.

5. **Why, and the same mistake again** (`--blame`, `--variants 3`). A grade
   says *that* a sentence is wrong and which rule it broke; the negative
   network wants to know *why* and to have seen the mistake more than once. So
   each failed sentence goes back to the teacher one final time, in one call
   for the whole round: it explains **why** the sentence is wrong - the rule,
   and what the student is doing instead - and writes `--variants` more short
   sentences that make the **same** mistake, each with the correct version
   beside it. Every pair is blamed exactly like the student's own mistake,
   diffed against its correction under the same reason, at `--variant-weight`
   (0.5) of its severity, because the student never actually wrote it. The
   negative network then knows the shape of the error rather than one sentence,
   and text it has never seen is already suspect:

   ```
   the network wrote:  the cat sat on the sun          (agreement, 2/10)
   why:                A plural subject takes a plural verb; the student drops the -s.
   same mistake:       the dogs sits on the mat   ->  the dogs sit on the mat
                       she walk to the shop       ->  she walks to the shop
   ```

   ```bash
   python -m radixnet negative why --text "the dogs sits on the mat"
   # verdict suspect, mostly 'agreement', worst at "sits"
   ```

   It is the only LLM call the tutor makes that nothing else uses: without
   `--blame` there is no negative network to teach, so the question is never
   asked, and `--variants 0` switches it off. The Go tutor asks it the same way
   (`radixnet-count tutor -blame -variants 2`), and a parity test runs both
   against one teacher to check they get the same answer and end with the same
   negative network.

   The same alignment is a command of its own, for a correction typed by hand:

   ```bash
   python -m radixnet correct --wrong "the cat sit on the mat" --right "the cat sits on the mat"
   python -m radixnet correct --wrong "he go to school" --right "he goes to school" --dry-run
   go/bin/radixnet-count correct --wrong "a apple a day" --right "an apple a day" --keep 0
   ```

### Counting paths, not edges

An edge is the right move in one sentence and the wrong one in another, so a
reward counted per edge blurs the two together. Every judgement is therefore
counted **per path**: the step in the company it kept - *the node that called
the edge's parent, and the edge it then took* - with three numbers, `seen`,
`correct` and `incorrect`. `log((correct + 0.5) / (incorrect + 0.5))` is then
added to that edge's weight (times `path_scale`, 1 by default) before the
softmax over the node's children, so the same step is cheap for the walk that
was right here and dear for the one that was wrong. The term is exactly zero
until something is judged, which is why a freshly trained model predicts as it
always did.

A context is born when a path is judged - a reward, a penalty, a correction's
blamed or taught steps - and every later traversal keeps its `seen` up to
date; an unjudged training pass never creates one, so training a corpus cannot
fill the table with the second-order counts of a whole language. `seen_ratio`
is how much of that edge's traffic came through that caller, `correct_ratio`
how much of the judged traffic was right. Splits carry their contexts with
them; a merge drops the ones that were never a choice (a unary chain has only
one way through). The searches carry the node they came from, so the beam, the
cheapest path and the sampler all see the context-aware costs.

```bash
python -m radixnet --model model.count.json paths --limit 20   # the judged steps and their counters
python -m radixnet --model model.count.json weights --path-scale 0   # ignore the path counters
go/bin/radixnet-count --model model.count.json paths
curl localhost:8000/api/paths?limit=20
```

```
after   step          correct  incorrect  seen  correct %  seen %  term
"the"   he  -> e c          3          1     7  75%        13%     +1.253
"a c"   ca  -> at           4          0     6  100%       22%     +2.197
```

### A node from where it stands

The same numbers read the other way round. An edge's counters say what that
step did; they do not say what it did *here*, among the other ways out of the
same node. `nodes` shares a node out over its neighbours - a row per previous
node, a row per next node - and each row carries:

* **seen %** - that edge's share of the traversals on its side of the node.
* **reward %** - that edge's share of the reward on its side, **signed**: the
  shares are taken over the magnitudes, so a penalty reads as a negative share
  of the pressure on the node and the two sides compare without the signs
  cancelling out. One arm holding all of it reads ±100%.
* **judged / of edge** - how much of that edge's traffic a judged context has
  been watching (`paths` above counts the same walks, keyed by who called
  them).
* **correct / wrong** and **correct %** - what those contexts came to, summed
  over every caller.

The denominators are the side's own, not the node's visits: a node is entered
without an in-edge whenever a text starts on it, so a node can be visited more
often than everything arriving at it adds up to.

```bash
python -m radixnet --model model.count.json nodes --limit 10        # the most visited nodes
python -m radixnet --model model.count.json nodes --node "at "      # one node, by label or by a trigram it holds
go/bin/radixnet-count --model model.count.json nodes --node "at "
curl 'localhost:8000/api/nodes?node=at%20'
```

```
"at "  visited 12x  (2 in, 3 out)
      node      seen  seen %  reward  reward %  judged  of edge  correct  wrong  correct %
----  --------  ----  ------  ------  --------  ------  -------  -------  -----  ---------
from  " cat"       7  58%     +0.00   0%             0  0%             0      0  -
from  "t sat"      5  42%     +1.00   100%           1  20%            1      0  100%
to    "t sat"      5  42%     +1.00   33%            1  20%            1      0  100%
to    "t on "      5  42%     +1.00   33%            1  20%            1      0  100%
to    "t ran "     2  17%     -1.00   -33%           1  50%            0      1  0%
```

That is the branch after *the cat* / *a cat*, once one answer has been
corrected: 42% of what leaves it goes to `t sat`, which earns a third of the
reward on that side and was right the one time it was judged, and 17% goes to
`t ran `, which carries a third of it as a penalty and was wrong. The Graph tab
shows the same table when a node is clicked.

`<back>` is an ordinary row here, because it is an ordinary edge: it says what
share of the walks leaving this node have learned to *go round* rather than
carry on, and the hand-over's reward sits beside the penalty on the step it was
about to loop through -

```
to    "t sat"       4  36%     -1.00   -50%
to    "<back>"      1   9%     +1.00   +50%
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
   taught. The marks alone already imply a plan - one lesson per weak point,
   worst first - and that is the floor: a weakness the teacher's plan skips
   takes the place of a lesson that drills nothing the card marked down, and
   an answer that cannot be read leaves the card's own plan standing (`source`
   says which wrote it).

6. **The prompt that teaches the next batch.** The plan ends in a `prompt`: the
   **brief**, two or three sentences addressed to the teacher who will write
   the next batch of exercises - the points of grammar to drill in order, the
   step up in difficulty, and what the sentences should be about. Start the
   next run with it (`--brief TEXT`, `{"brief": ...}`, or **Teach the next
   batch** in the Tutor tab) and every round of that run is written to it.

   The **incremental upgrade** in it is not the LLM's to invent - it is read
   off the marks, so a student who is failing never gets a harder exercise:

   | The last batch | Step | What the next one gets |
   |---|---|---|
   | 80% passed at 8/10 or better | `advance` | the next level up, openings one rung longer (`3 to 6` -> `5 to 8` -> `7 to 12` -> `10 to 16`), the pass mark +1 (capped at 9) |
   | 50% passed | `stretch` | the same level, openings one rung longer |
   | anything less | `hold` | nothing harder: the weak points drilled, and 3 correct sentences to imitate (`--drills`) |

   The teacher is given that step in words and asked to repeat it in the brief;
   the settings the plan carries are always the ones the report card earned.

7. **Auto run.** `--batches N` closes the loop without a human in it. One
   *batch* is `--rounds` rounds and the report card over them; between batches
   the run plans from that card, applies the plan to itself - the brief becomes
   the standing instruction, the upgrade sets the level, the openings, the pass
   mark and the drill sentences, and the single-focus pin is released - and
   teaches the next batch to it. `--batches 0` keeps going until Ctrl-C (the
   API job's **Stop**, the tab's Stop button). A batch that cannot be planned
   ends the run rather than repeating itself, and each batch's records carry
   its number (`{"kind": "batch", ...}` says what the next one is being taught
   to).

```bash
ollama pull llama3.2
python -m radixnet tutor --topic "everyday life" --rounds 5 --exercises 5
python -m radixnet tutor --topic "the sea" --focus "past tense" --drills 5 --threshold 7
python -m radixnet tutor --topic animals --dry-run            # set and mark, train nothing
python -m radixnet tutor --topic animals --rounds 3 --plan 3  # ... and plan the next three lessons
python -m radixnet tutor --topic animals --brief "Drill plural nouns and articles, worst first. Openings of 5 to 8 words."\
                         --level beginner --words "5 to 8" --plan 3    # teach the batch that plan asked for
python -m radixnet tutor --topic animals --rounds 3 --report lessons.json
make tutor TOPIC="everyday life" ROUNDS=5
make tutor-dry TOPIC="everyday life"
make tutor-plan TOPIC="everyday life" PLAN=3
python -m radixnet tutor --topic animals --rounds 2 --batches 5   # auto run: five batches, each planned from the last
python -m radixnet tutor --topic animals --batches 0              # ... until Ctrl-C
make tutor-auto TOPIC="everyday life" BATCHES=5
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
them, then the lesson plan under it: the step up, the brief for the next batch
("Teach the next batch" loads both into the settings) and a row per lesson
("Use this lesson" loads one). **Batches** turns that hand-off into an auto run
the server drives itself, the form following each batch as it starts; the
**Teacher** selector switches between Ollama
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

## Tool use: the network browses, Ollama sets the bar and teaches

`radixnet agent` and `radixnet explore` give the network **external tools** —
web search, web pages, a calculator, optionally a Python sandbox and the
uploaded files — and put a local LLM around it as the thing that keeps it
honest. The network cannot *decide* to call a function; it can only write
characters. So a tool call is text it writes, and the observation is text it
reads back:

```
TASK: How many legs does a cat have?
<tool>web_search {"query": "cat anatomy legs"}</tool>
<result>1. Cat - Wikipedia - https://en.wikipedia.org/wiki/Cat ...</result>
<tool>web_fetch {"url": "https://en.wikipedia.org/wiki/Cat"}</tool>
<result>Cat - Wikipedia The cat is a small domesticated carnivorous mammal ... four legs ...</result>
<answer>A cat has four legs.</answer>
```

That makes a whole attempt **one ordinary training text**, which is exactly
what 2NRL can reward or punish as a unit.

```
task -> acceptance criteria (Ollama, written first) -> the network calls tools -> judged against those criteria
     -> taught with the same real tools when it failed -> train on the failures, invert, fine-tune (2NRL)
```

Ollama plays four roles, and solving the task is the last one it is given:

1. **Criteria** — before anything is attempted, the LLM writes the handful of
   checkable statements a correct answer must satisfy (`--criteria 4`). The bar
   is therefore set independently of whatever the network happens to produce.
   A task file may carry its own criteria, and then the LLM is not asked.
2. **Mediator** — an untrained network writes noise. Whatever it emits that
   cannot be read as a call is handed to the LLM, which turns it into one valid
   call against the real tool schemas (Ollama's own tool-calling API where the
   model supports it, JSON otherwise). The repaired call is executed *and
   written into the transcript*, so what the network learns is always
   well-formed — and the share of calls it managed by itself (`own` /
   `autonomy` in the records) is the number that says whether it is learning.
   `--mediation always` never asks the network; `--mediation never` runs its
   broken call and learns from the failure.
3. **Judge** — the finished transcript is marked against the criteria written
   up front: every criterion met or not, a score, a critique. `--lenient`
   accepts an answer the judge calls correct even with a criterion unmet;
   `--no-judge` marks against a task's known `answer` instead.
4. **Teacher** — only when the network failed does the LLM solve the task, with
   the *same real tools*, so the demonstration is a transcript of things that
   actually happened. That transcript is the positive phase of 2NRL
   (`--no-teach` turns it off).

### Failure first, then invert

Learning follows section 9.1 of `DESIGN.md`. Every failed attempt gets a **gap**
in `[0, 1]`: the share of the acceptance criteria it missed, or how far below
the pass score the judge put it, whichever is worse — an attempt that answered
nothing has a gap of 1, and every failure keeps a floor of 0.1 so a near miss
still trains. With `--blatant-mode fail_invert` (the default) the negative phase
runs one pass per distinct weight, heaviest first, with the learning rates
multiplied by `min(--blatant-boost, 1 + gap / --blatant-margin)`: the worse the
attempt, the harder the network is pushed to reproduce it — to fail blatantly on
purpose — before the inversion turns that into avoidance and the correct
transcripts are the fine-tune pass. `activation` / `state` instead negate the
nodes along a failed transcript locally, and blatant failures then leave the
2NRL garbage set. Nothing failed: the correct run is rewarded and nothing is
inverted.

### The failures feed the negative network

`agent --blame` / `explore --blame` hands every failure the judge finds to the
[negative network](#the-negative-network-what-went-wrong-and-why), so the
network keeps a second graph of *what going wrong looks like here*. Each failure
is blamed at the granularity it happened at, because they are different
failures:

* the **transcript**, for the answer — and when a correct run of the same task
  exists (usually the teacher's demonstration) it rides along as the
  *correction*, so only the characters that differ from a run that worked are
  blamed. The shared task line and the calls that worked never become evidence;
* the **emission the mediator had to repair** (`bad-call`) — what the network
  actually wrote, which is *not* in the transcript (that holds the repaired
  call). Blaming the transcript for it would teach the network that a
  well-formed call is a mistake;
* the **call the network wrote itself that the tool refused** (`tool-error`).
  A call the mediator wrote is not the network's fault and is not blamed.

The reason comes from how far the attempt got (`no-call`, `bad-call`,
`tool-error`, `no-answer`, then the judge's own words) and the severity from its
gap. What it learns comes straight back: every candidate the network offers is
put through the negative network first, and one it recognises as a known failure
is passed over for the next candidate (`--no-avoid` turns that off). The task
records carry `negative_blamed`, `negative_edges` and `negative_reasons`, and
`radixnet negative reasons` shows the table.

### Browsing in a real Chrome

Plain fetching reads a document; it cannot read a page that draws itself.
`--browser` runs each page in a real headless **Chrome** and reads the DOM after
its scripts have run, so a search engine or a single-page app becomes readable:

```bash
python -m radixnet tools browser                       # what would be driven, and its versions
python -m radixnet tools call --tool web_fetch --arg url=https://example.com --browser
python -m radixnet explore --steps 20 --browser
```

WebDriver is an HTTP protocol, so nothing is installed: `radixnet` starts
`chromedriver` itself and talks to it with the standard library. It needs a
`chromedriver` and a Chrome **of the same major version** — the usual reason
this fails — and says so plainly when they differ. `$RADIXNET_CHROMEDRIVER` and
`$RADIXNET_CHROME` override the search, and `$RADIXNET_WEBDRIVER` (or
`endpoint=`) points at a WebDriver that is already running, such as a
`selenium/standalone-chrome` container or a Selenium Grid. The address guards
still run first; a browser executes whatever a page sends it, so untrusted
browsing belongs in the Docker image.

### Exploring on its own

`radixnet explore` takes the tasks away and lets the network choose them. Each
step it continues `TASK:` — the prefix every transcript it has learned starts
with — into whatever it is reaching for; the LLM turns that emission into one
concrete question browsing can settle, preferring the pages the network has come
across but not read yet. Then the ordinary cycle runs on it. Search results and
page links are filed as the frontier and the pages read as visited, so the
exploration compounds instead of circling. `--steps 0` runs until Ctrl-C, which
stops after the current step and saves.

```bash
ollama pull llama3.2
python -m radixnet tools list                                  # what the network can call
python -m radixnet tools call --tool web_fetch --arg url=https://example.com
python -m radixnet agent --tasks data/sample_tasks.txt         # solve a list of questions
python -m radixnet agent --tasks data/sample_tasks.jsonl --rounds 3 --report agent.json
python -m radixnet explore --steps 20 --seed-url https://en.wikipedia.org/wiki/Cat
python -m radixnet explore --steps 0                           # until Ctrl-C
```

Task files: one question per line (`.txt`, `#` comments), or `.json` / `.jsonl`
objects `{"id", "prompt", "criteria", "answer", "seeds"}` (`data/sample_tasks.*`).

**Safety.** The web tools accept `http` and `https` only, refuse URLs with
credentials, refuse any address that resolves into a private, loopback,
link-local or reserved range (`--allow-private` is for a local test server),
re-check every redirect hop, and cap both the response size (`--max-bytes`) and
the time (`--web-timeout`). Nothing is sent but a GET with a user agent — no
cookies, no credentials. The `calculator` evaluates an AST that allows numbers,
operators and the `math` functions and nothing else; `--python-tool` runs code in
the same sandbox as `codegen` (see its section for what that does and does not
contain). `--offline` removes the web tools altogether.

API: `GET /api/tools`, `POST /api/tools/call` (one call, no model),
`POST /api/agent/start` (job), `POST /api/agent/explore` (job),
`GET /api/agent/history`, `POST /api/agent/criteria` (the criteria for a task,
nothing attempted) and `POST /api/agent/solve` (one task through the loop with
no training, reporting the transcript, the verdict and the gap). The frontend's
**Agent** tab drives all of it: the mode, the tool list, the loop and mediation
options, the failure settings, a live log of criteria, proposals, tool calls
(tagged by who wrote each one) and attempts, and a table of finished tasks.

## MCP: the tools and the network, to any client

`radixnet mcp` speaks the **Model Context Protocol** on stdin / stdout, so any
MCP client — Claude Desktop, an editor, another agent — can use this instance.
It offers the external tools (browsing, the calculator, optionally the sandbox
and the uploaded files) *and* the network itself:

| Tool | What it does |
|---|---|
| `radixnet_predict` | continue a prefix (dijkstra, beam or sample) |
| `radixnet_generate` | whole texts from the prediction search |
| `radixnet_score` | how likely the network thinks a text is, per character |
| `radixnet_stats` | size, compression, training history, backend |
| `radixnet_judge` | the negative network's verdict: has this way of going wrong been seen here before, and why |
| `radixnet_solve` | one task through the whole agent loop — acceptance criteria, tool calls, a judged answer |

```bash
python -m radixnet --model model.json mcp        # tools + the network
python -m radixnet mcp --no-model --offline      # the calculator alone
python -m radixnet mcp --browser                 # browsing in a real Chrome
make go-mcp                                      # the Go server, same protocol
```

The **Go port speaks it too** (`go/radixnet/mcp.go`, `radixnet-count mcp`): the
same protocol revision, the same tool names and schemas, and the same answers
down to the error text, so a client cannot tell which one it is connected to.
It offers `radixnet_solve` only when an LLM is reachable, and `radixnet_judge`
only when the negative network is there — the same rule the Python server
follows. **So does the Rust port** (`rust/src/mcp.rs`, `radixnet mcp`), held to
Python line for line over one message stream (`tests/test_rust_parity_agent.py`).

Point a client at either the usual way:

```json
{"mcpServers": {"radixnet": {"command": "python", "args": ["-m", "radixnet", "--model", "model.json", "mcp"]}}}
{"mcpServers": {"radixnet": {"command": "go/bin/radixnet-count", "args": ["--model", "model.count.json", "mcp"]}}}
```

MCP is JSON-RPC 2.0 over a stream, so this is the standard library and nothing
else. A tool that fails comes back as a result with `isError`, not a protocol
error, so the client can show it to its model; nothing is ever written to stdout
but the protocol, and the log goes to stderr.

## Three models: RadixNet, the count / reward model and the resonant model

The selector at the top of the frontend (and `--kind` in the CLI, `POST
/api/model/select` in the API) chooses the algorithm.  All three live on the same
self-compressing cyclic graph and share encoding, prefix location, sampling,
scoring, compression, checkpoints and persistence; a model file records its
kind, so `load` always restores the right one.  A fourth kind, `word`, is the
count / reward model over an alphabet whose symbols are **words** rather than
characters - the same graph, the same weights, the same search (see *Word
n-grams* below).

| | RadixNet (`radix`) | Count / reward (`count`) | Resonant (`resonant`) |
|---|---|---|---|
| edge weight | learned by the one-hop rule together with the per-node sine activations | a **dual frequency function**: the edge's share of its node's traversals, all time (`R_all`) and inside a sliding window of the last N traversals (`R_recent`), plus rewards and the judged paths; no gradient, no learning rate | the edge's share of its node's traversals **plus a resonance**: `amp_scale · log share + reward_scale · reward + resonance_scale · coherence · cos(phase − mu)`, where `mu` is the mean phase at which the edge fired and `coherence` how consistently; no gradient, no learning rate |
| training | epochs over mini-batches with `lr` / `act_lr` (and their schedules) | every epoch counts one more traversal of each text's path (all time, in the sliding window and in the global total) | every epoch walks each text carrying its **phase** and counts each traversal into its edge's circular mean; the cycle decisions the text made train the metacognitive layer beside it |
| feedback (thumbs, 2NRL, codegen judge, adversarial review) | train on the bad texts, invert, fine-tune on the good ones | `punish`: reward −= `strength` on every edge of a bad path; `reward`: a traversal plus reward += `strength`; nothing is inverted | `punish` penalises **and decoheres** a path (its phase lock is scrambled), `reward` rewards and sharpens; 2NRL trains on the bad texts, inverts, then relocks on the good ones |
| `invert` | negates every weight and every node's activation (`a` and `k`) | flips the sign of every reward | rotates every edge's mean phase by `pi` - what resonated now cancels - and flips the layer with it |
| prediction | Dijkstra's cheapest path (or sampling) | a beam search that returns the **top K** (most likely) and **bottom K** (least likely) continuations of one prefix in one call; the best one is the prediction | **k-best** over `(node, chars, phase)`: Dijkstra with K labels per state, so the K cheapest walks come back *exactly* and each one carries its own path, which is what lets a **phase-locked cycle be handed to the metacognitive layer**. `dijkstra` (one label) is kept for the single-path guarantee and is cycle-blind; `beam` is kept for the bottom K |

```bash
python -m radixnet --kind count train --data data/sample_corpus.txt --epochs 3     # -> model.count.json
python -m radixnet --model model.count.json predict --prefix 'the quick' --length 10 --k 5
python -m radixnet --model model.count.json feedback --good-text 'the quick brown fox' --bad-text 'zzz qqq' --strength 2
```

### Word n-grams: the same model over an alphabet of words

`--encoding word:3:1` (the **Words** tab in the frontend, `SPEC-WordNGrams.md`,
D-071 and D-073) keeps every one of those rules and changes only what a *unit*
is.  Not one of the graph's structural rules mentions a character, so a word
n-gram model is not a different model - it is the same one with the encoding's
`unit` dial turned: a gram is three words rather than three characters, and a
label is text either way.  Compression then does to word chains what it does to
character chains - a repeated phrase becomes **one node whose label is that
phrase**.

```bash
python -m radixnet --kind count --encoding word:3:1 --model model.word.json train \
    --data data/sample_corpus.txt --epochs 3
python -m radixnet --model model.word.json predict --prefix 'the cat sat on' --length 6
python -m radixnet --model model.word.json words --limit 20   # the alphabet its grams are made of
make word-demo                                                # train, predict, generate, list it
```

There is **no vocabulary**: a gram is text, so the alphabet a graph has read is
whatever its grams are made of, and there is nothing to freeze, prune or learn.
Tokenising is the whitespace split and nothing else (punctuation stays attached,
case is kept).  Two things that costs, both deliberate: a word encoding
**normalises whitespace**, so a corpus whose whitespace carries meaning (source
code, base64, a waveform) belongs on the character encoding; and a word it has
never read makes a gram it has never seen, which shows up in
`unknown_transitions`.  Everything counted in units is counted in **words** -
`--length 6` emits six words, `score`'s `per_char` is per word - and `units` in
`stats`, the API and the frontend says which.  Python, Go and Rust read and
write the same file: the format never changed, and the encoding rides in the
graph document as a three-key block written only when it is not the default.

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
each edge's all-time and recent share.  Every one of those counts is a cyclic
counter (below): it is set back to 0 at `10^15` and the reset is counted, and
the ratios above are formed from the exact totals.

The server keeps the model of each kind in memory: switching kinds parks the
active model (unsaved work included) and brings the other one back, loading
its file (`model.json` / `model.count.json`) or creating a fresh one the
first time.

### The resonant model: an analog phase, and cycles that hand off

`RadixNet` reads *"a brain is an analog computer, so sine waves are how
information is encoded"* as a **pointwise** sine - every node passes its state
through `-sin(z/3)`.  This model reads the other half: a sine has a **phase**,
phases **add** along a path, and signals that meet in phase reinforce while
signals that meet in antiphase cancel.

A walk therefore carries one number more than the node it stands on: its phase,
one of `--buckets` positions on a ring.  Every trigram advances it by a fixed
amount - a **clock** (`buckets / period`: with the default one character is one
bucket, so the phase says where in the rhythm the walk is) plus a **kick**,
`--kick-scale` times a stable hash of the trigram itself.  With
`kick_scale = 0` (the default) the phase is pure position: dense and quickly
learned.  Turn it up and the phase becomes a rolling signature of the whole
path - long-range context on a three-character graph - at the price of far
sparser statistics per phase.

An edge does not learn a phase offset; it learns *the phases at which it was
actually taken*, as a circular mean.  That gives `mu` (where it fires) and
**coherence** (how consistently, in `[0, 1]`) - free confidence, measuring how
context-dependent a transition is with nothing added to measure it.  An
incoherent edge falls back to plain frequency; a coherent one is cheap in phase
and dear out of phase.

**What that buys.**  Train on `"the cat sat down"` and `"a big cat ran away"`.
Three contexts reach the node `"at "`, and phase-free its three children are
exactly `1/3` each - the model cannot tell them apart.  Per phase they are not:

```
bucket 0   t sat 0.154   t down 0.154   t ran away 0.691
bucket 3   t sat 0.097   t down 0.807   t ran away 0.097
bucket 5   t sat 0.807   t down 0.097   t ran away 0.097
```

so the model continues `"the cat "` with `"sat down"` and `"a big cat "` with
`"ran away"` - while the identical model with `--resonance-scale 0` answers
`"ran away"` to both.

**The search: Dijkstra with K labels per state.**  One label per state is what
makes Dijkstra a shortest path — and also what makes it blind: a single label
cannot say *which* walk reached the state, so there is no path for the
metacognitive layer to look at.  Letting a state be settled up to `k` times
fixes both at once.  The `k` goals pop in cost order and are the `k` cheapest
walks **exactly**, and every label reads back to its own path, so the cycle it
is standing in is visible.  `k = 1` is Dijkstra to the expansion; above that the
cost grows with `k`, not with the width of a frontier, because the search still
stops at the `k`-th finished walk:

```
prefix "the ", k = 5, to the end of a text     states expanded
  dijkstra (k = 1)                                   49
  k-best                                            159     exact
  beam                                             1312     an approximation
```

That is the default for both `predict` and `generate`.  `dijkstra` is kept for
the single-label guarantee, and `beam` because it is the only one that can
answer the **bottom** half — in a cyclic graph the worst walk is unboundedly bad
(loop once more and it is worse), so "least likely" needs a frontier's bound
rather than a goal count.

**Cycles are a decision, not a hazard.**  Coming back to a node at a *new* phase
is progress: the signal has moved on.  Coming back at the *same* phase is a loop
that would repeat for ever.  Only the second kind is a cycle worth deciding
about, and when the search meets one it stops asking the graph and asks the
**metacognitive layer**, which holds a learned policy per cycle signature (the
re-entered node's first trigram and how long the loop is, e.g. `lol:4`):

* `ride` - go round again (right for `aaa`, `lol lol lol`, `----`, indentation),
* `escape` - take the cheapest child that does not close the loop,
* `abort` - stop here.

Those are learned by counting what the corpus did at that exact cycle, so a
cycle the corpus rides stays cheap to ride and one it never rides becomes
expensive.  Nothing is forbidden.

**The reflex and the memory.**  The `BACK` sentinel and this layer are the same
knowledge at two grains.  `BACK` is the reflex — an edge competing for a node's
probability, taught by voices that caught themselves repeating and backed out,
and when it wins, the branch is handed over and offers nothing.  The layer is
the memory of one particular cycle.  They are wired both ways:

* the node's hand-over probability enters the layer's policy as evidence
  **against** riding, on the same scale as everything else, so:

  ```
  nothing known                              ride   (weakly)
  6 hand-overs at the node, P(BACK) = 0.95   escape (the reflex speaks)
    + 1 observed ride at this exact cycle    escape (one observation is not enough)
    + 3 observed rides                       ride   (the memory outranks the reflex)
  ```

* and when `BACK` has handed a branch over, the search asks the layer what it
  remembers about cycles at *that node* — it has to be the node-level question,
  because a walk meeting the hand-over is on its first visit and has no cycle to
  name yet — and restores the branch when the answer is that it rode them.  With
  no memory, the hand-over stands.

Metacognition supervising the reflex is what the research note describes, and
that is the line where it happens.  (Going the other way — letting the corpus's
declines teach `BACK` — is a dial, `teach_back`, and it is **off**: a corpus
declines cycles constantly, `BACK` only ever rises, and its veto is per *node*
while the corpus's knowledge is per *child*. Measurements in `DESIGN.md`
§30.3.1.)  `info` reports the signatures learned, the
status bar shows coherence and cycles, and `invert` flips the layer with the
graph so 2NRL covers it too.

```bash
python -m radixnet --kind resonant train --data data/sample_corpus.txt --epochs 3   # -> model.resonant.json
python -m radixnet --model model.resonant.json predict --prefix 'the ' --length 20 --mode beam --k 5
python -m radixnet --model model.resonant.json weights --kick-scale 1.0 --buckets 16
python -m radixnet --model model.resonant.json info
```

`weights` shows or changes `--buckets`, `--period`, `--kick-scale`,
`--resonance-scale`, `--amp-scale`, `--reward-scale` and `--concentration`
(which shrinks a thinly observed edge's coherence, so one traversal is not
mistaken for certainty), and rejects another kind's options by name rather than
ignoring them.

## The traversal: follow the rewards, or avoid the punishments

Every search in this package - Dijkstra, the two beams, the k-best walk, the
sampler, and their phase-unrolled twins - reads the graph through one funnel:
`[(child, edge, cost)]` for a node, with `cost = -log P(child | parent)`. Which
cost function fills that list is the **traversal**, and it is an option
(`radixnet/penalty.py`, `go/radixnet/penalty.go`):

| `--traversal` | what the search is looking for |
|---|---|
| `reward` (the default) | what the model believes. The count / reward model's probability carries `exp(reward_scale · reward)`, so a path the tutor rewarded is cheap and the search **follows the rewards**. Every release before this option behaved exactly this way, and still does unless told otherwise. |
| `punishment` | what the model was punished for. The rewards leave the score altogether and only the **penalties** price the step, so the cheapest path is the one that accumulated the **least punishment**. Nothing the network was praised for makes a step cheaper here; only what it was corrected for makes one dearer. |

The traversal is *what* a search looks for; `--mode` is *how* it looks. They
are independent: every mode of every kind can run either traversal, and the
search code itself does not change - it is reading a different cost function.

### Why the two differ

Rewards and penalties are not symmetric evidence. A reward says *this was good
once*; a penalty says *this was wrong, and here is the correction*. The first
is an invitation to repeat a success and pulls the search towards whatever the
tutor happened to praise; the second is a boundary, and a walk that respects
every boundary it has been taught is not the same walk as one that chases every
reward it has been given. A model whose rewards are sparse (a handful of thumbs
up) but whose penalties are dense (a tutor that corrected a thousand sentences)
has far more to say in the second currency than in the first.

### The split it rests on

One hook, `RadixCyclicGraph.child_evidence` (`Graph.ChildEvidence` in Go),
splits an edge's evidence in two:

* **merit** - what speaks *for* the step with every reward taken out of it:
  frequency, structure, resonance. What the corpus did, not what a judge said
  about it.
* **penalty** (`>= 0`) - what speaks *against* it: the punishment the edge
  carries.

The step's score is `merit_scale · merit − penalty_scale · penalty` and the cost
is the usual `-log softmax` over the parent's children, so costs stay
non-negative (Dijkstra is still a shortest path), `exp(-cost)` is still a path's
probability, and the numbers stay comparable with the reward traversal's.
`--merit-scale 0` is the pure form: nothing but the punishment decides, and
among equally unpunished children the walk is indifferent.

Every kind implements the split in its own currency:

| model | merit | penalty |
|---|---|---|
| `RadixNet` (the sine model) | the positive part of `w · f_p · f_c` | the negative part of it |
| the count / reward model | the dual frequency function with the whole reward subtracted back out | `reward_scale · max(0, −reward)` |
| the resonant model | amplitude and resonance, rewards out | `reward_scale · max(0, −reward)` |
| the negative network | `log(1 + cleared text)` | `log(1 + net blame)` |

The sine model keeps no separate ledger of its punishments: 2NRL trains a
failure in and then inverts it, so what a punishment leaves behind *is* a
negative score on the edges of that path - which is why its penalty is read
straight off the score. There the two traversals coincide at the default scales
and part company as soon as `--penalty-scale` is raised: what was punished then
weighs more than what was learned. A judged path context
(`path_scale · log((correct + ½) / (incorrect + ½))`) splits the same way: the
part of it that says *this step was wrong here* is a penalty, the part that says
*this step was right here* is merit.

On the **negative network** the option reverses the network's whole purpose,
which is the point: its ordinary traversal predicts the likeliest way a prefix
goes wrong, and `--traversal punishment` walks the **least blamed** way through
the same failure structure instead.

```bash
radixnet feedback --good-text "the cat sat on the mat" --strength 4   # praise one branch
radixnet feedback --bad-text  "the cat ate the rat"    --strength 4   # correct another

radixnet predict --prefix "the cat" --length 16        # " sat on the mat" - it follows the reward
radixnet predict --prefix "the cat" --length 16 \
    --traversal punishment --merit-scale 0             # " on the mat"     - it only avoids the penalty
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --traversal punishment
```

At the fork after `the cat`, with one branch praised, one corrected and one
never judged, the two cost functions read:

| child | reward | reward traversal | punishment traversal | pure punishment |
|---|---|---|---|---|
| `"t sat"` | +4 | 0.701 | 1.041 | 1.105 |
| `"t on t"` | +4 | 0.701 | 1.041 | 1.105 |
| `"t ate t"` | −4 | 8.901 | 5.242 | 5.105 |
| `"t r"` | 0 | 4.901 | 1.242 | 1.105 |

The child nobody ever judged costs `4.901` under the rewards and `1.242` under
the punishments - almost exactly what the praised children cost, because the
praise buys nothing here. Only the corrected child stays dear.

In Python it is `predict(..., traversal="punishment", penalty_scale=1.0,
merit_scale=1.0)` on every kind, the same three arguments on `generate` and on
`NegativeFilter.predict` / `.generate`; a `Prediction` carries `traversal`,
saying which one ran. The Go port has the same option on `PredictOptions` /
`GenerateOptions` and the cross-language parity suite requires both sides to
walk the same least-punished paths at the same costs.

In the frontend it lives on the **Settings** tab, and the Predict and Generate
tabs show the same control: it is one setting, shared, so changing it on any of
the three changes it on all of them.

## Search and training methods

More ways to search, and more ways to train - every one of them **off by
default**, and a search or a run with all of them off is exactly the one it
always was, draw for draw and byte for byte. `SPEC-SearchAndTraining.md` is the
contract; the Python, Go and Rust implementations keep it number for number,
and `tests/test_go_parity.py` / `tests/test_rust_parity_methods.py` hold them to
the same graph, history and file.

**Sampling filters** narrow what a sampled step (`--mode sample`) draws from:
`--top-k K` keeps the K cheapest options, `--min-p P` those at least P times as
likely as the best one, and `--top-p P` (*nucleus* sampling) the smallest set of
cheapest options holding P of the probability mass - applied in that order. The
cheapest option always survives, and a step still draws once however few are
left, so a seeded walk is the same walk on every server.

**A diverse beam** (`--diversity X`, on `--mode beam`): in a compressed graph the
K cheapest paths are often one text with its ending varied. With a diversity the
beam keeps a larger pool of finished paths and picks from it by *maximal
marginal relevance* - the best path first, then each next one by its cost plus
`X ×` how much of it repeats a path already picked from the start. Costs are
reported as they are, so after the first entry the list is no longer in cost
order: that is the trade. Only the top K are spread.

```bash
python -m radixnet --seed 7 generate --mode sample --count 5 --temperature 1.2 --top-p 0.9 --min-p 0.05
python -m radixnet generate --mode beam --count 5 --diversity 2
```

**How a run walks its texts** (`train`, and `/api/train`):

| flag | what it does |
|---|---|
| `--order corpus\|shortest-first\|longest-first\|shuffle` | the order every epoch walks the texts in; `shuffle` draws a fresh order each epoch from the model's seed, without touching its random generator |
| `--curriculum C` | the first epoch walks the first C of the ordered texts and the share grows evenly to all of them by the last epoch - with `shortest-first`, *baby steps*. The graph still sees every text before the first epoch |
| `--replay-size N` | the model keeps a **replay buffer**: a uniform sample of N of every text it was ever trained on (bottom-k sampling by a hash of its seed), saved at the end of its file. 0 drops it; left out, it is kept as it is and offered this run's texts |
| `--replay R` | every epoch also rehearses R times as many texts from the buffer as the run has new ones, after the new ones, a fresh slice each epoch - so a new corpus does not wash an old one out. Rehearsed texts are not counted as trained |
| `--patience N`, `--min-delta X` | stop after N full epochs whose loss did not fall X below the best; the epoch that stops the run carries `"early_stop": true`. An epoch still inside the curriculum does not count |

```bash
python -m radixnet train --data first.txt --epochs 5 --replay-size 500
python -m radixnet train --data second.txt --epochs 8 --order shortest-first --curriculum 0.3 --replay 0.25 --patience 2
```

These belong to plain training on the kinds that learn by walking a list of
texts - the count, sine and resonant models (the Go port: the count model).
Thumbs up and down, 2NRL, the agent's punishment and the negative network's
blame walk their texts as they always have, and never touch the buffer. In the
frontend they are on the **Settings** tab, and the Predict, Generate and Train
tabs show the same controls; the Model settings tab shows the buffer.

### Training in reverse

`--reverse` (`reverse` on `/api/train`, **Read every text backwards** on the
Train tab) reads every text of a run backwards, in the model's units - its last
character first, or its last word on a word model - so the model learns what
comes *before* a text rather than what follows it. With `--whole-file` a file is
read from its end to its start; one text per line turns every line around, in
the order the lines came. Everything after the turn sees the reversed texts: the
order and the curriculum, the counters, and the replay buffer, which keeps them
as they were read. It works on every kind, the negative network included, and
the three ports write the same file, byte for byte (`SPEC-SearchAndTraining.md`
§9).

```bash
python -m radixnet --model backwards.json train --data book.txt --reverse --epochs 5
```

A model trained backwards is asked backwards: the query turned around, the
answer turned back. The frontend does both - turn on **Query backwards** (on the
Settings tab, and beside the traversal on Predict and Generate), type the *end*
of a text, and the answer reads the right way round, with what the model says
came before it highlighted; Generate's prefix becomes the text's ending. A
thumbs up or down sends the model's own, backwards, text, since that is what
feedback trains on. From the command line, turn the query around yourself -
`Encoding.reverse` does it in the model's units, and on a character model so
does `rev` - and read the answer the same way, since it comes back as the model
reads it (`"enin sevas emit ni hctits a"` is *a stitch in time saves nine*):

```bash
python -m radixnet --model backwards.json predict --prefix "$(echo 'saves nine' | rev)" --to-end
```

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

### Does it remember the picture? (`image tutor`)

An image was encoded into text and trained on, so **the right answer is on
file** and no LLM is needed to mark anything.  `image tutor` gives the network
the opening of that text - the header and a few characters of the payload, so
it knows which picture is wanted - and asks it to write the rest.  What comes
back is decoded and compared with the original: the mark out of 10 is the
agreement over the payload, and a failure is named (`unreadable`, `truncated`,
`overrun`, `garbled`, `blank`, `noise`, `drift`).

```bash
python -m radixnet image tutor photo.jpg --train --blame     # teach it, ask for it back, blame what it forgot
python -m radixnet image tutor photo.jpg --lead 32 --length 400    # a longer opening, only the first 400 characters asked for
```

With `--blame` the failures teach the negative network: the original text is
the correction, so **only the characters it actually got wrong** are blamed and
the payload it did remember clears blame (see
[the negative network](#the-negative-network-what-went-wrong-and-why)).

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

### Does it remember what you said? (`speech tutor`)

The utterance was encoded into text and trained on, so **the right answer is on
file** and no LLM is needed to mark anything.  `speech tutor` gives the network
the utterance's own token and the waveform header - nothing of the payload,
because the token already says which recording is wanted - and asks it to write
the samples back.  What comes back is run through the codec and compared with
the recording: the mark out of 10 is the agreement over the waveform, and a
failure is named (`unreadable`, `truncated`, `overrun`, `garbled`, `silence`,
`clipping`, `mishearing`, `distortion`).

```bash
python -m radixnet speech tutor clip.wav --text "the cat sat on the mat" --train --blame
python -m radixnet speech tutor clip.wav --length 400        # only the first 400 characters of the waveform
python -m radixnet speech tutor clip.wav --listen-back        # transcribe what it said back and compare the words
```

`--listen-back` decodes the recalled waveform and transcribes it, so one that
is a perfectly plausible sound but says *different words* is a `mishearing`
rather than a `distortion`; it needs a transcription backend and is off by
default.  With `--blame` the failures teach the negative network: the original
text is the correction, so **only the characters it actually got wrong** are
blamed and the waveform it did remember clears blame (see
[the negative network](#the-negative-network-what-went-wrong-and-why)).

A whole second of 8 kHz audio is ~10 700 characters, so `--length` is usually
what you want: it caps how much of the payload is asked for *and* what the
marking compares against, so a short quiz is still a fair one.

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
| `activation` / `state` | the local variant: no negative pass; every other node on a failed path has its activation (amplitude and offset together, so 0.5 really is neutral) or its trained node value `z` moved toward its negation by `g / (2 · margin)` - a slight attenuation for a slightly worse fake, a neutralised path at the margin, a full sign flip at twice it - so only that path's transitions turn unlikely. Fakes beyond the margin are *blatant* and leave the 2NRL garbage set; when every bad fake was blatant the generation skips the negative pass and the global inversion. |

`blatant_margin` (default 1.0 nats per character) is where a failure counts
as blatant; `blatant_boost` (default 4) caps the multiplier.  Generation
records and the Evolve tab's table show `failures`, `blatant`, the mean
boost / amount and whether a 2NRL pass ran.  Outside the loop the same
primitives are available directly: `RadixNet.two_nrl(bad, good,
bad_weights=[...])` and `model.invert_paths(texts, mode, amounts)`.

## Talking to something that answers back

`converse` has the model talk to itself, which is a good way to see what it
knows and a useless way to find out whether it *answers* anything: neither
voice can tell the other that its reply did not follow on.  `chat`
(`radixnet/chat.py`, the **Chat** tab) puts a real language model on the other
side of the line.

```bash
python -m radixnet --kind count chat --conversations 2 --turns 3 --topic animals
```

```
Partner: tell me about the cat
Model: the cat on the mat
    picked up "the cat"
Partner: and what about the dog
Model: the dog sat on the mat
    picked up "the dog"
conversation 1: 1/2 replies failed, mean mark 5.5000/10

#  replies  passed  failed  mean mark  overall  vetoed  blamed  learned  ended
-  -------  ------  ------  ---------  -------  ------  ------  -------  -----
1        2       1       1  5.5000     7.0000        0       1  2nrl     -
```

One conversation is four things:

1. the **partner** says a line.  It is told to keep it short, plain and easy to
   carry on from, because that is what a character-level model can reply to at
   all - the prompt is doing the model a favour, not flattering it;
2. the **model replies the only way it can**: the tail of that line is located
   in the graph and continued (`dialogue.reply`, the same search `converse`
   uses).  A reply is a real walk of the network, not a prompt trick, and when
   nothing follows the line the context loses a word at a time before the voice
   changes the subject.  With a negative network in hand the pair vetoes a reply
   *before it is spoken* (`--no-guard` turns that off);
3. they take turns for `--turns` exchanges;
4. the **judge** marks every reply out of 10 **against the line it answered** -
   not against a style guide - and gives the conversation as a whole a verdict
   of its own.

What the marks buy is the point.  The failures blame the negative network and
the passes clear it, as every other tutor here does; and then 2NRL trains the
positive model on both - *and on the partner's own lines*, because in that
conversation, at that moment, they are exactly what a good reply would have
looked like.  `--no-learn` marks without training, `--no-teach-partner` keeps
the partner's lines out of the positive phase.

A conversation can also end early, and the report says which way: the model had
nothing left to say, the guard vetoed everything it could say, or the partner
went quiet.

The Go port holds the same conversation: `radixnet-count chat`, `POST
/api/chat/start` and `GET /api/chat/history`, with `TestGoChatParity` running
both CLIs against one fake partner and asserting the same prompts, the same
transcripts, the same marks and the same two networks on disk afterwards.

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
| the **speech tutor** (`speech tutor --blame`, the Speech tab, `POST /api/speech/tutor {"blame": true}`) | it is asked to say back an utterance it was taught; what comes back is run through the codec and compared with the recording, the agreement over the waveform is the mark, and the original is the correction | `unreadable`, `truncated`, `overrun`, `garbled`, `silence`, `clipping`, `mishearing`, `distortion` |
| the **image tutor** (`image tutor --blame`, the Images tab, `POST /api/images/tutor {"blame": true}`) | the same, for a picture it was shown: the payload it writes back is compared with the encoded image | `unreadable`, `truncated`, `overrun`, `garbled`, `blank`, `noise`, `drift` |
| **itself, on a loop** (`negative auto`, the Negative tab's *Automatic* card, `POST /api/negative/auto`) | the model writes texts of its own and an LLM reviewer marks them, round after round - the same critique-to-reason and mark-to-severity rules as the reviewer above, with nobody typing anything in | the reviewed-text reasons |
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

### Feeding itself: `negative auto`

Everything above needs *something else* to be running - a tutor round, a code
problem, a review.  `negative auto` is that something else, on a loop, so the
Negative tab stops being the one place where a person has to type a failure in
by hand.  Each round:

1. the positive model writes `--count` texts of its own;
2. an LLM reviewer marks each one out of 10 and says what is wrong with it - a
   local **Ollama** model by default, ChatGPT with `--provider chatgpt`;
3. every text below the pass mark blames the negative network (the critique
   picks the reason, the mark sets the severity) and the texts it passed take
   blame off what they share with known failures.

Then it goes round again.

```bash
python -m radixnet negative auto --rounds 5 --count 8              # five rounds, reviewed by the local Ollama
python -m radixnet negative auto --rounds 0 --context "plain English about everyday life"   # until Ctrl-C
python -m radixnet negative auto --provider chatgpt --reviewer-model gpt-4o-mini
```

`--context` is the reviewer's yardstick - what the texts are *meant* to be -
and is worth setting, because "is this good?" means little without it.
`--rounds 0` runs until Ctrl-C, which finishes the round it is in and saves.
**The positive model is only read from**: nothing here trains, rewards or
inverts it, so the loop can be left running beside whatever else is teaching
it.

In the browser it is the Negative tab's **Automatic** card: press Start and the
round table, the reason table and the journal below it fill in by themselves as
the rounds land (`POST /api/negative/auto` starts the job,
`GET /api/negative/auto/history` is its record).

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

### The guard: both networks on every answer

The pair above is what `negative filter` runs when you ask for it.  It is also
what `generate`, `predict` and `converse` run *without* being asked: the same
two networks, in tandem, on every answer the model hands out.

```bash
python -m radixnet negative blame --text "a bird flew over the mat" --reason "mixed-up animals"
python -m radixnet generate --count 3 --mode sample --seed 1 --max-length 28
```

```
guard: model.negative.json (2.0000 blame over 1 reasons)
#    cost    prob  end  text
-  ------  ------  ---  ------------------------------
1  2.4456  0.0867  yes  "the hill"
2  5.0796  0.0062  no   "the dog sat sat sat sat sat "
3  5.0244  0.0066  yes  "the dog sat on the mat"

guard: 8 of 9 candidates passed the negative network
rule     risk    peak   ratio  reason            vetoed
-----  ------  ------  ------  ----------------  --------------------------
blame  1.0000  1.0000  0.0917  mixed-up animals  "a bird flew over the mat"
```

The model was asked for nine texts rather than three, the sentence it had been
taught to hate never reached the answer, and the veto says which fragment it
was and who blamed it.  The same happens to a continuation (`predict` keeps
the survivors in `top`, and comes back with nothing at all when every one of
them is vetoed) and to a reply (`converse` leaves it unsaid and the voice
looks for another one; each turn counts its own `vetoed`).

Nothing is filtered silently and nothing is filtered for free:

* with **no negative model file** beside the model, or one that has **never
  been taught a failure**, the guard stands aside - the answer is exactly what
  it was before, and `guard` is `null`.  An empty negative network is never
  created just to guard an answer;
* `--no-guard` (CLI), `{"guard": false}` (API) or the *Filter with the negative
  network* checkbox (frontend) hands out what the positive model wrote;
* `--threshold`, `--min-coverage` and `--over-sample` tune it per command, and
  the Negative tab's settings move it for the server.

The loops that *teach* the negative network are deliberately outside the
guard: the critic (`negative auto`), the tutor and evolve all sample the
positive model directly, because a reviewer that only ever saw what already
passed the filter would have nothing left to teach.

The negative model is an ordinary model file (`model.negative.json` beside the
model, `--negative PATH` to move it) and an ordinary model kind, so
`--kind negative train` blames, `info`, `checkpoints`, `save` / `load` and the
model selector at the top of the frontend all work on it as usual.

## Counters that never overflow

Every number the model only ever counts up - traversals, node and edge visit
counts, the negative network's failure counts, epochs, trained characters and
texts, 2NRL runs, feedback passes, judgements, the internal version stamps -
would eventually run out of the integer holding it:
64 bits in the Go port, and long before that the 53 bits of mantissa in the
JSON number that carries it through a model file, the API and the browser. So
none of them is an unbounded integer. Each is a two-digit **odometer**:

```
total = resets × 1_000_000_000_000_000 + value        (0 ≤ value < 1_000_000_000_000_000)
```

The count goes up as before; the moment it reaches the limit it is **set back
to 0** and `resets` - how often that has happened - goes up by one. Nothing is
lost: the exact number of events is still there, split over two numbers that
each stay small. Cycles are a feature here too.

The limit is `10^15` because it is exactly representable as a double (so a
counter survives a model file, an API response and a JavaScript number
unchanged), and because it leaves four orders of magnitude of head room under a
64-bit integer - enough that a whole epoch of counting can land on a counter
before the next wrap. `resets` wraps at the same limit, so the odometer itself
comes full circle after `10^30` events.

What this changes in practice:

* **Nothing in the numbers.** Weights, shares, probabilities, losses,
  predictions and rankings are computed from the exact totals, so a model that
  has wrapped behaves exactly as if its counters had grown forever.
* **Counting stays as fast as it was.** Wrapping never happens in a counting
  loop: the loops (in Go, from one goroutine per text) add to a plain integer,
  and a sweep at the end of each epoch - skipped after a single comparison
  until a counter can actually have reached the limit - moves whatever crossed
  it into the resets.
* **Model files carry both numbers** (`format_version` 2). Files written by
  earlier versions load unchanged, with their counts wrapped on the way in.
  Python and Go read and write the same fields, so a wrapped model still moves
  between the two implementations unchanged.
* **The UI shows both**: the status bar reads `traversals 12,345 (+2 resets)`
  once a counter has gone round, and the Graph tab's node and edge tooltips and
  the Negative tab's failure counts do the same. `total_traversals_resets`,
  `epochs_total_resets`, `count_resets`, `fails_resets` and friends are in
  `/api/status` and `/api/graph` next to the values.

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

## The encoding: n-grams of any size, groups of letters, words

Three dials decide how a text becomes the grams the graph is built from, fixed
when a model is created and carried in its file.  **Both implementations have
them** (`python -m radixnet` and `go/bin/radixnet-count` take the same flags,
build the same graph and read each other's files):

| flag | what it sets | default |
|---|---|---|
| `--units char\|word` | what one unit of text is: a character, or a whitespace-delimited word | `char` |
| `--ngram N` | how many units one gram holds - the *n* of the n-gram, any number | 3 |
| `--stride N` | how far apart two consecutive grams start: **1** slides them (they overlap by n-1), **n** cuts the text into non-overlapping groups | 1 |

`--encoding SPEC` sets all three at once - `unit[:n[:stride]]`, plus the names
`trigram`, `bigram`, `word-bigram`, `word-trigram` and the shorthand
`:groups` for a stride equal to n:

```bash
python -m radixnet --model m.json --encoding char:3:1 train --data book.txt   # the default: trigrams
python -m radixnet --model m.json --encoding char:5:1 train --data book.txt   # a sliding window of five
python -m radixnet --model m.json --encoding char:4:4 train --data book.txt   # groups of four letters
python -m radixnet --model m.json --encoding char:5:groups train --data book.txt   # ... and of five
python -m radixnet --model m.json --encoding word:2:1 train --data book.txt   # word bigrams
python -m radixnet --model m.json --encoding word:3:1 train --data book.txt   # word trigrams

go/bin/radixnet-count --model m.json --encoding word:2:1 train --data book.txt   # the same dial in Go
go/bin/radixnet-count --model m.json predict --prefix "the cat sat" --k 5        # ... and the same file
```

The four Python model kinds all take it (`--kind radix | count | negative |
resonant`), and so do the library constructors:

```python
from radixnet import Encoding, WORDS, new_model

net = new_model("count", seed=1, encoding=Encoding(unit=WORDS, n=2))   # word bigrams
net.train(["the cat sat on the mat"], epochs=3)
net.predict("the cat", length=3)["top"][0]["text"]                     # whole words
```

Everything downstream is then measured in that unit rather than in characters:
a node's label, `--length` and `--max-length`, the `chars` of a score, and the
spans a correction blames (a word model's diff marks whole words). On a word
model, `predict --prefix "the cat sat"` walks whole words and `--length 3`
means three more words. `info` and `GET /api/status` say which encoding a model
is in; over HTTP, `POST /api/reset` takes `{"encoding": "word:2:1"}` or
`{"unit": "word", "ngram": 2, "stride": 1}` on **both** servers.

The encoding is fixed for the model's life - every label in the graph is
written in it - so the flags apply to a **new** model, and both CLIs refuse
them (rather than ignoring them) when they disagree with the model they loaded.
A model that is not `char:3:1` writes an `encoding` block into its file, and
**both implementations read it**: `tests/test_go_parity.py::TestGoEncodingParity`
trains the same corpus on both sides under nine encodings and holds them to the
same graph, the same file and the same predictions.

## Go implementation of the count / reward model

`go/` holds a Go port of the count / reward model (`CountRewardNet`) **and of
the negative network**, a standalone module with a library (`go/radixnet`) and
a CLI (`go/cmd/radixnet-count`); the Python implementation stays as it is.
Model files are interchangeable: both sides read and write the
`radixnet-count` and `radixnet-negative` JSON formats, including the Mersenne
Twister state, so a model trained on one side continues on the other with
identical numbers, in every encoding (`tests/test_go_parity.py` trains the same corpus on both,
compares structure, counts, rewards, window, RNG state, predictions, generated
texts, scores and conversations, blames the same failures and corrections and
compares the verdicts character for character, and lets each side read the
other's files).

```bash
make go-build                                   # -> go/bin/radixnet-count (needs Go 1.24+)
go/bin/radixnet-count --model model.count.json train --data data/sample_corpus.txt --epochs 5
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --k 5
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --traversal punishment   # the least punished way on
go/bin/radixnet-count --model model.count.json generate --mode beam --count 5
go/bin/radixnet-count --model model.count.json converse --opening "the cat sat on the mat"
go/bin/radixnet-count --model model.count.json chat --conversations 2 --topic animals   # an LLM talks to it and marks it
go/bin/radixnet-count --model model.count.json tutor --topic "everyday life" --rounds 3   # Ollama teaches it English
go/bin/radixnet-count --model model.count.json tutor --topic animals --rounds 3 --blame    # ... and blames what it marks down
go/bin/radixnet-count --model model.count.json train --data book.txt --split paragraphs --workers 8
go/bin/radixnet-count --model model.count.json negative why --text "the the the the cat"
go/bin/radixnet-count --model model.count.json negative filter --count 3   # the pair: write, then veto
go/bin/radixnet-count --model model.count.json generate --count 3          # ... and the same pair on every answer
go/bin/radixnet-count --model model.count.json negative auto --rounds 5 --blame   # Ollama reviews, the failures are blamed
go/bin/radixnet-count --model model.count.json codegen --problems problems.txt --phase teacher   # write programs, run them, learn
go/bin/radixnet-count tools call --tool calculator --arg 'expression=2*(3+4)'   # the tools the network can call
go/bin/radixnet-count --model model.count.json agent --tasks tasks.txt          # it browses, an LLM judges, 2NRL follows
go/bin/radixnet-count --model model.count.json explore --steps 0               # ... and picks its own tasks, until Ctrl-C
go/bin/radixnet-count --model model.count.json evolve --data data/sample_corpus.txt --generations 0   # until Ctrl-C
go/bin/radixnet-count --model model.count.json ollama review --count 8 --blame
go/bin/radixnet-count --model model.count.json speech teach clip.wav --text "the cat sat on the mat" --train
go/bin/radixnet-count --model model.count.json speech tutor clip.wav --length 400 --blame   # does it remember?
go/bin/radixnet-count --model model.count.json image encode photo.png --size 128 --train
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --traversal least-punished  # walk by the blame
go/bin/radixnet-count --seed 1 bench --chars 200000           # how fast this build counts and predicts
go/bin/radixnet-count --model five.json --ngram 5 --stride 5 train --data book.txt   # groups of five letters
go/bin/radixnet-count --model words.json --encoding word:2:1 train --data book.txt   # word bigrams
go/bin/radixnet-count --model words.json predict --prefix "the cat sat" --k 5        # ... predicted in words
python -m radixnet --model model.count.json info    # the Python side reads the same file
python -m radixnet negative why --text "..." --negative model.count.negative.json   # ... and the same negative one
```

Commands: `train`, `predict`, `generate`, `score`, `feedback`, `2nrl`, `correct`,
`negative` (`blame` | `clear` | `why` | `filter` | `reasons` | `forget` | `auto`),
`invert`, `compress`, `weights`, `info`, `converse`, `chat`, `tutor`, `evolve`,
`ollama` (`models` | `corpus` | `review`), `chatgpt` (`models` | `ask`),
`image` (`info` | `encode` | `tutor` | `decode`),
`speech` (`info` | `teach` | `tutor` | `decode`),
`checkpoints`, `bench`, `serve`, `version`; `--traversal reward|least-punished`
on `predict`, `generate` and `bench`; `--blame`
(with `--negative PATH`) on `tutor`, `correct`, `evolve` and `ollama review`; global options `--model`,
`--json`, `--seed`, `--workers N` (a cap on the goroutines; 0, the default, is
none), `--exact` (atomic counting), `--out`, `--memlimit SIZE` (soft heap
limit, 80 % of the machine or container by default), `--memprofile PATH`,
`--encoding SPEC` / `--units char|word` / `--ngram N` / `--stride N` (a **new**
model's encoding - see below).
Where the goroutines go:

| phase | concurrency |
|---|---|
| reading a corpus | `--split lines\|paragraphs\|pages\|file` decides what one text is (`--page-lines` cuts pages when a file has no form feeds); every text is one unit of work. Files and ZIP archives are **streamed**, never loaded whole: `--chunk N` (default 8192) texts at a time, at most `--inflight N` chunks (default two per CPU) in flight, so the reader runs ahead but memory does not grow with the corpus; `--parallel-parts` reads every archive entry at once instead of in order |
| encoding, tracing texts through the structure, counting | **one goroutine per text** of a chunk (no pool unless `--workers N`); the counters are bumped with plain increments from all of them at once - racy by design, a collision loses an update. `--exact` uses atomic increments instead: no lost updates, the result identical to the sequential run and to Python |
| building the structure | the one sequential phase: node splits reshape a shared radix index, and Go aborts the process on concurrent map writes, so this is not a race that can be ignored; texts that already walk through the graph are detected in parallel and skipped |
| the sliding window | applied in corpus order after each chunk's parallel pass (its semantics are the order of traversals); exact in both modes |
| weights and edge costs | recomputed lazily, only the touched rows after feedback; a full recompute after structural changes runs on a goroutine per 64 nodes |
| loss, scoring many texts | parallel reductions / one goroutine per text; the loss is the traversal-weighted mean edge cost, so it needs no list of transitions |
| prediction | the top and the bottom beam run side by side - in turn under `--workers 1`, so a single-worker run is single-threaded end to end and means the same thing as the Rust port's |

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
status bar, locks the model selector to the count model, and the Train tab
gains a **Texts are** selector (`lines | paragraphs | pages`) so the pasted
text and the uploaded files are cut into the units the goroutines fan out over.
Every tab works: both servers now run the lessons, the negative network and its
automatic loop, the evolve loop, the Ollama corpus and review, code generation,
tool use, the chat loop and the image and speech encoders. The Go side's images use a
thumbnail rather than the diffusion VAE, and its speech needs the words to come
with the audio - which is what the page dictates anyway.

| endpoint | Go server |
|---|---|
| `GET /api/health`, `GET /api/status`, `GET /api/model`, `POST /api/model/select` (count only), `POST /api/model/weights` | as the Python server, plus `engine`, `workers` (0 = one goroutine per text), `goroutines`, `counting` (`racy` \| `exact`), `heap_bytes`, `memory_limit_bytes` |
| `POST /api/train` | `{texts \| text \| files, whole_file, split: lines \| paragraphs \| pages \| file, page_lines, epochs, auto_compress, chunk_size, inflight, parallel_parts}` -> a job; uploads stream through in chunks whatever their size; learning rates are accepted and ignored |
| `GET /api/job`, `POST /api/job/stop` | one job at a time (409 while it runs); a job holds the model between epochs only, so predictions and the status poll keep answering |
| `POST /api/predict`, `/api/generate`, `/api/converse`, `/api/score` | same bodies and results as the Python count model, the guard included: all three run the pair by default and answer with the same `guard` report, and `{"guard": false}` turns it off |
| `POST /api/2nrl`, `POST /api/feedback` | jobs with `strength` (penalties, then traversal + reward); `good_ratings` / `bad_ratings` (marks out of 10) or `good_weights` / `bad_weights` scale the reward and the penalty per text |
| `GET /api/negative`, `POST /api/negative/blame`, `/clear`, `/judge`, `/filter`, `/forget`, `/settings`, `/reset`, `/save` | the negative network, same bodies and results as the Python server: the failures with the tutor's reasons, the verdicts with their blamed fragments, and the pair (the count model writes, the negative network vetoes by blame, peak or the likelihood ratio). Its file is `model.negative.json` beside the model path, interchangeable with Python's; `POST /api/save` writes it alongside the model |
| `GET /api/tutor`, `POST /api/tutor/start`, `GET /api/tutor/history`, `POST /api/tutor/lesson`, `GET /api/chatgpt/models` | the English lessons, same bodies and records as the Python server: the teacher (`tutor_provider`: a local Ollama model or ChatGPT) sets and marks the exercises, the count / reward model answers them (`serve --ollama-url / --ollama-model / --chatgpt-url / --chatgpt-model` set the defaults, the key is the server's own `$OPENAI_API_KEY`). With `blame` every failed sentence also teaches the negative network what the teacher marked it down for |
| `POST /api/invert`, `/api/compress`, `/api/save`, `/api/load`, `/api/reset` | as the Python server (reset / load of another kind is refused); `reset` also takes the encoding of the fresh model - `{"encoding": "word:2:1"}`, or `{"unit", "ngram", "stride"}` |
| `GET /api/checkpoints`, `POST /api/checkpoints/save`, `POST /api/checkpoints/restore` | the Python `CheckpointManager` layout (`ckpt-<tag>-<step>.json.gz`, `latest.json`, `index.json`), so both servers can share a directory |
| `GET /api/uploads`, `POST /api/uploads` (JSON, multipart, raw), `POST /api/uploads/delete` | text files and ZIP archives of any size: multipart and raw bodies stream to disk, archives are inspected and read entry by entry with the same rules as the Python module |
| `GET /api/graph`, `GET /api/history`, `GET /api/paths`, `GET /api/nodes` | as the Python server (edges carry `reward`, `share`, `recent_share`, `recent_count`; the judged paths and the node ratios come back with the same counters, the same shares and the same order) |
| `POST /api/evolve/start`, `POST /api/evolve/stop`, `GET /api/evolve/history` | the self-upgrade loop, same bodies and records as the Python server: the model generates, a discriminator judges, 2NRL follows; `blatant_mode` picks how failures drive the update and `blame` lets the critic teach the negative network. The discriminator lives beside the model as `discriminator.json` |
| `GET /api/ollama/models`, `POST /api/ollama/corpus`, `POST /api/ollama/review` | a corpus written to order (`train` starts a job on the lines) and the adversarial review, which with `blame` teaches the negative network what failed and why |
| `POST /api/negative/auto`, `GET /api/negative/auto/history` | the Negative tab, automatic: a `critic` job of write → review → blame, same bodies and records as the Python server |
| `GET /api/images`, `POST /api/images/encode`, `/decode`, `/tutor` | images as text, same bodies and results as the Python server. The **thumbnail encoder only**: the Stable Diffusion one needs torch and diffusers, so `encoder: "sd"` is refused here with a message naming the Python side |
| `GET /api/speech`, `POST /api/speech/teach`, `/decode`, `/tutor` | the waveform as text and the recall tutor over it, same bodies and results as the Python server. **Transcription is Python-only** (faster-whisper / openai-whisper are Python packages), so send the words with the audio - which is what the browser's dictation does |
| `POST /api/codegen/start`, `GET /api/codegen/history`, `POST /api/codegen/solve`, `POST /api/codegen/run` | code generation, same bodies and results as the Python server: the teacher writes, the sandbox runs, the judge decides and 2NRL follows. The sandbox is the same Python bootstrap both languages run, so a program sees the same interpreter, the same limits and the same isolation whichever server started it. The count model pushes by `strength` rather than `neg_lr` / `pos_lr` / `batch_size` |
| `GET /api/tools`, `POST /api/tools/call` | the external tools the network can call by writing `<tool>name {...}</tool>`, and one direct call: the same names, parameters, descriptions and JSON schemas as the Python server, so a transcript written on one side is one the other reads. Browsing is the standard library either way, with the same guards (http / https only, no credentials, no private address unless allowed, a byte cap, redirects followed by hand) |
| `POST /api/agent/start`, `/api/agent/explore`, `GET /api/agent/history`, `POST /api/agent/criteria`, `/api/agent/solve` | tool use: the LLM writes the acceptance criteria, mediates what the network could not write itself, judges the answer and demonstrates when it failed; 2NRL follows. Same bodies and records as the Python server; the count model pushes by `strength` rather than `neg_lr` / `pos_lr` |
| `POST /api/chat/start`, `GET /api/chat/history` | an LLM converses with the model and marks every reply: the partner says a short line, the network replies by continuing it, the judge marks each reply against the line it answered and the conversation as a whole, the failures blame the negative network, the passes clear it, and 2NRL trains the model on both. Same bodies and records as the Python server; the count model pushes by `strength` rather than `neg_lr` / `pos_lr` / `batch_size` |
| `/api/schedule/preview` | 404 with a message naming the Python server - the only endpoint that is still Python-only |

`tests/test_go_parity.py::TestGoTutorParity` points both tutors at one fake
Ollama and asserts that they send the teacher the same prompts, get the same
marks and leave the model in the same state, so the two implementations of the
lessons cannot drift apart.  `TestGoCodeGenParity` does the same for code
generation: one fake teacher, both trainers, the same conversation prompt for
prompt, the same solution and the same blame on disk.  `TestGoToolsParity`
holds the two tool sets to the same signatures, the same JSON schemas and the
same answers - including the calculator's, down to `29.0` rather than `29` -
because the network learns the characters of a call and its result, so a
transcript written on one side has to be one the other can read.
`TestGoChatParity` points both chat loops at one fake partner and judge: the
same lines are said, the same replies come back, the same marks are given and
the same two networks are on disk afterwards.

`tests/test_go_parity.py` also starts the Go server and checks its answers
against the key sets the Python API tests assert on, loads the model it saves
in Python, trains from a ZIP upload with `split: paragraphs`, and reads its
checkpoints with the Python `CheckpointManager`.

## Rust implementation of the whole package, and the two ports measured against each other

`rust/` is a third implementation - a standalone crate with no dependencies -
written first to find out how much of what this model costs is the model and
how much is the language, and to build the least-punished traversal beside the
Go one, and grown since into a port of the whole package: every model kind,
the negative network, every teaching loop, the LLM clients, the tools, code
generation and the agent, images and speech, MCP and the WebDriver browser.
It reads and writes every model file byte
for byte as Python does, and `radixnet serve` answers the same JSON API the
Python and Go servers answer, so `frontend/dist` runs against it unmodified -
a tab appears when the route it needs is in `/api/status`.  `rust/README.md`
lists every module and the three things deliberately not ported (the torch
backend, the Stable Diffusion encoder, local Whisper).

```bash
make rust-build        # -> rust/target/release/radixnet{,-bench} (needs Rust 1.82+)
make rust-test         # cargo test, clippy, fmt --check
make rust-serve        # frontend/dist against the Rust model on http://HOST:PORT
make bench-compare     # both ports over one corpus -> bench/RESULTS.md
```

`bench/compare.py` hands both builds the same texts and the same prefixes (one
Python generator writes them, because neither language can reproduce the
other's RNG), trains both for the same epochs, punishes every Nth text so the
least-punished traversal has blame to walk by, and **checks the two against each
other before it reports a single timing**: same nodes, edges, trigrams,
transitions and expansions; same compression ratio, loss and total penalty; same
prediction at the same cost, to the bit.  A speed comparison between two
programs that computed different things is not a comparison.

On a 4-core Xeon, 2,000,000 characters, 3 epochs, 3,000 predictions, Go counting
with `--exact` (its racy default is within a few percent, and a benchmark of a
deliberate data race measures the race):

| | Go, one worker | Rust, one worker | Go, all cores | Rust, all cores |
|---|--:|--:|--:|--:|
| training | 5.9M transitions/s | **14.3M** | 7.1M | **20.6M** |
| prediction, by reward | 4.3k/s | **16.1k** | 3.7k | **16.1k** |
| prediction, least punished | 41k/s | **275k** | 41k | **246k** |

2.2-2.8x at counting and 3.8-6.2x at predicting against Go's own default
counting, more against `--exact`, on identical work.  Three
representation choices carry most of it and none is algorithmic - a trigram is a
packed integer rather than a fresh string, a node's children are read into a
buffer the search reuses, and the trigram index hashes with a cheap
non-cryptographic hash - which is why `rust/README.md` lists them beside the
table.  The full matrix, with the machine and the toolchain versions it was
measured on, is `bench/RESULTS.md`.

The other column in that table is the traversal rather than the language: the
least-punished search expands **15x fewer nodes** than the ordinary one on the
same model, because refusing a blamed step at the node prunes the beam - and it
disagrees with it on about **20%** of the continuations.  That is an effort
result, not a quality result; which answers are better is the tutor's question.

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

The traversal, on any kind:

```python
from radixnet import CountRewardNet, TRAVERSALS   # ("reward", "punishment", "least-punished")

net = CountRewardNet(seed=1)
net.train(["the cat sat on the mat", "the cat ate the rat", "the cat ran up the hill"], epochs=4)
net.reward(["the cat sat on the mat"], strength=4)   # praise one branch
net.punish(["the cat ate the rat"], strength=4)      # correct another

print(net.predict("the cat", length=16).text)                              # " sat on the mat"
print(net.predict("the cat", length=16, traversal="punishment",
                  merit_scale=0).text)                                     # " on the mat"
print(net.graph.child_evidence(8))    # [(child, edge, merit, penalty), ...] - the split it walks on
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

```python
from radixnet import blame, recall            # does it remember what it was taught?

waveform = [t for t in spoken["texts"] if "aud:" in t]
lessons = recall.quiz(net, waveform, length=400)          # the exercise is the token and the header
print(recall.report_card(lessons))                        # {"passed": 0, "mean_score": 4.3, "reasons": {"truncated": 1}, ...}
blame.teach_recall(negative, lessons, source="speech")    # the original is the correction: only the wrong characters are blamed
```

## Tests

```bash
make test           # python -m unittest discover -s tests -v (includes the Go parity test when `go` is on PATH)
make go-test        # cd go && go test -race ./...
make frontend-test  # cd frontend && npm test (node --test over the settings store; no dependencies)
```

## Layout

```
RadixCyclicNN/
  radixnet/           activation, counter, encoding, graph, backend(+torch), search, beam, phasesearch, penalty, model,
                      countnet, negative, resonance, metacog, blame, duo, diff, schedule, gan, checkpoint, bench,
                      cli, api, llm, ollama, chatgpt, tutor, recall, critic, codegen, tools, agent, browser, mcp,
                      vision, speech, dialogue, chat
  tests/              unittest suite
  frontend/           Vite + React app (dist/ is prebuilt and served by the API; src/storage.js remembers
                      the panels' settings in localStorage, test/ holds its node --test suite)
  go/                 Go port of the count / reward model and the negative network: radixnet/ (library), cmd/radixnet-count (CLI)
  rust/               Rust port of the whole package: src/ (crate, HTTP server and every area included), src/bin
                      (the CLI and the benchmark), tests/ (the model, the encodings, the word alphabet and the
                      server end to end)
  bench/              the two ports over one corpus: make_corpus.py, compare.py, RESULTS.md
  data/               sample_corpus.txt (correct data), sample_garbage.txt (bad data),
                      sample_problems.* (codegen), sample_tasks.* (agent / explore)
  docker/             container entrypoint (optional checkpoint resume)
  Dockerfile, docker-compose.yml, docker-compose.gpu.yml, .env.example, Makefile
  DESIGN.md           the specification
  SPEC-LeastPunished.md   the traversal that follows the blame (built)
  SPEC-EdgeDecay.md   a node's edges fading on the graph's own clock (proposed)
```

## Design decisions

* **`N*N`** is read as the node-to-node weight matrix: N nodes, N×N possible edges, stored sparsely as adjacency lists and exported as CSR for the backends.
* **"activation of the child × activation of the parent"** is the edge signal `W[p,c] · f_c(z_c) · f_p(z_p)`; the softmax over a parent's signals is the next-node distribution, and `-log` of it is the Dijkstra cost.
* **"accept the vanishing gradient, update the activation function instead"** means no back-propagation through depth. Every observed transition applies a one-hop gradient to the edge weight, the two node states and the four sine parameters of the parent and children, so the activation functions carry the learning.
* **"invert the network"** (2NRL) flips the sign of every edge weight and of every node's activation - amplitude `a` *and* offset `k`, since `f = a·sin(b(x−h)) + k` and flipping `a` alone leaves `−f + 2k`, which negates the unit only while `k` is 0 and `k` is learned. That negates every edge signal: the most likely continuation becomes the least likely. Two inversions are the identity.
* **Prediction prefers short, confident completions** because the cost is summed per edge; `--step-penalty` and `--length` / `--to-end` steer that, and `--mode sample` gives diverse output for the GAN loop.
* **The traversal is a cost function, not a search.** "Traverse by the punishments" could have been a fifth mode beside dijkstra / beam / kbest / sample, and would then have had to be written four times over and once more for the phase-unrolled graph. Every search already reads the graph through one funnel - `[(child, edge, cost)]` for a node - so the option replaces the funnel instead: the searches are untouched, every mode of every kind gains the traversal at once, and the costs stay `-log softmax` over the parent's children, which is what keeps Dijkstra exact and `exp(-cost)` a probability.
* **Self-compression is lossy on purpose**: merging a unary chain keeps the parent's parameters; the chain was deterministic (probability 1, cost 0), so predictions are unchanged.
* **The resonant model's phase is defined per trigram, not per node**, so a node's advance is the sum over the trigrams its label covers. A split and a merge move trigrams between labels but never change which trigrams exist, so compression leaves the phase exactly where it was - and the phase of any text is a function of the text alone, no walk needed.
* **A phase-locked cycle is the only cycle worth a decision**: returning to a node at a new phase is progress, returning at the same phase repeats for ever. That is what the metacognitive layer is asked about, and its answer is a cost, never a prohibition.
* **Metacognition and exactness are not a trade — the number of labels per state is.** One label per state makes a shortest path and makes it blind; K labels per state give the K cheapest walks exactly *and* give every label a path to look at. Dijkstra is the K=1 case of the search that replaced it, not a different algorithm.
* **Counters cycle rather than grow**: an integer that only counts up is a fault waiting to happen, so every one of them goes back to 0 at `10^15` and counts the reset. The pair is exact, both halves stay inside a double, and the wrapping is done by a sweep between epochs instead of a check on every increment - so the counting loops (and the Go port's goroutines) are untouched.

## License

See the `LICENSE` file at the repository root. Source-available, all rights reserved.
