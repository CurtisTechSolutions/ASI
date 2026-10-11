# ASI

## READ THE LICENSE!!!

You automatically agree to the license if you visit this repository, or to whatever extent is legally allowable in a court of law.

The license is to protect my IP. This is source-available and to prove I'm the original inventor and author of these solutions. For more information, view the [LICENSE](LICENSE)

## The gist

I put in months of continuous extreme effort to research AGI/ASI, and this is the culmination of that work. Overall, I solved multiple different variations of AGI/ASI, all derived from my research on the human brain, and potential variations to the design of the human brain. This research diverges from the norm significantly, in that I refuse all current research and reinvent the wheel.

All rights reserved; all research is 100% my own, and none of this repo is allowed to be used for profit, unless used by my own company or myself [My company (CTS)](https://www.curtistechsolutions.com)

## Details

It all started when I was frustrated with the current AGI/LLM hype. I was building a company/product called Amy AI (Executive Email Assistant). I grew extremely frustrated with how much AI was parroted as the "Next big thing", and "AI *this*", "AI *that*". I was looking for hard problems to solve at the time, so I said: "screw it, I'll do it myself!". So, that's what I did! I reverse-engineered my entire brain, from memory to executive function, delved into neuroscience, and more. Overall, I have many pages of notes, scribbles, diagrams, methodologies, and more. I've also optimized the training process with my own proprietary custom training algorithms.

## Repository guide

ASI contains several related model architectures, reusable tools, and experiments.
Each subproject has its own entry point and documentation; there is no single
application or repository-wide install command. Some directories contain runnable
implementations, while others preserve early ideas or specifications awaiting code.

- [Quick start](#quick-start)
- [Project index](#project-index)
- [Models and architectures](#models-and-architectures)
- [Shared packages and tools](#shared-packages-and-tools)
- [Experiments](#experiments)
- [Research](#research)
- [Early variations](#early-variations)
- [Working on a subproject](#working-on-a-subproject)

Start with **RadixCyclicNN** for the cyclic text model, **ModelKit** for its
CLI and application layer, **CyclicCortex** or **GREN** for game learning,
and **DISTIL** for the agent that learns and verifies tools. The experiments
and research documents explain the ideas behind those implementations and the
measurements used to evaluate them.

## Quick start

Use **Python 3.11 or newer** for the packaged Python projects. The following
text-model example runs from a full checkout with the standard library alone:

```bash
git clone https://github.com/CurtisTechSolutions/ASI.git
cd ASI/RadixCyclicNN

python3 -m radixnet train --data data/sample_corpus.txt --epochs 5 --lr 0.5 --batch-size 4
python3 -m radixnet predict --prefix "the quick brown" --length 20
python3 -m radixnet score --text "the cat sat on the mat"
```

Training writes `model.json` in the current directory. Later commands load it;
running training again continues from that model. `python3 -m radixnet --help`
lists the commands, and `python3 -m radixnet serve` starts the API and frontend.
The CLI finds the sibling `ModelKit/` checkout automatically.

The project-guide commands below start at the **repository root**. They use
Bash subshells so each command returns to the same directory; on Windows, use
WSL or enter the named directory and run its command separately. Targets using
`make` require Make and the shell used by that project's Makefile.

Keep the sibling directories together: ModelKit's model layer uses RadixCyclicNN,
phonetic encodings use PhoneticTokenizer, and GREN's game oracles use CyclicCortex.
External frameworks, engines, services, and build tools are noted per project.

## Project index

This index covers every top-level project and support directory. The six
subprojects inside `Experiments/` are listed individually [below](#experiments).
“Implementation” means code is present; the linked project README describes its
current behavior, results, and limitations.

| Directory | Purpose | Contents / status |
|---|---|---|
| [AbsorbingMarkovChain](#absorbingmarkovchain) | Learn a transition matrix with absorbing stopping states | Idea and empty code stub |
| [AudioImage](#audioimage) | Encode sound as a spectrogram image and decode it back | Python package, CLI, browser UI, tests |
| [CyclicCortex](#cycliccortex) | Route related games to regions with growing neural networks | Python implementation, game engines, CLI, tests |
| [DISTIL](#distil) | Decompose goals, grade memory, and build verified tools | Python agent, API, React UI, tests |
| [Experiments](#experiments) | Measure activations, growth, compression, inversion, and encodings | Six experiment subprojects |
| [FilterBankRadix](#filterbankradix) | Route text through an activation filter into radix-tree experts | Python implementation, comparison runs, tests |
| [GREN](#gren) | Discover a game's rules from refused actions | Python implementation, CLI, tests |
| [GTMNN](#gtmnn) | Combine populations of micro-networks through game theory | Design specification; no implementation yet |
| [ModelKit](#modelkit) | Reusable teaching, agent, serving, and interface layer | Python, Go, Rust, and React components |
| [PhoneticTokenizer](#phonetictokenizer) | Turn words and recordings into sound-based tokens | Python, Go, and Rust implementations |
| [RadixAcyclicNN](#radixacyclicnn) | Compare the cyclic model with a context tree without cycles | Python model, CLI, comparisons, tests |
| [RadixCyclicNN](#radixcyclicnn) | Learn and search a self-compressing cyclic graph | Python, Go, and Rust model implementations |
| [RadixDecayNN](#radixdecaynn) | Study memory reinforced by use and weakened by elapsed traversals | Python model, CLI, experiments, tests |
| [RadixTreeLLM](#radixtreellm) | Record the original radix-tree compression direction | README-only concept |
| [RadixTreeRNN](#radixtreernn) | Attach an RNN to each radix-tree node | Idea and empty code stub |
| [RadixTrieLLM_RNN](#radixtriellm_rnn) | Attach a small transformer to each radix-trie node | PyTorch prototype and demo |
| [Research](#research) | Explain the hypotheses, mathematics, and evidence | Four papers and an insights index |
| [TwoNRL_Chess](#twonrl_chess) | Test failure training and inversion on chess | NumPy experiment, Stockfish integration, results, visualization |
| [tools](#tools) | Build reproducible text corpora | Gutenberg and Wikipedia downloaders, offline tests |
| [utils](#utils) | Small shared helpers | Standard-library LRU cache |

## Models and architectures

### RadixCyclicNN

The main implementation of the radix-inspired cyclic graph. Repeated n-grams
share nodes, unary paths compress, and later observations can split them again.
The learned `radix` model uses per-node sine activations and a local one-hop
update rule. Other model kinds use traversal counts and rewards (`count`), word
n-grams (`word`), or phase and coherence (`resonant`). A separate negative model
records failed paths and their reasons.

The model supports training, prediction, generation, scoring, persistence,
multiple search methods, and character, word, phonetic, and acoustic encodings.
Python lives in `radixnet/`; `go/` contains the Go count/reward and negative
models, and `rust/` contains the Rust model port. The CLI, servers, teaching
loops, and frontend live in **ModelKit**, with compatibility entry points here.

**Start:** `(cd RadixCyclicNN && make demo)`.
**Needs:** Python 3.11+; PyTorch is optional for acceleration. Go and Rust are
needed only for their respective ports; sound encodings use PhoneticTokenizer.
See the [manual](RadixCyclicNN/README.md), [design](RadixCyclicNN/DESIGN.md),
and [decision record](RadixCyclicNN/DECISIONS.md).

### RadixAcyclicNN

A control for the cyclic architecture: a node represents a context, and every
edge moves deeper into a rooted radix tree. It retains sine activations, local
learning, compressed paths, shortest-path prediction, and 2NRL inversion, while
removing cycles from the stored structure. The project compares the resulting
memory, repetition, and prediction behavior with RadixCyclicNN on shared corpora.

`radixtree/` contains the model and CLI; `data/`, `results/`, and `tests/`
contain corpora, comparison artifacts, and checks of the tree and learning rule.

**Start:** `(cd RadixAcyclicNN && make demo)`; `make compare-quick` from that
directory runs the small cyclic-versus-acyclic comparison.
**Needs:** Python standard library; comparisons need the sibling RadixCyclicNN
checkout, and phonetic encodings need PhoneticTokenizer.
See the [README](RadixAcyclicNN/README.md) and [design](RadixAcyclicNN/DESIGN.md).

### RadixDecayNN

A radix context tree whose memory is a decaying `seen` count instead of learned
weights or activations. Reading and ordinary generation both reinforce visited
nodes; time advances in traversals rather than wall-clock seconds. Half-life,
linear, and no-decay modes let the same implementation measure habit formation,
forgetting, retention through use, and recovery through rereading.

`radixdecay/` supplies the model, CLI, and search. Generation changes memory;
`--quiet` queries and scoring avoid that reinforcement. Saved models include
the traversal clock and memory state.

**Start:** `(cd RadixDecayNN && make demo)`.
**Needs:** Python standard library; phonetic modes use PhoneticTokenizer.
See the [README and experiments](RadixDecayNN/README.md) and
[design](RadixDecayNN/DESIGN.md).

### FilterBankRadix

A two-stage text model: an activation filter chooses an address, and the radix
tree at that address predicts the next character. The filter learns from how
well each expert compresses a text segment. It passes an address to the trees,
with no gradient propagated back through the tree into the filter.

`fbradix/` contains the filter, expert bank, model persistence, and CLI. The
experiments compare learned routing with a single tree, frozen and random
routers, and other controls across context depths and corpus sizes. The results
include cases where the single tree wins as well as cases where routing helps.

**Start:** `(cd FilterBankRadix && make demo)`; `make quick` runs reduced sweeps.
**Needs:** Python standard library.
See the [README](FilterBankRadix/README.md), [design](FilterBankRadix/DESIGN.md),
and [results guide](FilterBankRadix/results/README.md).

### CyclicCortex

A graph of games grouped by similarity in their mechanics. Each region owns a
self-building neural network and a shared vocabulary of generalized actions.
Chess, checkers, Go, and Sudoku adapters turn game states and candidate actions
into features for legality and quality prediction.

`cortex/` includes the game rules, built-in opponents, growing networks,
checkpointing, region routing, auction and Shapley credit mechanisms, self-play,
and teacher distillation. The experiments distinguish learning legal moves from
playing strong moves. Game mechanics are currently supplied by the adapters;
GREN's discovered packages are not yet wired into this player.

**Start:** `(cd CyclicCortex && python3 -m cortex.cli map)` for the regions,
or `python3 -m cortex.cli demo` from that directory for training and play.
**Needs:** Python standard library. Stockfish is optional as a chess opponent
and teacher; built-in opponents are available without it.
See the [README](CyclicCortex/README.md) and [design](CyclicCortex/DESIGN.md).

### GREN

**Game Rule Encoder Network** identifies games by probing actions and learning
from refusals. It compares random, boundary, and information-gain probe
policies, records structured refusal codes, and builds signatures that can be
compared or retrieved through a compressed radix tree.

`gren/` includes oracles for CyclicCortex's four games, probe policies, evidence
collection, a growing network that adds inputs and refusal classes, and the
`GamePackage` output. This implementation discovers refusal patterns; the
broader design's operator induction and generated-code sandbox are still
unimplemented. Packages are produced, but CyclicCortex does not yet consume them.

**Start:** `(cd GREN && python3 -m gren.cli explore)`; `similar`, `tree`,
`package`, and `grow` show the other stages.
**Needs:** Python standard library and the sibling CyclicCortex checkout.
See the [README](GREN/README.md) and [design](GREN/DESIGN.md).

### GTMNN

**Game Theory + Micro Neural Networks** is the proposed player counterpart to
GREN. Its design combines a population of small networks through games,
auctions, equilibrium selection, and Shapley credit, with evolutionary
replacement and a second population for metacognition. The intended separation
is for GREN to describe a game and GTMNN to play it.

**Status:** specification only. The directory contains `README.md` and
`DESIGN.md`, with no runnable package, CLI, or tests. The Python package,
optional acceleration, API, and frontend described there are planned components.
See the [overview](GTMNN/README.md) and [full specification](GTMNN/DESIGN.md).

### DISTIL

**Distillation Into Subgoals, with Tool Induction and Learning** is an agent
that breaks tasks into verifiable goals, retrieves memory using credibility as
well as similarity, reasons over choices, and writes Python tools that must
pass checks before reuse. It also includes a casebook, exploration, memory
compression, goal tracking, and self-editing workflows.

`distil/` contains the agent, providers, memory and storage layers, toolsmith,
execution checks, MCP integration, CLI, and HTTP server. `ui/` holds the React
frontend source. The deterministic local provider supports an offline demo;
Ollama, OpenAI, and Anthropic providers allow model-backed runs.

**Start:** `(cd DISTIL && make demo)`; `make ui` serves the interface.
**Needs:** Python standard library for the local runtime; external providers
need their service/configuration, and rebuilding the frontend needs Node.js/npm.
See the [README](DISTIL/README.md) and [design](DISTIL/DESIGN.md).

## Shared packages and tools

### ModelKit

The application layer extracted from RadixCyclicNN so it can be reused. Its
model-free modules provide LLM clients, agent tools, browser access, MCP,
archive handling, speech/image utilities, and teaching interfaces. Its model
layer adds tutoring, agents, dialogue, negative-output filtering, evolution,
checkpoints, benchmarks, the CLI, and the HTTP API around `radixnet`.

There are Python modules in `modelkit/`, Go components in `go/`, Rust components
in `rust/`, and a React app in `frontend/`. Dependency direction runs from the
kit to the model; the model's core works independently. The `radixnet` CLI
entry point delegates to this kit.

**Start:** `(cd ModelKit && python3 -m modelkit --help)`.
**Needs:** Python 3.11+; the model layer needs RadixCyclicNN installed or beside
it. Media extras and external LLM/browser services depend on the feature used.
Go, Cargo, and Node.js/npm are only needed for the respective builds.
See the [README](ModelKit/README.md), [Python package guide](ModelKit/modelkit/README.md),
and [frontend guide](ModelKit/frontend/README.md).

### PhoneticTokenizer

Converts text into phonemes, syllable constituents, syllables, or word-level
sound tokens. It also supports decoding to spelling, pronunciation inspection,
phonotactic scoring, word blending, and speech synthesis. Its acoustic tokenizer
learns a codebook from recordings and maps WAV audio to discrete sound units
without requiring transcripts.

Python (`phonetok/`), Go (`go/`), and Rust (`rust/`) share the token formats and
data. RadixCyclicNN and the tree variants use this package for sound-based
encodings. A core pronunciation lexicon is bundled; the full CMU dictionary
can be supplied separately.

**Start:** `(cd PhoneticTokenizer && python3 -m phonetok tokenize "The cat sat.")`.
**Needs:** Python 3.11+ for the Python package, with no required third-party
runtime dependencies; `cmudict` is optional. Go/Cargo build the other ports.
See the [README](PhoneticTokenizer/README.md) and [design](PhoneticTokenizer/DESIGN.md).

### AudioImage

Converts WAV audio into a PNG spectrogram with time across the image, frequency
vertically, and magnitude encoded by pixel intensity. It can synthesize example
sounds, render annotated views, decode images back into WAV files, and report
round-trip measurements. Magnitude-only images reconstruct phase during
decoding; an optional RGB format stores phase explicitly.

`audioimage/` contains WAV/PNG handling, transforms, codecs, rendering, a CLI,
and a browser server with its frontend. `data/` supplies a sample recording,
and `tests/` covers the formats and processing stages.

**Start:** `(cd AudioImage && make demo)`.
**Needs:** Python 3.11+ and the standard library. NumPy is an optional faster
backend, detected when available.
See the [README](AudioImage/README.md) and [design](AudioImage/DESIGN.md).

### tools

Shared corpus downloaders rather than a model. `gutenberg_download.py` selects
books by catalog metadata, supports filtering and listing, and downloads plain
text. `wikipedia_download.py` supports resumable official dumps, text extraction,
and selected article retrieval. Both write manifests with source information and
checksums; `common.py` provides their HTTP, filename, and atomic-write helpers.

**Start:** `python3 tools/gutenberg_download.py --help` and
`python3 tools/wikipedia_download.py --help` from the repository root.
**Needs:** Python 3.10+ and the standard library; downloads need network access.
See the [commands and offline test instructions](tools/README.md).

### utils

Small shared helpers. Currently `lru_cache.py` provides an `OrderedDict`-based
least-recently-used cache with `get`, `put`, and capacity-based eviction. A cache
miss returns `-1`. The included dump/load helpers do not currently round-trip
correctly; the directory README explains that limitation.

**Use:** `from utils.lru_cache import LRUCache` from the repository root.
**Needs:** Python standard library.
See the [README](utils/README.md).

## Experiments

The [Experiments index](Experiments/README.md) collects six separate projects.
Each has its own dependencies and run instructions. Results belong to the
specific tasks, controls, and configurations recorded in those directories;
the root guide links to the evidence instead of duplicating changing tables.
The chess study lives separately at the repository root and is covered here too.

### ActivationFunctionTest

Compares sigmoid, ReLU, and sine activations in CartPole scripts, and includes a
separate self-building-network study. `self_building_sinewave.py` compares five
activation variants on chirp, spline, and step regression tasks, tracking error,
growth, and inactive units, with finite-difference gradient checks.

**Start:** `(cd Experiments/ActivationFunctionTest && python3 self_building_sinewave.py --quick)`.
**Needs:** only the standard library for that script. The earlier CartPole
scripts use Gymnasium with TensorFlow/NumPy or PyTorch/Matplotlib, depending on
the script. See the [README and measured limits](Experiments/ActivationFunctionTest/README.md).

### DepthCountedNormalisation

Tests whether gradient magnitude decay can be measured as a function of network
depth and compensated for. `depth_counted_normalisation.py` fits the decay in
log space, then compares normalization based on that estimate with baselines.
It is the experiment linked to the vanishing-gradient research paper.

**Start:** `(cd Experiments/DepthCountedNormalisation && python3 depth_counted_normalisation.py --quick)`.
**Needs:** Python standard library.
See the [README](Experiments/DepthCountedNormalisation/README.md) and
[paper](Research/VanishingGradientIsAFeature.md).

### NeuralCompression

Tests whether one network can represent many synthetic games, whether output
partitions help, and how similarity ordering affects transfer. Additional
scripts examine adding games, inversion around stalled learning, and moving
specialist knowledge into a shared network through joint training, grafting,
or replay. The findings include controls and negative results, including cases
where simple conditioning beats the partition scheme.

**Start:** `(cd Experiments/NeuralCompression && python3 experiment.py)`.
**Needs:** Python standard library. `partition.py`, `vanishing.py`, and
`consolidate.py` run the other studies.
See the [README](Experiments/NeuralCompression/README.md) and
[findings](Experiments/NeuralCompression/FINDINGS.md).

### SBNN_RNN_ActivationFunction

A compact **Self-Building Neural Network** example: a NumPy RNN with a learned
sine activation and a hidden layer that expands when loss improvement stalls.
`main.py` contains the activation, derivative checks, recurrent network, growth
logic, and a next-value prediction demo on sine-wave sequences.

**Start:** `(cd Experiments/SBNN_RNN_ActivationFunction && python3 main.py)`.
**Needs:** NumPy. Growth changes the existing computation and is not fully
reproducible; the project documents those prototype limitations.
See the [README](Experiments/SBNN_RNN_ActivationFunction/README.md).

### TwoNRL_CartPole

Tests **Double-Negative Reinforcement Learning (2NRL)** on CartPole: first train
a sine network to fail, invert it, then fine-tune on the positive task. It
includes a matched control, ablations of the failure weighting and exploration
rules, gradient/inversion checks, saved results, and table-generation tools.
The experiment separates what the sign flip achieves from later training.

**Start:** `(cd Experiments/TwoNRL_CartPole && make quick)` after installing
NumPy and Gymnasium (`make install` in that directory).
**Needs:** NumPy and Gymnasium.
See the [README](Experiments/TwoNRL_CartPole/README.md) and
[results guide](Experiments/TwoNRL_CartPole/results/README.md).

### UniversalGameEncoding

Encodes chess, Othello, Connect Four, tic-tac-toe, Nim, and Pig onto a common
tape for RadixCyclicNN. A decoder/referee checks generated moves. The study
varies how states and actions are spelled to measure effects on legality,
refusals, and move quality, including controls for 2NRL. An external Hexapawn
example shows how to add another game.

**Start:** `(cd Experiments/UniversalGameEncoding && make quick)`.
**Needs:** Python standard library and the sibling RadixCyclicNN checkout;
the chess rules are included, so this experiment needs neither Stockfish nor
python-chess.
See the [README](Experiments/UniversalGameEncoding/README.md),
[usage guide](Experiments/UniversalGameEncoding/USING.md), and
[encoding design](Experiments/UniversalGameEncoding/ENCODING.md).

### TwoNRL_Chess

The chess study uses a growing sine network to rank from-to moves, learns from
refusals, and uses Stockfish as both opponent and move-quality teacher. It
compares 2NRL with positive training at matched compute and includes a separate
rule curriculum with legality, movement, capture, check, safety, and threat
heads alongside quality prediction.

The directory includes training and benchmark scripts, held-out evaluation,
inversion/growth checks, saved runs, and `viz/` for a self-contained HTML view
of the network before and after inversion. Its README explains why refusal
handling and ranking can affect apparent playing strength.

**Start:** `(cd TwoNRL_Chess && make quick)` after installing NumPy, the `chess`
Python package, and a Stockfish executable. The project's `make install` also
attempts an apt-based Stockfish install on systems where it is missing.
**Needs:** NumPy, python-chess, and Stockfish for the full experiment.
See the [README](TwoNRL_Chess/README.md) and [visualization guide](TwoNRL_Chess/viz/README.md).

## Research

[`Research/`](Research/README.md) contains the arguments and mathematical
write-ups behind the implementations, together with links to the experiments
that support, refine, or contradict their claims.

| Document | Focus |
|---|---|
| [Cycles Are a Feature](Research/CyclesAreAFeature.md) | Cyclic graphs, repetition, compression, and metacognition |
| [Sine Wave Activation Function](Research/SineWaveActivationFunction.md) | The learned sine activation, its derivatives, and comparison with other activations |
| [Vanishing Gradient Is a Feature](Research/VanishingGradientIsAFeature.md) | Depth as information for measuring and compensating gradient decay |
| [2NRL](Research/2NRL.md) | Training on consistent failure, inversion, and positive fine-tuning |
| [Insights](Research/Insights.md) | Findings across projects and the evidence behind them |

This directory is documentation, not an executable package. The
[research index](Research/README.md) maps ideas to implementation files;
project-level `DESIGN.md` files describe individual architectures.

## Early variations

### AbsorbingMarkovChain

An early idea for learning the transition matrix of an absorbing Markov chain,
with absorbing states determining where a sequence stops. `README.py` preserves
the original note, and `main.py` is empty. There is no runnable implementation
or dependency setup yet. See the [README](AbsorbingMarkovChain/README.md).

### RadixTreeLLM

The original note choosing a radix tree for information compression. This
directory contains only a README, whose historical heading names the companion
RadixTrieLLM_RNN variation. Follow [that prototype](#radixtriellm_rnn) for code,
or RadixCyclicNN and RadixAcyclicNN for the later graph/tree implementations.
There is nothing to run here. See the [note](RadixTreeLLM/README.md).

### RadixTreeRNN

The proposed recurrent version of the per-node model: each radix-tree node
would own an RNN, while shared prefixes provide compression. `main.py` is
empty; this is a named variation awaiting implementation. The implemented
per-node transformer example is in RadixTrieLLM_RNN.
See the [README](RadixTreeRNN/README.md).

### RadixTrieLLM_RNN

A runnable early prototype in which each radix-trie node owns a small byte-level
transformer. `main.py` implements the transformer, compressed trie insertion
and splitting, search, parent-to-child context propagation, and next-byte
prediction. Its demo inserts a few related words and prints searches, an encoded
representation, and a predicted byte; it does not provide a trained language
model or the later graph architecture's training and serving stack.

**Start:** `(cd RadixTrieLLM_RNN && python3 main.py)` after installing PyTorch.
**Needs:** PyTorch (`torch`).
See the [README](RadixTrieLLM_RNN/README.md).

## Working on a subproject

Read the project's README first, then its design/specification files when
present. Keep dependencies and generated artifacts local to the project;
use a virtual environment when installing optional Python packages. There is
no root test runner: use the project's Makefile or its documented test command.
For example, from the repository root:

```bash
(cd RadixCyclicNN && make test-core)
(cd ModelKit && make test)
(cd CyclicCortex && python3 -m tests.test_v1)
(cd GREN && python3 -m tests.test_gren)
```

For experiment changes, retain the configuration, seeds, controls, and output
needed to reproduce a result. Use reduced runs before full sweeps, and consult
the project instructions before commands that regenerate committed result files
or tables. Link new results from the relevant experiment or research document,
and update this index when adding or moving a subproject.

All subprojects remain subject to the repository's [LICENSE](LICENSE).
