# LatticeFSM - run commands.  `make help` lists every target.
#
# Two ports of one model: the Rust crate in rust/ (the CLI and the HTTP server;
# `cargo` is the whole dependency list, no crates) and the Python reference
# package in latticefsm/ (standard library only) - and the React frontend in
# frontend/ (Vite; `npm` for the dev server and for rebuilding it, while
# frontend/dist is committed and served as it is).  The Rust targets are the
# default; the Python ones are prefixed py-, the frontend's frontend-.  Every
# target is deterministic given SEED.
#
#   make serve         the backend (the Rust API, :8000) and the frontend (Vite with hot reload, :5173) together
#   make serve-prod    one process: the API and the built frontend/dist on :8000
#   make translate-fit && make translate TEXT="cat bus"   English to English through a network of machines
#   make ngram-train && make ngram-predict TEXT="the brain is"   predict the next thing
#   make chain-train && make chain-predict TEXT="the bra"        one input a node; walks shorten with confidence
#   make autotrain                 the corpus, then rounds with a local LLM through Ollama
#
# Override any variable on the command line, e.g. `make demo LANGUAGE=ends-ab STATES=3`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
CARGO     ?= cargo
PY        ?= python3
BIN       ?= rust/target/release/latticefsm
LANGUAGE  ?= even-b
STATES    ?= 13
ALPHABET  ?= abcdefghijklm
LIFE      ?= 1000
BASELINE  ?= 1
SEED      ?= 1
EPISODES  ?= 4000
MODEL     ?= machine.json.gz
TEXT      ?= abab
AXIS      ?= states
I         ?= 0
J         ?= 12
STIM      ?=
CREDIT    ?=
TICKS     ?= 1000
AMOUNT    ?= 2
PORT      ?= 8000
FRONTEND_PORT ?= 5173
FRONTEND_HOST ?= 127.0.0.1
NPM       ?= npm
CORE      ?= core.json.gz
PRECISION ?= exact
SHELLS    ?=
EVERY     ?= 0
FROM_MIDDLE ?=
FOCUS     ?=
LEARN_FOCUS ?=
SKIP      ?=
REARRANGE ?= 0
HOST      ?= 127.0.0.1
RESULTS   ?= results
ARCH      ?= 13>13
NET       ?= network.json.gz
NET_ALPHABET ?= ab
TASK      ?=
EPOCHS    ?=
CREDIT_BY ?= shared
UNIT      ?= word
ORDER     ?= 3
NGRAM     ?= ngram.json.gz
CORPUS    ?= data/corpus
NODES     ?= 8
CONFIDENCE ?= 0.9
CHAIN     ?= chain.json.gz
REPLY     ?= a lattice of adaptive edges.
ROUNDS    ?= 3
OLLAMA_MODEL ?=
TOPICS    ?=
FILE      ?=
OUT       ?=
KIND      ?= image
DATA      ?= data/$(KIND)
COUNT     ?= 60

MACHINE_OPTS = --states $(STATES) --alphabet $(ALPHABET) --life $(LIFE) --baseline $(BASELINE) --seed $(SEED) --language $(LANGUAGE) \
               $(if $(filter-out 0,$(EVERY)),--compress-every $(EVERY) --compress-precision $(PRECISION),) \
               $(if $(FOCUS),--focus $(FOCUS),) $(if $(LEARN_FOCUS),--learn-focus,) $(if $(SKIP),--skip,) \
               $(if $(filter-out 0,$(REARRANGE)),--rearrange-every $(REARRANGE),)

.PHONY: help all install build test check rust-test lint fmt demo train run teach accuracy table stats tick stimulate \
        serve serve-backend serve-frontend serve-prod experiments results \
        compress expand core-run focus skip rearrange swap \
        network network-fit network-evaluate network-predict network-summary translate translate-fit py-network \
        media-data media-fit media-evaluate \
        ngram-train ngram-predict ngram-generate ngram-evaluate autotrain autotrain-corpus ollama-check fake-ollama \
        py-autotrain base64-encode base64-decode chain-train chain-until chain-predict chain-generate chain-sweep \
        chain-teach corpus \
        py-test py-demo py-train py-run py-stats py-experiments parity \
        frontend-install frontend-build frontend-dev frontend-preview frontend-test clean distclean

help: ## Show this help
	@echo "LatticeFSM make targets  (override like: make demo LANGUAGE=ends-ab STATES=3)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: LANGUAGE=$(LANGUAGE) STATES=$(STATES) ALPHABET=$(ALPHABET) LIFE=$(LIFE) BASELINE=$(BASELINE) SEED=$(SEED) EPISODES=$(EPISODES) MODEL=$(MODEL) PORT=$(PORT)"

##@ Setting up
install: build frontend/node_modules ## Build the Rust binary and install the frontend's dependencies: everything `make serve` needs

all: build frontend-build ## Build the Rust binary and rebuild frontend/dist

frontend/node_modules: frontend/package.json frontend/package-lock.json
	cd frontend && $(NPM) ci --no-audit --no-fund
	@touch $@

##@ Building and checking
build: ## Build the Rust crate (release) into rust/target/release/latticefsm
	cd rust && $(CARGO) build --release

rust-test: ## The Rust tests alone
	cd rust && $(CARGO) test

lint: ## clippy with warnings denied, and the formatter check
	cd rust && $(CARGO) clippy --all-targets -- -D warnings && $(CARGO) fmt --check

fmt: ## Format the Rust code
	cd rust && $(CARGO) fmt

test: ## The Rust tests, clippy and the formatter check; the Python tests, parity included; the frontend's tests
	cd rust && $(CARGO) test && $(CARGO) clippy --all-targets -- -D warnings && $(CARGO) fmt --check
	$(MAKE) --no-print-directory build
	$(PY) -m unittest tests.test_latticefsm
	cd frontend && npm test

check: test ## The same

##@ Looking at it (Rust)
demo: build ## Learn LANGUAGE by reward and punishment, be stimulated, let time pass
	$(BIN) demo $(MACHINE_OPTS) --episodes $(EPISODES)

train: build ## Teach LANGUAGE over EPISODES random strings into a fresh machine, and save it to MODEL
	$(BIN) train $(MACHINE_OPTS) --episodes $(EPISODES) --save $(MODEL)

run: build ## Read TEXT from MODEL (STIM=x runs at that stimulation; CREDIT=r rewards, negative punishes), saved back
	$(BIN) run "$(TEXT)" --load $(MODEL) $(if $(STIM),--stimulation $(STIM),) $(if $(CREDIT),--credit $(CREDIT),) $(if $(FROM_MIDDLE),--from-middle,) --save $(MODEL)

teach: build ## Traverse edge SOURCE SYMBOL TARGET deliberately and credit it AMOUNT, saved back
	$(BIN) teach $(SOURCE) $(SYMBOL) $(TARGET) --load $(MODEL) --amount $(AMOUNT) --save $(MODEL)

accuracy: build ## Classification accuracy of MODEL on random strings of LANGUAGE (nothing moves)
	$(BIN) accuracy --load $(MODEL) --language $(LANGUAGE) --seed $(SEED)

table: build ## The greedy transition table of MODEL (nothing moves)
	$(BIN) table --load $(MODEL)

stats: build ## Size, memory, clock and stimulation of MODEL
	$(BIN) stats --load $(MODEL)

tick: build ## Let TICKS pass in MODEL, and save it
	$(BIN) tick --load $(MODEL) --ticks $(TICKS) --save $(MODEL)

stimulate: build ## Raise MODEL's stimulation by AMOUNT, and save it
	$(BIN) stimulate --load $(MODEL) --amount $(AMOUNT) --save $(MODEL)

##@ The central node
compress: build ## Fold MODEL into its central node at PRECISION (exact|float32|float16) and save the code to CORE
	$(BIN) compress --load $(MODEL) --precision $(PRECISION) --core $(CORE) --expansion

expand: build ## Rebuild a machine from CORE, from the central node outward (SHELLS=k for only k shells), into MODEL
	$(BIN) expand --core $(CORE) $(if $(SHELLS),--shells $(SHELLS),) --save $(MODEL)

core-run: build ## Walk TEXT straight from CORE (FROM_MIDDLE=1 starts from the middle state)
	$(BIN) core-run "$(TEXT)" --core $(CORE) $(if $(FROM_MIDDLE),--from-middle,)

##@ Where a run starts and how it moves
focus: build ## Set MODEL's focus to FOCUS (0..1: 0 the top node, 1 the bottom; empty: none), and save it
	$(BIN) focus --load $(MODEL) $(if $(FOCUS),--level $(FOCUS),--none) $(if $(LEARN_FOCUS),--learn on,) --save $(MODEL)

skip: build ## Turn MODEL's skips on (SKIP=1) or off (SKIP empty), and save it
	$(BIN) skip $(if $(SKIP),on,off) --load $(MODEL) --save $(MODEL)

rearrange: build ## Let MODEL's nodes rearrange themselves toward the centre until settled, and save it
	$(BIN) rearrange --load $(MODEL) --full --save $(MODEL)

swap: build ## Swap two of MODEL's nodes: AXIS=states|symbols, I and J (state numbers, or symbols like a g), and save it
	$(BIN) rearrange --load $(MODEL) --swap $(AXIS) $(I) $(J) --save $(MODEL)

##@ Networks of machines (ARCH like 13>(13|13)>13; enc / dec read and write English)
NET_NEW = --arch "$(ARCH)" --alphabet $(NET_ALPHABET) --seed $(SEED) --life $(LIFE) --credit $(CREDIT_BY)
network: build ## A fresh network of ARCH over NET_ALPHABET, saved to NET, and its summary
	$(BIN) network new $(NET_NEW) --language $(LANGUAGE) --save $(NET)

network-fit: build ## Teach LANGUAGE over EPISODES strings (EPOCHS times) to NET (fresh from ARCH if NET is missing), saved back
	$(BIN) network fit $(if $(wildcard $(NET)),--load $(NET),$(NET_NEW)) --language $(LANGUAGE) --episodes $(EPISODES) \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) --save $(NET)

network-evaluate: build ## NET's accuracy on every string of LANGUAGE up to 6 symbols (or TASK's exact translations)
	$(BIN) network evaluate --load $(NET) $(if $(TASK),--task $(TASK),--language $(LANGUAGE))

network-predict: build ## NET's verdict on TEXT, layer by layer
	$(BIN) network predict "$(TEXT)" --load $(NET)

network-summary: build ## NET's blocks, their matrices and what they write
	$(BIN) network summary --load $(NET)

translate-fit: build ## English to English: teach TASK (plural|past; default plural) to a fresh ARCH=enc>dec network, saved to NET
	$(BIN) network fit --arch "$(if $(filter 13>13,$(ARCH)),enc>dec,$(ARCH))" --seed $(SEED) --task $(or $(TASK),plural) \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) --save $(NET)

translate: build ## English in, English out: NET's translation of TEXT (e.g. make translate TEXT="cat dog")
	$(BIN) network translate $(TEXT) --load $(NET)

py-network: ## The same network fit from the Python package (ARCH, LANGUAGE or TASK), saved to NET
	$(PY) -m latticefsm network fit $(NET_NEW) $(if $(TASK),--task $(TASK),--language $(LANGUAGE) --episodes $(EPISODES)) \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) --save $(NET)

media-data: build ## A synthetic labelled data set of KIND (text|image|audio|video) in DATA (COUNT examples)
	$(BIN) network make-data --kind $(KIND) --out $(DATA) --count $(COUNT) --seed $(SEED)

media-fit: build ## Teach DATA/labels.tsv to a fresh network (ARCH, default KIND>13), every 5th held out, saved to NET
	$(BIN) network fit --arch "$(if $(filter 13>13,$(ARCH)),$(KIND)>13,$(ARCH))" --data $(DATA)/labels.tsv --seed $(SEED) \
	  --discount 0.95 $(if $(EPOCHS),--epochs $(EPOCHS),) --save $(NET)

media-evaluate: build ## NET's accuracy on DATA/labels.tsv, taught and held out
	$(BIN) network evaluate --load $(NET) --data $(DATA)/labels.tsv

##@ Predicting the next thing (N-grams; UNIT=word|char|sound|base64, ORDER=N, CORPUS=files or dirs)
ngram-train: build ## Train an ORDER-gram model of UNITs on CORPUS (every 10th line held out and measured), saved to NGRAM
	$(BIN) ngram train --unit $(UNIT) --order $(ORDER) --data $(CORPUS) --save $(NGRAM)

ngram-predict: build ## NGRAM's likeliest next tokens after TEXT
	$(BIN) ngram predict "$(TEXT)" --load $(NGRAM)

ngram-generate: build ## TEXT continued by NGRAM (TEMPERATURE=0: the likeliest each time; OUT=file: a base64 model's bytes)
	$(BIN) ngram generate "$(TEXT)" --load $(NGRAM) --temperature $(or $(TEMPERATURE),0) $(if $(OUT),--out $(OUT) --max-tokens 4096,)

ngram-evaluate: build ## NGRAM's held-out perplexity and next-token accuracy on CORPUS
	$(BIN) ngram evaluate --load $(NGRAM) --data $(CORPUS)

autotrain: build ## Train on CORPUS, then ROUNDS rounds with Ollama (OLLAMA_MODEL, TOPICS); keeps what it wrote in CORPUS/ollama.txt
	$(BIN) ngram autotrain --ollama --unit $(UNIT) --order $(ORDER) --data $(CORPUS) --rounds $(ROUNDS) \
	  $(if $(OLLAMA_MODEL),--model $(OLLAMA_MODEL),) $(if $(TOPICS),--topics "$(TOPICS)",) --save $(NGRAM)

autotrain-corpus: build ## The same without Ollama: the corpus only, measured
	$(BIN) ngram autotrain --unit $(UNIT) --order $(ORDER) --data $(CORPUS) --save $(NGRAM)

py-autotrain: ## autotrain from the Python package
	$(PY) -m latticefsm ngram autotrain --ollama --unit $(UNIT) --order $(ORDER) --data $(CORPUS) --rounds $(ROUNDS) \
	  $(if $(OLLAMA_MODEL),--model $(OLLAMA_MODEL),) $(if $(TOPICS),--topics "$(TOPICS)",) --save $(NGRAM)

ollama-check: build ## Is Ollama there ($$OLLAMA_HOST or 127.0.0.1:11434), which models, which autotrain would use
	$(BIN) ngram ollama $(if $(OLLAMA_MODEL),--model $(OLLAMA_MODEL),)

fake-ollama: ## A stand-in Ollama on :11500 for trying autotrain without a model (OLLAMA_HOST=127.0.0.1:11500 make autotrain)
	$(PY) tests/fake_ollama.py --port 11500

chain-train: build ## A chain of NODES nodes of letters on CORPUS (zip, tar, gz, html, docx... opened), saved to CHAIN
	$(BIN) chain train --nodes $(NODES) --confidence $(CONFIDENCE) --data $(CORPUS) --save $(CHAIN)

chain-until: build ## chain-train, then drilled until every input is predicted right that can be (N of N)
	$(BIN) chain train --nodes $(NODES) --confidence $(CONFIDENCE) --data $(CORPUS) --until --save $(CHAIN)

chain-predict: build ## The input nodes CHAIN fills from TEXT, the walk back through them, the likeliest next inputs
	$(BIN) chain predict "$(TEXT)" --load $(CHAIN)

chain-generate: build ## CHAIN continuing TEXT, and how deep every walk went
	$(BIN) chain generate "$(TEXT)" --load $(CHAIN)

chain-sweep: build ## Accuracy against input nodes traversed, threshold by threshold, on CORPUS's held-out lines
	$(BIN) chain sweep --load $(CHAIN) --data $(CORPUS)

chain-teach: build ## Teach CHAIN that REPLY follows TEXT, round after round, until every input of it is right
	$(BIN) chain teach "$(TEXT)" "$(REPLY)" --load $(CHAIN) --save $(CHAIN)

corpus: build ## What CORPUS is read as: every document (archives opened) and its lines
	$(BIN) corpus $(CORPUS) --show 2

base64-encode: build ## FILE (any kind) as base64, to OUT or the screen
	$(BIN) base64 encode $(FILE) $(if $(OUT),--out $(OUT),)

base64-decode: build ## FILE of base64 back to bytes, to OUT or the screen
	$(BIN) base64 decode $(FILE) $(if $(OUT),--out $(OUT),)

##@ Serving
# The backend serves the API (and the built frontend/dist) on HOST:PORT; the frontend dev server serves the app on
# FRONTEND_HOST:FRONTEND_PORT with hot reload and proxies /api to the backend.  Each runs in its own process group, so
# Ctrl-C - or either one exiting - stops both.
SERVE_BACKEND = $(BIN) serve $(MACHINE_OPTS) --host $(HOST) --port $(PORT) --frontend-dir frontend/dist \
                $(if $(wildcard $(MODEL)),--load $(MODEL),)
SERVE_FRONTEND = cd frontend && VITE_PROXY_TARGET=http://$(HOST):$(PORT) exec $(NPM) run dev -- \
                 --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort

serve: build frontend/node_modules ## The backend and the frontend together: open http://FRONTEND_HOST:FRONTEND_PORT/ (Ctrl-C stops both)
	@set -m; \
	$(SERVE_BACKEND) & backend=$$!; \
	stop() { for p in $$backend $${frontend:-}; do kill -- -$$p 2>/dev/null; kill $$p 2>/dev/null; done; wait 2>/dev/null; }; \
	trap 'stop' EXIT; \
	trap 'echo " stopping the backend and the frontend" >&2; trap - EXIT; stop; exit 0' INT TERM; \
	for _ in $$(seq 1 100); do \
	  (exec 3<>/dev/tcp/$(HOST)/$(PORT)) 2>/dev/null && break; \
	  kill -0 $$backend 2>/dev/null || { echo "the backend did not start (is port $(PORT) free?)" >&2; exit 1; }; \
	  sleep 0.1; \
	done; \
	( $(SERVE_FRONTEND) ) & frontend=$$!; \
	echo ""; \
	echo "  backend   http://$(HOST):$(PORT)/api/   (and the built app at http://$(HOST):$(PORT)/)"; \
	echo "  frontend  http://$(FRONTEND_HOST):$(FRONTEND_PORT)/   - open this one: hot reload, /api proxied to the backend"; \
	echo "  Ctrl-C stops both"; \
	echo ""; \
	wait -n; status=$$?; \
	echo "one of the two stopped (exit $$status); stopping the other" >&2; \
	exit $$status

serve-backend: build ## The backend alone: the API and the built frontend/dist on http://HOST:PORT/ (MODEL if it exists)
	$(SERVE_BACKEND)

serve-frontend: frontend/node_modules ## The frontend dev server alone, proxying /api to a backend on HOST:PORT
	$(SERVE_FRONTEND)

serve-prod: build ## One process, no npm: the API and the committed frontend/dist on http://HOST:PORT/
	$(SERVE_BACKEND)

##@ The experiments README.md quotes
experiments: build ## Learning, stimulation, adaptation, compression, walk, network, media and ngram, written to RESULTS/ - ten seconds
	mkdir -p $(RESULTS)
	$(BIN) experiment --which all --life $(LIFE) --out $(RESULTS) | tee $(RESULTS)/experiments.log

results: experiments py-experiments ## Both ports' experiments, every results file regenerated

py-experiments: ## The same eight from the Python reference, written to RESULTS/python/ - about six minutes
	mkdir -p $(RESULTS)/python
	$(PY) -m latticefsm experiment --which all --life $(LIFE) --out $(RESULTS)/python | tee $(RESULTS)/python/experiments.log

##@ The Python reference
py-test: ## The Python tests (parity with the Rust crate included when it is built)
	$(PY) -m unittest tests.test_latticefsm

py-demo: ## The demo from the Python package
	$(PY) -m latticefsm demo $(MACHINE_OPTS) --episodes $(EPISODES)

py-train: ## Teach LANGUAGE from the Python package, saved to MODEL
	$(PY) -m latticefsm train $(MACHINE_OPTS) --episodes $(EPISODES) --save $(MODEL)

py-run: ## Read TEXT from MODEL with the Python package
	$(PY) -m latticefsm run "$(TEXT)" --load $(MODEL) $(if $(STIM),--stimulation $(STIM),) $(if $(CREDIT),--credit $(CREDIT),)

py-stats: ## Size, memory, clock and settings of MODEL, from the Python package
	$(PY) -m latticefsm stats --load $(MODEL)

parity: build ## The two ports held to each other: the deterministic experiments, machine and network files, media symbols
	$(PY) -m unittest tests.test_latticefsm.TestRustParity tests.test_latticefsm.TestRustNetworkParity \
	  tests.test_latticefsm.TestRustMediaParity tests.test_latticefsm.TestRustNGramParity -v

##@ The frontend (React, Vite; frontend/dist is committed, so `make serve-prod` needs none of this)
frontend-install: ## Install the frontend's dependencies (npm ci)
	cd frontend && $(NPM) ci --no-audit --no-fund
	@touch frontend/node_modules

frontend-build: frontend/node_modules ## Rebuild frontend/dist, which the backend serves
	cd frontend && $(NPM) run build

frontend-dev: serve-frontend ## The same as serve-frontend

frontend-preview: frontend/node_modules ## Vite's preview of the built dist on FRONTEND_PORT, proxying /api to HOST:PORT
	cd frontend && VITE_PROXY_TARGET=http://$(HOST):$(PORT) $(NPM) run preview -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort

frontend-test: ## The frontend's unit tests (node --test; no npm install needed)
	cd frontend && $(NPM) test

clean: ## Remove build output, __pycache__ and saved machines (results, frontend/dist and node_modules are kept)
	cd rust && $(CARGO) clean
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f machine.json machine.json.gz core.json core.json.gz network.json network.json.gz ngram.json ngram.json.gz

distclean: clean ## clean, and the frontend's node_modules too
	rm -rf frontend/node_modules
