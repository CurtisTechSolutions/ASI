# LatticeFSM - run commands.  `make help` lists every target.
#
# Two ports of one model: the Rust crate in rust/ (the CLI and the HTTP server;
# `cargo` is the whole dependency list, no crates) and the Python reference
# package in latticefsm/ (standard library only) - and the React frontend in
# frontend/ (Vite; `npm` only for rebuilding it, frontend/dist is committed and
# served as it is).  The Rust targets are the default; the Python ones are
# prefixed py-, the frontend's frontend-.  Every target is deterministic given
# SEED.
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
STIM      ?=
CREDIT    ?=
TICKS     ?= 1000
AMOUNT    ?= 2
PORT      ?= 8000
CORE      ?= core.json.gz
PRECISION ?= exact
SHELLS    ?=
EVERY     ?= 0
FROM_MIDDLE ?=
HOST      ?= 127.0.0.1
RESULTS   ?= results

MACHINE_OPTS = --states $(STATES) --alphabet $(ALPHABET) --life $(LIFE) --baseline $(BASELINE) --seed $(SEED) --language $(LANGUAGE) \
               $(if $(filter-out 0,$(EVERY)),--compress-every $(EVERY) --compress-precision $(PRECISION),)

.PHONY: help build test check demo train run teach accuracy table stats tick stimulate serve experiments \
        compress expand core-run \
        py-test py-demo py-train py-run py-experiments parity \
        frontend-install frontend-build frontend-dev frontend-test clean

help: ## Show this help
	@echo "LatticeFSM make targets  (override like: make demo LANGUAGE=ends-ab STATES=3)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: LANGUAGE=$(LANGUAGE) STATES=$(STATES) ALPHABET=$(ALPHABET) LIFE=$(LIFE) BASELINE=$(BASELINE) SEED=$(SEED) EPISODES=$(EPISODES) MODEL=$(MODEL) PORT=$(PORT)"

##@ Building and checking
build: ## Build the Rust crate (release) into rust/target/release/latticefsm
	cd rust && $(CARGO) build --release

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

##@ Serving
serve: build ## The HTTP API and the frontend on http://HOST:PORT/ (MODEL if it exists, else a fresh machine)
	$(BIN) serve $(MACHINE_OPTS) --host $(HOST) --port $(PORT) --frontend-dir frontend/dist $(if $(wildcard $(MODEL)),--load $(MODEL),)

##@ The experiments README.md quotes
experiments: build ## Learning, stimulation, adaptation and compression, written to RESULTS/ - a second
	mkdir -p $(RESULTS)
	$(BIN) experiment --which all --life $(LIFE) --out $(RESULTS) | tee $(RESULTS)/experiments.log

py-experiments: ## The same four from the Python reference, written to RESULTS/python/ - about two minutes
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

parity: build ## The two ports held to each other: the deterministic experiments and a machine file, both ways
	$(PY) -m unittest tests.test_latticefsm.TestRustParity -v

##@ The frontend (React, Vite; frontend/dist is committed, so none of this is needed to run `make serve`)
frontend-install: ## Install the frontend's dependencies (npm ci)
	cd frontend && npm ci --no-audit --no-fund

frontend-build: ## Rebuild frontend/dist, which `make serve` serves
	cd frontend && npm run build

frontend-dev: ## Vite dev server with hot reload on http://localhost:5173, proxying /api to VITE_PROXY_TARGET (default :8000)
	cd frontend && npm run dev

frontend-test: ## The frontend's unit tests (node --test; no npm install needed)
	cd frontend && npm test

clean: ## Remove build output, __pycache__, node_modules and saved machines (results and frontend/dist are kept)
	cd rust && $(CARGO) clean
	rm -rf frontend/node_modules
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f machine.json machine.json.gz core.json core.json.gz
