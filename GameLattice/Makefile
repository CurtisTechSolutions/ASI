# GameLattice - run commands.  `make help` lists every target.
#
# The Rust crate in rust/ is the build (DESIGN.md §13): `cargo` is the whole dependency list, and the two models it
# combines are the sibling crates ../LatticeFSM/rust and ../RadixCyclicNN/rust.  The React frontend in frontend/
# (Vite; `npm` for the dev server and for rebuilding it, while frontend/dist is committed and served as it is) is
# DESIGN.md §16.  Every target is deterministic given SEED.
#
#   make serve         the backend (the Rust API, :8000) and the frontend (Vite with hot reload, :5173) together
#   make serve-prod    one process: the API and the built frontend/dist on :8000

SHELL := /bin/bash
.DEFAULT_GOAL := help

CARGO    ?= cargo
BIN      ?= rust/target/release/gamelattice
LANGUAGE ?= even-b
EPISODES ?= 4000
STATES   ?= 13
WINDOW   ?= 8
N        ?= 1
SEED     ?= 1
SEEDS    ?= 1,2,3
GAMES    ?= 400
BABBLE   ?= 200
RESULTS  ?= results
ARMS     ?=
HOST     ?= 127.0.0.1
PORT     ?= 8000
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 5173
NPM      ?= npm

.PHONY: help build test rust-test server-test lint fmt demo train plain nim experiments e1 e2 clean \
        serve serve-backend serve-frontend serve-prod frontend-install frontend-build frontend-dev frontend-preview \
        frontend-test

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: LANGUAGE=$(LANGUAGE) EPISODES=$(EPISODES) STATES=$(STATES) WINDOW=$(WINDOW) SEED=$(SEED) SEEDS=$(SEEDS) HOST=$(HOST) PORT=$(PORT)"

build: ## Build the Rust crate (release) into rust/target/release/gamelattice
	cd rust && $(CARGO) build --release

test: ## The Rust tests (the server's included), clippy with warnings denied, the formatter check; the frontend's tests
	cd rust && $(CARGO) test --release && $(CARGO) clippy --release --all-targets -- -D warnings && $(CARGO) fmt --check
	cd frontend && $(NPM) test

rust-test: ## The Rust tests alone
	cd rust && $(CARGO) test --release

server-test: ## The HTTP API's tests alone (rust/tests/test_server.rs)
	cd rust && $(CARGO) test --release --test test_server

lint: ## clippy with warnings denied, and the formatter check
	cd rust && $(CARGO) clippy --release --all-targets -- -D warnings && $(CARGO) fmt --check

fmt: ## Format the Rust code
	cd rust && $(CARGO) fmt

demo: build ## A body learns LANGUAGE: the encoder names the lattice's states, the oracle credits every step
	$(BIN) demo --language $(LANGUAGE) --episodes 1000 --states $(STATES) --window $(WINDOW) --n $(N) --seed $(SEED)

train: build ## The same over EPISODES random strings, printing held-out accuracy as it goes
	$(BIN) train --language $(LANGUAGE) --episodes $(EPISODES) --states $(STATES) --window $(WINDOW) --n $(N) --seed $(SEED)

plain: build ## The control: the lattice alone with the per-step oracle, the encoder only naming beside it
	$(BIN) train --language $(LANGUAGE) --episodes $(EPISODES) --states $(STATES) --seed $(SEED) --plain

nim: build ## Babble BABBLE games, then play GAMES games of Nim against a random opponent
	$(BIN) nim --games $(GAMES) --babble $(BABBLE) --states $(STATES) --window $(WINDOW) --seed $(SEED)

e1: build ## E1: the four languages, every arm, SEEDS seeds, written to RESULTS/e1.json (about two minutes)
	mkdir -p $(RESULTS)
	$(BIN) experiment --which e1 --episodes $(EPISODES) --states $(STATES) --window $(WINDOW) --n $(N) --seeds $(SEEDS) \
	  $(if $(ARMS),--arms $(ARMS),) --out $(RESULTS) | tee $(RESULTS)/e1.log

e2: build ## E2: Nim by window width, SEEDS seeds, written to RESULTS/e2.json (under a minute)
	mkdir -p $(RESULTS)
	$(BIN) experiment --which e2 --games $(GAMES) --babble $(BABBLE) --states $(STATES) --seeds $(SEEDS) --out $(RESULTS) \
	  | tee $(RESULTS)/e2.log

experiments: e1 e2 ## Both experiments

##@ Serving
# The backend serves the API (and the built frontend/dist) on HOST:PORT; the frontend dev server serves the app on
# FRONTEND_HOST:FRONTEND_PORT with hot reload and proxies /api to the backend.  Each runs in its own process group, so
# Ctrl-C - or either one exiting - stops both.
SERVE_BACKEND = $(BIN) serve --language $(LANGUAGE) --states $(STATES) --window $(WINDOW) --n $(N) --seed $(SEED) \
                --host $(HOST) --port $(PORT) --frontend-dir frontend/dist --results-dir $(RESULTS)
SERVE_FRONTEND = cd frontend && VITE_PROXY_TARGET=http://$(HOST):$(PORT) exec $(NPM) run dev -- \
                 --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort

frontend/node_modules: frontend/package.json frontend/package-lock.json
	cd frontend && $(NPM) ci --no-audit --no-fund
	@touch $@

serve: build frontend/node_modules ## The backend and the frontend together: open http://FRONTEND_HOST:FRONTEND_PORT/ (Ctrl-C stops both)
	@set -m; \
	$(SERVE_BACKEND) & backend=$$!; \
	stop() { trap - EXIT; trap '' INT TERM; for p in $$backend $${frontend:-}; do kill -TERM -- -$$p 2>/dev/null; done; wait $$backend $${frontend:-} 2>/dev/null; }; \
	trap 'stop' EXIT; \
	trap 'echo " stopping the backend and the frontend" >&2; exit 0' INT TERM; \
	ready=0; \
	for _ in $$(seq 1 100); do \
	  (exec 3<>/dev/tcp/$(HOST)/$(PORT)) 2>/dev/null && { ready=1; break; }; \
	  kill -0 $$backend 2>/dev/null || { echo "the backend did not start (is port $(PORT) free?)" >&2; exit 1; }; \
	  sleep 0.1; \
	done; \
	[ $$ready -eq 1 ] || { echo "the backend did not become ready on $(HOST):$(PORT)" >&2; exit 1; }; \
	( $(SERVE_FRONTEND) ) & frontend=$$!; \
	echo ""; \
	echo "  backend   http://$(HOST):$(PORT)/api/   (and the built app at http://$(HOST):$(PORT)/)"; \
	echo "  frontend  http://$(FRONTEND_HOST):$(FRONTEND_PORT)/   - open this one: hot reload, /api proxied to the backend"; \
	echo "  Ctrl-C stops both"; \
	echo ""; \
	while kill -0 $$backend 2>/dev/null && kill -0 $$frontend 2>/dev/null; do sleep 0.1; done; \
	if kill -0 $$backend 2>/dev/null; then wait $$frontend; else wait $$backend; fi; status=$$?; \
	echo "one of the two stopped (exit $$status); stopping the other" >&2; \
	exit $$status

serve-backend: build ## The backend alone: the API and the built frontend/dist on http://HOST:PORT/
	$(SERVE_BACKEND)

serve-frontend: frontend/node_modules ## The frontend dev server alone, proxying /api to a backend on HOST:PORT
	$(SERVE_FRONTEND)

serve-prod: build ## One process, no npm: the API and the committed frontend/dist on http://HOST:PORT/
	$(SERVE_BACKEND)

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

clean: ## Remove build output (results, frontend/dist and frontend/node_modules are kept)
	cd rust && $(CARGO) clean
