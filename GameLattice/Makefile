# GameLattice - run commands.  `make help` lists every target.
#
# The Rust crate in rust/ is the build (DESIGN.md §13): `cargo` is the whole dependency list, and the two models it
# combines are the sibling crates ../LatticeFSM/rust and ../RadixCyclicNN/rust.  Every target is deterministic
# given SEED.

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

.PHONY: help build test lint fmt demo train plain nim experiments e1 e2 clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: LANGUAGE=$(LANGUAGE) EPISODES=$(EPISODES) STATES=$(STATES) WINDOW=$(WINDOW) SEED=$(SEED) SEEDS=$(SEEDS)"

build: ## Build the Rust crate (release) into rust/target/release/gamelattice
	cd rust && $(CARGO) build --release

test: ## The Rust tests, clippy with warnings denied, and the formatter check
	cd rust && $(CARGO) test --release && $(CARGO) clippy --release --all-targets -- -D warnings && $(CARGO) fmt --check

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

clean: ## Remove build output
	cd rust && $(CARGO) clean
