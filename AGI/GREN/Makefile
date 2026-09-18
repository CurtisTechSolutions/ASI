# GREN — run commands.  `make help` lists every target.
#
# GREN learns a game's identity from what it REFUSES.  Every target here is
# zero-dependency python3; it probes CyclicCortex's game adapters, so that
# directory must be present beside this one.
#
# Override any variable on the command line, e.g. `make explore BUDGET=3000`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY      ?= python3
CLI     := $(PY) -m gren.cli

# Probes per game, the policy that picks them, the seed, and the hidden width
# of the network the `grow` target builds.
BUDGET  ?= 1500
POLICY  ?= eig
SEED    ?= 0
HIDDEN  ?= 24

# Where `make package` writes the handoff CyclicCortex reads.  Relative to
# THIS directory.
HANDOFF ?= ../CyclicCortex/data/gren_packages.json

COMMON  = --budget $(BUDGET) --policy $(POLICY) --seed $(SEED)

.PHONY: help check test explore similar policies tree grow package regress transfer all clean

help: ## Show this help
	@echo "GREN make targets  (override variables like: make explore BUDGET=3000)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

##@ Setup
check: ## Byte-compile the package and its tests (a syntax check, no tools needed)
	$(PY) -m compileall -q gren tests && echo "ok"

##@ Tests
test: ## Verdicts, the radix tree, growth identities, and the derived vocabulary
	$(PY) -m tests.test_gren

##@ Probing — what does each game refuse?
explore: ## Probe every game and report its refusal taxonomy
	$(CLI) explore $(COMMON)

similar: ## Discovered similarity against CyclicCortex's hand-written mechanic sets
	$(CLI) similar $(COMMON)

policies: ## Failure rate and rule coverage by probe policy (random, boundary, eig, learned)
	$(CLI) policies --budget $(BUDGET) --seed $(SEED)

tree: ## The radix tree over discovered signatures — trie mechanics, path compression
	$(CLI) tree $(COMMON)

grow: ## One Self-Building network growing its inputs and outputs across all four games
	$(CLI) grow $(COMMON) --hidden $(HIDDEN)

regress: ## Goal regression -- backwards from the goal to subgoals
	$(CLI) regress --trials 6

transfer: ## Do the RULES transfer between games? Per-code, through the shared vocabulary
	$(CLI) transfer --trials 3

##@ The handoff
package: ## Build a GamePackage per game and write the handoff to HANDOFF
	$(CLI) package $(COMMON) --out $(HANDOFF)

##@ Everything
all: check test explore similar tree regress package ## Check, test, then every report and the handoff

clean: ## Remove byte-code caches (the handoff lives in CyclicCortex/data/ and is untouched)
	rm -rf __pycache__ gren/__pycache__ tests/__pycache__
	@echo "ok"
