# CyclicCortex — run commands.  `make help` lists every target.
#
# One Self-Building network per REGION of a cyclic similarity graph, where
# distance is similarity.  Everything here is zero-dependency python3.
# Stockfish is used as chess opponent and teacher when it is on PATH, and the
# built-in engine stands in when it is not.
#
# The --discovered flags make the cortex build its map from GREN's measured
# signatures instead of the hand-written `mechanics` sets in cortex/games.py.
# Write that handoff first with `make -C ../GREN package`.
#
# Override any variable on the command line, e.g. `make demo EPISODES=1200`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY       ?= python3
CLI      := $(PY) -m cortex.cli

GAMES    ?= chess checkers go sudoku
SEED     ?= 0

# Training positions per game, self-play games, plies per played game,
# opponent search depth, and seeds per measurement.
EPISODES ?= 400
ROUNDS   ?= 40
PLIES    ?= 60
DEPTH    ?= 1
SEEDS_N  ?= 3

# Add DISCOVERED=1 to any target to route on GREN's measurement instead, and
# DISCOVERED_VOCAB=1 to take its derived input vocabulary too.  The vocabulary
# is opt-in because it is measured to cost transfer -- see README.md.
DISC     := $(if $(DISCOVERED_VOCAB),--discovered --discovered-vocab,$(if $(DISCOVERED),--discovered,))
COMMON   = --games $(GAMES) --seed $(SEED) --episodes $(EPISODES) $(DISC)

.PHONY: help check test test-v1 test-discovered demo map map-discovered play sudoku \
        transfer credit discovered denominators information all clean

help: ## Show this help
	@echo "CyclicCortex make targets  (override variables like: make demo EPISODES=1200)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

##@ Setup
check: ## Byte-compile the package and its tests (a syntax check, no tools needed)
	$(PY) -m compileall -q cortex tests && echo "ok"

##@ Tests
test: test-v1 test-discovered ## Both suites

test-v1: ## Regions, SBNN growth identities, Shapley credit, the rules themselves
	$(PY) -m tests.test_v1

test-discovered: ## The GREN handoff: does the measured map reproduce the hand-written one?
	$(PY) -m tests.test_discovered

##@ The cortex
demo: ## The tour: build the cortex, train, play chess, add sudoku, show every region
	$(CLI) demo --episodes $(EPISODES) --rounds $(ROUNDS) --plies $(PLIES) --seed $(SEED) $(DISC)

map: ## The similarity graph, its regions, and the routing log
	$(CLI) map --games $(GAMES) $(DISC)

map-discovered: ## The same map, built from GREN's measurement instead of the hand-written sets
	$(CLI) map --games $(GAMES) --discovered --discovered-vocab

play: ## Play a full game against the engine — PLIES plies, opponent depth DEPTH
	$(CLI) play --episodes $(EPISODES) --plies $(PLIES) --depth $(DEPTH) --seed $(SEED) $(DISC)

sudoku: ## Train the sudoku region and solve a puzzle with it
	$(CLI) sudoku --episodes $(EPISODES) --seed $(SEED) $(DISC)

##@ Measurements
transfer: ## What one region's learning is worth to another game
	$(CLI) transfer $(COMMON)

credit: ## Exact Shapley credit over regions, plus the routing auction
	$(CLI) credit $(COMMON)

discovered: ## Discovered map vs hand-written, and what the shared vocabulary is worth
	$(CLI) discovered --games $(GAMES) --games-n $(SEEDS_N) --episodes $(EPISODES)

denominators: ## Does a generalised vocabulary let one network learn chess and checkers?
	$(PY) common_denominator.py

information: ## What information does move legality actually require?
	$(PY) information.py

##@ Everything
all: check test map discovered ## Check, test, then the map and the handoff comparison

clean: ## Remove byte-code caches (data/ is kept — it holds GREN's handoff)
	rm -rf __pycache__ cortex/__pycache__ tests/__pycache__
	@echo "ok"
