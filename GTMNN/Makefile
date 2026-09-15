# GTMNN — run commands.  `make help` lists every target.
#
# A population of micro networks that play a game against each other; the
# prediction is the equilibrium and credit is the Shapley value. Zero-dependency
# python3. `transfer` and `modifier` read GREN's handoff and CyclicCortex's game
# adapters, so those directories must be present beside this one.
#
# Override any variable on the command line, e.g. `make demo MICROS=512`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY     ?= python3
CLI    := $(PY) -m gtmnn.cli

# Population size, feature width, receptive field per micro, seats per stage
# game, solver iterations, Shapley permutations, epochs, learning rate.
MICROS ?= 192
FEATS  ?= 128
RF     ?= 12
SEATS  ?= 24
ITERS  ?= 16
PERMS  ?= 8
EPOCHS ?= 6
LR     ?= 0.05
SEED   ?= 0
TRIALS ?= 3
GENS   ?= 3

COMMON = --micros $(MICROS) --features $(FEATS) --rf $(RF) --seats $(SEATS) \
         --iters $(ITERS) --permutations $(PERMS) --seed $(SEED)

.PHONY: help check test demo game shapley cycles auction modifier transfer evolve all clean

help: ## Show this help
	@echo "GTMNN make targets  (override variables like: make demo MICROS=512)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

##@ Setup
check: ## Byte-compile the package and its tests (a syntax check, no tools needed)
	$(PY) -m compileall -q gtmnn tests && echo "ok"

##@ Tests
test: ## 36 tests -- the Shapley axioms are the load-bearing ones
	$(PY) -m tests.test_gtmnn

##@ The population
demo: ## Train a population; both losses, gini, diversity, cycles, meta, Shapley
	$(CLI) demo $(COMMON) --epochs $(EPOCHS) --lr $(LR)

game: ## One stage game inspected -- bids, price, equilibrium, loads, coalitions
	$(CLI) game $(COMMON) --epochs $(EPOCHS)

evolve: ## Cull the bankrupt, breed the winners, score evolutionary stability
	$(CLI) evolve $(COMMON) --generations $(GENS)

##@ The claims, checked
shapley: ## Efficiency, null player, symmetry and exactness, numerically
	$(CLI) shapley

cycles: ## Rock-paper-scissors: the cycle detector and the time-average
	$(CLI) cycles

auction: ## Sweep a micro's bid -- no misreport beats the truth
	$(CLI) auction --seed $(SEED)

##@ The GREN handoff
modifier: ## Modifier cosine against the Ochiai similarity of the mechanic sets
	$(CLI) modifier

transfer: ## Train on chess, evaluate checkers -- with and without the modifier
	$(CLI) transfer $(COMMON) --trials $(TRIALS) --lr $(LR)

##@ Everything
all: check test shapley cycles modifier ## Check, test, then every claim that is fast

clean: ## Remove byte-code caches
	rm -rf __pycache__ gtmnn/__pycache__ tests/__pycache__
	@echo "ok"
