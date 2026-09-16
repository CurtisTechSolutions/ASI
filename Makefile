# ASI — one entry point for the repository.  `make help` lists every target.
#
# Each project owns its own Makefile and its own defaults; this one delegates
# with `make -C` rather than repeating them, so every command has exactly one
# home.  `make gren`, `make cortex`, `make compression`, `make audioimage`,
# `make radixcyclic` and `make cartpole` list what each project offers.
#
# Variables set on the COMMAND LINE flow down into the project Makefiles, so
# `make handoff BUDGET=3000` reaches GREN and `make demo EPISODES=1200` reaches
# CyclicCortex.  Their defaults live with them, not here.
#
# Everything under Tests and The handoff is pure standard-library python3 and
# runs with nothing installed.  `make deps` reports what the rest needs.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY ?= python3

PROJECTS := GREN GTMNN CyclicCortex Experiments/NeuralCompression AudioImage \
            RadixCyclicNN Experiments/TwoNRL_CartPole DISTIL TwoNRL_Chess

.PHONY: help deps check test test-quick test-gren test-cortex test-audioimage \
        test-radixcyclic test-cartpole test-gtmnn test-distil test-chess handoff \
        package discovered demo map depth-normalisation activation two-sine \
        gren gtmnn cortex compression audioimage radixcyclic cartpole distil \
        chess all clean

help: ## Show this help
	@echo "ASI make targets  (override variables like: make handoff BUDGET=3000)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

##@ Setup
deps: ## Report what the optional dependencies are, and whether they are here
	@echo "Pure standard library — always available:"
	@echo "    GREN, CyclicCortex, NeuralCompression, AudioImage, RadixCyclicNN (python side),"
	@echo "    DISTIL, Experiments/, and \`make activation\`"
	@echo
	@$(PY) -c "import importlib.util as u; \
	  [print('    %-12s %s' % (m, 'present' if u.find_spec(m) else 'MISSING  -> %s' % t)) \
	   for m, t in (('numpy', 'make test-cartpole, SBNN_RNN_ActivationFunction'), \
	                ('gymnasium', 'make test-cartpole, ActivationFunctionTest'), \
	                ('torch', 'RadixTrieLLM_RNN, RadixCyclicNN --backend torch'))]"
	@command -v stockfish >/dev/null && echo "    stockfish    present" \
		|| echo "    stockfish    MISSING  -> CyclicCortex falls back to its own engine"
	@command -v go >/dev/null && echo "    go           present" \
		|| echo "    go           MISSING  -> make -C RadixCyclicNN go-build"

check: ## Byte-compile every pure-python project (a syntax check, no tools needed)
	@$(MAKE) --no-print-directory -C GREN check
	@$(MAKE) --no-print-directory -C GTMNN check
	@$(MAKE) --no-print-directory -C CyclicCortex check
	@$(MAKE) --no-print-directory -C Experiments/NeuralCompression check
	@$(MAKE) --no-print-directory -C DISTIL check
	@$(PY) -m compileall -q Experiments/DepthCountedNormalisation utils && echo "ok"

##@ Tests
test: test-gren test-gtmnn test-cortex test-distil test-audioimage test-radixcyclic ## Every suite that needs nothing installed (minutes: RadixCyclicNN trains models)

test-quick: test-gren test-gtmnn test-cortex test-distil ## GREN, GTMNN, CyclicCortex and DISTIL — the fast loop

test-gren: ## GREN: verdicts, the radix tree, growth identities, the derived vocabulary
	@$(MAKE) --no-print-directory -C GREN test

test-gtmnn: ## GTMNN: the Shapley axioms, the cycle detector, the auction, the modifier
	@$(MAKE) --no-print-directory -C GTMNN test

test-cortex: ## CyclicCortex: regions, SBNN growth, Shapley, and the GREN handoff
	@$(MAKE) --no-print-directory -C CyclicCortex test

test-audioimage: ## AudioImage's own suite
	@$(MAKE) --no-print-directory -C AudioImage test

test-radixcyclic: ## RadixCyclicNN's own suite
	@$(MAKE) --no-print-directory -C RadixCyclicNN test

test-cartpole: ## TwoNRL_CartPole's 14 proofs (needs numpy + gymnasium)
	@$(MAKE) --no-print-directory -C Experiments/TwoNRL_CartPole test

test-distil: ## DISTIL's 288 tests — offline, no key, no network
	@$(MAKE) --no-print-directory -C DISTIL test

test-chess: ## TwoNRL_Chess's correctness proofs (needs numpy + python-chess; Stockfish to grade)
	@$(MAKE) --no-print-directory -C TwoNRL_Chess test

##@ The GREN -> CyclicCortex handoff  (the one pipeline that spans two projects)
handoff: ## Probe every game, then rebuild the cortex's map from what was measured
	@$(MAKE) --no-print-directory -C GREN package
	@$(MAKE) --no-print-directory -C CyclicCortex discovered

package: ## Just the GREN half: probe every game and write the handoff
	@$(MAKE) --no-print-directory -C GREN package

discovered: ## Just the CyclicCortex half: the discovered map against the hand-written one
	@$(MAKE) --no-print-directory -C CyclicCortex discovered

##@ Shortcuts  (the two you land on; each project's Makefile has the rest)
demo: ## The CyclicCortex tour: train, play chess, add sudoku, show every region
	@$(MAKE) --no-print-directory -C CyclicCortex demo

map: ## The similarity graph, its regions, and the routing log
	@$(MAKE) --no-print-directory -C CyclicCortex map

##@ Experiments with no project Makefile of their own
depth-normalisation: ## Is the vanishing gradient a measurement problem?  (minutes)
	cd Experiments/DepthCountedNormalisation && $(PY) depth_counted_normalisation.py

activation: ## A self-building network measured on its activation function  (minutes)
	cd Experiments/ActivationFunctionTest && $(PY) self_building_sinewave.py

two-sine: ## Adam's two moments read as two interfering waves  (~15s with --quick)
	cd Experiments/TwoSineOptimizer && $(PY) test_two_sine_optimizer.py --quick

##@ Projects  (each lists its own targets)
gren: ## GREN — learn a game's identity from what it refuses
	@$(MAKE) --no-print-directory -C GREN help

gtmnn: ## GTMNN — a population of micros playing a game; credit is the Shapley value
	@$(MAKE) --no-print-directory -C GTMNN help

cortex: ## CyclicCortex — one network per region of a cyclic similarity graph
	@$(MAKE) --no-print-directory -C CyclicCortex help

compression: ## NeuralCompression — can one network hold many games?
	@$(MAKE) --no-print-directory -C Experiments/NeuralCompression help

audioimage: ## AudioImage — audio encoded into a 2D plane, and decoded back
	@$(MAKE) --no-print-directory -C AudioImage help

radixcyclic: ## RadixCyclicNN — the count / reward model, Go port, frontend, Docker
	@$(MAKE) --no-print-directory -C RadixCyclicNN help

cartpole: ## TwoNRL_CartPole — fail on purpose, invert, fine-tune
	@$(MAKE) --no-print-directory -C Experiments/TwoNRL_CartPole help

distil: ## DISTIL — a self-improving agent whose memory is the embedding layer
	@$(MAKE) --no-print-directory -C DISTIL help

chess: ## TwoNRL_Chess — an SBNN that learns the rules by breaking them
	@$(MAKE) --no-print-directory -C TwoNRL_Chess help

##@ Everything
all: check test ## Byte-compile everything, then run every suite that needs nothing installed

clean: ## Remove caches and generated scratch everywhere (committed results and the handoff are kept)
	@for d in $(PROJECTS); do $(MAKE) --no-print-directory -C $$d clean >/dev/null; done
	@find . -name __pycache__ -type d -prune -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "ok"
