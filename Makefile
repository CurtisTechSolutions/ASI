# ASI — one entry point for every project in the repository.  `make help` lists
# every target.
#
# Everything under Tests, GREN, CyclicCortex, Compression and Experiments is
# pure standard-library python3 and runs with nothing installed.  `make deps`
# reports what the remaining targets need and whether it is here.
#
# Sub-projects that carry their own Makefile (AudioImage, RadixCyclicNN,
# TwoNRL_CartPole) are delegated to rather than duplicated — `make audioimage`
# lists theirs.
#
# Override any variable on the command line, e.g. `make handoff BUDGET=3000`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY        ?= python3
SEED      ?= 0
GAMES     ?= chess checkers go sudoku
HANDOFF   ?= CyclicCortex/data/gren_packages.json

# GREN: probes per game, and the policy that picks them
BUDGET    ?= 1500
POLICY    ?= eig
HIDDEN    ?= 24

# CyclicCortex: training positions per game, self-play games, plies per game,
# opponent search depth, and seeds per measurement
EPISODES  ?= 400
ROUNDS    ?= 40
PLIES     ?= 60
DEPTH     ?= 1
SEEDS_N   ?= 3

GREN_CLI  := cd GREN && $(PY) -m gren.cli
CX_CLI    := cd CyclicCortex && $(PY) -m cortex.cli

.PHONY: help deps check audioimage radixcyclic cartpole test test-quick test-gren test-cortex test-audioimage test-radixcyclic \
        test-cartpole handoff package discovered explore similar policies tree grow \
        demo map map-discovered play sudoku transfer credit compression partition \
        vanishing consolidate compression-all denominators information \
        depth-normalisation activation all clean

help: ## Show this help
	@echo "ASI make targets  (override variables like: make handoff BUDGET=3000)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: SEED=$(SEED)  BUDGET=$(BUDGET)  POLICY=$(POLICY)  EPISODES=$(EPISODES)"
	@echo "          GAMES=\"$(GAMES)\"  HANDOFF=$(HANDOFF)"

##@ Setup
deps: ## Report what the optional dependencies are and whether they are here
	@echo "Pure standard library — always available:"
	@echo "    GREN, CyclicCortex, NeuralCompression, AudioImage, RadixCyclicNN (python side),"
	@echo "    Research/experiments, and \`make activation\`"
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

check: ## Byte-compile every pure-python package (a syntax check, no tools needed)
	$(PY) -m compileall -q GREN/gren GREN/tests CyclicCortex/cortex CyclicCortex/tests \
		NeuralCompression Research/experiments utils && echo "ok"

##@ Tests
test: test-gren test-cortex test-audioimage test-radixcyclic ## Every suite that needs nothing installed (minutes: RadixCyclicNN trains models)

test-quick: test-gren test-cortex ## Just GREN and CyclicCortex (~40s) — the fast loop

test-gren: ## GREN: verdicts, the radix tree, growth identities, the derived vocabulary
	cd GREN && $(PY) -m tests.test_gren

test-cortex: ## CyclicCortex: regions, SBNN growth, Shapley, and the GREN handoff
	cd CyclicCortex && $(PY) -m tests.test_v1 && $(PY) -m tests.test_discovered

test-audioimage: ## AudioImage's own suite
	$(MAKE) -C AudioImage test

test-radixcyclic: ## RadixCyclicNN's own suite
	$(MAKE) -C RadixCyclicNN test

test-cartpole: ## TwoNRL_CartPole's 14 proofs (needs numpy + gymnasium)
	$(MAKE) -C TwoNRL_CartPole test

##@ The GREN -> CyclicCortex handoff
handoff: package discovered ## Probe every game, then rebuild the cortex's map from what was measured

package: ## GREN probes every game and writes the GamePackages into CyclicCortex/data/
	$(GREN_CLI) package --budget $(BUDGET) --policy $(POLICY) --seed $(SEED) \
		--out ../$(HANDOFF)

discovered: ## Compare the discovered map against the hand-written one, and measure what sharing is worth
	$(CX_CLI) discovered --games $(GAMES) --games-n $(SEEDS_N) --episodes $(EPISODES)

##@ GREN — learn a game's identity from what it refuses
explore: ## Probe every game: what does each one refuse, and how often?
	$(GREN_CLI) explore --budget $(BUDGET) --policy $(POLICY) --seed $(SEED)

similar: ## Discovered similarity against CyclicCortex's hand-written sets
	$(GREN_CLI) similar --budget $(BUDGET) --policy $(POLICY) --seed $(SEED)

policies: ## Failure rate and rule coverage by probe policy
	$(GREN_CLI) policies --budget $(BUDGET) --seed $(SEED)

tree: ## The radix tree over discovered signatures (trie mechanics, path compression)
	$(GREN_CLI) tree --budget $(BUDGET) --policy $(POLICY) --seed $(SEED)

grow: ## One Self-Building network growing its inputs and outputs across all four games
	$(GREN_CLI) grow --budget $(BUDGET) --policy $(POLICY) --seed $(SEED) --hidden $(HIDDEN)

##@ CyclicCortex — one network per region of a cyclic similarity graph
demo: ## The tour: build the cortex, train, play chess, add sudoku, show every region
	$(CX_CLI) demo --episodes $(EPISODES) --rounds $(ROUNDS) --plies $(PLIES) --seed $(SEED)

map: ## The similarity graph, its regions, and the routing log
	$(CX_CLI) map --games $(GAMES)

map-discovered: ## The same map, built from GREN's measurement instead of the hand-written sets
	$(CX_CLI) map --games $(GAMES) --discovered --discovered-vocab

play: ## Play a full game against the engine (PLIES plies, opponent depth DEPTH)
	$(CX_CLI) play --episodes $(EPISODES) --plies $(PLIES) --depth $(DEPTH) --seed $(SEED)

sudoku: ## Train the sudoku region and solve a puzzle with it
	$(CX_CLI) sudoku --episodes $(EPISODES) --seed $(SEED)

transfer: ## What one region's learning is worth to another game
	$(CX_CLI) transfer --games $(GAMES) --episodes $(EPISODES) --seed $(SEED)

credit: ## Exact Shapley credit over regions, plus the routing auction
	$(CX_CLI) credit --games $(GAMES) --episodes $(EPISODES) --seed $(SEED)

##@ Neural compression — can one network hold many games?
compression: ## Split the output range, one band per game, plus a selector
	cd NeuralCompression && $(PY) experiment.py

partition: ## Partition layout: what dust costs, and what a relayout costs
	cd NeuralCompression && $(PY) partition.py

vanishing: ## The vanishing gradient as a feature: invert out of it
	cd NeuralCompression && $(PY) vanishing.py

consolidate: ## The dual network: train granularly, consolidate into the second
	cd NeuralCompression && $(PY) consolidate.py

compression-all: compression partition vanishing consolidate ## Every compression experiment, in order

##@ Experiments
denominators: ## Does a generalised vocabulary let one network learn chess and checkers?
	cd CyclicCortex && $(PY) common_denominator.py

information: ## What information does move legality actually require?
	cd CyclicCortex && $(PY) information.py

depth-normalisation: ## Is the vanishing gradient a measurement problem?  (minutes)
	cd Research/experiments && $(PY) depth_counted_normalisation.py

activation: ## A self-building network measured on its activation function  (minutes)
	cd ActivationFunctionTest && $(PY) self_building_sinewave.py

##@ Sub-projects  (each carries its own Makefile — these list its targets)
audioimage: ## AudioImage: audio encoded into a 2D plane and decoded back
	$(MAKE) -C AudioImage help

radixcyclic: ## RadixCyclicNN: the count / reward model, Go port, frontend and Docker stack
	$(MAKE) -C RadixCyclicNN help

cartpole: ## TwoNRL_CartPole: fail on purpose, invert, fine-tune
	$(MAKE) -C TwoNRL_CartPole help

##@ Everything
all: check test ## Byte-compile everything, then run every suite that needs nothing installed

clean: ## Remove byte-code caches everywhere (results and handoffs are kept)
	find . -name __pycache__ -type d -prune -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "ok"
