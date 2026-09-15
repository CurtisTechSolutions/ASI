# 2NRL on chess - run commands.  `make help` lists every target.
#
# Every arm runs the same loop against the same Stockfish with the same engine
# budget, the same number of gradient updates and the same learning-rate
# schedule.  What changes between them is what the first block of updates
# trains toward, and whether the network is inverted afterwards.
#
# Override any variable on the command line, e.g. `make run SEEDS="0 1 2 3"`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY          ?= python3
EXPERIMENT  ?= $(PY) experiment.py
SEEDS       ?= 0 1 2
ROUNDS      ?= 20
GAMES       ?= 8
EVAL_GAMES  ?= 10
JOBS        ?= 4
OPENINGS    ?= 12
RESULTS     ?= results
WEIGHTS     ?= weights
POSITIONS   ?= 400
VIZ_SEED    ?= 0

COMMON = --seeds $(SEEDS) --rounds $(ROUNDS) --games $(GAMES) \
         --eval-games $(EVAL_GAMES) --jobs $(JOBS) --weights $(WEIGHTS)

.PHONY: help install check test data run entropy repulsion all-arms per-round phase-depth no-growth viz viz-data bench bench-all quick all summary clean

help: ## Show this help
	@echo "2NRL on chess make targets  (override variables like: make run SEEDS=\"0 1\")"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: SEEDS=$(SEEDS) ROUNDS=$(ROUNDS) GAMES=$(GAMES) JOBS=$(JOBS)"

##@ Setup
install: ## numpy + python-chess, and Stockfish itself
	$(PY) -m pip install numpy
	$(PY) -m pip install --only-binary :all: chess
	@command -v stockfish >/dev/null 2>&1 || ls /usr/games/stockfish >/dev/null 2>&1 \
		|| { echo "installing stockfish (needs sudo on most systems)"; \
		     (apt-get install -y stockfish || sudo apt-get install -y stockfish); }

check: ## Byte-compile everything and report what is installed
	$(PY) -m compileall -q sbnn.py features.py moves.py agent.py engine.py \
		dataset.py experiment.py benchmark.py summarize.py test_sbnn.py
	@$(PY) -c "import numpy, chess; from engine import find_stockfish; \
		print('numpy', numpy.__version__, '/ chess', chess.__version__, '/', find_stockfish())"

data: ## Build the held-out exam: POSITIONS positions, every legal move graded
	$(PY) dataset.py --positions $(POSITIONS) --depth 8

##@ Proofs
test: ## The correctness proofs: gradients, exact inversion, identity growth, the fast path
	$(PY) test_sbnn.py

##@ Experiments
run: ## The headline: 2NRL against the same thing without 2NRL
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl positive --out $(RESULTS)/main.json

entropy: ## The H(q) ladder of §11: worst move, own failure, random move
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl-worst 2nrl 2nrl-random \
		--out $(RESULTS)/entropy.json

repulsion: ## Arm E: the same negatives pushed away instead of inverted
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms repulsion --out $(RESULTS)/repulsion.json

all-arms: ## Every arm in one go
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl 2nrl-worst 2nrl-random positive repulsion \
		--out $(RESULTS)/all_arms.json

per-round: ## Ablation: §9.3's perpetual loop - invert inside every round
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl positive --schedule per-round --out $(RESULTS)/per_round.json

phase-depth: ## Ablation (§12 item 4): how long should phase 1 be?
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl --neg-fraction 0.15 \
		--out $(RESULTS)/phase_depth_15.json
	$(EXPERIMENT) $(COMMON) --arms 2nrl --neg-fraction 0.50 \
		--out $(RESULTS)/phase_depth_50.json

no-growth: ## Ablation: freeze the architecture, so nothing self-builds
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --arms 2nrl positive --no-growth \
		--out $(RESULTS)/no_growth.json

quick: ## One seed, a few rounds - a smoke test of the whole loop
	$(EXPERIMENT) --seeds 0 --rounds 4 --games 4 --eval-games 4 --jobs 2 \
		--arms 2nrl positive --weights $(WEIGHTS)

##@ Visualisation
viz: ## Capture the phase-boundary networks, export the data, and build viz/index.html
	$(EXPERIMENT) --arms 2nrl --seeds $(VIZ_SEED) --rounds $(ROUNDS) --games $(GAMES) \
		--eval-games 4 --jobs 1 --weights $(WEIGHTS) --checkpoints viz/checkpoints \
		--out $(RESULTS)/viz_run.json
	$(PY) export_viz.py --results $(RESULTS) --seed $(VIZ_SEED)
	$(PY) viz/build.py
	@echo "open viz/index.html"

viz-data: ## Rebuild viz/index.html from the checkpoints already captured
	$(PY) export_viz.py --results $(RESULTS) --seed $(VIZ_SEED)
	$(PY) viz/build.py

##@ Benchmarking
bench: ## Play the trained arms against each other and a common opponent
	@mkdir -p $(RESULTS)
	$(PY) benchmark.py --weights $(WEIGHTS) --seeds $(SEEDS) --openings $(OPENINGS) \
		--arms 2nrl positive --out $(RESULTS)/benchmark.json

bench-all: ## The same, over every arm that has trained weights
	@mkdir -p $(RESULTS)
	$(PY) benchmark.py --weights $(WEIGHTS) --seeds $(SEEDS) --openings $(OPENINGS) \
		--arms 2nrl 2nrl-worst 2nrl-random positive repulsion \
		--out $(RESULTS)/benchmark.json

##@ Everything
all: test run entropy repulsion per-round bench-all summary viz ## Proofs, every arm, the tables, then the page

summary: ## Rebuild the README tables from whatever is in RESULTS/
	$(PY) summarize.py --results $(RESULTS)

clean: ## Remove byte-code caches (RESULTS/ and viz/ are kept - they back the README)
	rm -rf __pycache__
