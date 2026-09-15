# 2NRL on CartPole - run commands.  `make help` lists every target.
#
# Every target under "Versions" runs the same three-phase cycle (fail on
# purpose -> invert -> fine-tune) and changes one thing: what drives how far
# the network ventures out while it is learning to fail.  Each writes its own
# JSON into RESULTS/ and reproduces the matching row of README.md.
#
# Override any variable on the command line, e.g. `make margin SEEDS="0 1 2 3"`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY              ?= python3
EXPERIMENT      ?= $(PY) twonrl_cartpole.py
SEEDS           ?= 0 1 2
NEG_UPDATES     ?= 150
POS_UPDATES     ?= 60
ABL_POS_UPDATES ?= 1
RESULTS         ?= results

COMMON = --seeds $(SEEDS) --neg-updates $(NEG_UPDATES)

.PHONY: help install check test run margin speed no-venture no-weights quick all summary clean

help: ## Show this help
	@echo "2NRL on CartPole make targets  (override variables like: make margin SEEDS=\"0 1\")"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: SEEDS=$(SEEDS) NEG_UPDATES=$(NEG_UPDATES) POS_UPDATES=$(POS_UPDATES) RESULTS=$(RESULTS)"
	@echo "          Ablations fine-tune for ABL_POS_UPDATES=$(ABL_POS_UPDATES) update(s): they are"
	@echo "          measuring what the inversion alone buys, not the final score."

##@ Setup
install: ## Install the two dependencies (numpy, gymnasium)
	$(PY) -m pip install numpy gymnasium

check: ## Byte-compile everything and report the installed versions
	$(PY) -m compileall -q sinenet.py twonrl_cartpole.py test_invert.py summarize.py
	@$(PY) -c "import numpy, gymnasium; print('numpy', numpy.__version__, '/ gymnasium', gymnasium.__version__)"

##@ Proofs
test: ## The 14 proofs: analytic gradients, exact inversion, the parity no-op
	$(PY) test_invert.py

##@ Versions  (what drives the venture-out radius in the negative phase)
run: margin ## Alias for `margin` - the headline run

margin: ## Graded loss |target - achieved| sets the radius   -> 499.4 inverted
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --pos-updates $(POS_UPDATES) --grading margin --control \
		--out $(RESULTS)/twonrl_results.json

speed: ## Ratio delta L_min/L instead - saturates at the cap   -> 344.5 inverted
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --pos-updates $(POS_UPDATES) --grading speed --control \
		--out $(RESULTS)/grading_speed_results.json

no-venture: ## Never venture out, standard resets  -> 36.4, the policy collapses
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --pos-updates $(ABL_POS_UPDATES) --fixed-starts \
		--out $(RESULTS)/ablation_no_venture.json

no-weights: ## Drop two_nrl's bad_weights - every failure trains equally hard
	@mkdir -p $(RESULTS)
	$(EXPERIMENT) $(COMMON) --pos-updates $(ABL_POS_UPDATES) --no-delta-weights \
		--out $(RESULTS)/ablation_no_weights.json

quick: ## One seed, few updates - a smoke test of the whole cycle
	$(EXPERIMENT) --seeds 0 --neg-updates 60 --pos-updates 2

##@ Everything
all: test margin speed no-venture no-weights summary ## Proofs, every version, then the tables

summary: ## Rebuild the README tables from whatever is in RESULTS/
	$(PY) summarize.py

clean: ## Remove byte-code caches (RESULTS/ is kept - it backs the README)
	rm -rf __pycache__
