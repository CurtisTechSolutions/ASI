# FilterBankRadix - run commands.  `make help` lists every target.
#
# Standard library only: no install step, nothing to download, `python3` is the
# whole dependency list.  Every target is deterministic given SEEDS.
#
# Override any variable on the command line, e.g. `make experiment BITS=3 DEPTH=4`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY         ?= python3
FB         ?= $(PY) -m fbradix.cli
BITS       ?= 2
DEPTH      ?= 5
ROUNDS     ?= 3
EPOCHS     ?= 3
SEEDS      ?= 3
PER_SOURCE ?= 200
RESULTS    ?= results

COMMON = --bits $(BITS) --depth $(DEPTH) --rounds $(ROUNDS) --epochs $(EPOCHS)

.PHONY: help test check demo route generate train corpus quick experiment main depth probe scale all clean

help: ## Show this help
	@echo "FilterBankRadix make targets  (override like: make demo BITS=3 DEPTH=4)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: BITS=$(BITS) ($$((1 << $(BITS))) experts) DEPTH=$(DEPTH) ROUNDS=$(ROUNDS) EPOCHS=$(EPOCHS) SEEDS=$(SEEDS)"

##@ Checks
test: ## Run the 25 tests (seconds)
	$(PY) -m tests.test_fbradix

check: ## Finite-difference checks of both learning rules
	$(FB) check

##@ Looking at it
demo: ## Fit a small bank and print what each address ended up holding
	$(FB) $(COMMON) demo

route: ## Show the filter's responses and the address for one piece of text
	$(FB) $(COMMON) route

generate: ## Sample characters, re-routing after every one
	$(FB) $(COMMON) generate

train: ## Fit one arm (ARM=learned|single|oracle|frozen|monotone|roundrobin|unbalanced|deep|deep-oracle)
	$(FB) $(COMMON) train --arm $(or $(ARM),learned)

corpus: ## Rebuild the corpus snapshot from the pinned repository files
	$(FB) --per-source $(PER_SOURCE) corpus

##@ The experiment
quick: ## One seed, small corpus, two depths - about a minute
	$(FB) experiment --quick

experiment: ## Every sweep, every arm (slow: the numbers in README.md)
	$(FB) experiment --sweep all --seeds $(SEEDS) --per-source $(PER_SOURCE) \
		--bits $(BITS) --depth $(DEPTH) --rounds $(ROUNDS) --epochs $(EPOCHS)

main: ## Just the headline table: every arm, every seed
	$(FB) experiment --sweep main --seeds $(SEEDS) --per-source $(PER_SOURCE)

depth: ## Just the depth sweep: does routing matter more with less context?
	$(FB) experiment --sweep depth --per-source $(PER_SOURCE)

probe: ## Where the routing gap is: the signal, the address code, or the rule?
	$(FB) experiment --sweep probe --per-source $(PER_SOURCE)

scale: ## Just the scale sweep: does more text per expert rescue the bank?
	$(FB) experiment --sweep scale

all: check test experiment ## Checks, tests, then the full experiment

clean: ## Remove __pycache__ (results and the corpus snapshot are kept)
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
