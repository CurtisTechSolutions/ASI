# RadixDecayNN - run commands.  `make help` lists every target.
#
# Standard library only: no install step, nothing to download, `python3` is the
# whole dependency list.  Every target is deterministic given SEED.
#
# Override any variable on the command line, e.g. `make demo LIFE=2000 DECAY=linear`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY        ?= python3
RD        ?= $(PY) -m radixdecay
DATA      ?= data/sample_corpus.txt
READINGS  ?= 1
DEPTH     ?=
LIFE      ?= 10000
DECAY     ?= half-life
MIN_SEEN  ?= 0.5
UNIT      ?= char
SEED      ?= 1
MODEL     ?= model.json.gz
PREFIX    ?= the quick
LENGTH    ?= 30
TIMES     ?= 3
LIVES     ?= 1
RESULTS   ?= results

MODEL_OPTS = --data $(DATA) --readings $(READINGS) --life $(LIFE) --decay $(DECAY) --min-seen $(MIN_SEEN) \
             --unit $(UNIT) --seed $(SEED) $(if $(DEPTH),--depth $(DEPTH),)

.PHONY: help test demo read train say predict cheapest generate score accuracy stats tick experiments experiments-prose clean

help: ## Show this help
	@echo "RadixDecayNN make targets  (override like: make demo LIFE=2000 DECAY=linear)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: DATA=$(DATA) LIFE=$(LIFE) DECAY=$(DECAY) MIN_SEEN=$(MIN_SEEN) DEPTH=$(if $(DEPTH),$(DEPTH),none) UNIT=$(UNIT) SEED=$(SEED)"

##@ Checks
test: ## Run the tests (seconds, standard library)
	$(PY) -m unittest tests.test_radixdecay

##@ Looking at it
demo: ## Read the sample corpus and watch it remember, say, forget and recover
	$(RD) demo $(MODEL_OPTS)

read: ## Read DATA into a fresh model and save it to MODEL
	$(RD) read $(MODEL_OPTS) --save $(MODEL)

train: read ## The same, under the sibling models' name

say: ## Say PREFIX to its end TIMES over from MODEL, and save what that did to it
	$(RD) say --load $(MODEL) "$(PREFIX)" --length $(LENGTH) --to-end --times $(TIMES) --save $(MODEL)

predict: say ## The same, under the sibling models' name

cheapest: ## The cheapest continuation of PREFIX from MODEL (one search), saved back
	$(RD) cheapest --load $(MODEL) "$(PREFIX)" --length $(LENGTH) --to-end --save $(MODEL)

generate: ## Three texts said from START, sampled, saved back
	$(RD) generate --load $(MODEL) --count 3 --sample-seed $(SEED) --save $(MODEL)

score: ## Bits per unit of DATA under MODEL (nothing moves)
	$(RD) score --load $(MODEL) --texts $(DATA)

accuracy: ## Next-token accuracy of the token-by-token walk over windows, on DATA (nothing moves)
	$(RD) accuracy --load $(MODEL) --texts $(DATA)

stats: ## Size, memory and clock of MODEL
	$(RD) stats --load $(MODEL)

tick: ## Let LIVES lives pass in MODEL, and save it
	$(RD) tick --load $(MODEL) --lives $(LIVES) --save $(MODEL)

##@ The experiments README.md quotes
experiments: ## Habit, forgetting, use and recovery on the sample corpus - seconds
	$(RD) experiment $(MODEL_OPTS) --which all --out $(RESULTS) | tee $(RESULTS)/experiments.log

experiments-prose: ## The same four on the 800-line prose corpus, with a life of 100 000 - about a minute
	$(RD) experiment $(MODEL_OPTS) --data data/corpus.txt --life 100000 --which all --out $(RESULTS)/prose | tee $(RESULTS)/prose/experiments.log

clean: ## Remove __pycache__ and saved models (results and the corpus are kept)
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f model.json model.json.gz
