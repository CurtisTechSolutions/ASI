# RadixAcyclicNN - run commands.  `make help` lists every target.
#
# Standard library only: no install step, nothing to download, `python3` is the
# whole dependency list.  Every target is deterministic given SEED.
#
# Override any variable on the command line, e.g. `make demo DEPTH=8 EPOCHS=20`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY        ?= python3
RT        ?= $(PY) -m radixtree
DATA      ?= data/sample_corpus.txt
EPOCHS    ?= 10
LR        ?= 0.5
ACT_LR    ?= 0.05
BATCH     ?= 32
DEPTH     ?=
MIN_COUNT ?= 1
SEED      ?= 1
MODEL     ?= model.json.gz
PREFIX    ?= the quick
LENGTH    ?= 30
RESULTS   ?= results

TRAIN_OPTS = --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --act-lr $(ACT_LR) --batch-size $(BATCH) \
             --min-count $(MIN_COUNT) --seed $(SEED) $(if $(DEPTH),--depth $(DEPTH),)

.PHONY: help test check demo train predict generate score stats 2nrl corpus window compare compare-quick compare-all clean

help: ## Show this help
	@echo "RadixAcyclicNN make targets  (override like: make demo DEPTH=8 EPOCHS=20)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: DATA=$(DATA) EPOCHS=$(EPOCHS) LR=$(LR) BATCH=$(BATCH) DEPTH=$(if $(DEPTH),$(DEPTH),none) MIN_COUNT=$(MIN_COUNT) SEED=$(SEED)"

##@ Checks
test: ## Run the tests (about half a minute, standard library)
	$(PY) -m unittest tests.test_radixtree

check: ## Finite-difference checks of the one-hop rule
	$(RT) check

##@ Looking at it
demo: ## Train on the sample corpus and see what the tree does
	$(RT) demo $(TRAIN_OPTS)

train: ## Train a tree on DATA and save it to MODEL
	$(RT) train $(TRAIN_OPTS) --save $(MODEL)

predict: ## Continue PREFIX from the saved MODEL (to an END leaf)
	$(RT) predict --load $(MODEL) "$(PREFIX)" --length $(LENGTH) --to-end

generate: ## Sample three texts from the saved MODEL
	$(RT) generate --load $(MODEL) --count 3 --sample-seed $(SEED)

score: ## Bits per character of DATA under the saved MODEL
	$(RT) score --load $(MODEL) --texts $(DATA)

stats: ## Size and shape of the saved MODEL
	$(RT) stats --load $(MODEL)

2nrl: ## Train on the garbage, invert, fine-tune on the corpus (2NRL)
	$(RT) 2nrl --data $(DATA) --epochs 2 --lr 1.0 --batch-size 1 --neg-lr 1.0 --pos-lr 1.0 --neg-epochs 2 --pos-epochs 2

##@ The comparison with RadixCyclicNN
corpus: ## Rebuild the prose corpus snapshot from the pinned research papers
	$(RT) corpus

window: ## Next-token accuracy of the token-by-token walk over windows, trained on the prose corpus with one fifth held out
	$(RT) accuracy --data data/corpus.txt --heldout-every 5 --epochs $(EPOCHS) --lr $(LR) --act-lr $(ACT_LR) \
		--batch-size $(BATCH) --seed $(SEED) --windows 1,2,3,4,6,8,all --out $(RESULTS)/window_results.json

compare-quick: ## Both models on the sample corpus - seconds
	$(RT) compare --data data/sample_corpus.txt --heldout-every 0 --seed $(SEED) --out $(RESULTS)/sample_results.json

compare: ## Both models on the prose corpus, one fifth held out - about a quarter of an hour
	$(RT) compare --seed $(SEED) --epochs $(EPOCHS) --lr $(LR) --act-lr $(ACT_LR) --batch-size $(BATCH) \
		--min-count $(MIN_COUNT) $(if $(DEPTH),--depth $(DEPTH),) --out $(RESULTS)/prose_results.json

compare-all: compare-quick ## Every run README.md quotes: the sample corpus, then the prose corpus plain, trusted twice, bounded to eight and to two
	$(RT) compare --seed $(SEED) --out $(RESULTS)/prose_results.json
	$(RT) compare --seed $(SEED) --min-count 2 --out $(RESULTS)/prose_min2_results.json
	$(RT) compare --seed $(SEED) --depth 8 --out $(RESULTS)/prose_depth8_results.json
	$(RT) compare --seed $(SEED) --depth 2 --out $(RESULTS)/prose_depth2_results.json

clean: ## Remove __pycache__ and saved models (results and the corpus are kept)
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f model.json model.json.gz
