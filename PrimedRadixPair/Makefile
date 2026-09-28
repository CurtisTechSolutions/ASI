# PrimedRadixPair - run commands.  `make help` lists every target.
#
# Standard library only: no install step, `python3` is the whole dependency list.
# The phonetic codecs read the sibling ../PhoneticTokenizer checkout; the gpt2
# codec reads data/gpt2/{encoder.json,vocab.bpe} (see `make gpt2-files`).
# Override any variable on the command line, e.g. `make prime CODEC=syllables L=2`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY        ?= python3
RP        ?= $(PY) -m radixpair
MODEL     ?= model.json.gz
CODEC     ?= phones
L         ?= 3
DATA      ?= ../RadixCyclicNN/data/sample_corpus.txt
TOKDATA   ?= $(DATA)
TOP       ?=
PREFIX    ?= the cat sat on the
SPELL     ?=
LENGTH    ?= 20
MODE      ?= dijkstra
TRAVERSAL ?= reward
TEXT      ?= the cat sat on the mat
STRENGTH  ?= 1
UNITS     ?= 200000
SAMPLES   ?= 50
CKPT_DIR  ?= checkpoints
GPT2_DIR  ?= data/gpt2
GPT2_URL  ?= https://openaipublic.blob.core.windows.net/gpt-2/models/124M

TOPFLAG = $(if $(TOP),--top $(TOP),)

.PHONY: help test check prime train reward punish feedback predict generate score weights info bench compare feedback-bench rungs gpt2-files clean

help: ## Show this help
	@echo "PrimedRadixPair make targets  (override like: make prime CODEC=syllables L=2)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: MODEL=$(MODEL) CODEC=$(CODEC) L=$(L) DATA=$(DATA)  (phones in, phones out; a prediction also shows the English it spells)"

##@ Checks
test: ## Run the test suite (seconds)
	$(PY) -m unittest discover -s tests -t . -v

check: ## The brute-force oracle: prime small trees literally and compare with the arithmetic
	$(RP) check

##@ The model
prime: ## Make a primed model file (CODEC, L; TOKDATA for bpe / syllables --vocabulary corpus / a capped gpt2)
	$(RP) prime --model $(MODEL) --codec $(CODEC) --L $(L) $(TOPFLAG) --tokenizer-data $(TOKDATA)

train: ## Count DATA into the count tree
	$(RP) train --model $(MODEL) --data $(DATA)

reward: ## Reward TEXT at STRENGTH
	$(RP) reward --model $(MODEL) --text "$(TEXT)" --strength $(STRENGTH)

punish: ## Punish TEXT at STRENGTH (a negative reward)
	$(RP) punish --model $(MODEL) --text "$(TEXT)" --strength $(STRENGTH)

predict: ## Continue PREFIX by LENGTH units (MODE, TRAVERSAL)
	$(RP) predict --model $(MODEL) --prefix "$(PREFIX)" --length $(LENGTH) --mode $(MODE) --traversal $(TRAVERSAL)

generate: ## A whole text from PREFIX
	$(RP) generate --model $(MODEL) --prefix "$(PREFIX)" --length $(LENGTH) --traversal $(TRAVERSAL)

score: ## Bits per unit of TEXT, and the reward tree's readings
	$(RP) score --model $(MODEL) --text "$(TEXT)" --per-unit

weights: ## Show the settings a prediction reads
	$(RP) weights --model $(MODEL)

info: ## What the model is and what it holds
	$(RP) info --model $(MODEL)

##@ Benches
bench: ## Throughput: counting and crediting UNITS random units
	$(RP) bench throughput --codec $(CODEC) --L $(L) --units $(UNITS)

compare: ## Bits per unit on a held-out split of DATA under backoff all / deepest / none
	$(RP) bench compare --codec $(CODEC) --L $(L) --data $(DATA) $(TOPFLAG)

feedback-bench: ## The mat / log case under both traversals
	$(RP) bench feedback --codec $(CODEC) --L $(L)

rungs: ## Rewards at every level against the final nodes only, on DATA
	$(RP) bench rungs --codec $(CODEC) --L $(L) --data $(DATA) --samples $(SAMPLES)

##@ Data
gpt2-files: ## Fetch GPT-2's encoder.json and vocab.bpe into data/gpt2 (needs network access)
	mkdir -p $(GPT2_DIR)
	curl -sSL -o $(GPT2_DIR)/encoder.json $(GPT2_URL)/encoder.json
	curl -sSL -o $(GPT2_DIR)/vocab.bpe $(GPT2_URL)/vocab.bpe
	@ls -la $(GPT2_DIR)

clean: ## Remove model files, checkpoints and caches
	rm -rf $(MODEL) $(CKPT_DIR) __pycache__ radixpair/__pycache__ tests/__pycache__
