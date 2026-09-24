# Universal game encoding on RadixCyclicNN - run commands.  `make help` lists every target.
#
# Nothing to install.  The experiment needs the standard library and the
# network in ../../RadixCyclicNN; set RADIXNET_PATH if this directory has moved.
#
# Override any variable on the command line, e.g. `make rules SEEDS="0 1 2 3"`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY         ?= python3
EXPERIMENT ?= $(PY) experiment.py
SEEDS      ?= 0 1 2
RESULTS    ?= results
QUICKDIR   ?= results-quick

COMMON = --seeds $(SEEDS) --out-dir $(RESULTS)

.PHONY: help check test test-full quick all rules phi mix phase layout multi twonrl summary readme example show clean

help: ## Show this help
	@echo "Universal game encoding  (override variables like: make phi SEEDS=\"0 1\")"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: SEEDS=$(SEEDS) RESULTS=$(RESULTS)"
	@echo "          A full 'make all' is roughly an hour; 'make quick' is a few minutes."

##@ Checks
check: ## Byte-compile everything and confirm radixnet can be found
	$(PY) -m compileall -q uge experiment.py tests.py
	@$(PY) -c "from uge.net import import_radixnet, radixnet_root; import_radixnet(); print('radixnet found at', radixnet_root() or 'an installed package')"

test: ## The test suite: perft to depth 3, the codebooks, the partition, round trips, the referee
	$(PY) tests.py

test-full: ## The same, plus perft to depth 4 and every spec combination
	$(PY) tests.py --full

##@ Arms
rules: ## Six games, one encoder: how far above guessing does the tape get the network?
	$(EXPERIMENT) rules $(COMMON)

phi: ## The abstraction dial, from a move-only tape to an injective one
	$(EXPERIMENT) phi $(COMMON)

mix: ## One token, split between a hash of the position and a summary of the last move
	$(EXPERIMENT) mix $(COMMON)

phase: ## Does a trigram belonging to two window phases matter?  (plain vs disjoint)
	$(EXPERIMENT) phase $(COMMON)

layout: ## Does the arrangement of digits inside a token matter?
	$(EXPERIMENT) layout $(COMMON)

multi: ## One network, six games, the header token as the selector
	$(EXPERIMENT) multi $(COMMON)

twonrl: ## 2NRL on the refused moves, against a matched positive control
	$(EXPERIMENT) twonrl $(COMMON)

##@ Runs
quick: ## Every arm at a tenth of the corpus and one seed - for checking it runs
	@echo "writing to $(QUICKDIR)/ so the three-seed numbers in README.md survive"
	$(EXPERIMENT) all --quick --seeds 0 --out-dir $(QUICKDIR)

all: ## Every arm at full size, every seed - this is what README.md quotes
	$(EXPERIMENT) all $(COMMON)

##@ Output
summary: ## Turn results/*.json into the markdown tables in README.md
	@$(EXPERIMENT) summary --out-dir $(RESULTS)

readme: ## The same, spliced into README.md between its <!--NAME--> markers
	@$(EXPERIMENT) summary --out-dir $(RESULTS) --into-readme

example: ## Run USING.md's tutorial: add a game (Hexapawn), train on it, play it
	$(PY) examples/hexapawn.py

show: ## Print one encoded tape per game, decoded back, to see the format
	@$(PY) -c "import sys; sys.path.insert(0, '.'); \
import uge; from uge.tape import TapeSpec; from uge.codec import decode; \
from uge.corpus import self_play; from uge.teachers import make_policy; \
[print(f'--- {n} ---', (sp := TapeSpec(n, uge.get_game(n).index, phi_bits=12)).describe(), \
(c := self_play(uge.get_game(n), sp, 1, make_policy('random'), seed=1)).tapes[0][:120], \
decode(uge.get_game(n), sp, c.tapes[0]).to_dict(), sep='\n') for n in uge.game_names()]"

clean: ## Remove the quick-run results and the bytecode (never $(RESULTS))
	rm -rf $(QUICKDIR) uge/__pycache__ __pycache__
