# NeuralCompression — run commands.  `make help` lists every target.
#
# Four experiments asking whether ONE network can hold many games, and what the
# vanishing gradient is good for.  Every result they print is reproduced in
# FINDINGS.md.  Zero-dependency python3; each takes seconds to a few minutes.
#
# Override any variable on the command line, e.g. `make all PY=python3.12`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY ?= python3

.PHONY: help check experiment partition vanishing consolidate all clean

help: ## Show this help
	@echo "NeuralCompression make targets  (override variables like: make all PY=python3.12)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} \
		 /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

##@ Setup
check: ## Byte-compile every experiment (a syntax check, no tools needed)
	$(PY) -m compileall -q . && echo "ok"

##@ Experiments  (in the order FINDINGS.md tells them)
experiment: ## Split the output range, one band per game, plus a selector
	$(PY) experiment.py

partition: ## Partition layout: what dust costs, and what a relayout costs
	$(PY) partition.py

vanishing: ## The vanishing gradient as a feature — invert out of it, and rotating inversion
	$(PY) vanishing.py

consolidate: ## The dual network: train granularly, consolidate into the second
	$(PY) consolidate.py

##@ Everything
all: experiment partition vanishing consolidate ## Every experiment, in order

clean: ## Remove byte-code caches
	rm -rf __pycache__
	@echo "ok"
