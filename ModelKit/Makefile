# ModelKit - run commands.  `make help` lists every target.
#
# Everything around a model, in the model's ports: Python (modelkit/), Go (go/),
# Rust (rust/) and the React frontend (frontend/).  The model layer is built on
# the RadixCyclicNN model beside this checkout (MODEL_DIR); its Makefile keeps
# the day-to-day commands (make serve, make tutor, ...) and runs them from here.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY        ?= python3
GO        ?= go
CARGO     ?= cargo
MODEL_DIR ?= ../RadixCyclicNN

# the phonetic tokenizer and the model beside this checkout, for the tests and the model layer
export PYTHONPATH := $(PYTHONPATH):../PhoneticTokenizer:$(MODEL_DIR)

.PHONY: help install check test test-layers go-build go-test go-parity rust-build rust-test rust-parity parity \
        frontend-install frontend-build frontend-dev frontend-test clean

help: ## Show this help
	@echo "ModelKit make targets"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the kit and the model beside it in editable mode (zero runtime dependencies)
	$(PY) -m pip install -e . -e $(MODEL_DIR)

check: ## Byte-compile the kit and check that its layers hold
	$(PY) -m compileall -q modelkit
	$(PY) -m unittest tests.test_layers

test: ## Run the Python suite (the Go and Rust parity suites skip without go / cargo)
	$(PY) -m unittest discover -s tests -v

test-layers: ## The model-free layer loads with no model, and the model never imports the kit
	$(PY) -m unittest tests.test_layers -v

go-build: ## Build the Go CLI into go/bin/radixnet-count
	cd go && $(GO) build -o bin/radixnet-count ./cmd/radixnet-count

go-test: ## Vet and test the Go module with the race detector
	cd go && $(GO) vet ./... && $(GO) test -race ./...

go-parity: ## The Go port held to the Python model's output
	$(PY) -m unittest tests.test_go_parity -v

rust-build: ## Build the Rust CLI and benchmark into rust/target/release (needs Rust 1.82+)
	cd rust && $(CARGO) build --release

rust-test: ## Test the Rust crate, run clippy and the formatter check
	cd rust && $(CARGO) test && $(CARGO) clippy --all-targets -- -D warnings && $(CARGO) fmt --check

rust-parity: ## The Rust port held to the Python model's output (every tests/test_rust_parity*.py)
	$(PY) -m unittest discover -s tests -p 'test_rust_parity*.py' -t . -v

parity: go-parity rust-parity ## Both ports held to Python's output

frontend-install: ## Install the frontend dependencies (npm)
	cd frontend && npm install --no-audit --no-fund

frontend-build: ## Rebuild frontend/dist, which every port's server serves
	cd frontend && npm run build

frontend-dev: ## Vite dev server with hot reload on http://localhost:5173 (proxies /api to VITE_PROXY_TARGET)
	cd frontend && npm run dev

frontend-test: ## The frontend's unit tests (node --test, no npm install needed)
	cd frontend && npm test

clean: ## Remove Python caches and build output
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf go/bin
