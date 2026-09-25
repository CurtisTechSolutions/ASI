# PhoneticTokenizer - run commands.  `make help` lists every target.
#
# Three ports of one tokenizer: Python (phonetok/), Go (go/) and Rust (rust/).
# They read the same two data files and the parity targets hold them to the
# same output, token for token.

PY ?= python3
GO ?= go

.PHONY: help install install-cmudict test check demo eval hear replay codebook go-build go-test rust-build \
        rust-test rust-clippy parity sync-data test-all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install the Python package in editable mode (zero runtime dependencies)
	$(PY) -m pip install -e .

install-cmudict: ## Install with the full CMU Pronouncing Dictionary (135 000 words)
	$(PY) -m pip install -e ".[cmudict]"

test: ## Run the Python test suite (plain unittest; the port comparisons run when go / cargo are on PATH)
	$(PY) -m unittest discover -s tests -v

check: ## Byte-compile the package and say what the lexicon holds
	$(PY) -m compileall -q phonetok && $(PY) -m phonetok lexicon

TEXT ?= The quick brown fox jumps over the lazy dog.
demo: ## Tokenize TEXT at every level, then read it back
	$(PY) -m phonetok tokenize "$(TEXT)"
	$(PY) -m phonetok tokenize --level constituent "$(TEXT)"
	$(PY) -m phonetok tokenize --level syllable "$(TEXT)"
	$(PY) -m phonetok explain "$(TEXT)"
	$(PY) -m phonetok ipa "$(TEXT)"
	$(PY) -m phonetok coin --count 5 --seed 1

eval: ## Score the letter-to-sound rules against the full dictionary (needs cmudict or PHONETOK_LEXICON)
	$(PY) -m phonetok eval

WAV ?= build/said.wav
hear: ## The acoustic units of WAV (default: the synthesizer saying TEXT), then spoken back into build/replay.wav
	@mkdir -p build
	@test -f "$(WAV)" || $(PY) -m phonetok say "$(TEXT)" --out "$(WAV)"
	$(PY) -m phonetok hear "$(WAV)"
	$(PY) -m phonetok hear "$(WAV)" | $(PY) -m phonetok replay --out build/replay.wav --polish 8

UNITS ?= q2 q28 q55 q5 q60 q3
replay: ## Speak UNITS back through the vocoder into build/replay.wav
	@mkdir -p build
	$(PY) -m phonetok replay $(UNITS) --out build/replay.wav --polish 8

codebook: ## Relearn the bundled codebook (phonetok/data/acoustic.tsv) from the synthesizer's speech, then sync it
	$(PY) tests/make_codebook.py
	$(MAKE) sync-data

go-build: ## Build the Go port's command line into go/bin/phonetok
	cd go && $(GO) build -o bin/phonetok ./cmd/phonetok

go-test: ## Run the Go tests (the parity fixture included)
	cd go && $(GO) vet ./... && $(GO) test ./...

rust-build: ## Build the Rust port's command line into rust/target/release/phonetok
	cd rust && cargo build --release

rust-test: ## Run the Rust tests (the parity fixture included)
	cd rust && cargo test --release

rust-clippy: ## Clippy and the formatter check for the Rust port
	cd rust && cargo clippy --release --all-targets -- -D warnings && cargo fmt --check

parity: ## Regenerate tests/parity.json from Python, then hold the Go and Rust ports to it
	$(PY) tests/make_parity.py
	$(MAKE) go-test
	$(MAKE) rust-test
	$(PY) -m unittest tests.test_parity -v

sync-data: ## Copy the shared data files into the Go module (Rust reads them in place)
	cp phonetok/data/core.dict go/phonetok/data/core.dict
	cp phonetok/data/rules.lts go/phonetok/data/rules.lts
	cp phonetok/data/voice.tsv go/phonetok/data/voice.tsv
	cp phonetok/data/acoustic.tsv go/phonetok/data/acoustic.tsv

test-all: test go-test rust-test ## Every suite of every port

clean: ## Remove build products
	rm -rf build go/bin rust/target phonetok/__pycache__ tests/__pycache__
