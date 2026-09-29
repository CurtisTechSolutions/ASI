# LatentRadixPair in Rust, with the web frontend.
CARGO ?= cargo
BIN ?= target/release/latentpair
DOCS ?= $(wildcard ../../*/README.md ../../*/DESIGN.md ../../*/PRD.md ../../README.md)
STEPS ?= 2000
LEVELS ?= 4,4:4,4:4,4
TOKENIZER ?= tokenizer.json.gz
MODEL ?= model.json.gz
ADDR ?= 127.0.0.1:8080

.PHONY: build test demo tokenizer model bench serve fixtures frontend frontend-dev clean

## the binary embeds web/dist, so run `make frontend` first when the frontend changed (needs Node)
build:
	$(CARGO) build --release

## the React frontend (Vite + TypeScript) into web/dist, which the binary embeds
frontend:
	cd web && npm install --no-audit --no-fund && npm run build

## the Vite dev server with hot reload, proxying /api to a running `latentpair serve` on 127.0.0.1:8080
frontend-dev:
	cd web && npm install --no-audit --no-fund && npm run dev

## unit tests, the parity tests against the Go implementation's fixtures, and the server's compile
test:
	$(CARGO) test

## the whole loop on this repository's documents, in about a minute
demo: build
	$(BIN) demo --steps 400 $(DOCS)

## train the tokenizer on the documents
tokenizer: build
	$(BIN) tokenizer train --out $(TOKENIZER) --steps $(STEPS) --levels $(LEVELS) $(DOCS)

## prime the trees for the tokenizer and read the documents
model: build
	$(BIN) prime --tokenizer $(TOKENIZER) --out $(MODEL)
	$(BIN) train --model $(MODEL) $(DOCS)

## held-out bits per byte against raw byte contexts
bench: build
	$(BIN) bench --tokenizer $(TOKENIZER) $(DOCS)

## the web frontend and JSON API on $(ADDR)
serve: build
	$(BIN) serve --model $(MODEL) --addr $(ADDR)

## regenerate the parity fixtures from the Go implementation (needs Go)
fixtures:
	cd .. && go run ./cmd/fixtures

clean:
	$(CARGO) clean
