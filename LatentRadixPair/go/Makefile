# LatentRadixPair in Go: a trained context tokenizer under a primed radix pair.
GO ?= go
BIN ?= bin/latentpair
DOCS ?= $(wildcard ../../*/README.md ../../*/DESIGN.md ../../*/PRD.md ../../README.md)
STEPS ?= 2000
LEVELS ?= 4,4:4,4:4,4
TOKENIZER ?= tokenizer.json.gz
MODEL ?= model.json.gz

.PHONY: build test vet demo tokenizer model bench clean

build:
	$(GO) build -o $(BIN) ./cmd/latentpair

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

## the whole loop on this repository's documents, in about a minute
## (the Rust port and the web frontend live in ../rust)
demo: build
	$(BIN) demo --steps 400 $(DOCS)

## train the tokenizer on the documents (about four minutes at 2000 steps)
tokenizer: build
	$(BIN) tokenizer train --out $(TOKENIZER) --steps $(STEPS) --levels $(LEVELS) $(DOCS)

## prime the trees for the tokenizer and read the documents
model: build
	$(BIN) prime --tokenizer $(TOKENIZER) --out $(MODEL)
	$(BIN) train --model $(MODEL) $(DOCS)

## held-out bits per byte against raw byte contexts
bench: build
	$(BIN) bench --tokenizer $(TOKENIZER) $(DOCS)

clean:
	rm -rf bin
