# LatentRadixPair: the Go implementation in go/, the Rust port and the web frontend in rust/.
.PHONY: test go rust serve

## both implementations' tests
test:
	$(MAKE) -C go test
	$(MAKE) -C rust test

go:
	$(MAKE) -C go build

rust:
	$(MAKE) -C rust build

## build the frontend and the Rust backend, make a model if there is none, and serve it
serve:
	$(MAKE) -C rust serve
