# rust

The Rust port of the kit: crate `modelkit`, everything built on the model's
crate (`radixnet`, in `../../RadixCyclicNN/rust`) — the teaching loops, the LLM
clients, the agent and its tools, images and speech, MCP, the HTTP API the
frontend talks to — and the two binaries, `radixnet` (the CLI) and
`radixnet-bench`. It depends on `radixnet` and `phonetok` by path and on
nothing else; the model's crate never depends on this one.

`../../RadixCyclicNN/rust/README.md` is the manual for both crates: which file
holds what, what is deliberately not ported, and how the two are held to
Python.

## Building and testing

From `..`:

```bash
make rust-build        # cargo build --release -> rust/target/release/radixnet{,-bench}
make rust-test         # cargo test, cargo clippy, cargo fmt --check
make rust-parity       # every tests/test_rust_parity*.py: the port held to Python
rust/target/release/radixnet --model model.count.json serve   # the API and this kit's frontend
```

`serve` looks for the frontend at `frontend/dist` under the working
directory: run it from `..`, or pass `--frontend-dir` (the model's
`make rust-serve` does).
