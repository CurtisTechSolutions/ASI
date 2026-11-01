# GameLattice

**A RadixCyclicNN whose step is a game, feeding a LatticeFSM whose every
state is one of the encoder's outputs, read out by decoders that are
separate from the lattice - with a body around it.** The senses are its
encoders, the effectors its decoders, and every effector's output comes
back in through a sense. [`DESIGN.md`](DESIGN.md) is the contract;
[`results/README.md`](results/README.md) is what it measured.

**Status: a first build in Rust.** The crate in `rust/` (`gamelattice`,
standard library only, depending on the sibling crates `../LatticeFSM/rust`
and `../RadixCyclicNN/rust`) has the encoder (§4), the lattice kinds (§5),
the decoder protocol with its four mechanisms (§6), the credit landings
(§7), the efference copy (§8), a text sense over the four regular languages
and a move tape over Nim, the gate, the clock, the selector, babbling, and
experiments E1 and E2 of §17. The Python reference the design planned comes
after the numbers, not before (§21.8, decided by the author: Rust first).
Not built yet: the senses of §9 beyond text and the tape, the association
lattice, the curriculum, persistence of a whole body, the server.

## Quick start

```bash
cd GameLattice
make demo                                   # even-b: the encoder names the lattice's states, the oracle credits each step
make train LANGUAGE=mod3-a EPISODES=4000    # held-out accuracy every tenth of the way
make plain LANGUAGE=mod3-a                  # the control: the lattice alone, per-step oracle
make nim                                    # babble 200 games, then 400 against a random opponent
make e1 && make e2                          # the measurements, to results/
make test                                   # cargo test, clippy with warnings denied, the formatter check
```

Needs Cargo. Everything is deterministic given `SEED`.

## What is here, in one paragraph

`GameGraph` (`rust/src/encoder.rs`) is a `radixnet::Graph` as it is, with a
belief over the lattice's `S` states plus abstain on every node, keyed by
where the node sits in the walk window (the current gram, the one before,
older, a child) and by the state the lattice was in. At every position the
window's nodes are sold seats in a uniform-price auction with a reserve,
play the congestion game over the states (`rust/src/game.rs`: right and rare
takes the pot, regret matching plus, sampled Shapley in antithetic pairs),
and the aggregate's argmax is the named state. Learning, the label is known
before the game: the lattice's own prediction when the oracle says it is
right, else the right state the lattice's row ranks highest. The lattice
(`rust/src/lattice.rs`) is a `latticefsm::Machine` with a kind, a
topographic flag, an abstain floor and a reasons table; its row is credited
per step by the oracle through `credit::land`, which lands a sharp verdict
on one edge in full, a dull one discounted within the run, and a reafferent
step at exactly zero. The decoders (`rust/src/decoders/`) are readouts,
repertoires on edges, propose-and-replay through a referee, and the taste
gate; the Nim referee (`rust/src/referee.rs`) refuses, replays and grades.

## The numbers

E1, the four languages, three seeds, held-out: the game arm **0.961**
against the plain oracle lattice's **0.961** (LatticeFSM's marks: 0.894
whole-run, 0.924 oracle, 0.944 explore). The stage game matches the lattice
alone and does not beat it. E2, Nim by window width: wins 0.47-0.50 against
a random opponent, first-try legality 0.13-0.21, under 0.04 bits of
information about the position in the named state - contradicted as built,
with the diagnosis in `results/README.md`.

## The pieces

| file | what |
|---|---|
| `rust/src/game.rs` | the stage game: seats, the auction, the payoffs (correctness, an oracle's set of right states, belief, a refusal), regret matching plus, the best-response cycle check, sampled Shapley, the aggregate |
| `rust/src/encoder.rs` | `GameGraph`: the radix graph with beliefs per node and bucket, the window, `name`, `settle`, `think_at`, the diversity and wealth readouts |
| `rust/src/lattice.rs` | `GameLattice`: a machine with a kind, the topographic flag, the abstain floor, the reasons table |
| `rust/src/verdict.rs`, `credit.rs` | the one verdict record; where it lands |
| `rust/src/forward.rs`, `efference.rs` | the forward model's quiet prediction and its lesson; the efference copy and its three returns |
| `rust/src/decoders/` | the protocol and the four mechanisms |
| `rust/src/referee.rs` | the language oracle; Nim |
| `rust/src/select.rs`, `gate.rs`, `clock.rs` | the release rule; the reserves and surges; the shared clock |
| `rust/src/body.rs`, `babble.rs` | the organism wired for a language and for Nim; babbling |
| `rust/src/experiment.rs` | E1 and E2 |
| `rust/tests/` | the design choices as tests: the game's invariants, quiet classification, no collapse, lessons move the row and agreement does not, reafference at zero, sharp on one edge, a decoder must name its sense, the release rule, the topographic axis, Nim, taste |
