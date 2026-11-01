# GameLattice

**A RadixCyclicNN whose step is a game, feeding a LatticeFSM whose every
state is one of the encoder's outputs, read out by decoders that are
separate from the lattice — with a body around it.** The senses are its
encoders, the effectors its decoders, and every effector's output comes
back in through a sense.

**Status:** specification only. [`DESIGN.md`](DESIGN.md) is the contract;
there is no runnable package, CLI or test yet. The package it describes,
`gamelattice`, imports the siblings `RadixCyclicNN/radixnet`,
`LatticeFSM/latticefsm` and `Experiments/UniversalGameEncoding/uge`, with
`PhoneticTokenizer/phonetok` and `ModelKit/modelkit` as extras for the ear,
the mouth, the hands and the monitors.

What the design does, in one paragraph: the encoder keeps the radix gram
graph, its compression, its cycles and its `BACK` / `THINK` sentinels, but at
every position the nodes of the walk window play a congestion game whose
outcome names a lattice state; the lattice is one class in three kinds — a
sense lattice per organ, an act lattice per effector whose row is the
forward model, an association lattice beside them; decoders are typed by
output and produce it by one of four mechanisms (repertoires on edges,
propose-and-replay through a referee, a readout, a gate); every emission is
copied into the alphabet of the sense that reads it back, and what returns
matching the copy earns nothing. `DESIGN.md` §7 treats all of it as codes:
every organ a source code with a rate, every lattice a channel with a state
capacity, every decoder a channel decoder whose parity check is the referee.

Nothing enters the design without a number already in the repository that
it must beat and a control that would show the body reading bought nothing
(`DESIGN.md` §3, §17). The decisions only the author can make are listed in
§21; the corrections to the earlier discussion are on the record in §20.
