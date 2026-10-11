"""Networks of machines - lattices connected like the layers of a neural network.

A machine reads a string and walks its matrix; a :class:`Layer` makes that a
block that also *writes* one: by default the state each move lands in, one
symbol per move, so the next block reads the walk the previous one took.  The
symbols a layer writes are its states' letters (:func:`state_symbols`: ``a``
for state 0, ``b`` for 1, ..., then ``A``-``Z`` and ``0``-``9``), and the
next machine is built over exactly that alphabet: a 13-state layer feeds a
machine whose matrix is ``S x 13 x S``.

Blocks compose, as in Keras:

* :class:`Layer` - one machine.  ``emit="states"`` writes every state the
  walk lands in; ``emit="final"`` writes only the last, one symbol.
* :class:`Sequential` - blocks one after another, each reading what the one
  before it wrote.
* :class:`Parallel` - blocks side by side, each reading the same input.  What
  they write is merged (``merge="concat"`` one after another, or
  ``"interleave"`` symbol by symbol), each branch's letters shifted past the
  branches before it so the next block can tell them apart.  When a parallel
  block is the network's last, its verdict is a vote of its branches
  (``vote="majority"``, ``"any"`` or ``"all"``).

* :class:`PhoneticEncoder` - English in: the text read as the sounds it is
  made of by the phonetic tokenizer of this repository (``phonetic.py``), a
  network's first block.
* :class:`PhoneticDecoder` - English out: a layer whose states are sounds, so
  what it writes is phonemes, spelled back as English.  With both ends, a
  network translates English to English (``translate``, ``fit_pairs``):
  ``"enc>13>dec"`` reads ``cat`` as ``K AE T``, walks it, writes ``K AE T S``
  and says ``cats``.

Blocks nest, so any series-parallel graph is a network.  :class:`Network`
holds one and is the model: ``compile`` (the output layers' accepting states,
the credit and how it is shared back), ``fit`` / ``evaluate`` / ``predict`` /
``summary`` / ``save`` and :func:`load_network`.  A network can also be
written as a string: ``"13>(13|13)>13"`` is a 13-state layer, then two
13-state layers side by side, then a 13-state layer; ``13f`` emits only the
final state.

Training is the single machine's, spread over the layers.  Every machine
walks (traversing its edges); the network's verdict is the last block's; it
is rewarded when it is right and punished when it is wrong, and the credit
travels back through the blocks as it travels back along a run: the last
block receives it in full, each block before it ``layer_discount`` times the
one after, and every branch of a parallel block the block's share.  Each
machine spends what it receives on its own path, discounted back from its
end as always.  A network of one layer is exactly the machine it holds.

A decoder is taught differently, because it has a right answer at every step,
not only at the end: each sound it wrote that is not the target's is punished,
and the target's path is taught edge by edge (``Machine.teach``) from the
boundary ``#`` it starts in; the blocks before it are rewarded by the share of
sounds it got right (never punished: see ``teach_pair``).
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable, Sequence

from .machine import DEFAULT_ALPHABET, DEFAULT_STATES, Machine

__all__ = ["CREDITS", "EMITS", "MERGES", "SYMBOLS", "VOTES", "Block", "Layer", "Network", "Parallel",
           "PhoneticDecoder", "PhoneticEncoder", "Sequential",
           "load_network", "parse_architecture", "state_symbols", "teach_network"]

SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
"""The letters blocks write: state ``i`` is ``SYMBOLS[i]``.  A block writes at most 62 distinct symbols."""

EMITS = ("states", "final")
MERGES = ("concat", "interleave")
VOTES = ("majority", "any", "all")
CREDITS = ("shared", "own")
LAYER_DISCOUNT = 0.8

_FORMAT = "latticefsm-network"
_FORMAT_VERSION = 1


def _text(symbols: Sequence[str]) -> str:
    """Symbols written out: run together when each is one character, else separated by spaces."""
    return ("" if all(len(s) == 1 for s in symbols) else " ").join(symbols)


def state_symbols(n: int) -> list[str]:
    """The first ``n`` letters of :data:`SYMBOLS`: what a block with ``n`` distinct outputs writes."""
    if not (1 <= n <= len(SYMBOLS)):
        raise ValueError(f"a block writes 1 to {len(SYMBOLS)} distinct symbols, not {n}")
    return list(SYMBOLS[:n])


class Block:
    """A part of a network: it is built over an input alphabet, reads a string, and writes one."""

    kind = "block"
    name: str

    def build(self, alphabet: Sequence[str], seeds: list[int], names: dict[str, int]) -> list[str]:
        """Make the block's machines over ``alphabet``; returns the alphabet the block writes."""
        raise NotImplementedError

    def forward(self, symbols: list[str], quiet: bool, temperature: float | None,
                trace: list[dict] | None) -> tuple[list[str], bool]:
        """Read ``symbols``: what the block writes, and its verdict."""
        raise NotImplementedError

    def credit(self, amount: float, layer_discount: float, own: tuple[bool, float, float] | None = None) -> int:
        """Credit the latest forward pass; returns how many edges received some.  ``own`` - the label, the reward and
        the punishment - credits the block's output layers by their own verdicts instead of ``amount``."""
        raise NotImplementedError

    def layers(self) -> list[Layer]:
        """Every layer in the block, depth first."""
        raise NotImplementedError

    def output_layers(self) -> list[Layer]:
        """The layers whose verdict is the block's: the last layer, or every branch's last."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def _name(self, names: dict[str, int]) -> None:
        if not getattr(self, "name", None):
            names[self.kind] = names.get(self.kind, 0) + 1
            self.name = f"{self.kind}_{names[self.kind]}"


class Layer(Block):
    """One machine as a block.  ``states`` and the machine's keyword arguments make it; the alphabet is the input's."""

    kind = "lattice"

    def __init__(self, states: int = DEFAULT_STATES, emit: str = "states", name: str | None = None,
                 accepting: Iterable[int] | None = None, labels: Sequence[str] | None = None,
                 **machine_options) -> None:
        if emit not in EMITS:
            raise ValueError(f"emit must be one of {EMITS}, got {emit!r}")
        if labels is not None:
            states = len(labels)
        elif not (1 <= states <= len(SYMBOLS)):
            raise ValueError(f"a layer has 1 to {len(SYMBOLS)} states, not {states}")
        for k in ("alphabet", "states"):
            if k in machine_options:
                raise ValueError(f"a layer's {k} is not a machine option here")
        self.states = int(states)
        self.emit = emit
        self.name = name or ""
        self.accepting = None if accepting is None else [int(s) for s in accepting]
        self.machine_options = dict(machine_options)
        self.machine: Machine | None = None
        self.last_accepted = False
        self._out = list(labels) if labels is not None else state_symbols(self.states)

    def build(self, alphabet, seeds, names):
        self._name(names)
        options = dict(self.machine_options)
        options.setdefault("seed", seeds[0] + seeds[1])
        seeds[1] += 1
        self.machine = Machine(self.states, list(alphabet), accepting=self.accepting or (), **options)
        self.machine.greedy_first = True
        return list(self._out)

    def forward(self, symbols, quiet, temperature, trace):
        run = self.machine.run(symbols, temperature=temperature, quiet=quiet)
        if self.emit == "states":
            out = [self._out[t.target] for t in run.transitions]
        else:
            out = [self._out[run.final]]
        self.last_accepted = run.accepted
        self.last_run = run
        if trace is not None:
            trace.append(self._step(symbols, run, out))
        return out, run.accepted

    def _step(self, symbols, run, out) -> dict:
        return {"layer": self.name, "read": _text(symbols), "states": run.states, "wrote": _text(out),
                "accepted": run.accepted}

    def credit(self, amount, layer_discount, own=None):
        if own is not None:
            amount = own[1] if self.last_accepted == own[0] else -own[2]
        return self.machine.credit(amount)

    def layers(self):
        return [self]

    def output_layers(self):
        return [self]

    def to_dict(self):
        return {"type": "layer", "name": self.name, "states": self.states, "emit": self.emit,
                "machine": self.machine.to_dict()}

    @classmethod
    def _from_dict(cls, data: dict) -> Layer:
        if data.get("type") == "decoder":
            layer = PhoneticDecoder(labels=data["labels"], stress=bool(data.get("stress", False)),
                                    name=data.get("name") or None)
            if layer.states != int(data["states"]):
                raise ValueError(f"decoder {layer.name}: {data['states']} states, but {layer.states} labels")
        else:
            layer = cls(int(data["states"]), emit=data.get("emit", "states"), name=data.get("name") or None)
        layer.machine = Machine.from_dict(data["machine"])
        layer.machine.greedy_first = True
        if layer.machine.n_states != layer.states:
            raise ValueError(f"layer {layer.name}: {layer.states} states, but its machine has {layer.machine.n_states}")
        return layer


class PhoneticDecoder(Layer):
    """English out: a layer whose states are sounds (``phonetic.inventory``), so what it writes is phonemes - each
    the state a step lands in, starting from the boundary ``#`` - which ``Network.translate`` spells as English.

    It writes one sound per symbol it reads, so it skips nothing, keeps its states where they are (no
    rearrangement: a state *is* its sound) and always starts from ``#``."""

    kind = "decoder"
    _FIXED = ("skip", "rearrange_every", "focus", "learn_focus", "start")

    def __init__(self, stress: bool = False, name: str | None = None, accepting: Iterable[int] | None = None,
                 labels: Sequence[str] | None = None, **machine_options) -> None:
        from .phonetic import inventory
        for k in self._FIXED:
            if machine_options.get(k):
                raise ValueError(f"a decoder's {k} is fixed: it writes one sound per symbol, from '#'")
            machine_options.pop(k, None)
        self.stress = bool(stress)
        super().__init__(emit="states", name=name, accepting=accepting,
                         labels=list(labels) if labels is not None else inventory(self.stress), **machine_options)

    def targets(self, text: str) -> list[int]:
        """The sounds of ``text`` as this decoder's states (any sound it has no state for dropped)."""
        from .phonetic import tokenizer
        index = {s: i for i, s in enumerate(self._out)}
        return [index[t] for t in tokenizer(self.stress).tokens(text) if t in index]

    def english(self, sounds: Sequence[str]) -> str:
        from .phonetic import tokenizer
        return tokenizer(self.stress).decode(list(sounds))

    def _step(self, symbols, run, out) -> dict:
        step = super()._step(symbols, run, out)
        step["english"] = self.english(out)
        return step

    def teach_targets(self, targets: Sequence[int], reward: float, punish: float) -> tuple[int, int]:
        """Teach the latest run its target sounds (padded with ``#`` to its length, or cut to it): every sound it
        wrote wrong is punished, then the target's path is taught edge by edge from ``#``.  Returns (right, of)."""
        m, run = self.machine, self.last_run
        n = len(run.transitions)
        want = (list(targets) + [0] * n)[:n]
        right = 0
        for t, w in zip(run.transitions, want):
            if t.target == w:
                right += 1
            else:
                m.lattice[t.source, t.symbol, t.target].credit(-punish, m.clock, m.life, widen=m.reward_widening,
                                                               narrow=m.punish_narrowing)
        prev = 0
        for t, w in zip(run.transitions, want):
            m.teach(prev, t.symbol, w, reward)
            prev = w
        return right, n

    def to_dict(self):
        d = super().to_dict()
        d.update(type="decoder", stress=self.stress, labels=list(self._out))
        return d


class PhoneticEncoder(Block):
    """English in: the text as the sounds it is made of (``phonetic.tokenizer``), followed by ``pad`` boundaries
    ``#`` - room for whatever comes after it to write more sounds than it read.  A network's first block."""

    kind = "encoder"

    def __init__(self, stress: bool = False, pad: int = 2, name: str | None = None) -> None:
        from .phonetic import inventory
        if pad < 0:
            raise ValueError(f"pad must be >= 0, got {pad}")
        self.stress = bool(stress)
        self.pad = int(pad)
        self.name = name or ""
        self.labels = inventory(self.stress)
        self._known = set(self.labels)
        self.last_text = ""

    def encode(self, text: str) -> list[str]:
        from .phonetic import tokenizer
        self.last_text = text
        return [t for t in tokenizer(self.stress).tokens(text) if t in self._known] + ["#"] * self.pad

    def english(self, sounds: Sequence[str]) -> str:
        from .phonetic import tokenizer
        return tokenizer(self.stress).decode(list(sounds))

    def build(self, alphabet, seeds, names):
        self._name(names)
        return list(self.labels)

    def forward(self, symbols, quiet, temperature, trace):
        if trace is not None:
            trace.append({"layer": self.name, "read": self.last_text, "states": [], "wrote": _text(symbols),
                          "accepted": False, "english": self.english(symbols)})
        return list(symbols), False

    def credit(self, amount, layer_discount, own=None):
        return 0

    def layers(self):
        return []

    def output_layers(self):
        return []

    def to_dict(self):
        return {"type": "encoder", "name": self.name, "stress": self.stress, "pad": self.pad}


class Sequential(Block):
    """Blocks one after another: each reads what the one before it wrote."""

    kind = "sequential"

    def __init__(self, blocks: Sequence[Block], name: str | None = None) -> None:
        if not blocks:
            raise ValueError("a sequential block needs at least one block")
        self.blocks = list(blocks)
        self.name = name or ""

    def build(self, alphabet, seeds, names):
        self._name(names)
        for b in self.blocks:
            alphabet = b.build(alphabet, seeds, names)
        return alphabet

    def forward(self, symbols, quiet, temperature, trace):
        accepted = False
        for b in self.blocks:
            symbols, accepted = b.forward(symbols, quiet, temperature, trace)
        return symbols, accepted

    def credit(self, amount, layer_discount, own=None):
        n = 0
        for b in reversed(self.blocks):
            n += b.credit(amount, layer_discount, own)
            amount *= layer_discount
            own = None
        return n

    def layers(self):
        return [l for b in self.blocks for l in b.layers()]

    def output_layers(self):
        return self.blocks[-1].output_layers()

    def to_dict(self):
        return {"type": "sequential", "name": self.name, "blocks": [b.to_dict() for b in self.blocks]}


class Parallel(Block):
    """Blocks side by side, each reading the same input; what they write merged, their verdicts voted on."""

    kind = "parallel"

    def __init__(self, branches: Sequence[Block], merge: str = "concat", vote: str = "majority",
                 name: str | None = None) -> None:
        if len(branches) < 2:
            raise ValueError("a parallel block needs at least two branches")
        if merge not in MERGES:
            raise ValueError(f"merge must be one of {MERGES}, got {merge!r}")
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}, got {vote!r}")
        self.branches = list(branches)
        self.merge = merge
        self.vote = vote
        self.name = name or ""
        self._maps: list[dict[str, str]] = []

    def build(self, alphabet, seeds, names):
        self._name(names)
        outs = [b.build(alphabet, seeds, names) for b in self.branches]
        self._wire(outs)
        return state_symbols(sum(len(o) for o in outs))

    def _wire(self, outs: list[list[str]]) -> None:
        total = sum(len(o) for o in outs)
        if total > len(SYMBOLS):
            raise ValueError(f"parallel block {self.name}: its branches write {total} distinct symbols, more than "
                             f"{len(SYMBOLS)}")
        self._maps, offset = [], 0
        for o in outs:
            self._maps.append({s: SYMBOLS[offset + i] for i, s in enumerate(o)})
            offset += len(o)

    def forward(self, symbols, quiet, temperature, trace):
        outs, verdicts = [], []
        for b, m in zip(self.branches, self._maps):
            out, accepted = b.forward(symbols, quiet, temperature, trace)
            outs.append([m[s] for s in out])
            verdicts.append(accepted)
        if self.merge == "concat":
            merged = [s for o in outs for s in o]
        else:
            merged = [o[i] for i in range(max(map(len, outs), default=0)) for o in outs if i < len(o)]
        yes = sum(verdicts)
        accepted = (yes * 2 > len(verdicts) if self.vote == "majority"
                    else yes > 0 if self.vote == "any" else yes == len(verdicts))
        return merged, accepted

    def credit(self, amount, layer_discount, own=None):
        return sum(b.credit(amount, layer_discount, own) for b in self.branches)

    def layers(self):
        return [l for b in self.branches for l in b.layers()]

    def output_layers(self):
        return [l for b in self.branches for l in b.output_layers()]

    def to_dict(self):
        return {"type": "parallel", "name": self.name, "merge": self.merge, "vote": self.vote,
                "branches": [b.to_dict() for b in self.branches]}


def _block_from_dict(data: dict) -> Block:
    t = data.get("type")
    if t in ("layer", "decoder"):
        return Layer._from_dict(data)
    if t == "encoder":
        return PhoneticEncoder(stress=bool(data.get("stress", False)), pad=int(data.get("pad", 2)),
                               name=data.get("name") or None)
    if t == "sequential":
        return Sequential([_block_from_dict(b) for b in data["blocks"]], name=data.get("name") or None)
    if t == "parallel":
        p = Parallel([_block_from_dict(b) for b in data["branches"]], merge=data.get("merge", "concat"),
                     vote=data.get("vote", "majority"), name=data.get("name") or None)
        p._wire([_written(b) for b in p.branches])
        return p
    raise ValueError(f"unknown block type {t!r}")


# ---- the architecture string ---------------------------------------------------------------------------------------

def parse_architecture(text: str, merge: str = "concat", vote: str = "majority", **layer_options) -> Block:
    """``"13>(13|13)>13"``: ``>`` one block after another, ``(a|b|...)`` side by side, a number a layer of that
    many states (``13f``: emitting only its final state); ``enc`` a :class:`PhoneticEncoder` and ``dec`` a
    :class:`PhoneticDecoder`.  ``layer_options`` go to every layer's machine."""
    src = text.replace(" ", "")
    pos = 0

    def fail(msg: str):
        raise ValueError(f"architecture {text!r}: {msg} at position {pos}")

    def peek() -> str:
        return src[pos] if pos < len(src) else ""

    def seq() -> Block:
        nonlocal pos
        blocks = [atom()]
        while peek() == ">":
            pos += 1
            blocks.append(atom())
        return blocks[0] if len(blocks) == 1 else Sequential(blocks)

    def atom() -> Block:
        nonlocal pos
        if peek() == "(":
            pos += 1
            alts = [seq()]
            while peek() == "|":
                pos += 1
                alts.append(seq())
            if peek() != ")":
                fail("expected ')'")
            pos += 1
            return alts[0] if len(alts) == 1 else Parallel(alts, merge=merge, vote=vote)
        for word in ("enc", "dec"):
            if src.startswith(word, pos):
                pos += len(word)
                if word == "enc":
                    return PhoneticEncoder()
                return PhoneticDecoder(**{k: v for k, v in layer_options.items() if k not in PhoneticDecoder._FIXED})
        start = pos
        while peek().isdigit():
            pos += 1
        if start == pos:
            fail("expected a number of states, enc, dec or '('")
        states = int(src[start:pos])
        emit = "states"
        if peek() == "f":
            pos += 1
            emit = "final"
        return Layer(states, emit=emit, **layer_options)

    block = seq()
    if pos != len(src):
        fail(f"unexpected {peek()!r}")
    return block


# ---- the model -----------------------------------------------------------------------------------------------------

class Network:
    """A network of machines: built over an input alphabet, compiled, fitted, evaluated, saved.

    Its quiet greedy passes (``predict``, ``evaluate``, ``trace``, ``translate``) are deterministic: a tie goes to the
    lowest state, and nothing is drawn (``Machine.greedy_first``).

    ``layers`` is a block, a list of blocks (one after another), or an architecture string
    (:func:`parse_architecture`; ``merge``, ``vote`` and ``layer_options`` apply to it).  Each layer's machine is
    seeded ``seed`` plus its index, depth first, unless its options say otherwise.
    """

    def __init__(self, layers: Block | Sequence[Block] | str, alphabet: Sequence[str] = DEFAULT_ALPHABET,
                 seed: int = 1, name: str = "network", merge: str = "concat", vote: str = "majority",
                 layer_options: dict | None = None, _built: bool = False) -> None:
        if isinstance(layers, str):
            self.architecture = layers.replace(" ", "")
            root = parse_architecture(layers, merge=merge, vote=vote, **(layer_options or {}))
        elif isinstance(layers, Block):
            self.architecture = ""
            root = layers
        else:
            self.architecture = ""
            root = Sequential(list(layers))
        if not isinstance(root, Sequential):
            root = Sequential([root])
        self.root = root
        encoders = _encoders(root)
        if encoders and encoders[0] is not root.blocks[0] or len(encoders) > 1:
            raise ValueError("an encoder can only be the network's first block")
        self.encoder: PhoneticEncoder | None = encoders[0] if encoders else None
        if not root.layers():
            raise ValueError("a network needs at least one layer")
        self.alphabet = list(self.encoder.labels) if self.encoder else list(alphabet)
        if len(set(self.alphabet)) != len(self.alphabet) or not self.alphabet:
            raise ValueError("the input alphabet is one or more distinct symbols")
        self.seed = int(seed)
        self.name = name
        self.layer_discount = LAYER_DISCOUNT
        self.reward = 1.0
        self.punish = 1.0
        self.credit = "shared"
        self.epochs = 0
        self.examples = 0
        self.words: list[str] = []
        """The English words of the pairs it was taught, sorted: the tokenizer hears them before it spells anything, so
        a word it was taught is spelled back as itself (its memory of the words it sounded out) in every process."""
        self._primed: set[tuple[bool, int]] = set()
        if not _built:
            self.output_alphabet = self.root.build(self.alphabet, [self.seed, 0], {})
        self._index = {s: i for i, s in enumerate(self.alphabet)}

    # ---- compiling -------------------------------------------------------------------------------------------------

    def compile(self, accepting: Iterable[int] | None = None, layer_discount: float = LAYER_DISCOUNT,
                reward: float = 1.0, punish: float = 1.0, credit: str = "shared") -> Network:
        """Set the output layers' accepting states (when given), how much of the credit each block further back
        receives, how much a right verdict is rewarded and a wrong one punished, and whose verdict the output layers
        are credited by: the network's (``"shared"``) or each one's own (``"own"``; the layers before them always
        receive the network's)."""
        if credit not in CREDITS:
            raise ValueError(f"credit must be one of {CREDITS}, got {credit!r}")
        if not (0.0 <= layer_discount <= 1.0):
            raise ValueError(f"layer_discount must be in [0, 1], got {layer_discount}")
        if accepting is not None:
            acc = [int(s) for s in accepting]
            for layer in self.root.output_layers():
                m = layer.machine
                for s in acc:
                    if not (0 <= s < m.n_states):
                        raise ValueError(f"accepting state {s} is not one of {layer.name}'s 0..{m.n_states - 1}")
                for i, st in enumerate(m.lattice.states):
                    st.accepting = i in acc
        self.layer_discount = float(layer_discount)
        self.reward = abs(float(reward))
        self.punish = abs(float(punish))
        self.credit = credit
        return self

    # ---- reading ---------------------------------------------------------------------------------------------------

    @property
    def decoder(self) -> PhoneticDecoder | None:
        """The network's last block, when it is a decoder: the network then writes English."""
        last = self.root.blocks[-1]
        return last if isinstance(last, PhoneticDecoder) else None

    def _symbols(self, text: str | Sequence[str]) -> list[str]:
        if self.encoder is not None:
            if not isinstance(text, str):
                text = " ".join(text)
            return self.encoder.encode(text)
        syms = [s for s in text if not (isinstance(s, str) and s.isspace())]
        for s in syms:
            if s not in self._index:
                raise ValueError(f"{s!r} is not in the network's alphabet {''.join(self.alphabet)}")
        return syms

    def forward(self, text: str | Sequence[str], quiet: bool = False,
                temperature: float | None = None) -> tuple[list[str], bool]:
        """Read ``text`` through every block: what the last block wrote, and the network's verdict."""
        return self.root.forward(self._symbols(text), quiet, temperature, None)

    def trace(self, text: str | Sequence[str], temperature: float = 0.0) -> dict:
        """The greedy quiet pass, layer by layer: what each read, the states it walked, what it wrote."""
        steps: list[dict] = []
        self._prime()
        symbols = self._symbols(text)
        out, accepted = self.root.forward(symbols, True, temperature, steps)
        shown = text if self.encoder is not None else _text(symbols)
        output = self.decoder.english(out) if self.decoder is not None else _text(out)
        return {"text": shown, "layers": steps, "output": output, "accepted": accepted}

    def predict(self, x: Iterable[str], temperature: float = 0.0) -> list[bool]:
        """The verdict on each string: the greedy (or, at a temperature, a sampled) quiet pass.  Nothing moves."""
        return [self.forward(s, quiet=True, temperature=temperature)[1] for s in x]

    def evaluate(self, x: Sequence[str], y: Sequence[bool], temperature: float = 0.0) -> float:
        """The share of ``x`` whose verdict is ``y``.  Nothing moves."""
        if len(x) != len(y):
            raise ValueError("x and y differ in length")
        if not x:
            return 0.0
        return sum(p == bool(t) for p, t in zip(self.predict(x, temperature), y)) / len(x)

    # ---- learning --------------------------------------------------------------------------------------------------

    def teach(self, text: str | Sequence[str], label: bool) -> bool:
        """One example: every machine walks it, and the network is rewarded if its verdict is ``label`` and punished
        if not.  Returns whether it was right."""
        _, accepted = self.forward(text)
        right = accepted == bool(label)
        own = (bool(label), self.reward, self.punish) if self.credit == "own" else None
        self.root.credit(self.reward if right else -self.punish, self.layer_discount, own)
        self.examples += 1
        return right

    def fit(self, x: Sequence[str], y: Sequence[bool], epochs: int = 1,
            validation_data: tuple[Sequence[str], Sequence[bool]] | None = None, verbose: bool = False) -> dict:
        """Teach every example of ``x`` in order, ``epochs`` times.  Returns the history: per epoch, the share
        classified right while it was taught (``accuracy``) and, given ``validation_data``, measured quietly after it
        (``val_accuracy``)."""
        if len(x) != len(y):
            raise ValueError("x and y differ in length")
        history: dict[str, list[float]] = {"accuracy": []}
        if validation_data is not None:
            history["val_accuracy"] = []
        for e in range(epochs):
            right = sum(self.teach(s, t) for s, t in zip(x, y))
            self.epochs += 1
            history["accuracy"].append(right / len(x) if x else 0.0)
            if validation_data is not None:
                history["val_accuracy"].append(self.evaluate(*validation_data))
            if verbose:
                line = f"epoch {self.epochs}: accuracy {history['accuracy'][-1]:.3f}"
                if validation_data is not None:
                    line += f", val_accuracy {history['val_accuracy'][-1]:.3f}"
                print(line)
        return history

    # ---- English to English ----------------------------------------------------------------------------------------

    def _check_translates(self) -> PhoneticDecoder:
        if self.encoder is None or self.decoder is None:
            raise ValueError("translating needs an encoder first and a decoder last (e.g. \"enc>13>dec\")")
        return self.decoder

    def _prime(self) -> None:
        from .phonetic import tokenizer
        for block in (self.encoder, self.decoder):
            if block is not None and self.words and (block.stress, len(self.words)) not in self._primed:
                tokenizer(block.stress).tokens(" ".join(self.words))
                self._primed.add((block.stress, len(self.words)))

    def sounds(self, text: str, temperature: float = 0.0) -> list[str]:
        """The sounds the network writes for ``text`` (greedy, quiet), the trailing boundaries dropped."""
        dec = self._check_translates()
        self._prime()
        out, _ = self.forward(text, quiet=True, temperature=temperature)
        while out and out[-1] == "#":
            out.pop()
        return out

    def translate(self, text: str, temperature: float = 0.0) -> str:
        """English in, English out: ``text`` read as sounds, walked, and what the decoder writes spelled back."""
        sounds = self.sounds(text, temperature)
        self._prime()
        return self._check_translates().english(sounds)

    def _target(self, text: str) -> list[str]:
        dec = self._check_translates()
        out = [dec._out[i] for i in dec.targets(text)]
        while out and out[-1] == "#":
            out.pop()
        return out

    def teach_pair(self, x: str, y: str) -> tuple[int, int]:
        """One English pair: every machine walks ``x``; the decoder is taught ``y``'s sounds step by step, and the
        blocks before it are rewarded by the share it got right - rewarded only: punishing them for the decoder's
        mistakes keeps reshuffling the walks the decoder is learning to read, and it never learns them.  Returns
        (sounds right, sounds written)."""
        dec = self._check_translates()
        new = {w.lower() for w in f"{x} {y}".split()} - set(self.words)
        if new:
            self.words = sorted(set(self.words) | new)
        self.forward(x)
        right, n = dec.teach_targets(dec.targets(y), self.reward, self.punish)
        if n:
            amount = self.reward * (right / n) * self.layer_discount
            for b in reversed(self.root.blocks[:-1]):
                b.credit(amount, self.layer_discount)
                amount *= self.layer_discount
        self.examples += 1
        return right, n

    def evaluate_pairs(self, pairs: Sequence[tuple[str, str]]) -> float:
        """The share of pairs whose translation has exactly the target's sounds (so a spelling the tokenizer chooses
        differently is not counted wrong).  Nothing moves."""
        if not pairs:
            return 0.0
        return sum(self.sounds(x) == self._target(y) for x, y in pairs) / len(pairs)

    def fit_pairs(self, pairs: Sequence[tuple[str, str]], epochs: int = 1,
                  validation_data: Sequence[tuple[str, str]] | None = None, verbose: bool = False) -> dict:
        """Teach every pair in order, ``epochs`` times.  The history, per epoch: the share of sounds right while
        taught (``sounds``), and the share of pairs exactly right after the epoch on ``pairs`` (``exact``) and on
        ``validation_data`` (``val_exact``)."""
        history: dict[str, list[float]] = {"sounds": [], "exact": []}
        if validation_data is not None:
            history["val_exact"] = []
        for _ in range(epochs):
            right = total = 0
            for x, y in pairs:
                r, n = self.teach_pair(x, y)
                right, total = right + r, total + n
            self.epochs += 1
            history["sounds"].append(right / total if total else 0.0)
            history["exact"].append(self.evaluate_pairs(pairs))
            if validation_data is not None:
                history["val_exact"].append(self.evaluate_pairs(validation_data))
            if verbose:
                print(f"epoch {self.epochs}: " + ", ".join(f"{k} {v[-1]:.3f}" for k, v in history.items()))
        return history

    # ---- describing ------------------------------------------------------------------------------------------------

    def layers(self) -> list[Layer]:
        return self.root.layers()

    def summary(self) -> str:
        """The blocks, nested, with each layer's matrix and what it writes; the totals."""
        rows: list[tuple[str, str, str, str, str]] = []

        def walk(b: Block, depth: int) -> None:
            pad = "  " * depth
            if isinstance(b, PhoneticEncoder):
                rows.append((f"{pad}{b.name}", f"English -> {len(b.labels)} sounds", "sounds", "", ""))
            elif isinstance(b, Layer):
                m = b.machine
                s, a, t = m.lattice.shape
                touched = len(list(m.lattice.touched()))
                emits = "sounds" if isinstance(b, PhoneticDecoder) else b.emit
                rows.append((f"{pad}{b.name}", f"{s} x {a} x {t}", emits, f"{s * a * t}", f"{touched}"))
            elif isinstance(b, Sequential):
                if depth >= 0:
                    rows.append((f"{pad}{b.name}", "sequential", "", "", ""))
                for c in b.blocks:
                    walk(c, depth + 1)
            else:
                rows.append((f"{pad}{b.name}", f"parallel: {b.merge}, {b.vote}", "", "", ""))
                for c in b.branches:
                    walk(c, depth + 1)

        for c in self.root.blocks:
            walk(c, 0)
        head = ("block", "matrix", "emits", "edges", "touched")
        widths = [max(len(r[i]) for r in rows + [head]) for i in range(5)]
        fmt = lambda r: "  ".join(v.ljust(widths[i]) if i < 3 else v.rjust(widths[i]) for i, v in enumerate(r)).rstrip()
        layers = self.layers()
        accepting = sorted({i for l in self.root.output_layers() for i in l.machine.accepting})
        reads = "English" if self.encoder else "".join(self.alphabet)
        writes = "English" if self.decoder else f"{len(self.output_alphabet)} symbols"
        lines = [f'network "{self.name}"' + (f" ({self.architecture})" if self.architecture else "")
                 + f": reads {reads}, writes {writes}",
                 fmt(head), fmt(tuple("-" * w for w in widths))]
        lines += [fmt(r) for r in rows]
        edges = sum(len(l.machine.lattice.edges) for l in layers)
        touched = sum(len(list(l.machine.lattice.touched())) for l in layers)
        lines += [f"{len(layers)} layers, {edges} edges, {touched} touched; output accepting {accepting}; "
                  f"{self.credit} credit, layer discount {self.layer_discount:g}, reward {self.reward:g}, punish {self.punish:g}; "
                  f"{self.examples} examples taught over {self.epochs} epochs"]
        return "\n".join(lines)

    # ---- saving ----------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"format": _FORMAT, "version": _FORMAT_VERSION, "name": self.name, "architecture": self.architecture,
                "alphabet": list(self.alphabet), "seed": self.seed, "layer_discount": self.layer_discount,
                "reward": self.reward, "punish": self.punish, "credit": self.credit, "words": list(self.words), "epochs": self.epochs, "examples": self.examples,
                "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> Network:
        if data.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        root = _block_from_dict(data["root"])
        net = cls(root, alphabet=data["alphabet"], seed=int(data.get("seed", 1)), name=data.get("name", "network"),
                  _built=True)
        net.architecture = data.get("architecture", "")
        first = net.layers()[0].machine
        if net.encoder is None and list(first.alphabet) != net.alphabet:
            raise ValueError("the first layer's alphabet is not the network's")
        net.output_alphabet = _written(net.root)
        net.layer_discount = float(data.get("layer_discount", LAYER_DISCOUNT))
        net.reward = float(data.get("reward", 1.0))
        net.punish = float(data.get("punish", 1.0))
        net.credit = data.get("credit", "shared")
        net.words = sorted(str(w) for w in data.get("words", []))
        if net.credit not in CREDITS:
            raise ValueError(f"credit must be one of {CREDITS}, got {net.credit!r}")
        net.epochs = int(data.get("epochs", 0))
        net.examples = int(data.get("examples", 0))
        return net

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)


def _written(b: Block) -> list[str]:
    if isinstance(b, Layer):
        return list(b._out)
    if isinstance(b, PhoneticEncoder):
        return list(b.labels)
    if isinstance(b, Sequential):
        return _written(b.blocks[-1])
    return state_symbols(sum(len(m) for m in b._maps))


def _encoders(b: Block) -> list[PhoneticEncoder]:
    if isinstance(b, PhoneticEncoder):
        return [b]
    if isinstance(b, Sequential):
        return [e for c in b.blocks for e in _encoders(c)]
    if isinstance(b, Parallel):
        return [e for c in b.branches for e in _encoders(c)]
    return []


def load_network(path: str) -> Network:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return Network.from_dict(json.load(f))


def teach_network(net: Network, lang_name: str, episodes: int, rng: random.Random, max_length: int = 6,
                  alphabet: Sequence[str] = "ab") -> int:
    """Teach ``episodes`` random strings of a language, drawn exactly as ``experiment.teach_language`` draws them, so
    a one-layer network and the machine it holds see the same strings.  Returns how many were classified right."""
    from .languages import language
    lang = language(lang_name)
    right = 0
    for _ in range(episodes):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_length)))
        right += net.teach(s, lang.member(s))
    return right
