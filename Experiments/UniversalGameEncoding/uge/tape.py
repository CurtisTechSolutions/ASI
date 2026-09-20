"""The universal layer: a game's play written as text the trigram network eats.

Nothing in this file knows what a game *is*.  It knows three things:

1.  RadixNet's window is ``W = 3`` characters with stride 1
    (:mod:`radixnet.encoding`), and every trigram of the corpus lives in
    **exactly one node** (:class:`radixnet.graph.RadixCyclicGraph`).
2.  Therefore a token that is exactly ``W`` characters wide makes one phase of
    the sliding window a whole token: the graph's nodes become game objects and
    its edges become game transitions.  Anything narrower or wider smears a
    move across phases and the graph learns the *spelling* instead of the game.
3.  Three characters out of a 64-symbol alphabet is ``262_144`` codes, which is
    more than the action space of any game in :mod:`uge.games`.

So the whole universal layer is: an alphabet, a bijection ``int <-> token``,
and a tape grammar::

    tape := head body tail
    head := one token  - which game, which state abstraction, which layout
    body := (phi-token* action-token)*   - one group per ply
    tail := one token  - how it ended

Every token is ``W`` characters, so a tape is a multiple of ``W`` and every
prefix that ends on a token boundary is a legal question to ask the network.

The phase problem, which is not obvious and decides everything
--------------------------------------------------------------
A stride-1 window over a ``W``-wide token grid produces ``W`` **phases**:

===========  =======================================
phase 0      ``(c0, c1, c2)``    - a whole token
phase 1      ``(c1, c2, c0')``   - a seam
phase 2      ``(c2, c0', c1')``  - a seam
===========  =======================================

The graph indexes a trigram by its *characters* and nothing else, so two
occurrences in **different phases** that happen to spell the same three
characters become one node - and that node's children are then a mixture of
"the rest of this token" and "the start of the next one".  A walk that enters
in one phase can leave in another, and the three characters it emits are not a
token at all.

Measured on this directory's corpora only 1-4 % of *distinct* trigrams are
ambiguous that way, and it still wrecks the small games, because the ambiguity
is not spread evenly.  A game with a seven-action space pads two of its three
digits with a constant, every token then ends in the same character, and the
seams collide with each other systematically rather than by accident.  The
first version of this file measured ``nim`` proposing a syntactically
impossible move as its **top** choice, from exactly this.

:data:`DISJOINT` removes it by construction: the last character of every token
comes from a reserved 21-symbol alphabet that appears nowhere else, so the
phase of a trigram is written on its face - phase 0 has the reserved character
last, phase 1 has it in the middle, phase 2 has it first - and no trigram can
be read two ways.  The cost is capacity (``43 * 43 * 21 = 38_829`` codes per
token instead of ``262_144``, and 15 bits of ``phi`` per token instead of 18)
and the *structured* layouts: chess's ``(from, to, promotion)`` needs 64 values
in two slots and 64 > 43, so under :data:`DISJOINT` chess falls back to a flat
code.  That trade - meaningful seams against unambiguous ones - is what
``experiment.py phase`` measures.

Every kind of token needs its own region of the code space
-----------------------------------------------------------
The graph identifies a token by its three characters and by nothing else - not
by where it sits in the tape, not by what it means.  So two *kinds* of token
that encode to the same three characters are **one node**, with children drawn
from both roles, and a walk that arrives as one leaves as the other.

This is the single most expensive lesson in this directory, because it was
learned three times:

1. The outcome token ``draw`` and connect four's "column 1" were both code 1,
   so every tape's ending was also a legal move, and a Dijkstra prediction from
   the header reached END after two tokens.
2. The header of game 0 was code 0, which was also that game's action 0.
3. ``phi`` wrote the position's own code, and the **empty board** hashes to
   code 0, which was also "play cell 0" - so the node every tic-tac-toe game
   starts at was the node for a move.  An injective abstraction, which can only
   carry more information than a hashed one, measured *worse*: 0.72 against
   0.99 on legality at known positions.

The fix is a partition of the code space, which :func:`uge.codec.phi_region`
computes and ``tests.py partition`` asserts:

===================  ==========================================================
``[0, bound)``       actions.  ``bound`` is what the game's own layout reaches
                     (:meth:`uge.game.Game.token_bound`), which is
                     ``action_count`` for a flat layout and more for a
                     structured one.
``[bound, meta)``    ``phi``.  What is left over, which is why a game with a
                     large action space gets a smaller abstraction: chess keeps
                     13 bits, everything else 15.
``[meta, capacity)`` the header and the outcome tokens - 2056 codes at the top.
===================  ==========================================================

The one free choice that is left
--------------------------------
``phi`` - how much of the state is written down.  ``_locate`` in
:mod:`radixnet.model` conditions a prediction on the **last trigram of the
prefix alone**, so the network is first-order in the window whatever we do.
What that window *holds* is ours to choose:

* ``phi = none``   - moves only: the model is a bigram over moves.
* ``phi = k bits`` - a digest of the position before every move, so the model
  is a policy over an abstraction, ``P(a | phi(s))``.
* ``phi_exact``    - the game's own state code rather than a hash, when it fits
  in one token: the graph becomes a transposition table, perfect on positions
  it has seen and silent on the rest.  ``tictactoe`` (3**9 = 19_683 boards) and
  ``nim`` (384) can do this; the other four cannot.

A hash is also a poor abstraction: it has no **locality**, so two nearly
identical positions land in unrelated buckets and there is no such thing as a
near miss.  :attr:`TapeSpec.phi_prev_bits` spends part of the same token on a
*meaningful* summary of the move just played
(:meth:`uge.game.Game.action_summary`) instead, which does generalise.

**And the abstraction has to fit in one token.**  ``_locate`` conditions on the
last trigram of the prefix, so when ``phi`` spans two tokens only the second
one is read and the first is tape the model pays for and never uses.  That caps
the model's memory at :attr:`Codebook.bits` - 18 under :data:`PLAIN`, 15 under
:data:`DISJOINT`, which is what phase-unambiguity costs.  ``experiment.py phi``
sweeps straight through the cliff at 16 bits rather than taking anyone's word
for it.

:mod:`uge.metrics` measures that curve.  It is the answer to "how do you encode
a game" that this directory exists to produce.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = [
    "ALPHABET",
    "BIG",
    "HEADER_CODES",
    "OUTCOME_CODES",
    "RESERVED",
    "CODES",
    "Codebook",
    "DEFAULT_LAYOUT",
    "DISJOINT",
    "LAYOUTS",
    "MAX_PHI_TOKENS",
    "OUTCOMES",
    "PLAIN",
    "RADIX",
    "STOP",
    "TapeError",
    "TapeSpec",
    "W",
    "code_to_token",
    "codebook",
    "digest_codes",
    "digits_token",
    "is_token",
    "outcome_code",
    "outcome_name",
    "phase_ambiguity",
    "token_digits",
    "token_to_code",
    "tokens",
    "trim_to_tokens",
]

W = 3
"""Token width, in characters.  It is ``radixnet.encoding.WINDOW`` and it is not a coincidence."""

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
"""base64url.  64 symbols, and - the point - none of them is ``<``, ``>`` or ``/``,
so no token can ever collide with RadixNet's ``<s>``, ``</s>`` or ``<back>`` sentinels."""

RADIX = len(ALPHABET)
CODES = RADIX**W

BIG = ALPHABET[:43]
STOP = ALPHABET[43:]
"""The split :data:`DISJOINT` uses.  ``43 * 43 * 21 = 38_829`` is the largest
``x * x * (64 - x)`` there is, so this is the most capacity a two-set split of
base64url can give - and it is comfortably more than chess's 20_480 actions."""

_VALUE = {c: i for i, c in enumerate(ALPHABET)}

LAYOUTS = ("struct", "lsb", "msb")
"""How an action becomes :data:`W` characters.

``lsb`` / ``msb`` are the two flat orders: the action's index in mixed radix,
least or most significant digit first.  ``struct`` lets the *game* lay the
fields out itself (:meth:`uge.game.Game.struct_token`) and falls back to ``lsb``
for a game that does not care.  Chess is why it exists: ``from`` and ``to`` are
64 values each, so under :data:`PLAIN` a chess move is exactly three base64
digits with nothing wasted, and the seams between tokens then read "given where
the last move landed, where does the next one start" - recapture structure,
handed to the network for free."""

DEFAULT_LAYOUT = "lsb"

MAX_PHI_TOKENS = 4
"""A state abstraction may occupy at most this many tokens.

More than one is always a waste - see the module docstring - but the limit is
not 1, because ``experiment.py phi`` has to be able to *show* that."""

OUTCOME_CODES = 8
HEADER_CODES = 2048
RESERVED = HEADER_CODES + OUTCOME_CODES
"""Codes reserved at the top of every codebook for the header and outcome tokens.

The header is an 11-bit *hash* of the spec rather than a packing of its fields:
there are more fields than bits, and spending real capacity on a readable
header would cost chess abstraction it cannot spare.  Two different specs share
a header with probability 1/2048, which matters only if both were fed to the
same network - and the arm that does that (``experiment.py multi``) varies the
game, which is in the hash."""


class TapeError(ValueError):
    """A malformed token, an out-of-range code, or a tape that is not a whole number of tokens."""


# ---------------------------------------------------------------------------
# codebooks: int <-> token
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Codebook:
    """A mixed-radix bijection between integers and :data:`W`-character tokens.

    ``slots`` is the alphabet each character position draws from.  When the
    first and last of those alphabets are disjoint the phase of a trigram is
    readable from the trigram itself, which is the whole reason this is a class
    and not a :func:`divmod`.
    """

    name: str
    slots: tuple[str, ...]

    @property
    def radices(self) -> tuple[int, ...]:
        return tuple(len(s) for s in self.slots)

    @property
    def capacity(self) -> int:
        c = 1
        for r in self.radices:
            c *= r
        return c

    @property
    def bits(self) -> int:
        """Whole bits of digest one token can carry."""
        return self.capacity.bit_length() - 1

    def action_capacity(self) -> int:
        """Codes below the reserved meta region - shared between actions and ``phi``."""
        return self.capacity - RESERVED

    def phi_capacity(self, action_bound: int) -> int:
        """Codes left for ``phi`` once the actions have taken theirs."""
        return max(0, self.action_capacity() - action_bound)

    def phi_bits_for(self, action_bound: int) -> int:
        """Whole bits of abstraction a game with this action bound can carry in one token."""
        room = self.phi_capacity(action_bound)
        return room.bit_length() - 1 if room > 0 else 0

    def meta(self, index: int, count: int) -> str:
        """The ``index``-th of the ``count`` codes reserved at the very top."""
        if not 0 <= index < count:
            raise TapeError(f"meta index {index} out of range [0, {count})")
        return self.encode(self.capacity - count + index, "lsb")

    def meta_index(self, token: str, count: int) -> int | None:
        """Inverse of :meth:`meta`, or ``None`` when the token is not in that region."""
        code = self.decode(token, "lsb")
        if code is None or code < self.capacity - count:
            return None
        return code - (self.capacity - count)

    @property
    def disjoint(self) -> bool:
        """Whether no trigram of a tape written with this codebook can be read in two phases."""
        return not (set(self.slots[0]) & set(self.slots[W - 1]))

    def encode(self, code: int, order: str = "lsb") -> str:
        if not isinstance(code, int) or isinstance(code, bool):
            raise TapeError(f"code must be an int, got {type(code).__name__}")
        if not 0 <= code < self.capacity:
            raise TapeError(f"code {code} out of range [0, {self.capacity}) for codebook {self.name}")
        slots = self.slots if order == "lsb" else tuple(reversed(self.slots))
        out = []
        for alphabet in slots:
            code, r = divmod(code, len(alphabet))
            out.append(alphabet[r])
        if order == "msb":
            out.reverse()
        return "".join(out)

    def decode(self, token: str, order: str = "lsb") -> int | None:
        """``None`` rather than an exception: a *predicted* token is expected to be junk sometimes."""
        if not isinstance(token, str) or len(token) != W:
            return None
        slots = self.slots if order == "lsb" else tuple(reversed(self.slots))
        chars = token if order == "lsb" else token[::-1]
        code, scale = 0, 1
        for ch, alphabet in zip(chars, slots):
            i = alphabet.find(ch)
            if i < 0:
                return None  # that character is not legal in that slot
            code += i * scale
            scale *= len(alphabet)
        return code


PLAIN = Codebook("plain", (ALPHABET, ALPHABET, ALPHABET))
"""All 64 symbols in every slot: maximum capacity, ambiguous phases."""

DISJOINT = Codebook("disjoint", (BIG, BIG, STOP))
"""A reserved last character: 15 % of the capacity, and no phase ambiguity at all."""

_CODEBOOKS = {"plain": PLAIN, "disjoint": DISJOINT}


def codebook(name: str) -> Codebook:
    try:
        return _CODEBOOKS[name]
    except KeyError:
        raise TapeError(f"unknown codebook {name!r}; expected one of {sorted(_CODEBOOKS)}") from None


def code_to_token(code: int, layout: str = "lsb", book: Codebook = PLAIN) -> str:
    """``code`` as exactly :data:`W` characters.  ``struct`` falls back to ``lsb`` here."""
    return book.encode(code, "msb" if layout == "msb" else "lsb")


def token_to_code(token: str, layout: str = "lsb", book: Codebook = PLAIN) -> int | None:
    """Inverse of :func:`code_to_token`; ``None`` when the token is not one."""
    return book.decode(token, "msb" if layout == "msb" else "lsb")


def digits_token(values: tuple[int, ...] | list[int]) -> str:
    """A token written digit by digit over the full alphabet: ``(from, to, promo) -> "cQA"``.

    Only :data:`PLAIN` can express this - it needs 64 values in every slot - so
    the structured layouts are a :data:`PLAIN`-only feature, which is the trade
    the module docstring describes.
    """
    if len(values) != W:
        raise TapeError(f"a token needs exactly {W} digits, got {len(values)}")
    out = []
    for v in values:
        if not 0 <= v < RADIX:
            raise TapeError(f"digit {v} out of range [0, {RADIX})")
        out.append(ALPHABET[v])
    return "".join(out)


def token_digits(token: str) -> tuple[int, ...]:
    """Inverse of :func:`digits_token`."""
    if not isinstance(token, str) or len(token) != W:
        raise TapeError(f"a token is exactly {W} characters, got {token!r}")
    try:
        return tuple(_VALUE[c] for c in token)
    except KeyError:
        raise TapeError(f"{token!r} is not made of alphabet characters") from None


def is_token(token: str) -> bool:
    """Whether ``token`` is :data:`W` characters all drawn from :data:`ALPHABET`."""
    return isinstance(token, str) and len(token) == W and all(c in _VALUE for c in token)


def tokens(text: str) -> list[str]:
    """Split a tape into its tokens; raises unless the length is a multiple of :data:`W`."""
    if len(text) % W:
        raise TapeError(f"a tape is a whole number of {W}-character tokens, got {len(text)} characters")
    return [text[i : i + W] for i in range(0, len(text), W)]


def trim_to_tokens(text: str) -> str:
    """Drop a trailing partial token.

    A *predicted* tape is cut wherever the search stopped emitting, which is
    almost never a token boundary - :func:`radixnet.encoding.repair_base64` is
    the same job for the media formats.  Here the repair is one line, because
    the grid is fixed.
    """
    return text[: len(text) - len(text) % W]


def phase_ambiguity(tapes: list[str]) -> dict:
    """How much of a corpus can be read in more than one phase.

    ``distinct`` counts trigram *types*, ``weighted`` counts occurrences, and it
    is ``weighted`` that predicts whether the model's ranking will be poisoned:
    a collision on a trigram that occurs once is a curiosity, one on a trigram
    that occurs in every game is a defect.
    """
    phases: dict[str, set[int]] = {}
    counts: dict[str, int] = {}
    for tape in tapes:
        for i in range(len(tape) - W + 1):
            gram = tape[i : i + W]
            phases.setdefault(gram, set()).add(i % W)
            counts[gram] = counts.get(gram, 0) + 1
    ambiguous = [g for g, p in phases.items() if len(p) > 1]
    total = sum(counts.values())
    return {
        "trigrams": len(phases),
        "ambiguous": len(ambiguous),
        "distinct": len(ambiguous) / max(1, len(phases)),
        "weighted": sum(counts[g] for g in ambiguous) / max(1, total),
    }


# ---------------------------------------------------------------------------
# the state abstraction, phi
# ---------------------------------------------------------------------------


def digest_codes(key: bytes, bits: int, room: int, salt: bytes = b"") -> list[int]:
    """``bits`` bits of a stable hash of ``key``, split into codes that each fit in ``room``.

    ``blake2b`` rather than ``hash()``: Python salts string hashing per process,
    and a corpus that encodes differently on every run is not a corpus.
    ``salt`` is mixed in ahead of the key - :mod:`uge.codec` puts the result the
    game ended in there when result-conditioning is on, so that the same
    position reached in a won game and in a lost one becomes two different
    nodes and ``P(a | phi(s), result)`` is a thing the network can learn.

    The codes are offsets *within* the ``phi`` region; the caller adds the
    region's base.
    """
    if bits <= 0:
        return []
    if room < 2:
        raise TapeError("no room left for phi: the action space fills the codebook")
    per = max(1, room.bit_length() - 1)
    n_tokens = -(-bits // per)
    if n_tokens > MAX_PHI_TOKENS:
        raise TapeError(f"{bits} bits needs {n_tokens} tokens of {per} bits, the limit is {MAX_PHI_TOKENS}")
    raw = hashlib.blake2b(salt + b"\x00" + key, digest_size=max(8, -(-bits // 8))).digest()
    value = int.from_bytes(raw, "big") & ((1 << bits) - 1)
    out = []
    for _ in range(n_tokens):
        out.append(value & ((1 << per) - 1))
        value >>= per
    return out


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

OUTCOMES = ("ongoing", "draw", "first", "second", "broken")
"""``broken`` is a tape the game refused: the decoder stopped before a terminal state."""


def outcome_code(name: str) -> int:
    try:
        return OUTCOMES.index(name)
    except ValueError:
        raise TapeError(f"unknown outcome {name!r}; expected one of {OUTCOMES}") from None


def outcome_name(code: int | None) -> str:
    return OUTCOMES[code] if code is not None and 0 <= code < len(OUTCOMES) else "broken"


# ---------------------------------------------------------------------------
# the tape spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TapeSpec:
    """Everything the encoder and the decoder have to agree on, in one object.

    It is also the **header token**, which is the first token of every tape and
    therefore a single node in the graph that every tape of this game and this
    configuration passes through.  One network trained on six games has six
    header nodes and the walk picks its game at the first step: the selector of
    ``Experiments/NeuralCompression`` experiment 1, except that here it falls
    out of the encoding rather than being bolted on.
    """

    game: str
    game_index: int
    phi_bits: int = 15
    layout: str = DEFAULT_LAYOUT
    book: str = "disjoint"
    condition_result: bool = False
    with_outcome: bool = True
    phi_exact: bool = False
    """Write the game's own state code instead of a hash.  Only for a game whose
    state space fits in one token (:meth:`uge.game.Game.state_code`)."""
    phi_prev_bits: int = 0
    """How many of ``phi_bits`` are spent on a summary of the **previous action**
    rather than on a hash of the position.

    The token is one abstraction of a fixed size, and this splits it.  It exists
    because of what the ``phi`` sweep shows: a hash has no **locality** - two
    nearly identical positions land in unrelated buckets - so for a game whose
    state space dwarfs the corpus a hashed ``phi`` can only recognise positions
    it has already seen, and chess at 13 bits covers a fraction of them.  A
    summary of the last move (:meth:`uge.game.Game.action_summary`; for chess,
    the square it landed on) is not a hash, generalises across positions, and
    costs the same bits.  ``phi_prev_bits = phi_bits`` is history only, ``0`` is
    position only, and ``experiment.py mix`` sweeps between them."""

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise TapeError(f"unknown layout {self.layout!r}; expected one of {LAYOUTS}")
        book = self.codebook  # validates the name
        if not 0 <= self.game_index < 8:
            raise TapeError(f"game_index {self.game_index} does not fit the header's 3 bits")
        if self.phi_bits < 0:
            raise TapeError(f"phi_bits must be >= 0, got {self.phi_bits}")
        if self.phi_tokens > MAX_PHI_TOKENS:
            raise TapeError(f"phi_bits={self.phi_bits} needs more than {MAX_PHI_TOKENS} tokens")
        if book.disjoint and self.structured:
            raise TapeError(
                f"layout {self.layout!r} needs the full alphabet in every slot; "
                f"codebook {book.name!r} reserves one - use layout='lsb' or book='plain'"
            )
        if not 0 <= self.phi_prev_bits <= max(self.phi_bits, 0):
            raise TapeError(
                f"phi_prev_bits={self.phi_prev_bits} must be between 0 and phi_bits={self.phi_bits}"
            )
        if self.phi_prev_bits and self.phi_exact:
            raise TapeError("phi_exact writes the state code itself; it has no bits to lend to history")
        if self.phi_exact and self.condition_result:
            raise TapeError(
                "phi_exact writes the state code itself, so there is no room to mix a result into it; "
                "use a hashed phi for result conditioning"
            )

    @property
    def codebook(self) -> Codebook:
        return codebook(self.book)

    @property
    def structured(self) -> bool:
        """Whether this spec asks for a layout only :data:`PLAIN` can express."""
        return self.layout in ("struct", "msb")

    def phi_tokens_for(self, action_bound: int) -> int:
        """How many tokens the abstraction occupies for a game with this action bound."""
        if self.phi_exact:
            return 1
        if not self.phi_bits:
            return 0
        per = max(1, self.codebook.phi_bits_for(action_bound))
        return -(-self.phi_bits // per)

    @property
    def phi_tokens(self) -> int:
        """How many tokens a state abstraction occupies, ignoring the action region.

        An upper bound used for validation only; the encoder and the decoder
        both go through :meth:`phi_tokens_for`, which knows what the game has
        already taken.
        """
        if self.phi_exact:
            return 1
        return -(-self.phi_bits // self.codebook.bits) if self.phi_bits else 0

    def group_for(self, action_bound: int) -> int:
        """Tokens per ply for a game with this action bound: the abstraction plus the action."""
        return self.phi_tokens_for(action_bound) + 1

    @property
    def group(self) -> int:
        """Tokens per ply, ignoring the action region - see :meth:`phi_tokens`."""
        return self.phi_tokens + 1

    def header(self) -> str:
        """The node every tape of this spec starts at, from the reserved header region.

        An 11-bit hash of everything that changes the encoding, so two specs
        that differ anywhere get different headers and a network fed several
        tape formats keeps them apart from the very first token.
        """
        ident = (
            f"{self.game}|{self.game_index}|{self.phi_bits}|{self.phi_prev_bits}|"
            f"{self.layout}|{self.book}|{int(self.condition_result)}|{int(self.phi_exact)}"
        )
        index = int.from_bytes(hashlib.blake2b(ident.encode(), digest_size=2).digest(), "big")
        return self.codebook.meta(OUTCOME_CODES + (index % HEADER_CODES), RESERVED)

    def outcome_token(self, name: str) -> str:
        """The tail token, from the last 8 codes of the codebook."""
        return self.codebook.meta(outcome_code(name), OUTCOME_CODES)

    def read_outcome(self, token: str) -> str | None:
        """``None`` when ``token`` is not one of the outcome tokens."""
        index = self.codebook.meta_index(token, OUTCOME_CODES)
        return None if index is None else outcome_name(index)

    def describe(self) -> str:
        phi = "exact" if self.phi_exact else ("none" if self.phi_bits == 0 else f"{self.phi_bits}b")
        if self.phi_prev_bits:
            phi += f"({self.phi_prev_bits}prev)"
        tail = "+result" if self.condition_result else ""
        return f"{self.game}/phi={phi}{tail}/{self.layout}/{self.book}"
