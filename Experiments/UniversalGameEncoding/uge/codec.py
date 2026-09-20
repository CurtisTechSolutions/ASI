"""The encoder and the decoder - and the decoder is the referee.

Encoding is the easy half: walk the play, write a token per object.

Decoding is the half that matters, and it is **not string manipulation**.  A
tape is decoded by replaying it through the game's own transition function from
its own initial state, which makes decoding

* **total** - every tape decodes to *something*, because the replay stops the
  moment the rules refuse;
* **self-checking** - the ply at which it stopped, and why, is the output;
* **the training signal** - a tape that broke at ply ``k`` is a tape the game
  refused, which is exactly the negative that ``Research/2NRL.md`` §6.5 wants
  ("a rule is invisible while you are obeying it") and it costs nothing to
  produce.

``radixnet.vision`` needs :func:`radixnet.encoding.repair_base64` to survive a
garbled prediction.  Here the repair is :func:`uge.tape.trim_to_tokens` plus
the replay: cut to the token grid, then let the rules cut the rest.

Four ways a tape can be wrong, and the decoder tells them apart
---------------------------------------------------------------
``header``   the first token is not this game's header - the walk picked the
             wrong game, which only happens when one network holds several.
``syntax``   a token in an action slot does not name an action at all.
``illegal``  it names an action the position does not allow.
``phi``      the state token disagrees with the state the replay is in.  This
             is the tape's own checksum: the network wrote down a position it
             is not actually in, so it has desynchronised from the game even
             though every move so far was legal.

Where the tail token stops being the tail
-----------------------------------------
A tape ends with an outcome token, and a game whose last state is terminal ends
the replay before reaching it - so the tail is never ambiguous for a tape that
came out of :func:`encode`.  It *is* ambiguous for a tape the network
**predicted**, which can stop anywhere, and worse for a game that was
adjudicated rather than finished: the replay is still hungry when the tail
arrives and will happily read ``draw`` as a move.  :func:`_tail_at` decides
the question the only two ways it can be decided from the tape alone - the
token is last, and it parses as an outcome - and a prediction that loses to
both costs at most one ply of survival, which the README states rather than
hides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .game import Game
from .tape import W, TapeError, TapeSpec, digest_codes, trim_to_tokens

__all__ = [
    "BREAKS",
    "DecodeResult",
    "Play",
    "decode",
    "encode",
    "encode_prefix",
    "outcome_of",
    "phi_region",
    "result_salt",
    "state_token",
]

BREAKS = ("none", "header", "truncated", "syntax", "illegal", "phi")
"""Why a replay stopped.  ``none`` means the tape ran out exactly at a terminal state."""

_SALTS = {1: b"W", 0: b"D", -1: b"L"}


def result_salt(spec: TapeSpec, winner: int | None, mover: int) -> bytes:
    """What gets mixed into ``phi`` when result-conditioning is on.

    The result *from the point of view of the player about to move*.  A
    position reached in a game its mover went on to win and the same position
    in a game they lost become two different nodes, so ``P(a | phi(s), result)``
    is learnable and at play time you simply ask for the winning one.  The
    trick costs one byte and no architecture: the return is carried in the
    state abstraction rather than in a conditioning input the network does not
    have.
    """
    if not spec.condition_result:
        return b""
    if winner is None:
        return b"?"
    if winner < 0:
        return _SALTS[0]
    return _SALTS[1] if winner == mover else _SALTS[-1]


def phi_region(game: Game, spec: TapeSpec) -> tuple[int, int]:
    """``(base, room)`` of the ``phi`` region: the codes above the actions and below the meta tokens."""
    base = game.token_bound(spec)
    return base, spec.codebook.phi_capacity(base)


def state_token(
    game: Game,
    spec: TapeSpec,
    state: Any,
    winner: int | None = None,
    prev_action: Any = None,
) -> str:
    """The ``phi`` tokens for ``state`` - ``""`` when the tape is moves only.

    Written into the ``phi`` region of the code space, never into the actions'.
    A ``phi`` token that shares three characters with an action token is the
    *same node* in the graph, and the empty board hashing to "play cell 0" is
    the version of that bug which cost this directory the most time - see
    :mod:`uge.tape`.

    With ``phi_exact`` the game's own state code is written instead of a hash,
    so two positions share a token only if they *are* the same position and the
    abstraction stops abstracting.  That is the far end of the dial, and only
    ``tictactoe`` and ``nim`` are small enough to reach it.
    """
    if not spec.phi_exact and spec.phi_bits == 0:
        return ""
    base, room = phi_region(game, spec)
    book = spec.codebook
    if spec.phi_prev_bits:
        return _mixed_token(game, spec, state, winner, prev_action, base, room)
    if spec.phi_exact:
        code = game.state_code(state)
        if code is None:
            raise TapeError(f"{game.name} has no exact state code; use a hashed phi")
        if code >= room:
            raise TapeError(
                f"{game.name}: state code {code} does not fit the {room} codes left for phi "
                f"in codebook {spec.book!r} once {base} are spoken for by actions"
            )
        return book.encode(base + code, "lsb")
    salt = result_salt(spec, winner, game.to_move(state))
    codes = digest_codes(game.state_key(state), spec.phi_bits, room, salt)
    return "".join(book.encode(base + c, "lsb") for c in codes)


def _mixed_token(game, spec, state, winner, prev_action, base: int, room: int) -> str:
    """``phi`` split between a summary of the last action and a hash of the position.

    The high bits are :meth:`uge.game.Game.action_summary` of the move that was
    just played (0 at the start of the game, which is its own bucket and a true
    statement: nothing was played), the low bits a hash of the state.  One
    token, two kinds of information, and the split is the knob.
    """
    hash_bits = spec.phi_bits - spec.phi_prev_bits
    prev = 0 if prev_action is None else 1 + game.action_summary(prev_action)
    prev &= (1 << spec.phi_prev_bits) - 1
    salt = result_salt(spec, winner, game.to_move(state))
    low = digest_codes(game.state_key(state), hash_bits, 1 << hash_bits, salt) if hash_bits else [0]
    code = (prev << hash_bits) | (low[0] if low else 0)
    if code >= room:
        raise TapeError(
            f"{game.name}: phi code {code} does not fit the {room} codes left for phi; "
            f"lower phi_bits or use codebook 'plain'"
        )
    return spec.codebook.encode(base + code, "lsb")


def outcome_of(game: Game, state: Any) -> str:
    """The tail token's name for a state: ``draw`` / ``first`` / ``second`` / ``ongoing``."""
    w = game.winner(state)
    if w is None:
        return "ongoing"
    if w < 0:
        return "draw"
    return "first" if w == 0 else "second"


@dataclass(slots=True)
class Play:
    """One game as the encoder sees it: the states walked, the actions taken, how it ended."""

    states: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    outcome: str = "ongoing"

    @property
    def plies(self) -> int:
        return len(self.actions)

    @property
    def winner(self) -> int | None:
        return {"draw": -1, "first": 0, "second": 1}.get(self.outcome)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def encode(game: Game, spec: TapeSpec, play: Play) -> str:
    """``play -> tape``.

    ``len(play.states)`` is ``len(play.actions) + 1`` for a complete play: the
    terminal state has no action after it, and its ``phi`` token is not written
    (nothing follows it, so it would be a node with no children and a lie about
    where the game can go).
    """
    if len(play.states) != len(play.actions) + 1:
        raise TapeError(
            f"a play has one more state than actions, got {len(play.states)} and {len(play.actions)}"
        )
    winner = play.winner
    parts = [spec.header()]
    prev = None
    for state, action in zip(play.states, play.actions):
        parts.append(state_token(game, spec, state, winner, prev))
        parts.append(game.action_token(action, spec))
        prev = action
    if spec.with_outcome:
        parts.append(spec.outcome_token(play.outcome))
    return "".join(parts)


def encode_prefix(
    game: Game,
    spec: TapeSpec,
    states: list[Any],
    actions: list[Any],
    assume_winner: int | None = None,
) -> str:
    """The tape **up to and including** the current position's ``phi`` token.

    This is the question you put to the network: its next three characters are
    a move.  ``assume_winner`` is the result to condition on when
    result-conditioning is on - at play time you ask for the one where the
    mover wins.
    """
    if len(states) != len(actions) + 1:
        raise TapeError(f"expected {len(actions) + 1} states, got {len(states)}")
    parts = [spec.header()]
    prev = None
    for state, action in zip(states, actions):
        parts.append(state_token(game, spec, state, assume_winner, prev))
        parts.append(game.action_token(action, spec))
        prev = action
    parts.append(state_token(game, spec, states[-1], assume_winner, prev))
    return "".join(parts)


# ---------------------------------------------------------------------------
# decode - the replay
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecodeResult:
    """What the replay found.

    ``ok`` means the rules never refused - every token the replay reached named
    an action the position allowed.  ``complete`` is the stronger thing: ``ok``
    *and* the replay ended in a terminal state, so the tape is a whole game
    rather than a legal fragment of one.  Adjudicated games (:mod:`uge.corpus`
    caps chess at a ply limit) are ``ok`` but not ``complete``, and counting
    them as failures would be counting the cap as a rule violation.
    """

    ok: bool = False
    plies: int = 0
    """How many actions were accepted before the replay stopped - the survival length."""
    actions: list[Any] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    break_kind: str = "none"
    break_token: str = ""
    """The token that ended the replay, ``""`` when the tape simply ran out."""
    terminal: bool = False
    outcome: str = "ongoing"
    """What the replay reached, not what the tail token claims."""
    claimed_outcome: str | None = None
    """What the tail token claims, when there is one.  A disagreement is a finding, not an error."""
    complete: bool = False
    tokens_read: int = 0
    trimmed: int = 0
    """Characters dropped as a trailing partial token."""
    phi_checked: bool = False

    @property
    def state(self) -> Any:
        return self.states[-1] if self.states else None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "complete": self.complete,
            "plies": self.plies,
            "break_kind": self.break_kind,
            "break_token": self.break_token,
            "terminal": self.terminal,
            "outcome": self.outcome,
            "claimed_outcome": self.claimed_outcome,
            "tokens_read": self.tokens_read,
            "trimmed": self.trimmed,
            "phi_checked": self.phi_checked,
        }


def _is_outcome_token(spec: TapeSpec, token: str) -> bool:
    return spec.read_outcome(token) is not None


def _tail_at(spec: TapeSpec, group: int, toks: list[str]) -> int:
    """Index of the tail token, or ``len(toks)`` when there is not one to read."""
    if not spec.with_outcome or len(toks) < 2:
        return len(toks)
    if (len(toks) - 2) % group:
        return len(toks)  # the token count is not header + plies + tail
    return len(toks) - 1 if _is_outcome_token(spec, toks[-1]) else len(toks)


def decode(
    game: Game,
    spec: TapeSpec,
    text: str,
    max_plies: int = 1024,
    assume_winner: int | None = None,
    check_phi: bool = True,
) -> DecodeResult:
    """``tape -> play``, by replaying it through the rules.

    ``assume_winner`` supplies the salt when result-conditioning is on; without
    it the ``phi`` tokens cannot be recomputed and are not checked
    (``phi_checked`` says which happened).
    """
    raw = str(text or "")
    body = trim_to_tokens(raw)
    result = DecodeResult(trimmed=len(raw) - len(body))
    toks = [body[i : i + W] for i in range(0, len(body), W)]
    state = game.initial()
    result.states.append(state)
    prev_action = None

    if not toks or toks[0] != spec.header():
        result.break_kind = "header"
        result.break_token = toks[0] if toks else ""
        return result
    result.tokens_read = 1
    i = 1
    body_end = _tail_at(spec, spec.phi_tokens_for(game.token_bound(spec)) + 1, toks)

    n_phi = spec.phi_tokens_for(game.token_bound(spec))
    verify = (
        check_phi
        and n_phi > 0
        and not (spec.condition_result and assume_winner is None)
    )
    result.phi_checked = verify

    while i < body_end and result.plies < max_plies:
        if game.winner(state) is not None:
            break  # the game is over; whatever follows is the tail token
        if n_phi:
            if i + n_phi > body_end:
                result.break_kind = "truncated"
                break
            written = "".join(toks[i : i + n_phi])
            if verify:
                expected = state_token(game, spec, state, assume_winner, prev_action)
                if written != expected:
                    result.break_kind, result.break_token = "phi", written
                    break
            i += n_phi
            result.tokens_read += n_phi
        if i >= body_end:
            result.break_kind = "truncated"
            break
        token = toks[i]
        action = game.token_action(token, spec)
        if action is None:
            result.break_kind, result.break_token = "syntax", token
            break
        if action not in game.legal(state):
            result.break_kind, result.break_token = "illegal", token
            break
        state = game.apply(state, action)
        prev_action = action
        result.actions.append(action)
        result.states.append(state)
        result.plies += 1
        result.tokens_read += 1
        i += 1

    result.terminal = game.winner(state) is not None
    result.outcome = outcome_of(game, state)
    if result.break_kind == "none" and body_end < len(toks):
        result.claimed_outcome = spec.read_outcome(toks[body_end]) or "broken"
        result.tokens_read += 1
    result.ok = result.break_kind == "none"
    result.complete = result.ok and result.terminal
    return result
