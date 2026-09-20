"""What a game has to be for the universal encoder to encode it.

Every finite, discrete, sequential game is the same object:

    G = (S, s0, A, legal: S -> 2^A, tau: S x A -> S, terminal, payoff)

a **labelled transition system with payoffs**.  A play is a walk through it.
A walk through a labelled graph is precisely what RadixCyclicNN stores, so
"how do I encode a game" is not a question about strings - it is the question
of *which projection of the walk gets written down*, and that is
:mod:`uge.codec`'s job, not this file's.

This file is the interface.  A game supplies the seven parts above plus three
encoding hooks, and gets the encoder, the decoder, the referee, the metrics
and the training corpus for free.  Six games implement it
(:mod:`uge.games`): chess, othello, connect four, tic-tac-toe, nim and pig -
the last one to show that chance is not an extension of the scheme but an
instance of it.

Defaults are provided for everything a game does not care about.  The smallest
possible game is ``initial``, ``legal``, ``apply``, ``winner`` and
``action_count``; the rest has a sensible default here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .tape import LAYOUTS, Codebook, TapeError, TapeSpec, code_to_token, is_token, token_to_code

__all__ = ["Game", "REGISTRY", "get_game", "game_names", "register"]


class Game(ABC):
    """A labelled transition system with payoffs, plus three encoding hooks."""

    name: str = "game"
    players: int = 2
    index: int = 0
    """Position in :data:`REGISTRY`, and the first digit of the header token."""

    # -- the transition system ------------------------------------------------

    @abstractmethod
    def initial(self) -> Any:
        """``s0``.  States must be hashable or at least copyable; nothing here mutates one."""

    @abstractmethod
    def legal(self, state: Any) -> list[Any]:
        """``legal(s)`` - every action the rules accept, in a deterministic order."""

    @abstractmethod
    def apply(self, state: Any, action: Any) -> Any:
        """``tau(s, a)`` - a **new** state.  Never mutates ``state``."""

    @abstractmethod
    def winner(self, state: Any) -> int | None:
        """``0`` / ``1`` for a won terminal state, ``-1`` for a drawn one, ``None`` while the game is on."""

    def is_terminal(self, state: Any) -> bool:
        return self.winner(state) is not None

    @abstractmethod
    def to_move(self, state: Any) -> int:
        """Whose turn it is: ``0`` or ``1``.  A chance node still answers with the player it interrupts."""

    # -- chance ---------------------------------------------------------------

    def is_chance(self, state: Any) -> bool:
        """Whether the next action belongs to the dealer rather than to a player.

        The universal scheme does not special-case chance: a chance node is a
        node whose mover happens to be the world, its outcomes are ordinary
        actions with ordinary tokens, and the tape records which one happened.
        The network then *learns* the die instead of being told it, which is
        exactly what it does with the rules.
        """
        return False

    def chance_weights(self, state: Any) -> list[float]:
        """Probabilities over ``legal(state)`` at a chance node; uniform by default."""
        actions = self.legal(state)
        return [1.0 / len(actions)] * len(actions) if actions else []

    # -- the encoding hooks ---------------------------------------------------

    @property
    @abstractmethod
    def action_count(self) -> int:
        """Size of ``A``.  Must be ``<= uge.tape.CODES``."""

    def action_index(self, action: Any) -> int:
        """``a -> int`` in ``[0, action_count)``.  The default assumes actions *are* ints."""
        return int(action)

    def index_action(self, index: int) -> Any:
        """Inverse of :meth:`action_index`.  The default assumes actions are ints."""
        return index

    def struct_token(self, action: Any) -> str:
        """The game's own field layout for an action - one base64 digit per field.

        Overridden by any game whose action decomposes naturally into three
        fields of at most 64 values (chess: ``from``, ``to``, ``promotion``).
        The default is the flat ``lsb`` code, so a game that does not care pays
        nothing and ``layout="struct"`` is simply ``layout="lsb"`` for it.
        """
        from .tape import PLAIN  # noqa: PLC0415 - struct layouts are a PLAIN-only feature

        return code_to_token(self.action_index(action), "lsb", PLAIN)

    def struct_action(self, token: str) -> Any | None:
        """Inverse of :meth:`struct_token`; ``None`` when the token names no action."""
        from .tape import PLAIN  # noqa: PLC0415 - struct layouts are a PLAIN-only feature

        return self._flat_action(token, "lsb", PLAIN)

    def action_token(self, action: Any, spec: TapeSpec) -> str:
        """``a -> W characters``, in the spec's layout and codebook."""
        if spec.layout == "struct":
            return self.struct_token(action)
        return code_to_token(self.action_index(action), spec.layout, spec.codebook)

    def token_action(self, token: str, spec: TapeSpec) -> Any | None:
        """``W characters -> a``, or ``None`` when the token is not a syntactically valid action.

        ``None`` is the first of the two ways a predicted tape can be wrong,
        and the cheaper one: the token does not name an action *at all*.  The
        second is a token that names an action the position does not allow,
        which only ``legal`` can catch.  :mod:`uge.metrics` counts them apart.
        """
        if not is_token(token):
            return None
        if spec.layout == "struct":
            return self.struct_action(token)
        return self._flat_action(token, spec.layout, spec.codebook)

    def _flat_action(self, token: str, layout: str, book: Codebook) -> Any | None:
        code = token_to_code(token, layout, book)
        if code is None or code >= self.action_count:
            return None
        return self.index_action(code)

    @abstractmethod
    def state_key(self, state: Any) -> bytes:
        """A canonical, **injective** encoding of the state - what ``phi`` hashes.

        Injective matters: two positions that share a key are one node in the
        graph whatever ``phi_bits`` says, and the abstraction stops being a
        dial we control.  Include everything the rules read - for chess that is
        castling rights, the en passant square and the halfmove clock, not just
        the pieces - and as little else as possible: the move *number* is left
        out, so a transposition hashes as a transposition.
        """

    action_summary_bound: int = 64
    """How many values :meth:`action_summary` can take."""

    def action_summary(self, action: Any) -> int:
        """A small, *meaningful* code for an action - what ``phi_prev_bits`` writes down.

        Meaningful is the whole point, and it is what separates this from a
        hash: the summary has to put actions that behave alike near each other,
        because the bits spent on it buy generalisation only if positions that
        share a summary really do play alike.  Chess answers with the square the
        move landed on, so "a piece was just taken on e5" is one bucket across
        every position in which it happened; a hash of the move index would
        scatter those and buy nothing.

        The default truncates the action index, which is meaningful for the
        games whose actions *are* small integers (a column, a cell, a die face)
        and meaningless for anything larger - so a game with a big action space
        should override it.
        """
        return self.action_index(action) % self.action_summary_bound

    def token_bound(self, spec: TapeSpec) -> int:
        """Exclusive upper bound on the codes this game's **action** tokens occupy.

        The base of the ``phi`` region, and therefore the thing that decides how
        much abstraction a game can afford.  A flat layout reaches exactly
        ``action_count``; a structured one spreads the same actions over a wider
        range (tic-tac-toe writes ``(col, row, 0)``, whose largest code is 130
        rather than 8), so it is computed once per layout and cached.
        """
        if spec.layout != "struct":
            return self.action_count
        cached = self._struct_bounds.get(self.name)
        if cached is None:
            from .tape import PLAIN  # noqa: PLC0415

            cached = 1 + max(
                PLAIN.decode(self.struct_token(self.index_action(i)), "lsb") or 0
                for i in range(self.action_count)
            )
            self._struct_bounds[self.name] = cached
        return cached

    _struct_bounds: dict[str, int] = {}

    def state_code(self, state: Any) -> int | None:
        """An injective ``int`` for the state, or ``None`` when the state space is too large.

        This is ``phi`` with the abstraction taken out: a game that can answer
        it gets a lossless one-token state and the graph becomes a
        transposition table, which is the far end of the dial
        (:data:`uge.tape.TapeSpec.phi_exact`).  ``tictactoe`` answers with a
        base-3 board code and ``nim`` with a mixed-radix heap code; the other
        four are far too large and answer ``None``.
        """
        return None

    # -- play -----------------------------------------------------------------

    def heuristic(self, state: Any) -> float:
        """Static evaluation from **player 0's** point of view; 0 when a game has nothing to say.

        Only the teachers in :mod:`uge.teachers` read it.  A game with a flat
        heuristic simply gets a random teacher, which is a legitimate corpus:
        the rules are still in it.
        """
        return 0.0

    def render(self, state: Any) -> str:
        """A human-readable board; used by ``experiment.py --show`` and by failing tests."""
        return repr(state)

    def action_str(self, action: Any) -> str:
        """A human-readable action, for the same two readers."""
        return str(action)

    # -- housekeeping ---------------------------------------------------------

    def validate(self) -> None:
        """Cheap self-check: the action space fits every codebook and round-trips through it."""
        from .tape import _CODEBOOKS  # noqa: PLC0415

        state = self.initial()
        for book_name, book in _CODEBOOKS.items():
            if self.action_count > book.action_capacity():
                raise TapeError(
                    f"{self.name}: action space {self.action_count} exceeds the "
                    f"{book.action_capacity()} non-meta codes of codebook {book_name!r}"
                )
            for layout in LAYOUTS:
                if book.disjoint and layout in ("struct", "msb"):
                    continue  # a structured layout needs the whole alphabet in every slot
                spec = TapeSpec(self.name, self.index, layout=layout, book=book_name)
                if self.state_code(state) is not None:
                    TapeSpec(self.name, self.index, layout=layout, book=book_name, phi_exact=True)
                for action in self.legal(state):
                    token = self.action_token(action, spec)
                    back = self.token_action(token, spec)
                    if back != action:
                        raise TapeError(
                            f"{self.name}/{layout}/{book_name}: "
                            f"{action!r} -> {token!r} -> {back!r} is not a round trip"
                        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} |A|={self.action_count}>"


REGISTRY: dict[str, Game] = {}


def register(game: Game) -> Game:
    """Add ``game`` to the registry and stamp it with its index (the header token's first digit)."""
    if game.name in REGISTRY:
        raise ValueError(f"a game named {game.name!r} is already registered")
    game.index = len(REGISTRY)
    REGISTRY[game.name] = game
    return game


def get_game(name: str) -> Game:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown game {name!r}; have {sorted(REGISTRY)}") from None


def game_names() -> list[str]:
    """Registry order - which is also header-token order."""
    return list(REGISTRY)
