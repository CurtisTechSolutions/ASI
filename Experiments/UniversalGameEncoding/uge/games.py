"""Six games, one interface.

They were chosen to break the encoder if it is going to break:

==============  =========  ==================================================
game            ``|A|``    what it is here to test
==============  =========  ==================================================
``chess``       20480      a large structured action space: under
                           :data:`~uge.tape.PLAIN` a move is exactly
                           ``(from, to, promotion)``, one base64 digit each,
                           with nothing wasted.  It is also the game whose
                           action space is big enough to squeeze ``phi``
``othello``     65         a mid-sized action space with a *pass* - an action
                           that is legal only sometimes and changes no disc
``connect4``    7          a tiny action space, where **guessing is a strong
                           baseline**: nearly every one of the seven codes is
                           legal, so ``legal@1`` alone would flatter a model
                           that had learned nothing
``tictactoe``   9          small enough for an **injective** ``phi``
                           (``3 ** 9 = 19_683`` boards in one token): the one
                           game where the abstraction dial reaches the end
``nim``         21         a **solved** game - the optimal move is computable,
                           so "did it play well" has an exact answer and not
                           just an opinion
``pig``         8          **chance**.  The die is not a special case, it is a
                           player whose actions go on the tape like anyone's
==============  =========  ==================================================

Every one of them is under 200 lines of rules, which is the argument for the
interface: the cost of adding a game is the cost of writing the game.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import chess_rules as C
from .game import Game, register
from .tape import digits_token, token_digits

__all__ = ["Chess", "Connect4", "Nim", "Othello", "Pig", "TicTacToe"]


# ---------------------------------------------------------------------------
# chess
# ---------------------------------------------------------------------------

_PIECE_VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

# a coarse central-control table; index by square in the mover's own frame
_CENTRE = [
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 2, 3, 4, 4, 3, 2, 1,
    2, 3, 5, 6, 6, 5, 3, 2,
    3, 4, 6, 8, 8, 6, 4, 3,
    3, 4, 6, 8, 8, 6, 4, 3,
    2, 3, 5, 6, 6, 5, 3, 2,
    1, 2, 3, 4, 4, 3, 2, 1,
    0, 1, 2, 3, 3, 2, 1, 0,
]


@lru_cache(maxsize=200_000)
def _chess_legal(pos: C.Position) -> tuple[tuple[int, int, int], ...]:
    return tuple(C.legal_moves(pos))


@lru_cache(maxsize=200_000)
def _chess_outcome(pos: C.Position) -> int | None:
    return C.outcome(pos)


class Chess(Game):
    """Full chess (bar threefold repetition - see :mod:`uge.chess_rules`), proved by perft.

    The action space is every ``(from, to, promotion)``, **legal or not**:
    ``64 * 64 * 5 = 20480``.  That is deliberate and it is
    ``Research/2NRL.md`` §6.5's argument - the network is not handed the legal
    moves, so a rule is something it can *break*, and breaking it is the only
    way the rule becomes visible.  About 30 of the 20480 are legal in a typical
    position, so a uniform guess is legal 0.15 % of the time.
    """

    name = "chess"
    players = 2

    @property
    def action_count(self) -> int:
        return 64 * 64 * 5

    def initial(self) -> C.Position:
        return C.from_fen(C.START_FEN)

    def legal(self, state: C.Position) -> list[tuple[int, int, int]]:
        return list(_chess_legal(state))

    def apply(self, state: C.Position, action: tuple[int, int, int]) -> C.Position:
        return C.make_move(state, action)

    def winner(self, state: C.Position) -> int | None:
        return _chess_outcome(state)

    def to_move(self, state: C.Position) -> int:
        return 0 if state.white else 1

    def action_index(self, action: tuple[int, int, int]) -> int:
        src, dst, promo = action
        return (src * 64 + dst) * 5 + promo

    def index_action(self, index: int) -> tuple[int, int, int]:
        rest, promo = divmod(index, 5)
        src, dst = divmod(rest, 64)
        return (src, dst, promo)

    def struct_token(self, action: tuple[int, int, int]) -> str:
        """``(from, to, promotion)`` as one base64 digit each - the whole point of ``W = 3``."""
        return digits_token(action)

    def struct_action(self, token: str) -> tuple[int, int, int] | None:
        src, dst, promo = token_digits(token)
        return None if promo > 4 else (src, dst, promo)

    def action_summary(self, action: tuple[int, int, int]) -> int:
        """The square the move landed on.

        The one piece of a chess move that means the same thing in every
        position: a capture on e5 is a capture on e5 whatever else is on the
        board, and the reply to it is usually a recapture on e5.  Sixty-four
        buckets, and they generalise, which is exactly what a hash of the
        position does not do.
        """
        return action[1]

    def state_key(self, state: C.Position) -> bytes:
        return state.key()

    def heuristic(self, state: C.Position) -> float:
        """Material, plus a little central control - enough for a teacher, not for a player."""
        score = 0.0
        for sq, piece in enumerate(state.board):
            if piece == ".":
                continue
            value = _PIECE_VALUE[piece.upper()] + _CENTRE[sq]
            score += value if piece.isupper() else -value
        return score / 100.0

    def render(self, state: C.Position) -> str:
        rows = [f"{r + 1} " + " ".join(state.board[r * 8 : r * 8 + 8]) for r in range(7, -1, -1)]
        return "\n".join(rows) + "\n  a b c d e f g h\n" + C.to_fen(state)

    def action_str(self, action: tuple[int, int, int]) -> str:
        src, dst, promo = action
        return C.square_name(src) + C.square_name(dst) + (C.PROMO_PIECES[promo].lower() if promo else "")


# ---------------------------------------------------------------------------
# tic-tac-toe
# ---------------------------------------------------------------------------

_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


class TicTacToe(Game):
    """3x3.  The one game here whose ``phi`` can be made injective inside a single token.

    ``3 ** 9 = 19_683`` boards - and whose turn it is need not be stored, being
    the parity of the marks already down - so with ``phi_exact`` the state
    abstraction stops being an abstraction and the graph becomes the game tree
    itself.  That is the far end of the dial, and it is why this game is in the
    set: everywhere else the end of the dial is unreachable and has to be
    argued about instead of measured.
    """

    name = "tictactoe"
    players = 2

    @property
    def action_count(self) -> int:
        return 9

    def initial(self) -> tuple[str, int]:
        return ("." * 9, 0)

    def legal(self, state: tuple[str, int]) -> list[int]:
        cells, _ = state
        return [] if self.winner(state) is not None else [i for i, c in enumerate(cells) if c == "."]

    def apply(self, state: tuple[str, int], action: int) -> tuple[str, int]:
        cells, mover = state
        mark = "XO"[mover]
        return (cells[:action] + mark + cells[action + 1 :], 1 - mover)

    def winner(self, state: tuple[str, int]) -> int | None:
        cells, _ = state
        for a, b, c in _LINES:
            if cells[a] != "." and cells[a] == cells[b] == cells[c]:
                return "XO".index(cells[a])
        return -1 if "." not in cells else None

    def to_move(self, state: tuple[str, int]) -> int:
        return state[1]

    def struct_token(self, action: int) -> str:
        return digits_token((action % 3, action // 3, 0))

    def struct_action(self, token: str) -> int | None:
        col, row, pad = token_digits(token)
        return None if pad or col > 2 or row > 2 else row * 3 + col

    def state_key(self, state: tuple[str, int]) -> bytes:
        return f"{state[0]}{state[1]}".encode()

    def state_code(self, state: tuple[str, int]) -> int:
        """The board in base 3: ``3 ** 9 = 19_683``, which fits one token in either codebook.

        Whose turn it is is *not* in the code and does not need to be - it is
        the parity of the marks already on the board - so the state space is
        the 19_683 boards rather than twice that, and it fits under
        :data:`~uge.tape.DISJOINT` where twice that would not.
        """
        code = 0
        for ch in state[0]:
            code = code * 3 + ".XO".index(ch)
        return code

    def render(self, state: tuple[str, int]) -> str:
        cells = state[0]
        return "\n".join(" ".join(cells[r * 3 : r * 3 + 3]) for r in range(3))


# ---------------------------------------------------------------------------
# connect four
# ---------------------------------------------------------------------------

C4_COLS, C4_ROWS = 7, 6


def _c4_lines() -> tuple[tuple[int, ...], ...]:
    lines = []
    for r in range(C4_ROWS):
        for c in range(C4_COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for k in range(4):
                    rr, cc = r + dr * k, c + dc * k
                    if 0 <= rr < C4_ROWS and 0 <= cc < C4_COLS:
                        cells.append(rr * C4_COLS + cc)
                if len(cells) == 4:
                    lines.append(tuple(cells))
    return tuple(lines)


C4_LINES = _c4_lines()


class Connect4(Game):
    """Seven columns.  The action space is so small that almost every token is illegal.

    ``|A| = 7`` out of ``64 ** 3`` codes, so a uniformly guessed token names an
    action at all with probability ``7 / 262144``.  Syntax failure dominates
    legality failure here, which is why :class:`uge.codec.DecodeResult` counts
    the two apart.
    """

    name = "connect4"
    players = 2

    @property
    def action_count(self) -> int:
        return C4_COLS

    def initial(self) -> tuple[str, int]:
        return ("." * (C4_COLS * C4_ROWS), 0)

    def legal(self, state: tuple[str, int]) -> list[int]:
        cells, _ = state
        if self.winner(state) is not None:
            return []
        top = (C4_ROWS - 1) * C4_COLS
        return [c for c in range(C4_COLS) if cells[top + c] == "."]

    def apply(self, state: tuple[str, int], action: int) -> tuple[str, int]:
        cells, mover = state
        for row in range(C4_ROWS):
            i = row * C4_COLS + action
            if cells[i] == ".":
                return (cells[:i] + "XO"[mover] + cells[i + 1 :], 1 - mover)
        raise ValueError(f"column {action} is full")

    def winner(self, state: tuple[str, int]) -> int | None:
        cells, _ = state
        for line in C4_LINES:
            a = cells[line[0]]
            if a != "." and all(cells[i] == a for i in line[1:]):
                return "XO".index(a)
        return -1 if "." not in cells else None

    def to_move(self, state: tuple[str, int]) -> int:
        return state[1]

    def struct_token(self, action: int) -> str:
        return digits_token((action, 0, 0))

    def struct_action(self, token: str) -> int | None:
        col, a, b = token_digits(token)
        return None if a or b or col >= C4_COLS else col

    def state_key(self, state: tuple[str, int]) -> bytes:
        return f"{state[0]}{state[1]}".encode()

    def heuristic(self, state: tuple[str, int]) -> float:
        """Open threes and twos, counted per line; the classic cheap Connect Four eval."""
        cells = state[0]
        score = 0.0
        for line in C4_LINES:
            xs = sum(cells[i] == "X" for i in line)
            os = sum(cells[i] == "O" for i in line)
            if xs and os:
                continue
            if xs:
                score += (1, 4, 32, 10000)[xs - 1]
            elif os:
                score -= (1, 4, 32, 10000)[os - 1]
        return score / 100.0

    def render(self, state: tuple[str, int]) -> str:
        cells = state[0]
        rows = [" ".join(cells[r * C4_COLS : (r + 1) * C4_COLS]) for r in range(C4_ROWS - 1, -1, -1)]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(C4_COLS))


# ---------------------------------------------------------------------------
# nim
# ---------------------------------------------------------------------------

NIM_HEAPS = (3, 5, 7)
NIM_MAX = max(NIM_HEAPS)


class Nim(Game):
    """Normal-play Nim on heaps of 3, 5 and 7: take any number from one heap, last to take wins.

    It is **solved** - the optimal move is any move that makes the xor of the
    heap sizes zero - so :mod:`uge.metrics` can ask a question no other game
    here answers exactly: not "was the move legal" or "did it match the
    teacher", but *was it optimal*.
    """

    name = "nim"
    players = 2

    @property
    def action_count(self) -> int:
        return len(NIM_HEAPS) * NIM_MAX

    def initial(self) -> tuple[tuple[int, ...], int]:
        return (NIM_HEAPS, 0)

    def legal(self, state: tuple[tuple[int, ...], int]) -> list[tuple[int, int]]:
        heaps, _ = state
        return [(h, n) for h, size in enumerate(heaps) for n in range(1, size + 1)]

    def apply(self, state, action):
        heaps, mover = state
        h, n = action
        after = list(heaps)
        after[h] -= n
        return (tuple(after), 1 - mover)

    def winner(self, state) -> int | None:
        heaps, mover = state
        return None if any(heaps) else 1 - mover  # the player who took the last object has just moved

    def to_move(self, state) -> int:
        return state[1]

    def action_index(self, action: tuple[int, int]) -> int:
        h, n = action
        return h * NIM_MAX + (n - 1)

    def index_action(self, index: int) -> tuple[int, int]:
        h, n = divmod(index, NIM_MAX)
        return (h, n + 1)

    def struct_token(self, action: tuple[int, int]) -> str:
        h, n = action
        return digits_token((h, n - 1, 0))

    def struct_action(self, token: str) -> tuple[int, int] | None:
        h, n, pad = token_digits(token)
        return None if pad or h >= len(NIM_HEAPS) or n >= NIM_MAX else (h, n + 1)

    def state_key(self, state) -> bytes:
        heaps, mover = state
        return bytes(heaps) + bytes([mover])

    def state_code(self, state) -> int:
        """Heap sizes in mixed radix, times the mover: ``4 * 6 * 8 * 2 = 384``."""
        heaps, mover = state
        code = 0
        for size, limit in zip(reversed(heaps), reversed(NIM_HEAPS)):
            code = code * (limit + 1) + size
        return code * 2 + mover

    def optimal(self, state) -> list[tuple[int, int]]:
        """Every move that leaves a xor of zero; all of them when the position is already lost."""
        heaps, _ = state
        best = [a for a in self.legal(state) if self._xor_after(heaps, a) == 0]
        return best or self.legal(state)

    @staticmethod
    def _xor_after(heaps: tuple[int, ...], action: tuple[int, int]) -> int:
        h, n = action
        after = list(heaps)
        after[h] -= n
        x = 0
        for size in after:
            x ^= size
        return x

    def heuristic(self, state) -> float:
        heaps, mover = state
        x = 0
        for size in heaps:
            x ^= size
        # a non-zero xor means the player to move can win; sign it for player 0
        edge = 1.0 if x else -1.0
        return edge if mover == 0 else -edge

    def render(self, state) -> str:
        heaps, mover = state
        return " ".join(f"[{i}]{'|' * n}" for i, n in enumerate(heaps)) + f"  to move: {mover}"


# ---------------------------------------------------------------------------
# othello
# ---------------------------------------------------------------------------

OTHELLO_PASS = 64
_OTH_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Othello(Game):
    """8x8 Reversi, pass included.

    The pass is the interesting part for an encoder: an action that is legal
    only when nothing else is, changes no disc, and hands the turn over.  It
    gets a token like any other action (``kind = 1``), which is the scheme's
    answer to every "but what about..." - if it is an element of ``A`` it has a
    code, and if it has a code it has a token.
    """

    name = "othello"
    players = 2

    @property
    def action_count(self) -> int:
        return 65

    def initial(self) -> tuple[str, int]:
        cells = ["."] * 64
        cells[27], cells[36] = "O", "O"
        cells[28], cells[35] = "X", "X"
        return ("".join(cells), 0)

    @staticmethod
    def _flips(cells: str, sq: int, mark: str, other: str) -> list[int]:
        if cells[sq] != ".":
            return []
        r0, c0 = divmod(sq, 8)
        out: list[int] = []
        for dr, dc in _OTH_DIRS:
            r, c = r0 + dr, c0 + dc
            run: list[int] = []
            while 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == other:
                run.append(r * 8 + c)
                r += dr
                c += dc
            if run and 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == mark:
                out.extend(run)
        return out

    def _placements(self, state: tuple[str, int]) -> list[int]:
        cells, mover = state
        mark, other = "XO"[mover], "XO"[1 - mover]
        return [sq for sq in range(64) if self._flips(cells, sq, mark, other)]

    def legal(self, state: tuple[str, int]) -> list[int]:
        if self.winner(state) is not None:
            return []
        moves = self._placements(state)
        return moves if moves else [OTHELLO_PASS]

    def apply(self, state: tuple[str, int], action: int) -> tuple[str, int]:
        cells, mover = state
        if action == OTHELLO_PASS:
            return (cells, 1 - mover)
        mark, other = "XO"[mover], "XO"[1 - mover]
        flips = self._flips(cells, action, mark, other)
        if not flips:
            raise ValueError(f"square {action} flips nothing")
        out = list(cells)
        out[action] = mark
        for sq in flips:
            out[sq] = mark
        return ("".join(out), 1 - mover)

    def winner(self, state: tuple[str, int]) -> int | None:
        cells, mover = state
        if self._placements(state) or self._placements((cells, 1 - mover)):
            return None
        xs, os = cells.count("X"), cells.count("O")
        return -1 if xs == os else (0 if xs > os else 1)

    def to_move(self, state: tuple[str, int]) -> int:
        return state[1]

    def struct_token(self, action: int) -> str:
        if action == OTHELLO_PASS:
            return digits_token((0, 0, 1))
        return digits_token((action % 8, action // 8, 0))

    def struct_action(self, token: str) -> int | None:
        file, rank, kind = token_digits(token)
        if kind == 1:
            return OTHELLO_PASS if file == 0 and rank == 0 else None
        return None if kind or file > 7 or rank > 7 else rank * 8 + file

    def action_summary(self, action: int) -> int:
        """The square played on; the pass gets its own bucket."""
        return action

    action_summary_bound = 65

    def state_key(self, state: tuple[str, int]) -> bytes:
        return f"{state[0]}{state[1]}".encode()

    def heuristic(self, state: tuple[str, int]) -> float:
        """Corners, then mobility, then discs - the standard ordering for a cheap Othello eval."""
        cells = state[0]
        score = 0.0
        for sq in (0, 7, 56, 63):
            if cells[sq] == "X":
                score += 25
            elif cells[sq] == "O":
                score -= 25
        score += 2 * (len(self._placements((cells, 0))) - len(self._placements((cells, 1))))
        score += 0.25 * (cells.count("X") - cells.count("O"))
        return score / 10.0

    def render(self, state: tuple[str, int]) -> str:
        cells = state[0]
        rows = [f"{r + 1} " + " ".join(cells[r * 8 : r * 8 + 8]) for r in range(7, -1, -1)]
        return "\n".join(rows) + "\n  a b c d e f g h"

    def action_str(self, action: int) -> str:
        return "pass" if action == OTHELLO_PASS else f"{'abcdefgh'[action % 8]}{action // 8 + 1}"


# ---------------------------------------------------------------------------
# pig - the chance game
# ---------------------------------------------------------------------------

PIG_TARGET = 100
PIG_HOLD, PIG_ROLL = 0, 1


class Pig(Game):
    """The dice game: roll to add to a turn total, a 1 wipes it, hold to bank it, 100 wins.

    Pig is in this set to make one point, and it is the point that decides
    whether "all games" is a claim or a slogan: **chance is not an extension of
    the encoding.**  A chance node is a node whose mover is the world, the six
    faces are six ordinary actions with six ordinary tokens, and what the tape
    records is which one happened.  Nothing in :mod:`uge.tape` or
    :mod:`uge.codec` knows that this game has a die in it.

    The consequence is that the network *learns* the die - the six faces are
    edges out of the chance node and their softmax converges on 1/6 - exactly
    as it learns which moves the board accepts.  A game with hidden information
    needs the same treatment one level up: ``phi`` hashes the mover's
    information set instead of the world state, and nothing else changes.
    """

    name = "pig"
    players = 2

    @property
    def action_count(self) -> int:
        return 8  # hold, roll, and the six faces

    def initial(self) -> tuple[int, int, int, int, bool]:
        return (0, 0, 0, 0, False)  # score0, score1, turn total, mover, pending roll

    def is_chance(self, state) -> bool:
        return state[4]

    def legal(self, state) -> list[int]:
        if self.winner(state) is not None:
            return []
        return [2, 3, 4, 5, 6, 7] if state[4] else [PIG_HOLD, PIG_ROLL]

    def apply(self, state, action: int):
        s0, s1, turn, mover, pending = state
        if pending:
            face = action - 1  # actions 2..7 are faces 1..6
            if face == 1:
                return (s0, s1, 0, 1 - mover, False)
            return (s0, s1, turn + face, mover, False)
        if action == PIG_ROLL:
            return (s0, s1, turn, mover, True)
        banked = (s0 + turn, s1) if mover == 0 else (s0, s1 + turn)
        return (banked[0], banked[1], 0, 1 - mover, False)

    def winner(self, state) -> int | None:
        s0, s1 = state[0], state[1]
        if s0 >= PIG_TARGET:
            return 0
        if s1 >= PIG_TARGET:
            return 1
        return None

    def to_move(self, state) -> int:
        return state[3]

    def struct_token(self, action: int) -> str:
        return digits_token((1 if action >= 2 else 0, action - 2 if action >= 2 else action, 0))

    def struct_action(self, token: str) -> int | None:
        kind, value, pad = token_digits(token)
        if pad:
            return None
        if kind == 0:
            return value if value <= 1 else None
        return value + 2 if value <= 5 else None

    def state_key(self, state) -> bytes:
        s0, s1, turn, mover, pending = state
        return bytes([min(s0, 255), min(s1, 255), min(turn, 255), mover, int(pending)])

    def heuristic(self, state) -> float:
        s0, s1, turn, mover, _ = state
        held = turn * 0.75  # a turn total is worth less than banked points: a 1 takes it all
        return (s0 - s1 + (held if mover == 0 else -held)) / 10.0

    def render(self, state) -> str:
        s0, s1, turn, mover, pending = state
        return f"{s0}-{s1}  turn={turn}  to move: {mover}{'  (rolling)' if pending else ''}"

    def action_str(self, action: int) -> str:
        return ("hold", "roll")[action] if action < 2 else f"die:{action - 1}"


CHESS = register(Chess())
OTHELLO = register(Othello())
CONNECT4 = register(Connect4())
TICTACTOE = register(TicTacToe())
NIM = register(Nim())
PIG = register(Pig())
