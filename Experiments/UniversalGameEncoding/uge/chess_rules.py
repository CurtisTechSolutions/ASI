"""Chess, in pure Python, with no dependencies - and proved by perft.

``TwoNRL_Chess/`` uses ``python-chess`` and Stockfish.  This directory cannot:
``Experiments/README.md`` says each experiment stands alone, and the whole
point of the universal encoder is that a game only has to supply
:class:`uge.game.Game` - so the rules have to be *here*, and they have to be
right, or every legality number measured against them is meaningless.

They are checked the only way chess rules can be checked: **perft**, the count
of leaf nodes at depth ``n`` from a position, against the published values for
the six standard positions (``tests.py perft``).  A movegen with one wrong
en-passant case passes a hundred hand-written tests and fails perft(3).

Board layout
------------
Squares are ``rank * 8 + file``, so ``a1 = 0``, ``h1 = 7``, ``a8 = 56``,
``h8 = 63``.  The board is a 64-character string (``.`` is empty, ``PNBRQK``
white, ``pnbrqk`` black): immutable, hashable, and cheap to copy - a move
builds a new one, so nothing ever has to be unmade.

What is deliberately not implemented
------------------------------------
**Threefold repetition.**  It is not a function of the position, it is a
function of the history, and :class:`uge.game.Game` asks ``winner(state)`` to
be a function of the state.  Draw by repetition is therefore an adjudication
applied by :mod:`uge.corpus` (a ply cap), not a rule here, and the file says so
rather than quietly returning the wrong answer.  The side effect is one the
repository likes: a repeated position is a genuine **cycle** in the graph
(``Research/CyclesAreAFeature.md``) instead of a terminal state.
"""

from __future__ import annotations

__all__ = [
    "KIWIPETE",
    "PERFT_POSITIONS",
    "Position",
    "START_FEN",
    "attacked",
    "from_fen",
    "in_check",
    "legal_moves",
    "make_move",
    "perft",
    "square_name",
    "to_fen",
]

WHITE_PIECES = "PNBRQK"
BLACK_PIECES = "pnbrqk"
EMPTY = "."

_FILES = "abcdefgh"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def square_name(sq: int) -> str:
    return f"{_FILES[sq & 7]}{(sq >> 3) + 1}"


def square_index(name: str) -> int:
    return (int(name[1]) - 1) * 8 + _FILES.index(name[0])


# ---------------------------------------------------------------------------
# precomputed step tables
# ---------------------------------------------------------------------------


def _steps(deltas: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    table = []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        row = []
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                row.append(nr * 8 + nf)
        table.append(tuple(row))
    return table


def _rays(deltas: list[tuple[int, int]]) -> list[tuple[tuple[int, ...], ...]]:
    table = []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        rays = []
        for df, dr in deltas:
            ray = []
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                ray.append(nr * 8 + nf)
                nf += df
                nr += dr
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return table


_KNIGHT_D = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING_D = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_BISHOP_D = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
_ROOK_D = [(0, 1), (1, 0), (0, -1), (-1, 0)]

KNIGHT_STEPS = _steps(_KNIGHT_D)
KING_STEPS = _steps(_KING_D)
BISHOP_RAYS = _rays(_BISHOP_D)
ROOK_RAYS = _rays(_ROOK_D)
QUEEN_RAYS = [BISHOP_RAYS[sq] + ROOK_RAYS[sq] for sq in range(64)]

# squares a white pawn on sq attacks, and the same for black
WHITE_PAWN_ATT = _steps([(-1, 1), (1, 1)])
BLACK_PAWN_ATT = _steps([(-1, -1), (1, -1)])

PROMO_PIECES = (None, "N", "B", "R", "Q")
"""Promotion codes 0-4; 0 is "no promotion"."""

# castling rights lost when a piece leaves or is captured on these squares
_ROOK_HOME = {0: "Q", 7: "K", 56: "q", 63: "k"}


class Position:
    """An immutable chess position.  Hashable, so ``legal_moves`` can be cached."""

    __slots__ = ("board", "white", "castling", "ep", "halfmove", "fullmove", "_hash")

    def __init__(
        self,
        board: str,
        white: bool,
        castling: str,
        ep: int | None,
        halfmove: int = 0,
        fullmove: int = 1,
    ) -> None:
        self.board = board
        self.white = white
        self.castling = castling
        self.ep = ep
        self.halfmove = halfmove
        self.fullmove = fullmove
        self._hash = hash((board, white, castling, ep, halfmove))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (
            self.board == other.board
            and self.white == other.white
            and self.castling == other.castling
            and self.ep == other.ep
            and self.halfmove == other.halfmove
        )

    def key(self) -> bytes:
        """The rule-relevant state, canonically.

        Pieces, side, castling rights, the en passant square and the halfmove
        clock - everything the rules read.  The fullmove number is left out, so
        a transposition hashes as a transposition.
        """
        return f"{self.board}|{'w' if self.white else 'b'}|{self.castling or '-'}|{self.ep}|{self.halfmove}".encode()

    def __repr__(self) -> str:
        return f"Position({to_fen(self)!r})"


# ---------------------------------------------------------------------------
# FEN
# ---------------------------------------------------------------------------


def from_fen(fen: str) -> Position:
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"not a FEN: {fen!r}")
    rows = parts[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"FEN needs 8 ranks, got {len(rows)}: {fen!r}")
    squares = [EMPTY] * 64
    for i, row in enumerate(rows):
        rank = 7 - i
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                if file > 7:
                    raise ValueError(f"rank {rank + 1} overflows in {fen!r}")
                squares[rank * 8 + file] = ch
                file += 1
        if file != 8:
            raise ValueError(f"rank {rank + 1} has {file} files in {fen!r}")
    castling = "".join(c for c in "KQkq" if c in parts[2])
    ep = None if parts[3] == "-" else square_index(parts[3])
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    return Position("".join(squares), parts[1] == "w", castling, ep, halfmove, fullmove)


def to_fen(pos: Position) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row, gap = "", 0
        for file in range(8):
            piece = pos.board[rank * 8 + file]
            if piece == EMPTY:
                gap += 1
            else:
                row += (str(gap) if gap else "") + piece
                gap = 0
        rows.append(row + (str(gap) if gap else ""))
    ep = "-" if pos.ep is None else square_name(pos.ep)
    return f"{'/'.join(rows)} {'w' if pos.white else 'b'} {pos.castling or '-'} {ep} {pos.halfmove} {pos.fullmove}"


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------


def attacked(board: str, sq: int, by_white: bool) -> bool:
    """Whether ``sq`` is attacked by ``by_white``'s pieces.  Colour of the occupant is irrelevant."""
    pieces = WHITE_PIECES if by_white else BLACK_PIECES
    pawn, knight, bishop, rook, queen, king = pieces
    # a pawn attacks sq iff sq is attacked from where that pawn stands, which is
    # the mirror of the squares sq itself would attack as a pawn of the other colour
    for s in (BLACK_PAWN_ATT if by_white else WHITE_PAWN_ATT)[sq]:
        if board[s] == pawn:
            return True
    for s in KNIGHT_STEPS[sq]:
        if board[s] == knight:
            return True
    for s in KING_STEPS[sq]:
        if board[s] == king:
            return True
    for ray in BISHOP_RAYS[sq]:
        for s in ray:
            piece = board[s]
            if piece != EMPTY:
                if piece == bishop or piece == queen:
                    return True
                break
    for ray in ROOK_RAYS[sq]:
        for s in ray:
            piece = board[s]
            if piece != EMPTY:
                if piece == rook or piece == queen:
                    return True
                break
    return False


def king_square(board: str, white: bool) -> int:
    return board.find("K" if white else "k")


def in_check(pos: Position) -> bool:
    """Whether the side to move is in check."""
    ks = king_square(pos.board, pos.white)
    return ks >= 0 and attacked(pos.board, ks, not pos.white)


# ---------------------------------------------------------------------------
# move generation
# ---------------------------------------------------------------------------


def pseudo_moves(pos: Position) -> list[tuple[int, int, int]]:
    """``(from, to, promo)`` moves that obey piece movement but may leave the king in check."""
    board = pos.board
    white = pos.white
    own = WHITE_PIECES if white else BLACK_PIECES
    theirs = BLACK_PIECES if white else WHITE_PIECES
    moves: list[tuple[int, int, int]] = []
    add = moves.append
    forward = 8 if white else -8
    start_rank = 1 if white else 6
    last_rank = 7 if white else 0

    for sq in range(64):
        piece = board[sq]
        if piece == EMPTY or piece not in own:
            continue
        kind = piece.upper()
        if kind == "P":
            one = sq + forward
            if 0 <= one < 64 and board[one] == EMPTY:
                if (one >> 3) == last_rank:
                    for promo in (1, 2, 3, 4):
                        add((sq, one, promo))
                else:
                    add((sq, one, 0))
                    if (sq >> 3) == start_rank:
                        two = one + forward
                        if board[two] == EMPTY:
                            add((sq, two, 0))
            for target in (WHITE_PAWN_ATT if white else BLACK_PAWN_ATT)[sq]:
                occupant = board[target]
                if occupant in theirs or (pos.ep is not None and target == pos.ep):
                    if (target >> 3) == last_rank:
                        for promo in (1, 2, 3, 4):
                            add((sq, target, promo))
                    else:
                        add((sq, target, 0))
        elif kind == "N":
            for target in KNIGHT_STEPS[sq]:
                if board[target] not in own:
                    add((sq, target, 0))
        elif kind == "K":
            for target in KING_STEPS[sq]:
                if board[target] not in own:
                    add((sq, target, 0))
        else:
            rays = BISHOP_RAYS[sq] if kind == "B" else ROOK_RAYS[sq] if kind == "R" else QUEEN_RAYS[sq]
            for ray in rays:
                for target in ray:
                    occupant = board[target]
                    if occupant == EMPTY:
                        add((sq, target, 0))
                    else:
                        if occupant in theirs:
                            add((sq, target, 0))
                        break

    # castling: rights, empty path, and the king may not start, cross or land in check
    rights = pos.castling
    home = 4 if white else 60
    if rights and board[home] == ("K" if white else "k"):
        enemy = not white
        for flag, rook_sq, empties, path in (
            ("K" if white else "k", home + 3, (home + 1, home + 2), (home, home + 1, home + 2)),
            ("Q" if white else "q", home - 4, (home - 1, home - 2, home - 3), (home, home - 1, home - 2)),
        ):
            if flag not in rights or board[rook_sq] != ("R" if white else "r"):
                continue
            if any(board[s] != EMPTY for s in empties):
                continue
            if any(attacked(board, s, enemy) for s in path):
                continue
            add((home, path[2], 0))
    return moves


def make_move(pos: Position, move: tuple[int, int, int]) -> Position:
    """Apply ``move`` without checking legality; returns a new :class:`Position`."""
    src, dst, promo = move
    board = list(pos.board)
    piece = board[src]
    kind = piece.upper()
    white = pos.white
    capture = board[dst] != EMPTY

    board[dst] = piece
    board[src] = EMPTY

    if kind == "P":
        if pos.ep is not None and dst == pos.ep and not capture:
            board[dst - 8 if white else dst + 8] = EMPTY  # en passant: the pawn is not on dst
            capture = True
        if promo:
            board[dst] = PROMO_PIECES[promo] if white else PROMO_PIECES[promo].lower()
    elif kind == "K" and abs(dst - src) == 2:
        rook_from, rook_to = (src + 3, src + 1) if dst > src else (src - 4, src - 1)
        board[rook_to] = board[rook_from]
        board[rook_from] = EMPTY

    castling = pos.castling
    if castling:
        if kind == "K":
            castling = "".join(c for c in castling if c.isupper() != white)
        else:
            lost = _ROOK_HOME.get(src)
            if lost and lost in castling:
                castling = castling.replace(lost, "")
        taken = _ROOK_HOME.get(dst)  # a rook captured on its home square also ends the right
        if taken and taken in castling:
            castling = castling.replace(taken, "")

    ep = None
    if kind == "P" and abs(dst - src) == 16:
        ep = (src + dst) // 2

    halfmove = 0 if (kind == "P" or capture) else pos.halfmove + 1
    return Position("".join(board), not white, castling, ep, halfmove, pos.fullmove + (0 if white else 1))


def legal_moves(pos: Position) -> list[tuple[int, int, int]]:
    """Every legal move, in a deterministic order (``from``, then ``to``, then promotion)."""
    out = []
    for move in pseudo_moves(pos):
        after = make_move(pos, move)
        ks = king_square(after.board, pos.white)
        if ks < 0 or not attacked(after.board, ks, not pos.white):
            out.append(move)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# terminal conditions
# ---------------------------------------------------------------------------


def insufficient_material(board: str) -> bool:
    """K vs K, K+minor vs K, and K+B vs K+B on the same colour: the drawn-by-force material."""
    pieces = [(i, c) for i, c in enumerate(board) if c != EMPTY and c not in "Kk"]
    if any(c.upper() in "PRQ" for _, c in pieces):
        return False
    if len(pieces) <= 1:
        return True
    if len(pieces) == 2:
        (i, a), (j, b) = pieces
        if a.upper() == "B" and b.upper() == "B" and a.isupper() != b.isupper():
            return ((i & 7) + (i >> 3)) % 2 == ((j & 7) + (j >> 3)) % 2
    return False


def outcome(pos: Position) -> int | None:
    """``None`` while the game is on; ``-1`` drawn; ``0`` / ``1`` for a win by white / black.

    Checkmate, stalemate, the fifty-move rule and insufficient material.
    Threefold repetition is *not* here - see the module docstring.
    """
    if not legal_moves(pos):
        return (1 if pos.white else 0) if in_check(pos) else -1
    if pos.halfmove >= 100 or insufficient_material(pos.board):
        return -1
    return None


# ---------------------------------------------------------------------------
# perft - the only proof that any of the above is right
# ---------------------------------------------------------------------------


def perft(pos: Position, depth: int) -> int:
    """Leaf nodes at ``depth``.  Compared against published values in ``tests.py``."""
    if depth <= 0:
        return 1
    moves = legal_moves(pos)
    if depth == 1:
        return len(moves)
    return sum(perft(make_move(pos, m), depth - 1) for m in moves)


PERFT_POSITIONS = (
    ("startpos", START_FEN, (20, 400, 8902, 197281)),
    ("kiwipete", KIWIPETE, (48, 2039, 97862)),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238)),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", (6, 264, 9467)),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", (44, 1486, 62379)),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", (46, 2079, 89890)),
)
"""The six standard perft positions and their published node counts (Chess Programming Wiki)."""
