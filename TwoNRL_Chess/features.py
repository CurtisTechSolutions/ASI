"""Encoding a chess position for the SBNN, always from one side's point of view.

The encoder is deliberately thin.  Piece placement, castling rights, whose turn
it is, whether the side to move is in check and whether it has any legal move
at all - that is the whole of it.  There is no material count, no mobility, no
king safety, no piece-square table: those are what the network is supposed to
find, and handing them over would measure the features rather than the method.
The last two flags are *rules*, not evaluation - together they are exactly
checkmate and stalemate, and they go in as inputs rather than as a hard-coded
terminal score so that the value of a mate is **learned** and therefore
**negates with the rest of the network** when 2NRL inverts it.

    0   .. 767   12 planes x 64 squares: our P N B R Q K, then theirs
    768 .. 771   castling rights: ours kingside/queenside, then theirs
    772          it is our turn in this position
    773          the side to move is in check
    774          the side to move has no legal move  (773 & 774 = checkmate)

Black's position is mirrored (``square ^ 56``) and the planes are ordered
*ours first*, so one set of weights plays both colours.  Every arm of the
experiment uses this same encoder, so nothing here can favour one over another.
"""

from __future__ import annotations

import chess
import numpy as np

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING)
PLANE_OF = {pt: i for i, pt in enumerate(PIECE_ORDER)}

N_PLANES = 12
N_SQUARES = 64
CASTLING_OFFSET = N_PLANES * N_SQUARES          # 768
TURN_OFFSET = CASTLING_OFFSET + 4               # 772
CHECK_OFFSET = TURN_OFFSET + 1                  # 773
NO_MOVES_OFFSET = CHECK_OFFSET + 1              # 774
FEATURE_DIM = NO_MOVES_OFFSET + 1               # 775


def encode(board: chess.Board, pov: chess.Color, out: np.ndarray | None = None) -> np.ndarray:
    """One position as ``FEATURE_DIM`` floats, oriented so ``pov`` is "us"."""
    x = np.zeros(FEATURE_DIM) if out is None else out
    mirror = pov == chess.BLACK
    for square, piece in board.piece_map().items():
        plane = PLANE_OF[piece.piece_type] + (0 if piece.color == pov else 6)
        x[plane * N_SQUARES + (square ^ 56 if mirror else square)] = 1.0
    opp = not pov
    x[CASTLING_OFFSET + 0] = float(board.has_kingside_castling_rights(pov))
    x[CASTLING_OFFSET + 1] = float(board.has_queenside_castling_rights(pov))
    x[CASTLING_OFFSET + 2] = float(board.has_kingside_castling_rights(opp))
    x[CASTLING_OFFSET + 3] = float(board.has_queenside_castling_rights(opp))
    x[TURN_OFFSET] = float(board.turn == pov)
    x[CHECK_OFFSET] = float(board.is_check())
    x[NO_MOVES_OFFSET] = float(not any(board.generate_legal_moves()))
    return x


def encode_children(board: chess.Board, moves: list[chess.Move], pov: chess.Color,
                    out: np.ndarray | None = None) -> np.ndarray:
    """``(len(moves), FEATURE_DIM)``: the position each move leads to, ``pov``'s view.

    ``board`` is left exactly as it was found.  The player does not use this -
    it scores *(position, move)* pairs, for the reasons in ``agent.py`` - and it
    is here so ``test_sbnn.py`` can build the position-scoring alternative and
    show its search breaking the inversion.
    """
    rows = np.zeros((len(moves), FEATURE_DIM)) if out is None else out
    for i, move in enumerate(moves):
        board.push(move)
        encode(board, pov, out=rows[i])
        board.pop()
    return rows


def material(board: chess.Board, pov: chess.Color) -> int:
    """Centipawn material balance from ``pov``, on the usual 1/3/3/5/9 scale.

    Only used for reporting - the network never sees it.
    """
    values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    total = 0
    for piece in board.piece_map().values():
        total += values[piece.piece_type] * (1 if piece.color == pov else -1)
    return total
