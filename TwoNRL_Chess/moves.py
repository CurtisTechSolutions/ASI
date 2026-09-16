"""The action space: every from-square to every to-square, legal or not.

``Research/2NRL.md`` §6.5 is the reason this file exists.

    A game is two things: rules and payoffs... I went after the first, and I
    went after it by losing on purpose... a rule is invisible while you are
    obeying it.  Play a legal move and the game says nothing.  Play an illegal
    one and the game answers exactly: not that, and here is the line.

So the network is not handed the legal moves.  It proposes a move out of all
``64 x 64 = 4096`` from-to pairs, the board refuses it or accepts it, and a
refusal is a training example - the cheapest and most informative kind, because
the rules are deterministic and a rule-failure is therefore the lowest-entropy
negative there is (§5).  A move is submitted, refused, and the next one down
the ranking is submitted, until one is accepted; **how many refusals that took
is the rule metric**, and it starts near 135 (what guessing costs when about 30
of 4096 moves are legal) and has to be driven to zero by learning.

Indices are in the mover's own frame - black's board is mirrored, exactly as in
``features.py`` - so one set of weights proposes for both colours and the
static half of the move encoding is a constant.

Move features, 184 of them, six of which are ever set::

    0   .. 63    from square
    64  .. 127   to square
    128 .. 142   file delta, -7 .. +7
    143 .. 157   rank delta, -7 .. +7
    158 .. 170   what is on the from square: empty, ours P N B R Q K, theirs
    171 .. 183   what is on the to square, the same thirteen

The deltas are geometry, not chess: nothing here says a knight moves in an L or
that a bishop cannot jump.  What the network has to discover is which
(piece, delta) pairs the board will accept, which is the rule set.

Under-promotion is outside the action space - a pawn reaching the last rank
promotes to a queen - so 4096 indices cover every move but three per promoting
pawn.
"""

from __future__ import annotations

import chess
import numpy as np

from features import FEATURE_DIM as BOARD_DIM

N_SQUARES = 64
N_ACTIONS = N_SQUARES * N_SQUARES            # 4096

FROM_OFFSET = 0
TO_OFFSET = 64
DFILE_OFFSET = 128
DRANK_OFFSET = 143
FROM_CONTENT_OFFSET = 158
TO_CONTENT_OFFSET = 171
MOVE_DIM = TO_CONTENT_OFFSET + 13            # 184
INPUT_DIM = BOARD_DIM + MOVE_DIM             # 959

# Per-action constants, in the mover's frame.
ACTION_FROM = np.arange(N_ACTIONS) // N_SQUARES
ACTION_TO = np.arange(N_ACTIONS) % N_SQUARES
ACTION_DFILE = (ACTION_TO % 8) - (ACTION_FROM % 8) + 7
ACTION_DRANK = (ACTION_TO // 8) - (ACTION_FROM // 8) + 7

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING)
_CONTENT_OF = {pt: i + 1 for i, pt in enumerate(PIECE_ORDER)}


def square_contents(board: chess.Board, pov: chess.Color) -> np.ndarray:
    """``(64,)`` of ``0`` empty, ``1..6`` ours, ``7..12`` theirs, in ``pov``'s frame."""
    states = np.zeros(N_SQUARES, dtype=np.int64)
    mirror = pov == chess.BLACK
    for square, piece in board.piece_map().items():
        code = _CONTENT_OF[piece.piece_type] + (0 if piece.color == pov else 6)
        states[square ^ 56 if mirror else square] = code
    return states


def to_square(pov_square: int, pov: chess.Color) -> int:
    return pov_square ^ 56 if pov == chess.BLACK else pov_square


def to_action(board: chess.Board, move: chess.Move, pov: chess.Color) -> int:
    """The action index of a real move (promotion piece ignored)."""
    f, t = move.from_square, move.to_square
    if pov == chess.BLACK:
        f, t = f ^ 56, t ^ 56
    return f * N_SQUARES + t


def to_move(board: chess.Board, action: int, pov: chess.Color) -> chess.Move:
    """The move an action index names, with a queen promotion where one is required."""
    f = to_square(action // N_SQUARES, pov)
    t = to_square(action % N_SQUARES, pov)
    promotion = None
    if board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7):
        promotion = chess.QUEEN
    return chess.Move(f, t, promotion=promotion)


def dense(board_x: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """``(len(actions), INPUT_DIM)``: the board vector plus each move's own features.

    The slow, obvious path.  :func:`preactivation` is the fast one, and
    ``test_sbnn.py`` asserts they agree exactly.
    """
    actions = np.asarray(actions, dtype=np.int64)
    out = np.zeros((len(actions), INPUT_DIM))
    out[:, :BOARD_DIM] = board_x
    rows = np.arange(len(actions))
    f, t = ACTION_FROM[actions], ACTION_TO[actions]
    out[rows, BOARD_DIM + FROM_OFFSET + f] = 1.0
    out[rows, BOARD_DIM + TO_OFFSET + t] = 1.0
    out[rows, BOARD_DIM + DFILE_OFFSET + ACTION_DFILE[actions]] = 1.0
    out[rows, BOARD_DIM + DRANK_OFFSET + ACTION_DRANK[actions]] = 1.0
    out[rows, BOARD_DIM + FROM_CONTENT_OFFSET + states[f]] = 1.0
    out[rows, BOARD_DIM + TO_CONTENT_OFFSET + states[t]] = 1.0
    return out


def preactivation(Wm: np.ndarray, states: np.ndarray) -> np.ndarray:
    """``(4096, hidden)``: the first layer's move contribution for every action.

    Each action sets exactly six inputs, so its contribution to the first
    layer is the sum of six rows of the weight matrix.  Six gathers replace a
    ``4096 x 184`` matrix product and the 31 MB of mostly-zero board columns
    that the dense path would tile.
    """
    return (Wm[FROM_OFFSET + ACTION_FROM]
            + Wm[TO_OFFSET + ACTION_TO]
            + Wm[DFILE_OFFSET + ACTION_DFILE]
            + Wm[DRANK_OFFSET + ACTION_DRANK]
            + Wm[FROM_CONTENT_OFFSET + states[ACTION_FROM]]
            + Wm[TO_CONTENT_OFFSET + states[ACTION_TO]])
