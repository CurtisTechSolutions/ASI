"""The player: an SBNN that scores a proposed move, and the batch it trains on.

The network scores a *(position, move)* pair, and the move can be anything -
the whole 4096-action space of ``moves.py``, legal or not.  Playing a turn is
therefore: rank all 4096, submit the best one, and keep submitting down the
ranking until the board accepts one.  Every refusal on the way is a failure the
network caused and can learn from, which is ``Research/2NRL.md`` §6.5's
"failing at the game to learn its rules" taken literally.

Why the network scores moves rather than positions
--------------------------------------------------
Scoring the *position a move leads to* is the obvious alternative, and it
cannot be used here for a plain reason - an illegal move leads to no position,
so the network could not even express a proposal the board would refuse.

There is a second reason, and it is 2NRL's.  A position-scoring player is only
as good as the search on top of it, and a search has to commute with negation
or the inversion stops being order-reversing one level above where it was
proved.  The natural opponent model does not commute: ``min_r(-v)`` is
``-max_r(v)``, not ``-min_r(v)``.  ``test_sbnn.py`` demonstrates it.  Scoring
the move directly keeps §4.3's operator exact all the way to the policy - the
inversion negates all 4096 scores at once, so the move the failure network most
wanted is exactly the one it now least wants.

What the network outputs
------------------------
Seven numbers, not one (``rules.HEADS``).  The first is the score the softmax
and the ranking have always used; the other six answer on the sine
activation's own two ends (+1 yes, -1 no) and carry the
rules and some elementary chess - is this move legal, is it even geometrically
possible, does it capture, does it check, does it hang the piece, is the piece
already hanging - each with exact ground truth from the board itself.  They all
read one trunk, so what the network learns about chess it learns once.

Proposing then ranks by::

    quality  +  rule_weight * legal_logit

which is "play the best move you believe the board will accept".  The two terms
negate together under §4.3's inversion, so the composite ranking inverts
exactly as the bare score did, and ``rule_weight = 0`` reproduces the old
behaviour to the bit.
"""

from __future__ import annotations

import chess
import numpy as np

from features import FEATURE_DIM as BOARD_DIM
from features import encode
from moves import (INPUT_DIM, N_ACTIONS, dense, preactivation,
                   square_contents, to_action, to_move)
from rules import LEGAL, N_HEADS, QUALITY
from sbnn import SineNet

NEG_INF = -1.0e30
MAX_REFUSALS = 1000         # give up and take a random legal move after this many
                            # (guessing needs ~135; 1000 makes giving up ~0.02% likely)


class Agent:
    """A :class:`SineNet` over (position, move) pairs, proposing into a closed door."""

    def __init__(self, net: SineNet, tau: float = 1.0, rule_weight: float = 1.0) -> None:
        self.net = net
        self.tau = tau
        # A one-output network predates the heads; there is nothing to weigh in
        # that case, so old saved networks keep working and rank as they always did.
        self.rule_weight = rule_weight if net.sizes[-1] > 1 else 0.0

    @property
    def n_heads(self) -> int:
        return self.net.sizes[-1]

    @staticmethod
    def build(hidden: list[int], seed: int, tau: float = 1.0,
              n_heads: int = N_HEADS, rule_weight: float = 1.0) -> "Agent":
        rng = np.random.default_rng(seed)
        return Agent(SineNet([INPUT_DIM, *hidden, n_heads], rng), tau, rule_weight)

    def rank_of(self, heads: np.ndarray) -> np.ndarray:
        """The ordering moves are proposed in: quality, plus belief in legality.

        Both columns are read-outs of the same negated units, so this whole
        expression negates when the network is inverted - ``argmax`` becomes
        ``argmin`` for the composite exactly as it did for the bare score.
        """
        if self.rule_weight == 0.0 or heads.shape[1] == 1:
            return heads[:, QUALITY]
        return heads[:, QUALITY] + self.rule_weight * heads[:, LEGAL]

    # ------------------------------------------------------------- scoring

    def action_heads(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every head for all 4096 actions.  ``((4096, heads), board vector, contents)``.

        The first layer is split: the board half is one vector shared by every
        action, and the move half is six gathers (see :func:`moves.preactivation`).
        Identical to running :meth:`dense_rows` through the network, and 11x
        faster - ``test_sbnn.py`` checks they agree exactly.
        """
        pov = board.turn
        x = encode(board, pov)
        states = square_contents(board, pov)
        first = self.net.layers[0]
        z = x @ first.W[:BOARD_DIM] + preactivation(first.W[BOARD_DIM:], states) + first.bias
        y = first.a * np.sin(first.b * (z - first.h)) + first.k
        for layer in self.net.layers[1:]:
            y = layer.forward(y, train=False)
        return y, x, states

    def action_scores(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The proposal ranking over all 4096 actions."""
        heads, x, states = self.action_heads(board)
        return self.rank_of(heads), x, states

    def dense_rows(self, board_x: np.ndarray, states: np.ndarray,
                   actions: np.ndarray) -> np.ndarray:
        return dense(board_x, states, actions)

    # ------------------------------------------------------------ proposing

    def propose(self, board: chess.Board, rng: np.random.Generator | None = None,
                temperature: float = 0.0, keep_refusals: int = 8) -> dict:
        """Submit moves down the ranking until the board accepts one.

        ``temperature > 0`` perturbs the ranking with Gumbel noise, which is
        sampling without replacement from the softmax - exploration that still
        produces a whole ordering to walk.
        """
        heads, board_x, states = self.action_heads(board)
        scores = self.rank_of(heads)
        ranked = scores if temperature <= 0.0 or rng is None else (
            scores / max(temperature, 1e-6) + rng.gumbel(size=N_ACTIONS))
        order = np.argsort(-ranked)

        refused: list[int] = []
        chosen, n_refused, move = -1, 0, None
        for action in order[:MAX_REFUSALS + 1]:
            move = to_move(board, int(action), board.turn)
            if board.is_legal(move):
                chosen = int(action)
                break
            n_refused += 1
            if len(refused) < keep_refusals:
                refused.append(int(action))
        gave_up = chosen < 0
        if gave_up:                       # rare, and only before the rules are learned
            legal = list(board.legal_moves)
            move = legal[int(rng.integers(len(legal)))] if rng is not None else legal[0]
        # Castling has two spellings the board accepts - king to rook square
        # and king two squares - and only the second is how the rest of the
        # world names it.  Settle on one before anybody looks the move up.
        move = board.parse_uci(board.uci(move))
        chosen = to_action(board, move, board.turn)
        return {"action": chosen, "move": move,
                "refused": refused, "n_refused": n_refused, "gave_up": gave_up,
                "scores": scores, "heads": heads,
                "board_x": board_x, "states": states,
                "top1_legal": n_refused == 0}


class Batch:
    """A ragged set of decisions, flattened into one matrix the SBNN can eat.

    Each sample is a position plus ``width`` candidate *actions*, and the slots
    are fixed so that every arm trains on the identical rows:

    ==== ==============================================================
    slot what it holds
    ==== ==============================================================
    0    the move the network played (legal, and graded)
    1    Stockfish's best move          - the correction, every arm's phase 3
    2    Stockfish's worst legal move   - the lowest-entropy failure
    3    a uniformly random legal move  - the high-entropy control
    4    the first move the board refused - the network's own rule failure
    5    a uniformly random action out of all 4096 - almost always illegal
    6..  more random legal moves, to fill the softmax out
    ==== ==============================================================

    All that differs between arms is which slot the first block aims at, and
    whether the network is inverted before the second.
    """

    FIELDS = ("played", "best", "worst", "rand", "refused", "wild", "loss_cp",
              "rank", "weight", "is_rule")

    def __init__(self, samples: list[dict], width: int) -> None:
        self.n = len(samples)
        self.width = width
        self.dim = samples[0]["X"].shape[-1] if samples else INPUT_DIM
        self.X = np.zeros((self.n * width, self.dim), dtype=np.float32)
        self.mask = np.zeros((self.n, width), dtype=bool)
        for name in self.FIELDS:
            setattr(self, name, np.zeros(self.n, dtype=float))
        for i, s in enumerate(samples):
            m = s["X"].shape[0]
            self.X[i * width:i * width + m] = s["X"]
            if m < width:                       # pad with slot 0, then mask it off
                self.X[i * width + m:(i + 1) * width] = s["X"][0]
            self.mask[i, :m] = True
            for name in self.FIELDS:
                getattr(self, name)[i] = s[name]
        for name in ("played", "best", "worst", "rand", "refused", "wild"):
            setattr(self, name, getattr(self, name).astype(int))

    def __len__(self) -> int:
        return self.n

    def subset(self, idx: np.ndarray) -> "Batch":
        out = object.__new__(Batch)
        out.n, out.width, out.dim = len(idx), self.width, self.dim
        rows = (idx[:, None] * self.width + np.arange(self.width)).ravel()
        out.X = self.X[rows]
        out.mask = self.mask[idx]
        for name in self.FIELDS:
            setattr(out, name, getattr(self, name)[idx])
        return out


def policy(net: SineNet, batch: Batch, tau: float, train: bool = True) -> np.ndarray:
    """Probabilities over each sample's candidate moves."""
    s = net.forward(np.asarray(batch.X, dtype=np.float64), train=train)[:, QUALITY]
    logits = np.where(batch.mask, s.reshape(batch.n, batch.width) / tau, NEG_INF)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits) * batch.mask
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_grad(p: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``d/dlogits`` of a weighted cross-entropy: train **toward** ``target``."""
    onehot = np.zeros_like(p)
    onehot[np.arange(len(target)), target] = 1.0
    return (p - onehot) * weights[:, None]


def unlikelihood_grad(p: np.ndarray, target: np.ndarray, weights: np.ndarray,
                      cap: float = 10.0) -> np.ndarray:
    """``d/dlogits`` of ``-log(1 - p_target)``: train **away from** ``target``.

    The repulsive control of ``Research/2NRL.md`` §11 arm E - the same
    negatives, used the conventional way.  It is Welleck's unlikelihood rather
    than a plain reversed gradient because a reversed gradient has no minimum
    and simply diverges; capping ``p/(1-p)`` keeps the comparison about
    *direction* rather than about who blows up first.
    """
    idx = np.arange(len(target))
    pt = np.clip(p[idx, target], 0.0, 1.0 - 1e-6)
    coef = np.clip(pt / (1.0 - pt), 0.0, cap) * weights
    onehot = np.zeros_like(p)
    onehot[idx, target] = 1.0
    return (onehot - p) * coef[:, None]


def apply_grad(net: SineNet, opt, batch: Batch, dlogits: np.ndarray, tau: float) -> None:
    """Push ``dlogits`` back through the quality head and take one Adam step.

    The rule heads get zero here: the softmax over candidate moves is about
    which move is *best*, and legality is taught by :func:`rules.mse_grad` on
    its own rows, at its own rate.  Keeping the two gradients on separate
    columns is what stops "illegal" and "bad" being the same signal again.
    """
    dy = np.zeros((batch.n * batch.width, net.sizes[-1]))
    dy[:, QUALITY] = (dlogits / (tau * max(batch.n, 1))).ravel()
    opt.step(net.backward(dy))


def nll(p: np.ndarray, target: np.ndarray) -> float:
    idx = np.arange(len(target))
    return float(-np.log(np.clip(p[idx, target], 1e-12, None)).mean())
