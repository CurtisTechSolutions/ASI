"""The rule curriculum: making the network *learn chess*, not merely stumble into it.

Until this module existed the network met the rules two rows at a time.  Every
decision handed the softmax one refused action and one wild one, buried in a
twelve-wide ranking whose target was *move quality* - so "this move is illegal"
and "this move is bad" arrived on the same wire, in the same units, and the
network had to disentangle them from about two bits per position.  It worked,
slowly: refusals fell from 660 to around 40, which is a long way from zero for
a rule set a beginner learns in an afternoon.

The trouble is that the board is a far richer teacher than that.  Every
position answers **all 4096 questions at once**, for free and exactly:
``board.is_legal`` is ground truth, there is no label noise, and no engine call
is needed.  Refusing to use that was leaving the entire rule set on the floor.

So this module turns the rules into what they are - a supervised classification
problem with perfect labels - and gives the network a set of heads to answer it
with.

The heads
---------

The output layer stops being a single score and becomes seven, all reading the
same trunk, so everything the network learns about chess is learned *once* and
shared:

======== ================================================ =====================
head     ground truth                                     what it forces
======== ================================================ =====================
quality  the graded softmax over candidates (as before)    payoffs
legal    ``board.is_legal(move)``                          the rules, entire
pseudo   ``board.is_pseudo_legal(move)``                   how pieces move
capture  ``board.is_capture(move)``                        what a move takes
check    ``board.gives_check(move)``                       forcing moves
safe     the to-square is unattacked afterwards            not hanging pieces
threat   the from-square is attacked right now             which pieces are loose
======== ================================================ =====================

``legal`` and ``pseudo`` factor the rule set in the way chess itself does::

    legal  ==  pseudo  AND  the move does not leave our own king in check

``pseudo`` is pure geometry and blocking - a knight's L, a bishop's diagonal, a
rook stopped by the piece in front of it, the pawn's peculiar habit of moving
one way and capturing another.  The gap between the two heads is precisely the
non-local part of the rules: pins, and the obligation to answer a check.  A
network that has ``pseudo`` and not ``legal`` has learned how the pieces move
and not yet that a pinned knight is nailed down.  Keeping them apart makes that
distinction *measurable* rather than a guess about what went wrong.

The last four are not rules at all, they are the first things anyone learns
after the rules, and they are here because the user asked for "more about
chess": the trunk that has to answer them cannot get away with memorising which
squares are usually fine.

The heads use the sine activation, not a sigmoid
------------------------------------------------

``Research/SineWaveActivationFunction.md`` is about replacing the sigmoid with
``f(x) = a*sin(b*(x - h)) + k``, so putting a logistic on top of these heads
would have been the one thing the paper exists to remove.  It is also
redundant: the network's last layer is a :class:`sbnn.SineDense` like every
other, so a head's output is **already** squashed by the sine, bounded in
``[k - |a|, k + |a|]`` - by default exactly ``[-1, 1]``.

So a head is read the way ``Experiments/ActivationFunctionTest/sinewave.py``
reads its output layer: the activated value directly, against a mean-squared
error, with no second squashing function anywhere.  The targets are the two
ends of the wave::

    +1   the answer is yes        -1   the answer is no

and the decision threshold is ``0``, exactly as ``logit > 0`` would have been.

This also makes the logical NOT cleaner rather than weaker.  A sigmoid needed
``sigma(-z) == 1 - sigma(z)`` to turn a negation into a complement; on the wave
the negation *is* the complement, because ``+1`` and ``-1`` are each other's
opposite and the activation is odd about ``h`` when ``k = 0``.  Negate a head
and every answer it gives is reversed, at the same distance from the threshold.
``test_sbnn.py`` checks it.

Which actions get asked about
-----------------------------

All 4096 per position would be 31 MB of mostly-zero features per position, and
it would also be a terrible curriculum: about 4066 of them are illegal, so the
head could answer "illegal" to everything and be 99.3% right.  Four groups are
sampled instead, and the second is the one that matters:

``legal``
    every legal move in the position (they are the positives, and there are
    only about thirty of them).
``hard``
    the highest-scoring **illegal** actions under the network as it stands -
    exactly the moves it is about to propose and have refused.  These are
    mined fresh every decision, so the negatives chase the network's current
    error instead of sitting still.  Refusals are literally the count of this
    set, so training on it is aimed straight at the metric.
``pin``
    pseudo-legal but illegal: moves that obey the geometry and leave the king
    in check.  There are only a handful in most positions and they are the
    hardest negative in chess, so they are sampled separately rather than left
    to chance.
``wild``
    uniform over all 4096, so the easy majority of the space stays represented
    and the head does not drift on the 99% it never sees.

Per position that is roughly ninety rows instead of two, with the negatives
concentrated where the network is currently wrong.
"""

from __future__ import annotations

import chess
import numpy as np

from moves import N_ACTIONS, to_move

# The output layer, in order.  Index 0 is the score the ranking and the softmax
# have always used; 1.. are the answer heads this module supervises, read on
# the sine activation's own two ends (+1 yes, -1 no).
HEADS = ("quality", "legal", "pseudo", "capture", "check", "safe", "threat")
N_HEADS = len(HEADS)
QUALITY, LEGAL, PSEUDO, CAPTURE, CHECK, SAFE, THREAT = range(N_HEADS)
RULE_HEADS = HEADS[1:]
N_RULE_HEADS = len(RULE_HEADS)

# `legal` and `pseudo` are defined for every action in the space, so they are
# trained on all of it.  The other four describe something a move *does*, which
# is only meaningful once the move is one the board would actually make, so
# they are masked to the legal rows.  Training them on nonsense would teach the
# trunk that the nonsense is worth representing.
LEGAL_ONLY = np.array([h in ("capture", "check", "safe", "threat") for h in RULE_HEADS])


def labels(board: chess.Board, actions: np.ndarray,
           pov: chess.Color) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth and mask for ``actions``: ``(n, 6)`` each.

    Costs no engine call and no guesswork - python-chess *is* the rule book, so
    every entry here is exact.  The mask is 0 where a head has nothing to say
    about that action (see :data:`LEGAL_ONLY`).
    """
    actions = np.asarray(actions, dtype=np.int64)
    n = len(actions)
    y = np.zeros((n, N_RULE_HEADS), dtype=np.float32)
    m = np.ones((n, N_RULE_HEADS), dtype=np.float32)
    them = not board.turn
    for i, action in enumerate(actions):
        move = to_move(board, int(action), pov)
        pseudo = board.is_pseudo_legal(move)
        legal = pseudo and board.is_legal(move)
        y[i, LEGAL - 1] = float(legal)
        y[i, PSEUDO - 1] = float(pseudo)
        if not legal:
            m[i, LEGAL_ONLY] = 0.0          # the four tactical heads say nothing
            continue
        y[i, CAPTURE - 1] = float(board.is_capture(move))
        y[i, CHECK - 1] = float(board.gives_check(move))
        # Does the piece land somewhere the opponent can take it?  Asked after
        # the move, because that is when it matters, and asked of the opponent
        # to move, which is who would take it.
        board.push(move)
        y[i, SAFE - 1] = float(not board.is_attacked_by(board.turn, move.to_square))
        board.pop()
        # Is the piece under attack where it stands?  A property of the square,
        # so it is the same for every move of that piece - which is the point:
        # "this knight is hanging" should raise the score of all of its moves.
        y[i, THREAT - 1] = float(board.is_attacked_by(them, move.from_square))
    return y, m


def curriculum(board: chess.Board, pov: chess.Color, ranking: np.ndarray,
               nprng: np.random.Generator, n_legal: int = 32, n_hard: int = 16,
               n_pin: int = 8, n_wild: int = 32) -> np.ndarray:
    """The actions this position will be asked about, as described in the module docstring.

    ``ranking`` is the network's own current ordering over all 4096 actions, so
    the ``hard`` group is mined from the network being trained rather than
    drawn at random.  That is the difference between a negative set that stays
    difficult and one the network outgrows in three rounds.
    """
    all_legal = {_action_of(board, m, pov) for m in board.legal_moves}
    legal_actions = np.array(sorted(all_legal), dtype=np.int64)
    if len(legal_actions) > n_legal:
        legal_actions = nprng.choice(legal_actions, size=n_legal, replace=False)
    chosen = set(int(a) for a in legal_actions)

    # `all_legal`, not `chosen`: in a position with more than `n_legal` legal
    # moves the ones that missed the sample are still legal, and mining them as
    # "the highest-scoring illegal actions" would put a legal move in the hard
    # negative group under a name that is simply false.
    illegal_rank = [int(a) for a in np.argsort(-ranking) if int(a) not in all_legal]
    hard = illegal_rank[:n_hard]
    chosen.update(hard)

    # Pseudo-legal but not legal: pins and unanswered checks.  Scanning the
    # ranking rather than all 4096 keeps this cheap and, again, prefers the
    # ones the network currently wants.
    pin: list[int] = []
    # Bounded scan: in a position with nothing pinned there are no pseudo-legal
    # illegal moves at all, and an unbounded search for them would run
    # `is_pseudo_legal` over the whole space on every decision.
    for action in illegal_rank[n_hard:n_hard + 512]:
        if len(pin) >= n_pin:
            break
        if board.is_pseudo_legal(to_move(board, action, pov)):
            pin.append(action)
    chosen.update(pin)

    wild = [int(a) for a in nprng.integers(N_ACTIONS, size=n_wild * 2) if int(a) not in chosen]
    chosen.update(wild[:n_wild])
    return np.array(sorted(chosen), dtype=np.int64)


def _action_of(board: chess.Board, move: chess.Move, pov: chess.Color) -> int:
    f, t = move.from_square, move.to_square
    if pov == chess.BLACK:
        f, t = f ^ 56, t ^ 56
    return f * 64 + t


YES, NO = 1.0, -1.0          # the two ends of the default wave


def targets(y: np.ndarray) -> np.ndarray:
    """Turn 0/1 ground truth into the activation's own two ends, -1 and +1."""
    return np.where(y > 0.5, YES, NO)


def mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error over the unmasked entries.

    The loss ``sinewave.py`` compiles with, on the value the sine activation
    already produced.  No second squashing function.
    """
    err = (pred - target) ** 2
    return float((err * mask).sum() / max(float(mask.sum()), 1.0))


def mse_grad(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
             pos_weight: np.ndarray | None = None) -> np.ndarray:
    """``d/dpred`` of that loss.

    ``pos_weight`` re-weights the ``+1`` answers per head.  Legal moves are
    about 3% of a uniformly drawn action set, and without the correction a head
    learns the prior - answer ``-1`` to everything - instead of the rule, which
    is the exact failure this module exists to remove.
    """
    g = 2.0 * (pred - target) * mask
    if pos_weight is not None:
        g = g * np.where(target > 0.0, pos_weight, 1.0)
    return g / max(float(mask.sum()), 1.0)


def head_report(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Per-head accuracy, plus the one number that says whether the rules are known.

    ``legal_auc`` is the probability that a random legal move outranks a random
    illegal one under the ``legal`` head.  It needs no threshold, it is not
    fooled by the 97:3 class imbalance the way accuracy is, and 0.5 is exactly
    "no idea" - so it is the honest version of "has it learned the rules".
    It reads the activated value directly, which is all a rank statistic needs.
    """
    out: dict[str, float] = {}
    for j, name in enumerate(RULE_HEADS):
        keep = mask[:, j] > 0
        if not keep.any():
            out[f"{name}_acc"] = float("nan")
            continue
        pred = scores[keep, j] > 0.0
        out[f"{name}_acc"] = float((pred == (target[keep, j] > 0.5)).mean())
    j = LEGAL - 1
    out["legal_auc"] = auc(scores[:, j], target[:, j] > 0.5)
    out["pseudo_auc"] = auc(scores[:, PSEUDO - 1], target[:, PSEUDO - 1] > 0.5)
    return out


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC; ``nan`` when one of the two classes is absent."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average the ranks of ties, or a head that has learned nothing scores 1.0
    # or 0.0 by accident of the sort order rather than 0.5.
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
