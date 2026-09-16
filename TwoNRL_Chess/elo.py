"""Rate the trained networks on a ladder, because a score against Stockfish is 0.000.

Stockfish beats every one of these networks every game, at any setting it has -
its own weakest is ``UCI_Elo 1320``, which is a club player. A rating cannot be
computed from a clean sweep: the likelihood is maximised at minus infinity. So
asking "what is this network's Elo" needs rungs it can actually reach, and this
builds them.

The ladder
----------
Four reference players, none of which is a neural network, spanning the range
from "knows only the rules" to "is a chess engine":

===================  =========================================================
``random``           uniform over the legal moves.  **The anchor, defined as 0**
``capture-greedy``   takes the most valuable piece it can, else plays at random
``material-1ply``    maximises its own material after its move
``material-2ply``    the midrange over the opponent's replies (measured at 128.7
                     centipawns a move against random's 289.0)
``sf-skill0-d1``     Stockfish, Skill Level 0, one ply of search
``sf-elo1320``       Stockfish at ``UCI_Elo 1320``, its own floor
===================  =========================================================

Every player - the networks included - plays everyone else over shared openings
with the colours swapped, and the ratings are the maximum-likelihood fit to the
whole cross-table under the usual logistic model, with a weak prior so that an
undefeated player stays finite. Confidence intervals are bootstrapped over
games.

What the number means, and does not
-----------------------------------
It is **Elo relative to a random legal mover**, not FIDE Elo. ``sf-elo1320``'s
fitted rating is on the same scale, so it says how much of the distance from
random to a club player each network has covered - and that is the honest way
to report a rating for a player this weak.

One thing has to be said plainly. In these games the board is asked for the
legal moves *on the network's behalf*: it proposes, and the harness walks down
its ranking until something is accepted. A network that needs 133 attempts to
find a legal move is not penalised for it here, because chess arbiters forfeit
an illegal move and that would end every game on move one. So this rating
measures **which legal move a network settles on**, and deliberately not
**whether it knows the rules** - which is what ``refusals per move`` measures,
and where these arms differ most.
"""

from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import os
import random
import time

import chess
import numpy as np

from agent import Agent
from benchmark import EnginePlayer, NetPlayer, RandomMover, openings, play
from engine import Judge, Opponent, find_stockfish
from sbnn import SineNet

VALUE = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


class MaterialPlayer:
    """Counts material, and nothing else.  ``plies`` 1 or 2.

    The 2-ply version uses the midrange over the opponent's replies, the same
    sign-symmetric aggregator ``agent.py`` argues for - here only because it is
    a reasonable weak opponent, not because anything is being inverted.
    """

    def __init__(self, plies: int = 1, seed: int = 0) -> None:
        self.plies = plies
        self.name = f"material-{plies}ply"
        self.rng = random.Random(seed)
        self.refusals: list[int] = []

    @staticmethod
    def _material(board: chess.Board, pov: chess.Color) -> float:
        total = 0
        for piece in board.piece_map().values():
            total += VALUE[piece.piece_type] * (1 if piece.color == pov else -1)
        return float(total)

    def pick(self, board: chess.Board) -> chess.Move:
        self.refusals.append(0)
        pov = board.turn
        best, best_score = None, -1e18
        for move in board.legal_moves:
            board.push(move)
            if self.plies == 1 or board.is_game_over(claim_draw=False):
                score = 1e5 if board.is_checkmate() else self._material(board, pov)
            else:
                vals = []
                for reply in board.legal_moves:
                    board.push(reply)
                    vals.append(self._material(board, pov))
                    board.pop()
                score = 0.5 * (min(vals) + max(vals)) if vals else (
                    1e5 if board.is_checkmate() else 0.0)
            board.pop()
            if score > best_score:
                best, best_score = move, score
        return best


class CaptureGreedy:
    """Takes the most valuable piece on offer, and otherwise moves at random."""

    name = "capture-greedy"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.refusals: list[int] = []

    def pick(self, board: chess.Board) -> chess.Move:
        self.refusals.append(0)
        best, best_value = None, 0
        for move in board.legal_moves:
            victim = board.piece_type_at(move.to_square)
            value = VALUE[victim] if victim else 0
            if board.is_en_passant(move):
                value = VALUE[chess.PAWN]
            if value > best_value:
                best, best_value = move, value
        return best or self.rng.choice(list(board.legal_moves))


class ArmPlayer:
    """One arm, playing with a different seed's network each game."""

    def __init__(self, arm: str, paths: list[str]) -> None:
        self.name = arm
        self.nets = [SineNet.load(p) for p in paths]
        self.agents = [Agent(n) for n in self.nets]
        self.index = 0
        self.refusals: list[int] = []

    def new_game(self, game_index: int) -> None:
        self.index = game_index % len(self.agents)

    def pick(self, board: chess.Board) -> chess.Move:
        out = self.agents[self.index].propose(board)
        self.refusals.append(out["n_refused"])
        return out["move"]


# ------------------------------------------------------------------ the fit


def fit_elo(games: list[tuple[int, int, float]], n: int, anchor: int = 0,
            prior: float = 2.0, tol: float = 1e-4, max_iters: int = 2000) -> np.ndarray:
    """Maximum-likelihood Elo under the logistic model, anchored at ``anchor``.

    Newton steps on the diagonal of the Hessian, iterated to convergence rather
    than for a fixed count.  Plain gradient ascent needs thousands of passes to
    settle when the field spans a thousand Elo, and a fixed budget that is
    generous for the point estimate but not for the bootstrap produces intervals
    that do not contain it - which is exactly the bug this replaces.

    ``prior`` adds that many virtual draws against a 0-rated phantom for every
    player, which keeps an undefeated or winless player's rating finite instead
    of running off to infinity.
    """
    if not games:
        return np.zeros(n)
    i = np.array([g[0] for g in games])
    j = np.array([g[1] for g in games])
    s = np.array([g[2] for g in games], dtype=float)
    r = np.zeros(n)
    scale = np.log(10.0) / 400.0
    for _ in range(max_iters):
        p = 1.0 / (1.0 + np.power(10.0, -(r[i] - r[j]) / 400.0))
        resid, curve = s - p, p * (1.0 - p)
        grad, hess = np.zeros(n), np.zeros(n)
        np.add.at(grad, i, resid)
        np.add.at(grad, j, -resid)
        np.add.at(hess, i, curve)
        np.add.at(hess, j, curve)
        pp = 1.0 / (1.0 + np.power(10.0, -r / 400.0))
        grad += prior * (0.5 - pp)
        hess += prior * pp * (1.0 - pp)
        step = (grad / scale) / np.maximum(hess, 1e-9)
        step = np.clip(step, -400.0, 400.0)          # no wild first move
        r += step
        r -= r[anchor]
        if np.abs(step).max() < tol:
            break
    return r


def bootstrap_elo(games, n, anchor=0, rounds=200, seed=7) -> np.ndarray:
    """Resample games with replacement and refit - to the same tolerance."""
    rng = np.random.default_rng(seed)
    out = np.zeros((rounds, n))
    idx = np.arange(len(games))
    for b in range(rounds):
        pick = rng.choice(idx, size=len(games), replace=True)
        out[b] = fit_elo([games[k] for k in pick], n, anchor)
    return out


# ------------------------------------------------------------------- run it


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default="weights")
    p.add_argument("--arms", nargs="+",
                   default=["2nrl", "positive", "2nrl-worst", "2nrl-random", "repulsion"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--openings", type=int, default=5)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--opening-seed", type=int, default=90210)
    p.add_argument("--judge-depth", type=int, default=6)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--stockfish", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = find_stockfish(args.stockfish)
    judge = Judge(path, depth=args.judge_depth)

    players: list = [RandomMover(seed=5)]
    players[0].name = "random"
    players.append(CaptureGreedy(seed=6))
    players.append(MaterialPlayer(1, seed=7))
    players.append(MaterialPlayer(2, seed=8))
    weak = Opponent(path, skill=0, depth=1)
    players.append(EnginePlayer(weak, name="sf-skill0-d1"))
    floor = Opponent(path, skill=20, depth=6, elo=1320)
    players.append(EnginePlayer(floor, name="sf-elo1320"))

    for arm in args.arms:
        paths = [os.path.join(args.weights, f"{arm}_seed{s}.npz") for s in args.seeds]
        paths = [q for q in paths if os.path.exists(q)]
        if paths:
            players.append(ArmPlayer(arm, paths))
        else:
            print(f"  (no trained networks for {arm}, skipping)")

    names = [pl.name for pl in players]
    anchor = names.index("random")
    fens = openings(args.openings, args.opening_plies, args.opening_seed)
    print(f"=== ladder: {len(names)} players, {len(fens)} openings, both colours ===")
    print("    " + ", ".join(names))

    started = time.time()
    games: list[tuple[int, int, float]] = []
    cross = np.zeros((len(names), len(names)))
    counts = np.zeros((len(names), len(names)))
    for a, b in itertools.combinations(range(len(names)), 2):
        pa, pb = players[a], players[b]
        for gi, fen in enumerate(fens):
            for a_white in (True, False):
                for pl in (pa, pb):
                    if hasattr(pl, "new_game"):
                        pl.new_game(gi * 2 + int(a_white))
                white, black = (pa, pb) if a_white else (pb, pa)
                g = play(white, black, fen, judge, None)
                score_a = g["white_score"] if a_white else 1.0 - g["white_score"]
                games.append((a, b, score_a))
                cross[a, b] += score_a
                cross[b, a] += 1.0 - score_a
                counts[a, b] += 1
                counts[b, a] += 1
        print(f"  {names[a]:>14s} vs {names[b]:<14s} "
              f"{cross[a, b] / max(counts[a, b], 1):.3f}")

    ratings = fit_elo(games, len(names), anchor)
    boot = bootstrap_elo(games, len(names), anchor, rounds=args.bootstrap)
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    order = np.argsort(-ratings)
    print(f"\n=== Elo, relative to a random legal mover at 0 "
          f"({len(games)} games, {time.time() - started:.0f}s) ===\n")
    print(f"{'player':<16}{'Elo':>8}{'95% CI':>18}{'score':>9}{'games':>7}{'refusals':>10}")
    rows = []
    for k in order:
        played = counts[k].sum()
        scored = cross[k].sum()
        pl = players[k]
        ref = (float(np.mean(pl.refusals)) if getattr(pl, "refusals", None) else 0.0)
        print(f"{names[k]:<16}{ratings[k]:8.0f}"
              f"{f'[{lo[k]:.0f}, {hi[k]:.0f}]':>18}"
              f"{scored / max(played, 1):9.3f}{int(played):7d}{ref:10.1f}")
        rows.append({"player": names[k], "elo": round(float(ratings[k]), 1),
                     "ci95": [round(float(lo[k]), 1), round(float(hi[k]), 1)],
                     "score": round(float(scored / max(played, 1)), 4),
                     "games": int(played), "refusals": round(ref, 2)})

    judge.close()
    weak.close()
    floor.close()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"players": names, "ratings": rows,
                       "anchor": "random", "games": len(games),
                       "cross_table": cross.tolist(), "counts": counts.tolist(),
                       "openings": len(fens),
                       # the raw results, so ratings can be refitted without
                       # replaying five hundred games
                       "results": [[int(a), int(b), float(sc)] for a, b, sc in games]},
                      fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
