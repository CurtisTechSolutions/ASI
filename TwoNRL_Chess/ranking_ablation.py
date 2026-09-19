"""Was the old Elo measuring the network, or the refusal walk?

This script exists because of a result that first looked like a regression.
Teaching the rules properly (``rules.py``) cut refusals from ~98 a move to ~1,
and the networks' rating against a random legal mover **fell**.  The obvious
reading - "learning the rules made it play worse" - is wrong, and this is the
experiment that shows what actually happened.

The trick is that the same saved network can be asked to play in two different
ways, with no retraining and not one weight changed:

``rule_weight = 0``
    rank on the quality head alone.  The network then needs about a hundred
    refusals to find a legal move, so the move it finally plays is whatever
    legality happens to allow a long way down an ordering that was never about
    legality.  That is close to a uniform draw over the legal moves, with a
    slight tilt toward the ones it liked.
``rule_weight = 1``
    rank on quality *plus* belief in legality.  Now it plays the move it
    actually wants, and finds it in a handful of tries.

So the comparison isolates one thing: whether the network's own policy is being
expressed, or averaged away by a walk down a ranking.

The answer, over 80 games a cell against a random legal mover:

    arm          ranked by        vs random   sigma    acpl   refusals
    positive     quality alone        0.631             772      108.7
    positive     + legal              0.463     2.2     710        4.2
    2nrl         quality alone        0.581             693      110.1
    2nrl         + legal              0.237     4.8     710        5.2
    2nrl-worst   quality alone        0.506             812      100.2
    2nrl-worst   + legal              0.356     2.0     811        4.4
    2nrl-random  quality alone        0.537             721      121.4
    2nrl-random  + legal              0.275     3.7     742        2.4
    repulsion    quality alone        0.619             535      571.1
    repulsion    + legal              0.388     3.1     693       18.8

    random       -                   0.481             774        0.0

Every arm falls, by 0.15 to 0.34, and every one of those falls is two to five
standard errors.  Read against the last row, the pattern is sharper than that:
ranked on quality alone every arm scores **above** a random legal mover, and
ranked with the legal head every arm scores **at or below** it.

The centipawn loss does not account for it.  It barely moves, it moves in both
directions, and it is not bad: three of the five policies give away *less* than
the random mover's 774 and none of them is far off.  So the network's own moves
are not worse by the grader, and its *games* are much worse.

That much is solid, and it is enough for the conclusion below.  One piece of it
is genuinely unexplained and is left that way rather than tidied: why the
refusal walk *beats* a random mover (0.63 against 0.48) when the two give away
the same 772 and 774 centipawns a move.  Some property of the distribution that
the walk produces is worth about 0.15 of score and is invisible to a
per-move centipawn average.  It is worth chasing and is not chased here.

What is not a mystery is the direction.  A player drawing near-uniformly from
the legal moves has no plan to be punished for; a player with a consistent bad
policy does.  Expressing a policy is only an improvement if the policy is good,
and at ~700 centipawns given away per move these are not.

Which means the earlier ratings - ``positive`` at 112 Elo, and every arm's
number before the rule heads existed - were not measuring what they appeared to.
A network needing 45 to 160 refusals a move was being randomised by its own
ignorance of the rules, and the rating was largely the rating of that
randomisation.  Learning the rules did not make these networks worse players.
It revealed that they were not playing.

The rules half of §6.5 is genuinely solved here.  The payoffs half is not, and
this is the measurement that stops the first from flattering the second.
"""

from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import glob
import json
import os

import chess
import numpy as np

from agent import Agent
from benchmark import RandomMover, openings, play
from engine import Judge, find_stockfish
from sbnn import SineNet


class WeightedArm:
    """One arm's networks, ranking at a chosen ``rule_weight``."""

    def __init__(self, arm: str, paths: list[str], weight: float) -> None:
        self.name = f"{arm} (rule_weight={weight:g})"
        self.agents = [Agent(SineNet.load(p), rule_weight=weight) for p in paths]
        self.index = 0
        self.refusals: list[int] = []

    def pick(self, board: chess.Board) -> chess.Move:
        out = self.agents[self.index].propose(board)
        self.refusals.append(out["n_refused"])
        return out["move"]


def run(weights: str, arms: list[str], rule_weights: list[float], n_openings: int,
        plies: int, seed: int, judge_depth: int, stockfish: str | None,
        baseline: bool = True) -> dict:
    judge = Judge(find_stockfish(stockfish), depth=judge_depth)
    books = openings(n_openings, plies=plies, seed=seed)
    out: dict[str, dict] = {}
    print(f"{'arm':<14}{'rule_weight':>12}{'vs random':>11}{'± se':>8}"
          f"{'acpl':>8}{'refusals':>10}{'games':>7}")
    for arm in arms:
        paths = sorted(glob.glob(os.path.join(weights, f"{arm}_seed*.npz")))
        if not paths:
            continue
        for weight in rule_weights:
            player = WeightedArm(arm, paths, weight)
            scores, acpls = [], []
            for i, fen in enumerate(books):
                for as_white in (True, False):
                    player.index = i % len(player.agents)
                    # A fresh opponent seeded per opening, so both cells of the
                    # comparison meet the identical random player.
                    rnd = RandomMover(seed=i)
                    a, b = (player, rnd) if as_white else (rnd, player)
                    r = play(a, b, fen, judge,
                             track=chess.WHITE if as_white else chess.BLACK)
                    scores.append(r["white_score"] if as_white else 1.0 - r["white_score"])
                    acpls.append(r["acpl"])
            s = np.array(scores)
            se = float(s.std() / max(np.sqrt(len(s)), 1))
            row = {"score": float(s.mean()), "stderr": se,
                   "acpl": float(np.mean(acpls)),
                   "refusals": float(np.mean(player.refusals)), "games": len(s)}
            out[player.name] = row
            print(f"{arm:<14}{weight:12.2f}{row['score']:11.3f}{se:8.3f}"
                  f"{row['acpl']:8.0f}{row['refusals']:10.1f}{row['games']:7d}")
    if baseline:
        # The anchor the whole table is read against: a random legal mover, put
        # through the identical games. Its score is a half by construction; its
        # centipawn loss is the number that matters, because it says what
        # picking a legal move at random costs - and therefore whether any of
        # these policies is worth having.
        scores, acpls = [], []
        for i, fen in enumerate(books):
            for as_white in (True, False):
                me, them = RandomMover(seed=1000 + i), RandomMover(seed=i)
                a, b = (me, them) if as_white else (them, me)
                r = play(a, b, fen, judge,
                         track=chess.WHITE if as_white else chess.BLACK)
                scores.append(r["white_score"] if as_white else 1.0 - r["white_score"])
                acpls.append(r["acpl"])
        s = np.array(scores)
        out["random legal mover"] = {
            "score": float(s.mean()), "stderr": float(s.std() / np.sqrt(len(s))),
            "acpl": float(np.mean(acpls)), "refusals": 0.0, "games": len(s)}
        r = out["random legal mover"]
        print(f"{'random':<14}{'-':>12}{r['score']:11.3f}{r['stderr']:8.3f}"
              f"{r['acpl']:8.0f}{0.0:10.1f}{r['games']:7d}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default="weights_rules")
    p.add_argument("--arms", nargs="+",
                   default=["positive", "2nrl", "2nrl-worst", "2nrl-random", "repulsion"])
    p.add_argument("--rule-weights", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--openings", type=int, default=40)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--opening-seed", type=int, default=99)
    p.add_argument("--judge-depth", type=int, default=6)
    p.add_argument("--stockfish", default=None)
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the random-legal-mover row the table is read against")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = run(args.weights, args.arms, args.rule_weights, args.openings,
               args.opening_plies, args.opening_seed, args.judge_depth, args.stockfish,
               baseline=not args.no_baseline)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"weights": args.weights, "openings": args.openings,
                       "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
