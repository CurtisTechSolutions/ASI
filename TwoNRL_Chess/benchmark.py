"""Benchmarking the trained arms: against each other, and against a common opponent.

``Research/2NRL.md`` §11.1 asks for matched compute, several seeds and reported
variance.  This is the measurement half of that.

Three readings, because no single one of them is honest on its own:

* **Head to head.**  The arm trained with 2NRL plays the arm trained without
  it, over the *same* openings with the colours swapped, so the pairing
  cancels the opening draw.  This is the comparison the question asks for and
  the only one where the two networks meet.
* **A common opponent.**  Both play the same skill-limited Stockfish from the
  same openings.  A head-to-head score can be a rock-paper-scissors artefact
  between two weak players; a common opponent cannot be.
* **A random mover.**  The floor - and a generous one, because it is handed the
  legal moves the networks had to learn.  Any arm that does not clearly beat it
  has not learned to play chess, whatever it scores against its sibling.

Refusals are counted here too.  A trained network still proposes into a closed
door sometimes, and a player that needs thirty attempts a move has not learned
the rules however well its games happen to go.

Every game is adjudicated the same way, by the same full-strength judge at the
same fixed depth, and no engine anywhere is given a time limit - only depths -
so the numbers do not move when the machine is busy.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import time

import chess
import numpy as np

from agent import Agent
from engine import Judge, Opponent, find_stockfish
from features import material
from moves import to_move
from sbnn import SineNet

RESIGN_CP = 1200
MAX_PLIES = 120


class RandomMover:
    """The floor: uniform over legal moves - and it is *given* them."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.refusals: list[int] = []

    def pick(self, board: chess.Board) -> chess.Move:
        self.refusals.append(0)
        return self.rng.choice(list(board.legal_moves))


class NetPlayer:
    """A trained SBNN, proposing greedily into the whole 4096-move action space."""

    def __init__(self, path: str, name: str | None = None) -> None:
        self.agent = Agent(SineNet.load(path))
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.path = path
        self.refusals: list[int] = []

    def pick(self, board: chess.Board) -> chess.Move:
        out = self.agent.propose(board)
        self.refusals.append(out["n_refused"])
        return out["move"]


class EnginePlayer:
    """Skill-limited Stockfish as an ordinary player."""

    def __init__(self, opponent: Opponent, name: str = "stockfish") -> None:
        self.opponent = opponent
        self.name = name
        self.refusals: list[int] = []

    def pick(self, board: chess.Board) -> chess.Move:
        self.refusals.append(0)
        return self.opponent.play(board)


def openings(n: int, plies: int, seed: int) -> list[str]:
    """``n`` opening positions, shared by every match so the pairing is exact."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            out.append(board.fen())
    return out


def play(white, black, fen: str, judge: Judge, track: int | None = None) -> dict:
    """One game from ``fen``.  ``track`` is the colour whose centipawn loss is measured."""
    board = chess.Board(fen)
    losses: list[float] = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        player = white if board.turn == chess.WHITE else black
        move = player.pick(board)
        if track is not None and board.turn == track:
            losses.append(judge.grade(board, move)["loss"])
        board.push(move)
        plies += 1
        if plies % 4 == 0:
            _, cp = judge.analyse(board)
            if abs(cp) > RESIGN_CP:                 # resign rather than play it out
                break
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        white_score = 1.0 if outcome.winner == chess.WHITE else (0.5 if outcome.winner is None else 0.0)
    else:
        _, cp = judge.analyse(board)
        cp = cp if board.turn == chess.WHITE else -cp        # to White's point of view
        white_score = 1.0 if cp > 150 else (0.0 if cp < -150 else 0.5)
    return {"white_score": white_score, "plies": plies,
            "acpl": float(np.mean(losses)) if losses else 0.0,
            "material": material(board, chess.WHITE)}


def match(a, b, fens: list[str], judge: Judge, track_acpl: bool = True) -> dict:
    """``a`` against ``b`` over every opening, both colours.  Scores are ``a``'s."""
    results, acpl, plies = [], [], []
    a.refusals = []
    for fen in fens:
        for a_is_white in (True, False):
            white, black = (a, b) if a_is_white else (b, a)
            track = (chess.WHITE if a_is_white else chess.BLACK) if track_acpl else None
            g = play(white, black, fen, judge, track)
            results.append(g["white_score"] if a_is_white else 1.0 - g["white_score"])
            acpl.append(g["acpl"])
            plies.append(g["plies"])
    r = np.array(results)
    n = len(r)
    se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"games": n, "score": float(r.mean()), "stderr": se, "ci95": 1.96 * se,
            "wins": int((r == 1.0).sum()), "draws": int((r == 0.5).sum()),
            "losses": int((r == 0.0).sum()),
            "acpl": float(np.mean(acpl)), "plies": float(np.mean(plies)),
            "refusals": float(np.mean(a.refusals)) if a.refusals else 0.0,
            "top1_legal": float(np.mean([x == 0 for x in a.refusals])) if a.refusals else 0.0,
            "elo": elo_of(float(r.mean()))}


def elo_of(score: float) -> float | None:
    """The usual conversion; ``None`` for a clean sweep, where it is unbounded."""
    if score <= 0.0 or score >= 1.0:
        return None
    return round(-400.0 * math.log10(1.0 / score - 1.0), 1)


def load_players(weights: str, arms: list[str], seeds: list[int]) -> dict[str, list[NetPlayer]]:
    players: dict[str, list[NetPlayer]] = {}
    for arm in arms:
        found = []
        for seed in seeds:
            path = os.path.join(weights, f"{arm}_seed{seed}.npz")
            if os.path.exists(path):
                found.append(NetPlayer(path, f"{arm}/s{seed}"))
        if not found:
            available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(weights, "*.npz")))
            raise SystemExit(f"no trained networks for arm '{arm}' in {weights}/ "
                             f"(found: {available or 'nothing'}) - run `make run` first")
        players[arm] = found
    return players


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default="weights")
    p.add_argument("--arms", nargs="+", default=["2nrl", "positive"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--openings", type=int, default=12)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--opening-seed", type=int, default=4242)
    p.add_argument("--opp-skill", type=int, default=0)
    p.add_argument("--opp-depth", type=int, default=4)
    p.add_argument("--judge-depth", type=int, default=6)
    p.add_argument("--pairings", choices=["matched", "all"], default="matched",
                   help="matched: seed i vs seed i;  all: every seed against every seed")
    p.add_argument("--stockfish", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = find_stockfish(args.stockfish)
    judge = Judge(path, depth=args.judge_depth)
    engine = EnginePlayer(Opponent(path, skill=args.opp_skill, depth=args.opp_depth))
    fens = openings(args.openings, args.opening_plies, args.opening_seed)
    players = load_players(args.weights, args.arms, args.seeds)
    started = time.time()
    report: dict = {"openings": len(fens), "opening_plies": args.opening_plies,
                    "opponent": engine.opponent.describe(), "judge_depth": args.judge_depth}

    # ---- head to head -----------------------------------------------------
    print(f"\n=== head to head ({len(fens)} openings, both colours) ===")
    head: dict = {}
    for left, right in itertools.combinations(args.arms, 2):
        pairs = (list(zip(players[left], players[right])) if args.pairings == "matched"
                 else list(itertools.product(players[left], players[right])))
        games = []
        for a, b in pairs:
            m = match(a, b, fens, judge, track_acpl=False)
            games.append(m)
            print(f"  {a.name:>16s} vs {b.name:<16s} "
                  f"{m['score']:.3f}  (+{m['wins']} ={m['draws']} -{m['losses']})")
        r = np.array([g["score"] for g in games])
        total = sum(g["games"] for g in games)
        se = float(np.sqrt(sum(g["stderr"] ** 2 * g["games"] ** 2 for g in games)) / total)
        head[f"{left} vs {right}"] = {
            "score": float(np.average(r, weights=[g["games"] for g in games])),
            "games": total, "stderr": se, "ci95": 1.96 * se,
            "wins": sum(g["wins"] for g in games), "draws": sum(g["draws"] for g in games),
            "losses": sum(g["losses"] for g in games), "pairings": len(games)}
        head[f"{left} vs {right}"]["elo"] = elo_of(head[f"{left} vs {right}"]["score"])
        h = head[f"{left} vs {right}"]
        print(f"  {left} vs {right}: score {h['score']:.3f} +- {h['ci95']:.3f}  "
              f"(+{h['wins']} ={h['draws']} -{h['losses']} over {h['games']} games)")
    report["head_to_head"] = head

    # ---- the common opponent, and the floor -------------------------------
    print(f"\n=== against the common opponent (Stockfish skill {args.opp_skill}, "
          f"depth {args.opp_depth}) and a random mover ===")
    common: dict = {}
    for arm in args.arms:
        per_seed = [match(pl, engine, fens, judge) for pl in players[arm]]
        floor = [match(pl, RandomMover(seed=17 + i), fens, judge)
                 for i, pl in enumerate(players[arm])]
        common[arm] = {
            "vs_stockfish": _pool(per_seed), "vs_random": _pool(floor),
            "per_seed": [{"name": pl.name, "score": m["score"], "acpl": m["acpl"]}
                         for pl, m in zip(players[arm], per_seed)]}
        s, f = common[arm]["vs_stockfish"], common[arm]["vs_random"]
        print(f"  {arm:<14s} vs stockfish {s['score']:.3f} +- {s['ci95']:.3f}  "
              f"acpl {s['acpl']:6.1f}  legal@1 {s['top1_legal']:.3f} "
              f"refusals {s['refusals']:5.1f}  |  vs random {f['score']:.3f} "
              f"+- {f['ci95']:.3f}")
    rnd = [match(RandomMover(seed=99 + i), engine, fens, judge) for i in range(len(args.seeds))]
    common["random"] = {"vs_stockfish": _pool(rnd)}
    print(f"  {'random':<14s} vs stockfish {common['random']['vs_stockfish']['score']:.3f}  "
          f"acpl {common['random']['vs_stockfish']['acpl']:6.1f}   (the floor)")
    report["common_opponent"] = common
    report["seconds"] = round(time.time() - started, 1)

    judge.close()
    engine.opponent.close()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")


def _pool(matches: list[dict]) -> dict:
    total = sum(m["games"] for m in matches)
    score = float(np.average([m["score"] for m in matches], weights=[m["games"] for m in matches]))
    se = float(np.sqrt(sum(m["stderr"] ** 2 * m["games"] ** 2 for m in matches)) / total)
    return {"score": score, "games": total, "stderr": se, "ci95": 1.96 * se,
            "wins": sum(m["wins"] for m in matches), "draws": sum(m["draws"] for m in matches),
            "losses": sum(m["losses"] for m in matches),
            "acpl": float(np.mean([m["acpl"] for m in matches])),
            "plies": float(np.mean([m["plies"] for m in matches])),
            "refusals": float(np.mean([m["refusals"] for m in matches])),
            "top1_legal": float(np.mean([m["top1_legal"] for m in matches])),
            "elo": elo_of(score)}


if __name__ == "__main__":
    main()
