"""The held-out evaluation set: positions with every legal move already graded.

Built once and cached, for two reasons.

**It is the same exam for every arm.**  ``Research/2NRL.md`` §10 is blunt that
2NRL has never been measured against a baseline, and §11 asks for matched
compute, multiple seeds and reported variance.  A fixed, shared, pre-graded set
of positions is what makes the comparison paired: every arm and every seed
answers the identical questions, so the difference between them is the method
and not the draw.

**It costs no engine calls at evaluation time.**  Each position stores a
centipawn score for *every* legal move (Stockfish at ``MultiPV = all``), so
scoring a network is a lookup.  Evaluating twelve runs twenty-five times each
would otherwise dominate the experiment.

Positions come from games between two Stockfish instances of deliberately
different strength, off random openings, so the set contains the ordinary
mistakes of weak play rather than only clean master positions.
"""

from __future__ import annotations

import argparse
import json
import os

import chess
import chess.engine

from engine import MATE_SCORE, find_stockfish

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heldout.json")


def _rank_all(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> dict[str, int]:
    """Every legal move's score in centipawns, from the side to move's view."""
    moves = list(board.legal_moves)
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=len(moves))
    scores: dict[str, int] = {}
    for info in infos:
        pv = info.get("pv") or []
        if pv:
            scores[board.uci(pv[0])] = int(info["score"].pov(board.turn).score(mate_score=MATE_SCORE))
    return scores


def build(path: str, n_positions: int, depth: int, stockfish: str, seed: int = 12345,
          opening_plies: int = 6, skills: tuple[int, int] = (0, 5),
          min_moves: int = 6, verbose: bool = True) -> dict:
    import random

    rng = random.Random(seed)
    players = [chess.engine.SimpleEngine.popen_uci(stockfish) for _ in skills]
    for engine, skill in zip(players, skills):
        engine.configure({"Threads": 1, "Hash": 16, "Skill Level": skill})
    grader = chess.engine.SimpleEngine.popen_uci(stockfish)
    grader.configure({"Threads": 1, "Hash": 64})

    positions: list[dict] = []
    seen: set[str] = set()
    game = 0
    while len(positions) < n_positions:
        board = chess.Board()
        for _ in range(opening_plies):                    # a random, legal opening
            board.push(rng.choice(list(board.legal_moves)))
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < 120:
            if board.legal_moves.count() >= min_moves and board.epd() not in seen:
                seen.add(board.epd())
                scores = _rank_all(grader, board, depth)
                if len(scores) == board.legal_moves.count():
                    positions.append({"fen": board.fen(), "scores": scores})
                    if verbose and len(positions) % 50 == 0:
                        print(f"  {len(positions)}/{n_positions} positions")
                    if len(positions) >= n_positions:
                        break
            board.push(players[ply % 2].play(board, chess.engine.Limit(depth=rng.choice([2, 3, 4]))).move)
            ply += 1
        game += 1
    for engine in players:
        engine.quit()
    grader.quit()

    blob = {"depth": depth, "seed": seed, "skills": list(skills), "games": game,
            "positions": positions}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(blob, fh)
    if verbose:
        print(f"wrote {len(positions)} graded positions to {path}")
    return blob


def load(path: str = DEFAULT_PATH) -> list[dict]:
    """``[{board, moves, cp, best_cp, best_index}]`` - decoded and ready to score."""
    with open(path) as fh:
        blob = json.load(fh)
    out = []
    for item in blob["positions"]:
        board = chess.Board(item["fen"])
        moves = list(board.legal_moves)
        cp = [item["scores"][board.uci(m)] for m in moves]
        best = max(cp)
        out.append({"board": board, "moves": moves, "cp": cp, "best_cp": best,
                    "best_moves": {board.uci(m) for m, c in zip(moves, cp) if c == best}})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--positions", type=int, default=400)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--stockfish", default=None)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    build(args.out, args.positions, args.depth, find_stockfish(args.stockfish), args.seed)


if __name__ == "__main__":
    main()
