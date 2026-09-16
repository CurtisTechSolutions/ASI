"""Stockfish in its two roles: the opponent that plays, and the judge that grades.

``Research/2NRL.md`` §6.5 says the helper's job is to write the rules down
before play, so that failure is *rule-shaped* rather than random - §5's
low-entropy condition is the whole mechanism.  Chess hands us both halves for
free, and that is why it is a good test bed for the claim:

* **The rules are stated up front.** ``python-chess`` supplies the legal moves,
  so the network never fails by playing an illegal one.  Every failure it can
  make is a *judgement* failure, which is exactly the kind §5 wants.
* **The judge is not an opinion.** Stockfish grades a move in centipawns
  against its own best line, the same way every time, at a fixed depth.  It is
  the ground-truth reward of §9 item 4 rather than somebody's mark out of ten.

Two processes, because the two roles want different strength.  The **judge**
runs at full strength so the grade is stable; the **opponent** is skill-limited
so the games are not over in ten moves.  Confusing the two would grade the
network against a deliberately weakened engine, and the labels would be noise.
"""

from __future__ import annotations

import chess
import chess.engine

MATE_SCORE = 10_000        # what a forced mate is worth when scores become ints
DEFAULT_PATHS = ("/usr/games/stockfish", "/usr/bin/stockfish", "stockfish")


def find_stockfish(path: str | None = None) -> str:
    """The first Stockfish that actually starts, or a clear error."""
    import shutil
    candidates = [path] if path else list(DEFAULT_PATHS)
    for candidate in candidates:
        if candidate and (shutil.which(candidate) or candidate.startswith("/")):
            resolved = shutil.which(candidate) or candidate
            try:
                engine = chess.engine.SimpleEngine.popen_uci(resolved)
                engine.quit()
                return resolved
            except Exception:
                continue
    raise RuntimeError(
        "no Stockfish binary found - `make install` (apt-get install stockfish) "
        f"or pass --stockfish PATH.  Tried: {candidates}")


class Judge:
    """Full-strength Stockfish, used only to grade - never to play.

    ``best`` and ``loss`` are the two things the experiment needs, and both are
    in centipawns from the point of view of the side to move.
    """

    def __init__(self, path: str, depth: int = 6, threads: int = 1, hash_mb: int = 32,
                 cache_size: int = 200_000) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.depth = depth
        self.limit = chess.engine.Limit(depth=depth)
        self._cache: dict[str, tuple[str | None, int]] = {}
        self.cache_size = cache_size
        self.calls = 0
        self.hits = 0

    def _key(self, board: chess.Board) -> str:
        return board._transposition_key().__str__() if False else board.epd()

    def analyse(self, board: chess.Board) -> tuple[chess.Move | None, int]:
        """``(best move, score)`` in centipawns for the side to move.

        A finished position has no best move and a score of 0 for a draw or
        ``-MATE_SCORE`` for being mated - the side to move has lost.
        """
        if board.is_game_over(claim_draw=False):
            outcome = board.outcome(claim_draw=False)
            return None, (0 if outcome is None or outcome.winner is None else -MATE_SCORE)
        key = self._key(board)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return (chess.Move.from_uci(cached[0]) if cached[0] else None), cached[1]
        info = self.engine.analyse(board, self.limit)
        self.calls += 1
        pv = info.get("pv") or []
        move = pv[0] if pv else None
        score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
        if len(self._cache) < self.cache_size:
            self._cache[key] = (move.uci() if move else None, int(score))
        return move, int(score)

    def score_after(self, board: chess.Board, move: chess.Move) -> int:
        """What ``move`` is worth, in centipawns, to the side that plays it."""
        board.push(move)
        try:
            _, score = self.analyse(board)
            return -score                 # the child's score is the opponent's
        finally:
            board.pop()

    def grade(self, board: chess.Board, move: chess.Move) -> dict:
        """One graded move: the best alternative, and how much ``move`` gave away.

        ``loss`` is centipawns lost against Stockfish's own choice and is the
        grade §6.1 weights by - the worse the failure, the harder the negative
        phase drives the network to reproduce it.  It is clipped at zero
        because a shallow search can rate a move above its own principal
        variation, which is search noise and not a gain.
        """
        best, best_score = self.analyse(board)
        same = best is not None and board.uci(move) == board.uci(best)
        played = best_score if same else self.score_after(board, move)
        return {"best": best, "best_score": best_score, "played_score": played,
                "loss": max(0, best_score - played)}

    def rank(self, board: chess.Board) -> dict[str, int]:
        """Every legal move scored in one search (``MultiPV = all``).

        This single call is what keeps the arms compute-matched.  From the one
        ranking come all four negative sets of ``Research/2NRL.md`` §11 and the
        positive target: the **best** move (the correction), the **worst** move
        (the low-entropy failure), the network's **own** move and its
        centipawn loss (the medium-entropy failure), and a **random** legal
        move (the high-entropy control).  No arm gets an engine call the
        others do not.
        """
        moves = list(board.legal_moves)
        if not moves:
            return {}
        key = "mpv:" + self._key(board)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        infos = self.engine.analyse(board, self.limit, multipv=len(moves))
        self.calls += 1
        scores: dict[str, int] = {}
        for info in infos:
            pv = info.get("pv") or []
            if pv:
                scores[board.uci(pv[0])] = int(info["score"].pov(board.turn).score(mate_score=MATE_SCORE))
        if len(scores) < len(moves):
            # A shallow MultiPV search can drop a line; fill the gaps so every
            # legal move has a score and the ranking is total.
            floor = min(scores.values()) if scores else 0
            for m in moves:
                scores.setdefault(board.uci(m), floor)
        if len(self._cache) < self.cache_size:
            self._cache[key] = scores
        return scores

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass


class Opponent:
    """Skill-limited Stockfish, used only to play - never to grade."""

    def __init__(self, path: str, skill: int = 0, depth: int = 4,
                 elo: int | None = None, threads: int = 1, hash_mb: int = 16) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        options: dict = {"Threads": threads, "Hash": hash_mb, "Skill Level": skill}
        if elo is not None:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.engine.configure(options)
        self.limit = chess.engine.Limit(depth=depth)
        self.skill, self.depth, self.elo = skill, depth, elo

    def play(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def describe(self) -> dict:
        return {"skill": self.skill, "depth": self.depth, "elo": self.elo}

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass
