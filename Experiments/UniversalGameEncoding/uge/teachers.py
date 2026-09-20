"""Where the corpus comes from: seeded policies that play the games.

``TwoNRL_Chess/`` has Stockfish for this.  This directory has no dependencies,
so the teacher is a depth-limited search over the game's own
:meth:`uge.game.Game.heuristic` - weak, but that is not a problem here.  The
experiment is about what the *encoding* carries, and a corpus of weak play
contains the rules exactly as completely as a corpus of strong play does.
Where play quality matters (``nim``) the game is solved and the exact answer is
available for free; where it cannot be had (``chess``) the README says so
rather than dressing a depth-2 search up as an opponent.

Minimax, not negamax, because two of the six games do not alternate turns the
way negamax assumes: ``pig`` interrupts a player's turn with the die, and
``othello`` hands the turn straight back on a pass.  Evaluating everything from
**player 0's** point of view costs one sign flip at the root and never gets
that wrong.
"""

from __future__ import annotations

import random
from typing import Any

from .game import Game

__all__ = ["Policy", "RandomPolicy", "SearchPolicy", "make_policy", "search_value"]

_WIN = 1e6
_INF = float("inf")


def search_value(game: Game, state: Any, depth: int, alpha: float = -_INF, beta: float = _INF) -> float:
    """Minimax value from **player 0's** point of view, alpha-beta pruned.

    A chance node is an expectation over its outcomes, which is the only thing
    :mod:`uge.games`'s ``pig`` needs and it needs no other special case.  Wins
    are scored ``_WIN + depth`` so that a mate in one beats a mate in three:
    without it the search is indifferent between winning now and winning later,
    and the corpus fills up with games that refuse to finish.
    """
    winner = game.winner(state)
    if winner is not None:
        return 0.0 if winner < 0 else (_WIN + depth if winner == 0 else -(_WIN + depth))
    if depth <= 0:
        return game.heuristic(state)
    actions = game.legal(state)
    if not actions:
        return game.heuristic(state)
    if game.is_chance(state):
        weights = game.chance_weights(state)
        return sum(
            p * search_value(game, game.apply(state, a), depth - 1)
            for a, p in zip(actions, weights)
            if p
        )
    if game.to_move(state) == 0:
        best = -_INF
        for a in actions:
            best = max(best, search_value(game, game.apply(state, a), depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    best = _INF
    for a in actions:
        best = min(best, search_value(game, game.apply(state, a), depth - 1, alpha, beta))
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


class Policy:
    """Something that picks an action.  A chance node is sampled, never chosen."""

    name = "policy"

    def act(self, game: Game, state: Any, rng: random.Random) -> Any:
        raise NotImplementedError

    @staticmethod
    def sample_chance(game: Game, state: Any, rng: random.Random) -> Any:
        actions = game.legal(state)
        return rng.choices(actions, weights=game.chance_weights(state), k=1)[0]


class RandomPolicy(Policy):
    """Uniform over the legal actions.  The floor every other number is measured against."""

    name = "random"

    def act(self, game: Game, state: Any, rng: random.Random) -> Any:
        if game.is_chance(state):
            return self.sample_chance(game, state, rng)
        return rng.choice(game.legal(state))


class SearchPolicy(Policy):
    """Depth-limited minimax over the game's heuristic, with an epsilon of noise.

    ``epsilon`` is not a weakness, it is the corpus's variety: a fully
    deterministic teacher plays one game per seed pair and a thousand games
    then teach the network one line.  At ``epsilon = 0.15`` the openings differ
    while the play stays recognisable.
    """

    def __init__(self, depth: int = 2, epsilon: float = 0.15) -> None:
        self.depth = int(depth)
        self.epsilon = float(epsilon)
        self.name = f"search{self.depth}"

    def act(self, game: Game, state: Any, rng: random.Random) -> Any:
        if game.is_chance(state):
            return self.sample_chance(game, state, rng)
        actions = game.legal(state)
        if len(actions) == 1:
            return actions[0]
        if self.epsilon and rng.random() < self.epsilon:
            return rng.choice(actions)
        maximizing = game.to_move(state) == 0
        scored = [(search_value(game, game.apply(state, a), self.depth - 1), a) for a in actions]
        best = max(s for s, _ in scored) if maximizing else min(s for s, _ in scored)
        return rng.choice([a for s, a in scored if s == best])


def make_policy(spec: str) -> Policy:
    """``"random"`` or ``"searchN"`` / ``"searchN:eps"``; the string goes straight into the results file."""
    if spec == "random":
        return RandomPolicy()
    if spec.startswith("search"):
        body = spec[len("search") :]
        depth, _, eps = body.partition(":")
        return SearchPolicy(int(depth or 2), float(eps) if eps else 0.15)
    raise ValueError(f"unknown policy {spec!r}; expected 'random' or 'searchN[:eps]'")
