"""Self-play, and the corpora it produces.

A corpus here is a list of :class:`uge.codec.Play` objects plus the tapes they
encode to, split into train and test by **game**, never by position.  The
distinction matters more than usual: with a state abstraction on the tape, a
position seen in training is a node in the graph, so a test set that shares
positions with the training set measures recall and calls it generalisation.
:func:`split` therefore holds whole games out, and :func:`unseen_positions`
goes further and keeps only the test positions whose ``phi`` token the training
corpus never wrote.

The ply cap
-----------
``uge.chess_rules`` deliberately leaves threefold repetition out (it is a
function of the history, and ``winner(state)`` is a function of the state), so
some chess games here do not end, and Pig played at random banks about three
points a turn so it takes a long time to reach 100.  :data:`PLY_CAPS` stops
them.  A capped game keeps the outcome ``ongoing``, because that is what it is -
it is not labelled a draw, which would put a result on the tape that the rules
never gave.  The decoder then reports such a tape as ``ok`` (the rules never
refused) but not ``complete`` (it never reached a terminal state), and
:mod:`uge.metrics` never charges the cap to the network.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .codec import Play, encode, outcome_of, state_token
from .game import Game
from .tape import TapeSpec
from .teachers import Policy, RandomPolicy

__all__ = [
    "Corpus",
    "PLY_CAPS",
    "encode_plays",
    "generate_plays",
    "play_game",
    "positions",
    "self_play",
    "split",
    "unseen_positions",
]

PLY_CAPS = {"chess": 120, "othello": 80, "connect4": 42, "tictactoe": 9, "nim": 20, "pig": 400}
"""Where each game is adjudicated.

Only ``chess`` and ``pig`` ever reach theirs, and for opposite reasons: chess
without threefold repetition can shuffle for ever, and Pig played at random
banks about three points a turn so it takes a long time to reach 100.  The four
others are bounded by their own rules (``connect4`` cannot exceed 42 plies)."""


def play_game(
    game: Game,
    policies: tuple[Policy, Policy],
    rng: random.Random,
    max_plies: int | None = None,
) -> Play:
    """One game.  ``max_plies`` adjudicates a draw; ``None`` uses :data:`PLY_CAPS`."""
    cap = PLY_CAPS.get(game.name, 200) if max_plies is None else max_plies
    state = game.initial()
    play = Play(states=[state])
    while game.winner(state) is None and play.plies < cap:
        policy = policies[game.to_move(state)]
        action = policy.act(game, state, rng)
        state = game.apply(state, action)
        play.actions.append(action)
        play.states.append(state)
    play.outcome = outcome_of(game, state)
    return play  # an adjudicated game stays "ongoing": that is what it is


@dataclass(slots=True)
class Corpus:
    """Plays, their tapes, and enough provenance to reproduce them."""

    game: str
    spec: TapeSpec
    plays: list[Play] = field(default_factory=list)
    tapes: list[str] = field(default_factory=list)
    teacher: str = ""
    seed: int = 0

    def __len__(self) -> int:
        return len(self.tapes)

    def stats(self) -> dict:
        plies = [p.plies for p in self.plays]
        outcomes: dict[str, int] = {}
        for p in self.plays:
            outcomes[p.outcome] = outcomes.get(p.outcome, 0) + 1
        return {
            "games": len(self.plays),
            "plies_total": sum(plies),
            "plies_mean": sum(plies) / max(1, len(plies)),
            "chars": sum(len(t) for t in self.tapes),
            "outcomes": outcomes,
            "teacher": self.teacher,
            "seed": self.seed,
        }


def generate_plays(
    game: Game,
    n_games: int,
    teacher: Policy | None = None,
    seed: int = 0,
    opponent: Policy | None = None,
    max_plies: int | None = None,
) -> list[Play]:
    """``n_games`` games - the expensive half, and the half no ``TapeSpec`` changes.

    One RNG per game, seeded from ``seed`` and the game's place in the list, so
    a corpus of 100 games has the same first 50 as a corpus of 50: a
    corpus-size sweep is then a sweep over one variable instead of two.  It
    also means a sweep over *encodings* can generate the games once and
    re-encode them, which is what ``experiment.py`` does and what makes the
    ``phi`` and ``mix`` arms finish in minutes instead of hours.
    """
    teacher = teacher or RandomPolicy()
    opponent = opponent or teacher
    plays = []
    for i in range(n_games):
        rng = random.Random((seed * 1_000_003) ^ (i * 2_654_435_761))
        plays.append(play_game(game, (teacher, opponent), rng, max_plies))
    return plays


def encode_plays(game: Game, spec: TapeSpec, plays: list[Play], teacher: str = "", seed: int = 0) -> Corpus:
    """The cheap half: the same games, written down under a different spec."""
    return Corpus(
        game=game.name,
        spec=spec,
        plays=list(plays),
        tapes=[encode(game, spec, play) for play in plays],
        teacher=teacher,
        seed=seed,
    )


def self_play(
    game: Game,
    spec: TapeSpec,
    n_games: int,
    teacher: Policy | None = None,
    seed: int = 0,
    opponent: Policy | None = None,
    max_plies: int | None = None,
) -> Corpus:
    """``n_games`` games encoded under ``spec`` - :func:`generate_plays` then :func:`encode_plays`."""
    teacher = teacher or RandomPolicy()
    plays = generate_plays(game, n_games, teacher, seed, opponent, max_plies)
    return encode_plays(game, spec, plays, teacher.name, seed)


def split(corpus: Corpus, test_fraction: float = 0.2) -> tuple[Corpus, Corpus]:
    """Hold whole games out, deterministically (the last ``test_fraction`` of the list)."""
    n_test = max(1, int(round(len(corpus) * test_fraction)))
    cut = len(corpus) - n_test
    train = Corpus(corpus.game, corpus.spec, corpus.plays[:cut], corpus.tapes[:cut], corpus.teacher, corpus.seed)
    test = Corpus(corpus.game, corpus.spec, corpus.plays[cut:], corpus.tapes[cut:], corpus.teacher, corpus.seed)
    return train, test


def positions(game: Game, corpus: Corpus, limit: int | None = None, stride: int = 1) -> list[tuple[Play, int]]:
    """``(play, ply)`` pairs: every position in the corpus at which somebody moved."""
    out: list[tuple[Play, int]] = []
    for play in corpus.plays:
        for i in range(0, play.plies, stride):
            out.append((play, i))
            if limit is not None and len(out) >= limit:
                return out
    return out


def unseen_positions(
    game: Game,
    spec: TapeSpec,
    train: Corpus,
    test: Corpus,
    limit: int | None = None,
) -> list[tuple[Play, int]]:
    """Test positions whose ``phi`` token the training corpus never wrote.

    The only honest generalisation set when the state is on the tape - and,
    when ``phi`` is a *hash*, a set the model is guaranteed to score zero on:
    the last trigram of the prefix **is** the ``phi`` token, so an unseen token
    is an unseen node and the model has nothing to say.  That is not a bug in
    the measurement, it is limitation 2 of the README stated as a set: a hash
    has no locality, so there is no such thing as a *near* miss.

    The set is only interesting where ``phi`` is not a hash - a move summary
    (``phi_prev_bits``) takes few enough values that a new position usually
    lands on a token the corpus already wrote - and with ``phi_bits = 0`` there
    is no state token at all, so this returns every test position.
    """
    if spec.phi_bits == 0:
        return positions(game, test, limit)
    seen = {
        state_token(game, spec, play.states[i], play.winner, play.actions[i - 1] if i else None)
        for play in train.plays
        for i in range(play.plies)
    }
    out = []
    for play in test.plays:
        for i in range(play.plies):
            token = state_token(game, spec, play.states[i], play.winner, play.actions[i - 1] if i else None)
            if token not in seen:
                out.append((play, i))
                if limit is not None and len(out) >= limit:
                    return out
    return out
