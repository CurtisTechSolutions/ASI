"""Universal game encoding on RadixCyclicNN - the six arms, and the numbers they produce.

Run:  python3 experiment.py <arm> [--quick] [--seeds 0 1 2] ...
      python3 experiment.py all --quick
      python3 experiment.py summary

===========  =====================================================================
arm          the question it answers
===========  =====================================================================
``rules``    can the network learn a game's **rules** from a tape - for six games,
             one encoder, no per-game code?
``phi``      how much of the position should the tape carry?  The sweep from a
             move-only tape to an injective one, including the cliff where the
             abstraction stops fitting in a single token.
``mix``      the same budget split between a hash of the position and a summary
             of the last move - because a hash has no locality and a move
             summary does.
``phase``    does the phase ambiguity of a stride-1 window over a token grid
             matter, and does the disjoint codebook fix it?
``layout``   does the *arrangement of digits inside a token* change what the
             network can learn?  (Chess, where ``(from, to, promo)`` is an
             option.)
``multi``    can one network hold all six games at once, with the header token as
             the selector?
``twonrl``   does 2NRL - train on the moves the board refused, invert, fine-tune -
             beat the same compute spent on correct moves?
===========  =====================================================================

Every arm writes ``results/<arm>.json``; ``summary`` turns those into the tables
in ``README.md``.  Nothing here needs anything installed beyond the standard
library and ``RadixCyclicNN/``.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uge
from uge import metrics
from uge.codec import Play, encode, encode_prefix
from uge.corpus import encode_plays, generate_plays, positions, split, unseen_positions
from uge.net import TRAIN_DEFAULTS, new_net
from uge.tape import DISJOINT, PLAIN, TapeSpec, phase_ambiguity
from uge.teachers import make_policy

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
"""Where an arm writes its JSON.  ``--out-dir`` moves it, and ``make quick``
uses that to write into ``results-quick/`` - a quick run is a check that the
code works, and it must not overwrite the three-seed numbers README.md quotes."""

GAMES = ["chess", "othello", "connect4", "tictactoe", "nim", "pig"]

TEACHERS = {
    "chess": "search1",
    "othello": "search2",
    "connect4": "search2",
    "tictactoe": "search3",
    "nim": "search2",
    "pig": "search2",
}
"""Depth per game, chosen so a corpus takes seconds rather than hours.

Chess gets depth 1 for the same reason ``TwoNRL_Chess`` gets Stockfish: the
search is the expensive part, and a *rules* corpus does not need a strong
opponent - a corpus of weak play contains the legal moves exactly as completely
as a corpus of strong play.  Where quality is the question (``nim``, which is
solved) the exact answer is available without a search at all."""

FULL = {"games": 200, "epochs": 8, "eval": 300, "rollouts": 3, "chess_games": 120}
QUICK = {"games": 60, "epochs": 4, "eval": 80, "rollouts": 1, "chess_games": 30}


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def default_spec(game_name: str, **overrides) -> TapeSpec:
    """The configuration every arm starts from unless it is the thing being varied.

    ``phi_bits`` defaults to the largest abstraction that still fits in **one**
    token once the game's actions have taken their share of the code space -
    14 bits for chess, 15 for everything else under ``disjoint``.  A second
    token would be tape the model pays for and never reads (:mod:`uge.tape`),
    so the default never spends one by accident; ``experiment.py phi`` asks for
    it deliberately.
    """
    game = uge.get_game(game_name)
    kwargs = {"layout": "lsb", "book": "disjoint"}
    kwargs.update(overrides)
    if "phi_bits" not in kwargs and not kwargs.get("phi_exact"):
        probe = TapeSpec(game_name, game.index, phi_bits=0, **kwargs)
        kwargs["phi_bits"] = probe.codebook.phi_bits_for(game.token_bound(probe))
    return TapeSpec(game_name, game.index, **kwargs)


_PLAYS: dict[tuple, list] = {}


def build(game_name: str, spec: TapeSpec, cfg: dict, seed: int):
    """Corpus, split, and the positions the arms evaluate on.

    The games themselves are cached on ``(game, count, teacher, seed)`` and
    re-encoded per spec, because a :class:`~uge.tape.TapeSpec` changes how a
    play is *written down* and not how it was *played*.  Without this the
    ``phi`` and ``mix`` arms would spend nearly all of their time replaying
    identical Othello games, and every arm would be comparing encodings across
    slightly different corpora as well.
    """
    game = uge.get_game(game_name)
    n = cfg["chess_games"] if game_name == "chess" else cfg["games"]
    teacher = TEACHERS[game_name]
    key = (game_name, n, teacher, seed)
    plays = _PLAYS.get(key)
    if plays is None:
        plays = generate_plays(game, n, make_policy(teacher), seed=seed)
        _PLAYS[key] = plays
    corpus = encode_plays(game, spec, plays, teacher, seed)
    train, test = split(corpus, 0.2)
    return game, corpus, train, test


def train_model(tapes: list[str], seed: int, epochs: int, **overrides):
    model = new_net(seed=seed)
    settings = dict(TRAIN_DEFAULTS, epochs=epochs)
    settings.update(overrides)
    model.train(tapes, **settings)
    return model


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> float:
    """Half-width of the 95 % interval of the mean; 0 for a single seed, and it says so."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(var / len(values))


def pool(records: list[dict], keys: list[str]) -> dict:
    """Mean and interval per key across seeds."""
    out = {"seeds": len(records)}
    for key in keys:
        values = [r[key] for r in records if key in r]
        if values:
            out[key] = mean(values)
            out[key + "_ci"] = ci95(values)
    return out


REPORT = [
    "coverage",
    "legal_at_1",
    "legal_at_1_covered",
    "legal_at_1_baseline",
    "legal_at_1_fallback",
    "legal_at_3",
    "refusals",
    "refusals_baseline",
    "teacher_match",
    "optimal_match",
    "greedy_plies",
    "greedy_terminal",
    "selfplay_plies",
    "selfplay_terminal",
    "selfplay_refusals",
    "free_plies",
    "free_best_plies",
    "ppl_nats_per_token",
    "ppl_unknown",
    "nodes",
    "edges",
    "trigrams",
    "compression_ratio",
]


def write(name: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def log(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# arm: rules
# ---------------------------------------------------------------------------


def arm_rules(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """Six games, one encoder: how far above guessing does the tape get the network?

    The headline number is ``refusals`` against ``refusals_baseline`` - how many
    moves the board refuses before accepting one, against what a uniform guess
    over the whole action space costs.  ``Research/2NRL.md`` §6.5's metric,
    computed here for six games instead of one, and with no engine involved.

    ``unseen`` repeats the whole battery on the test positions whose ``phi``
    token the training corpus never wrote, which is the only honest
    generalisation question once the state is on the tape.
    """
    out: dict = {"arm": "rules", "config": cfg, "seeds": seeds, "games": {}}
    for name in games:
        per_seed, per_seed_unseen, corpus_stats = [], [], None
        for seed in seeds:
            spec = default_spec(name)
            game, corpus, train, test = build(name, spec, cfg, seed)
            corpus_stats = corpus.stats()
            model = train_model(train.tapes, seed, cfg["epochs"])
            seen = positions(game, test, limit=cfg["eval"])
            per_seed.append(
                metrics.evaluate(model, game, spec, seen, test.tapes, rollouts=cfg["rollouts"])
            )
            fresh = unseen_positions(game, spec, train, test, limit=cfg["eval"])
            if fresh:
                per_seed_unseen.append(
                    metrics.evaluate(model, game, spec, fresh, test.tapes, rollouts=1)
                )
            log(f"  rules/{name} seed {seed}: "
                f"legal@1 {per_seed[-1]['legal_at_1']:.3f} "
                f"refusals {per_seed[-1]['refusals']:.2f} of {per_seed[-1]['refusals_baseline']:.1f}")
        out["games"][name] = {
            "spec": default_spec(name).describe(),
            "corpus": corpus_stats,
            "seen": pool(per_seed, REPORT),
            "unseen": pool(per_seed_unseen, REPORT) if per_seed_unseen else None,
            "action_count": uge.get_game(name).action_count,
        }
    return out


# ---------------------------------------------------------------------------
# arm: phi
# ---------------------------------------------------------------------------


def arm_phi(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """The abstraction dial, from a move-only tape to an injective one.

    ``phi_bits = 0`` is a bigram over moves; 6 to 15 bits are coarser and finer
    hashes of the position; ``exact`` is the game's own state code, for the two
    games small enough.  The interesting part is 16 to 30: a digest that needs a
    second token, where the *first* token stops being read at all because
    ``_locate`` conditions on the last trigram.  The tape gets longer and the
    model gets worse, which is the sharpest statement this directory has of what
    "first-order in the window" costs.
    """
    settings = [
        ("phi=0", {"phi_bits": 0}),
        ("phi=4", {"phi_bits": 4}),
        ("phi=8", {"phi_bits": 8}),
        ("phi=11", {"phi_bits": 11}),
        ("phi=14", {"phi_bits": 14}),
        ("phi=15", {"phi_bits": 15}),
        ("phi=20", {"phi_bits": 20}),
        ("phi=30", {"phi_bits": 30}),
        ("phi=exact", {"phi_exact": True}),
    ]
    out: dict = {"arm": "phi", "config": cfg, "seeds": seeds, "games": {}}
    for name in games:
        game = uge.get_game(name)
        rows = {}
        for label, kw in settings:
            if kw.get("phi_exact") and game.state_code(game.initial()) is None:
                continue
            per_seed = []
            chars = 0
            for seed in seeds:
                spec = default_spec(name, **kw)
                g, corpus, train, test = build(name, spec, cfg, seed)
                chars = corpus.stats()["chars"]
                model = train_model(train.tapes, seed, cfg["epochs"])
                seen = positions(g, test, limit=cfg["eval"])
                per_seed.append(metrics.evaluate(model, g, spec, seen, test.tapes, rollouts=1))
            rows[label] = pool(per_seed, REPORT) | {
                "corpus_chars": chars,
                "phi_tokens": spec.phi_tokens_for(game.token_bound(spec)),
            }
            log(f"  phi/{name} {label}: legal@1 {rows[label]['legal_at_1']:.3f} "
                f"cov {rows[label]['coverage']:.2f} nodes {rows[label]['nodes']:.0f}")
        out["games"][name] = rows
    return out


# ---------------------------------------------------------------------------
# arm: mix
# ---------------------------------------------------------------------------


def arm_mix(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """One token, split between the last move and the position.  Where should the bits go?

    The ``phi`` sweep leaves a hole.  A hash of the position has no
    **locality** - two nearly identical chess positions land in unrelated
    buckets - so the abstraction can recognise a position it has seen and can
    say nothing about one it has not, and chess at 13 bits covers a quarter of
    the test set.  Spending the same bits on a *summary of the move just
    played* (:meth:`uge.game.Game.action_summary`: for chess, the square it
    landed on) buys something a hash cannot, because the summary means the same
    thing in every position it appears in.

    The budget is fixed at the game's one-token maximum; only the split moves.
    ``prev = 0`` is the ``phi`` arm's best hashed setting and ``prev = bits`` is
    history alone, which is the move bigram with the moves bucketed.
    """
    out: dict = {"arm": "mix", "config": cfg, "seeds": seeds, "games": {}}
    for name in games:
        game = uge.get_game(name)
        probe = default_spec(name, phi_bits=0)
        bits = probe.codebook.phi_bits_for(game.token_bound(probe))
        rows = {}
        unseen_rows = {}
        for prev in sorted({0, bits // 4, bits // 2, (3 * bits) // 4, bits}):
            per_seed, per_seed_unseen = [], []
            for seed in seeds:
                spec = default_spec(name, phi_bits=bits, phi_prev_bits=prev)
                g, corpus, train, test = build(name, spec, cfg, seed)
                model = train_model(train.tapes, seed, cfg["epochs"])
                seen = positions(g, test, limit=cfg["eval"])
                per_seed.append(metrics.evaluate(model, g, spec, seen, test.tapes, rollouts=1))
                fresh = unseen_positions(g, spec, train, test, limit=cfg["eval"])
                if fresh:
                    per_seed_unseen.append(
                        metrics.evaluate(model, g, spec, fresh, test.tapes, rollouts=1)
                    )
            label = f"prev={prev}/{bits}"
            rows[label] = pool(per_seed, REPORT)
            rows[label]["unseen_fraction"] = mean(
                [1.0 - r["coverage"] for r in per_seed]
            )
            if per_seed_unseen:
                unseen_rows[label] = pool(per_seed_unseen, REPORT)
            log(f"  mix/{name} {label}: legal@1 {rows[label]['legal_at_1']:.3f} "
                f"cov {rows[label]['coverage']:.2f} refusals {rows[label]['refusals']:.2f}")
        out["games"][name] = {"bits": bits, "rows": rows, "unseen": unseen_rows}
    return out


# ---------------------------------------------------------------------------
# arm: phase
# ---------------------------------------------------------------------------


def arm_phase(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """Does it matter that a trigram can belong to more than one window phase?

    ``plain`` puts all 64 symbols in every slot; ``disjoint`` reserves the last
    character so the phase of a trigram is readable from the trigram.  The
    ambiguity is reported both ways - by trigram *type* and by *occurrence* -
    because only the second one predicts the damage.
    """
    books = [
        ("plain/lsb", {"book": "plain", "layout": "lsb"}),
        ("plain/struct", {"book": "plain", "layout": "struct"}),
        ("disjoint/lsb", {"book": "disjoint", "layout": "lsb"}),
    ]
    out: dict = {"arm": "phase", "config": cfg, "seeds": seeds, "games": {}}
    for name in games:
        game = uge.get_game(name)
        # one abstraction size for every codebook, or chess compares a two-token
        # phi against a one-token one and the answer is about the cliff, not the
        # phases: `disjoint` carries 13 bits a token where `plain` carries 17
        bits = min(
            TapeSpec(name, game.index, phi_bits=0, **kw).codebook.phi_bits_for(
                game.token_bound(TapeSpec(name, game.index, phi_bits=0, **kw))
            )
            for _label, kw in books
        )
        rows = {"phi_bits": bits}
        for label, kw in books:
            kw = dict(kw, phi_bits=bits)
            per_seed, ambiguity = [], None
            for seed in seeds:
                spec = default_spec(name, **kw)
                g, corpus, train, test = build(name, spec, cfg, seed)
                ambiguity = phase_ambiguity(corpus.tapes)
                model = train_model(train.tapes, seed, cfg["epochs"])
                seen = positions(g, test, limit=cfg["eval"])
                per_seed.append(metrics.evaluate(model, g, spec, seen, test.tapes, rollouts=1))
            rows[label] = pool(per_seed, REPORT) | {"ambiguity": ambiguity}
            log(f"  phase/{name} {label}: legal@1 {rows[label]['legal_at_1']:.3f} "
                f"ambiguous(weighted) {ambiguity['weighted']:.1%}")
        out["games"][name] = rows
    return out


# ---------------------------------------------------------------------------
# arm: layout
# ---------------------------------------------------------------------------


def arm_layout(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """Does the order of the digits inside a token change what can be learned?

    Only ``plain`` can express a structured layout, so this arm is run entirely
    under it and the comparison is between three ways of writing the *same*
    action as three characters.  Chess is the case that should show it: under
    ``struct`` the seams between tokens ask "the last move landed on this
    square - where does the next one start", and under ``msb`` they ask nothing
    until the last character.
    """
    layouts = ["struct", "lsb", "msb"]
    out: dict = {"arm": "layout", "config": cfg, "seeds": seeds, "games": {}}
    for name in games:
        game = uge.get_game(name)
        # one abstraction size for all three layouts, or the comparison is of two
        # things at once: a structured layout spreads the same actions over a
        # wider range of codes, which leaves phi fewer of them
        bits = min(
            TapeSpec(name, game.index, phi_bits=0, layout=lay, book="plain").codebook.phi_bits_for(
                game.token_bound(TapeSpec(name, game.index, phi_bits=0, layout=lay, book="plain"))
            )
            for lay in layouts
        )
        rows = {"phi_bits": bits}
        for layout in layouts:
            per_seed = []
            for seed in seeds:
                spec = default_spec(name, book="plain", layout=layout, phi_bits=bits)
                g, corpus, train, test = build(name, spec, cfg, seed)
                model = train_model(train.tapes, seed, cfg["epochs"])
                seen = positions(g, test, limit=cfg["eval"])
                per_seed.append(metrics.evaluate(model, g, spec, seen, test.tapes, rollouts=1))
            rows[layout] = pool(per_seed, REPORT)
            log(f"  layout/{name} {layout}: legal@1 {rows[layout]['legal_at_1']:.3f} "
                f"refusals {rows[layout]['refusals']:.2f}")
        out["games"][name] = rows
    return out


# ---------------------------------------------------------------------------
# arm: multi
# ---------------------------------------------------------------------------


def _cross_talk(model, game, spec, eval_positions, vocab: dict, name: str) -> float:
    """How often the shared network answers a position of one game with another game's move.

    The six games share a code space - only the header token is reserved per
    game - so nothing *structurally* stops a connect four position from being
    answered with a token that only ever appeared in Othello.  Whether that
    happens is the real question behind "can one network hold six games", and it
    is not answered by the legality numbers: a foreign token is usually illegal
    anyway, so it hides inside ``legal@1``.  This counts it directly.
    """
    others = set().union(*(v for k, v in vocab.items() if k != name)) - vocab[name]
    if not others:
        return 0.0
    hits = 0
    for play, ply in eval_positions:
        prefix = encode_prefix(game, spec, play.states[: ply + 1], play.actions[:ply], play.winner)
        ranked = metrics.rank_tokens(model, prefix)
        if ranked and ranked[0][0] in others:
            hits += 1
    return hits / max(1, len(eval_positions))


def arm_multi(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """One network, six games - and the header token is the whole selector.

    ``Experiments/NeuralCompression`` asked whether one network can hold many
    games and had to build a partitioned output and a selector to ask it.  Here
    the question needs no machinery at all: six corpora go into one
    ``RadixNet.train`` call and the first token of every tape says which game it
    is.  What is measured is whether each game's numbers survive the company,
    and what the shared graph costs against six separate ones.
    """
    out: dict = {"arm": "multi", "config": cfg, "seeds": seeds, "shared": {}, "separate": {},
                 "cross_talk": {}}
    shared_sizes, separate_sizes = [], []
    for seed in seeds:
        specs, data, vocab = {}, {}, {}
        for name in games:
            spec = default_spec(name)
            g, corpus, train, test = build(name, spec, cfg, seed)
            specs[name] = spec
            data[name] = (g, train, test)
            vocab[name] = {
                g.action_token(a, spec) for play in train.plays for a in play.actions
            }
        every_tape = [t for name in games for t in data[name][1].tapes]
        shared = train_model(every_tape, seed, cfg["epochs"])
        shared_sizes.append(shared.graph.num_nodes())
        separate_total = 0
        for name in games:
            g, train, test = data[name]
            seen = positions(g, test, limit=cfg["eval"])
            r = metrics.evaluate(shared, g, specs[name], seen, test.tapes, rollouts=1)
            out["shared"].setdefault(name, []).append(r)
            out["cross_talk"].setdefault(name, []).append(
                _cross_talk(shared, g, specs[name], seen, vocab, name)
            )
            alone = train_model(train.tapes, seed, cfg["epochs"])
            separate_total += alone.graph.num_nodes()
            out["separate"].setdefault(name, []).append(
                metrics.evaluate(alone, g, specs[name], seen, test.tapes, rollouts=1)
            )
            log(f"  multi/{name} seed {seed}: shared legal@1 {r['legal_at_1']:.3f} "
                f"alone {out['separate'][name][-1]['legal_at_1']:.3f}")
        separate_sizes.append(separate_total)
    out["shared"] = {k: pool(v, REPORT) for k, v in out["shared"].items()}
    out["separate"] = {k: pool(v, REPORT) for k, v in out["separate"].items()}
    out["cross_talk"] = {k: mean(v) for k, v in out["cross_talk"].items()}
    out["shared_nodes"] = mean(shared_sizes)
    out["separate_nodes"] = mean(separate_sizes)
    out["node_ratio"] = mean(separate_sizes) / max(1.0, mean(shared_sizes))
    return out


# ---------------------------------------------------------------------------
# arm: twonrl
# ---------------------------------------------------------------------------


def refused_tapes(model, game, spec, train, rng: random.Random, limit: int) -> tuple[list[str], list[str]]:
    """The negative set and its matched positive control, generated by the model itself.

    For a sampled position the model is asked to rank, and the **highest-ranked
    token the board refuses** is written onto the end of the prefix: a tape the
    game says no to, produced by the network rather than by us.  That is
    ``Research/2NRL.md`` §6.5's "losing on purpose" carried over to a game the
    paper did not use, and unlike the chess experiment it needs no engine - the
    rules refuse for free.

    A model that ranks nothing (an untrained one, or a position it has never
    seen) falls back to a uniformly random refused token, which is the literal
    "train on garbage" of §3.

    The second list is the control: the same prefixes with the move the teacher
    actually played.  Same count, same lengths, so the positive arm and the
    negative arm differ in *what* they train toward and nothing else.
    """
    game_obj = game
    bad, good = [], []
    pool_positions = positions(game_obj, train)
    rng.shuffle(pool_positions)
    for play, ply in pool_positions[:limit]:
        states, actions = play.states[: ply + 1], play.actions[:ply]
        state = states[-1]
        prefix = encode_prefix(game_obj, spec, states, actions, play.winner)
        legal = set(game_obj.legal(state))
        token = ""
        for candidate, _cost in metrics.rank_tokens(model, prefix):
            action = game_obj.token_action(candidate, spec)
            if action is None or action not in legal:
                token = candidate
                break
        if not token:
            for _ in range(64):
                code = rng.randrange(game_obj.action_count)
                candidate = game_obj.action_token(game_obj.index_action(code), spec)
                if game_obj.token_action(candidate, spec) not in legal:
                    token = candidate
                    break
        if not token:
            continue
        bad.append(prefix + token + spec.outcome_token("broken"))
        good.append(prefix + game_obj.action_token(play.actions[ply], spec) + spec.outcome_token("ongoing"))
    return bad, good


def arm_twonrl(cfg: dict, seeds: list[int], games: list[str]) -> dict:
    """2NRL against a matched positive control, on the rules of a game.

    Three arms, the same number of gradient passes over the same number of
    texts of the same length:

    ``2nrl``      train toward the refused moves, :meth:`RadixNet.invert`, then
                  fine-tune on the real games at a fifth of the rate.
    ``positive``  train toward the *correct* moves on the same prefixes, no
                  inversion, then the same fine-tune.  §11's arm D.
    ``local``     train toward the refused moves, then
                  :meth:`RadixNet.invert_paths` - the local inversion that only
                  touches the nodes the failures ran through - then the same
                  fine-tune.  The repository's own alternative to flipping
                  everything.

    Every arm starts from the *same* warm-started model, so the negatives are
    the same network's own failures in all three and the comparison is of what
    is done with them.
    """
    out: dict = {"arm": "twonrl", "config": cfg, "seeds": seeds, "games": {}}
    neg_epochs, pos_epochs = 3, 3
    for name in games:
        rows: dict[str, list] = {"2nrl": [], "positive": [], "local": [], "base": []}
        negatives = 0
        for seed in seeds:
            spec = default_spec(name)
            game, corpus, train, test = build(name, spec, cfg, seed)
            seen = positions(game, test, limit=cfg["eval"])
            rng = random.Random(seed * 7919 + 13)

            warm = train_model(train.tapes, seed, max(1, cfg["epochs"] // 2))
            bad, good = refused_tapes(warm, game, spec, train, rng, cfg["eval"])
            if not bad:
                log(f"  twonrl/{name} seed {seed}: no refusable positions, skipping")
                continue
            negatives += len(bad)
            base = metrics.evaluate(warm, game, spec, seen, test.tapes, rollouts=1)
            rows["base"].append(base)

            for arm in ("2nrl", "positive", "local"):
                model = type(warm).from_dict(warm.to_dict())  # the same warm start for all three
                if arm == "2nrl":
                    model.two_nrl(bad, train.tapes, neg_epochs=neg_epochs, pos_epochs=pos_epochs,
                                  neg_lr=0.05, pos_lr=0.01, batch_size=TRAIN_DEFAULTS["batch_size"])
                elif arm == "positive":
                    model.train(good, epochs=neg_epochs, lr=0.05,
                                batch_size=TRAIN_DEFAULTS["batch_size"], phase="positive")
                    model.train(train.tapes, epochs=pos_epochs, lr=0.01, act_lr=0.001,
                                batch_size=TRAIN_DEFAULTS["batch_size"], phase="positive")
                else:
                    model.train(bad, epochs=neg_epochs, lr=0.05,
                                batch_size=TRAIN_DEFAULTS["batch_size"], phase="negative")
                    model.invert_paths(bad, mode="activation")
                    model.train(train.tapes, epochs=pos_epochs, lr=0.01, act_lr=0.001,
                                batch_size=TRAIN_DEFAULTS["batch_size"], phase="positive")
                r = metrics.evaluate(model, game, spec, seen, test.tapes, rollouts=1)
                rows[arm].append(r)
                log(f"  twonrl/{name} seed {seed} {arm}: legal@1 {r['legal_at_1']:.3f} "
                    f"refusals {r['refusals']:.2f}")
        out["games"][name] = {
            "negatives": negatives / max(1, len(rows["base"])),
            **{arm: pool(records, REPORT) for arm, records in rows.items() if records},
        }
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _order(rows: dict) -> list:
    """Row labels in numeric order.  ``json.dump(sort_keys=True)`` writes ``phi=11``
    before ``phi=4``, which is correct for a string and useless for a sweep."""
    def key(label: str):
        digits = "".join(c if c.isdigit() else " " for c in label).split()
        return (0, int(digits[0])) if digits else (1, 0)

    return sorted(rows, key=lambda k: key(k))


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarise(into_readme: bool = False) -> None:
    """Turn ``results/*.json`` into the markdown tables ``README.md`` quotes.

    With ``into_readme`` the tables are spliced into the file between the
    ``<!--NAME-->`` / ``<!--/NAME-->`` markers instead of printed, so the
    README and the runs cannot drift apart - the same arrangement
    ``TwoNRL_Chess/summarize.py`` uses.  It is idempotent: a second run
    replaces what the first one wrote.
    """
    sections: dict[str, list[str]] = {}
    current = ["LOOSE"]

    def section(name: str) -> None:
        current[0] = name
        sections.setdefault(name, [])

    def print(*parts, **_kw):  # noqa: A001 - shadowed on purpose, see section()
        sections.setdefault(current[0], []).append(" ".join(str(p) for p in parts))

    def load(name):
        path = os.path.join(RESULTS, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    rules = load("rules")
    if rules:
        section("RULES")
        print("### rules\n")
        print("Six games, one encoder, and the same defaults for all of them:")
        print("a 15-bit hashed abstraction (13 for chess, whose 20 480 actions take")
        print("the code space it needs), the disjoint codebook, and the flat `lsb`")
        print("layout. Nothing is tuned per game - the later arms do that.\n")
        print("| game | `\\|A\\|` | coverage | legal@1 covered | guessing | refusals | refusals guessing | teacher match | unaided plies | self-play refusals | nodes |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, row in rules["games"].items():
            r = row["seen"]
            print(f"| `{name}` | {row['action_count']} | {_fmt(r['coverage'],2)} | **{_fmt(r['legal_at_1_covered'])}** | "
                  f"{_fmt(r['legal_at_1_baseline'])} | **{_fmt(r['refusals'],2)}** | {_fmt(r['refusals_baseline'],1)} | "
                  f"{_fmt(r['teacher_match'])} | {_fmt(r['greedy_plies'],1)} | "
                  f"{_fmt(r.get('selfplay_refusals'),2)} | {_fmt(r['nodes'],0)} |")
        print("\n`unaided plies` is how far the model gets playing both sides with **no**")
        print("second chance - its top token, or the rollout ends. `self-play refusals`")
        print("is the same rollout allowed to walk down its own ranking, measured on")
        print("**its own** positions rather than the teacher's, which by move ten are")
        print("not the same distribution.")
        print("\n**The same models on positions whose `phi` token the training corpus never wrote.**")
        print("Under a hashed `phi` this is a tautology and the point of quoting it:")
        print("the last trigram of the prefix *is* the `phi` token, so an unseen token")
        print("is an unseen node and the model is silent - a hash has no near misses.")
        print("The [`mix`](#mix) arm is where the abstraction is not a hash and this")
        print("set stops being trivial.\n")
        print("| game | coverage | legal@1 covered | guessing | refusals | refusals guessing |")
        print("|---|---:|---:|---:|---:|---:|")
        for name, row in rules["games"].items():
            u = row.get("unseen")
            if not u:
                continue
            print(f"| `{name}` | {_fmt(u['coverage'],2)} | {_fmt(u['legal_at_1_covered'])} | "
                  f"{_fmt(u['legal_at_1_baseline'])} | {_fmt(u['refusals'],2)} | {_fmt(u['refusals_baseline'],1)} |")

    phi = load("phi")
    if phi:
        section("PHI")
        print("### phi\n")
        print("From a move-only tape to an injective one.\n")
        print("**The cliff is not at two tokens, it is at the *last* token.** Only the")
        print("last trigram of the prefix conditions a prediction, so what matters is")
        print("how many bits land in the final token. Nim at 15 bits scores 0.94; at")
        print("20 bits it scores 0.51; at 30 bits it is back to 0.94. Twenty and thirty")
        print("both cost two tokens - but at 20 the second token carries five bits and")
        print("at 30 it carries fifteen. Othello is the same story and louder: 211")
        print("refusals a move at 20 bits against 8.6 at 30.\n")
        print("**An injective `phi` is not automatically the best one.** Tic-tac-toe's")
        print("`exact` row is *worse* than its 15-bit hash (0.65 against 0.99 at covered")
        print("positions) although it throws nothing away, because a hash uses the token")
        print("space evenly and a board code does not: at matched corpus size the hashed")
        print("tape has 875 distinct seam trigrams and the exact one has 677, so the")
        print("exact tape's seams are shared by more contexts and mix them. Uniformity")
        print("is worth something on its own, separately from injectivity.\n")
        for name, rows in phi["games"].items():
            print(f"\n**`{name}`**\n")
            print("| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for label in _order(rows):
                r = rows[label]
                print(f"| {label} | {r['phi_tokens']} | {_fmt(r['coverage'],2)} | {_fmt(r['legal_at_1'])} | "
                      f"{_fmt(r['legal_at_1_covered'])} | **{_fmt(r.get('legal_at_1_fallback'))}** | "
                      f"{_fmt(r.get('legal_at_1_baseline'))} | {_fmt(r['refusals'],2)} | "
                      f"{_fmt(r['teacher_match'])} | {_fmt(r['nodes'],0)} | {r['corpus_chars']} |")

    mix = load("mix")
    if mix:
        section("MIX")
        print("### mix\n")
        print("The same number of bits, split between a hash of the position (`prev=0`)")
        print("and a summary of the move just played (`prev=bits`). A hash has no")
        print("locality; a move summary does.\n")
        for name, block in mix["games"].items():
            print(f"\n**`{name}`** - {block['bits']} bits to spend\n")
            print("| split | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | refusals guessing | teacher match |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for label in _order(block["rows"]):
                r = block["rows"][label]
                print(f"| {label} | {_fmt(r['coverage'],2)} | **{_fmt(r['legal_at_1'])}** | "
                      f"{_fmt(r['legal_at_1_covered'])} | {_fmt(r.get('legal_at_1_fallback'))} | "
                      f"{_fmt(r.get('legal_at_1_baseline'))} | **{_fmt(r['refusals'],2)}** | "
                      f"{_fmt(r['refusals_baseline'],1)} | {_fmt(r['teacher_match'])} |")
            unseen = block.get("unseen") or {}
            if unseen:
                print("\nOn positions whose `phi` token the training corpus never wrote:\n")
                print("| split | coverage | legal@1 | legal@1 covered | refusals | refusals guessing |")
                print("|---|---:|---:|---:|---:|---:|")
                for label in _order(unseen):
                    r = unseen[label]
                    print(f"| {label} | {_fmt(r['coverage'],2)} | {_fmt(r['legal_at_1'])} | "
                          f"{_fmt(r['legal_at_1_covered'])} | {_fmt(r['refusals'],2)} | "
                          f"{_fmt(r['refusals_baseline'],1)} |")

    phase = load("phase")
    if phase:
        section("PHASE")
        print("### phase\n")
        print("`ambiguous occurrences` is the number that matters: a collision on a")
        print("trigram seen once is a curiosity, one on a trigram in every game is a")
        print("defect.\n")
        print("| game | codebook | ambiguous types | ambiguous occurrences | legal@1 covered | with fallback | refusals |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for name, rows in phase["games"].items():
            for label, r in rows.items():
                if not isinstance(r, dict) or "ambiguity" not in r:
                    continue  # the shared phi_bits every codebook of this game used
                a = r["ambiguity"]
                print(f"| `{name}` | {label} | {a['distinct']:.1%} | **{a['weighted']:.1%}** | "
                      f"{_fmt(r['legal_at_1_covered'])} | {_fmt(r.get('legal_at_1_fallback'))} | "
                      f"{_fmt(r['refusals'],2)} |")

    layout = load("layout")
    if layout:
        section("LAYOUT")
        print("### layout\n")
        print("Three ways of writing the *same* action as three characters, all under")
        print("the `plain` codebook, which is the only one that can express a")
        print("structured layout.\n")
        print("| game | layout | coverage | legal@1 | legal@1 covered | refusals | teacher match | nodes |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for name, rows in layout["games"].items():
            for label, r in rows.items():
                if not isinstance(r, dict):
                    continue  # the shared phi_bits this game's three layouts all used
                print(f"| `{name}` | {label} | {_fmt(r['coverage'],2)} | **{_fmt(r['legal_at_1'])}** | "
                      f"{_fmt(r['legal_at_1_covered'])} | {_fmt(r['refusals'],2)} | "
                      f"{_fmt(r['teacher_match'])} | {_fmt(r['nodes'],0)} |")

    multi = load("multi")
    if multi:
        section("MULTI")
        print("### multi\n")
        print("Six corpora into one `RadixNet.train` call. The header token is the")
        print("whole selector - `Experiments/NeuralCompression` needed a partitioned")
        print("output and a selector network to ask this question; here it falls out of")
        print("the encoding.\n")
        print("It works, and it is not free. The shared graph is smaller than six")
        print("separate ones, and every game pays for the company: refusals are worse")
        print("inside it for all six, and legality for five of the six (Connect Four's")
        print("top-1 improves and its refusals get five times worse). The last")
        print("column says why, and it is the same lesson as everywhere else in this")
        print("directory: the six games share a code space, only the header is reserved")
        print("per game, so nothing structurally stops a Connect Four position being")
        print("answered with a token that only ever appeared in Othello. It is not a")
        print("hypothetical - a quarter of the answers are exactly that, and the")
        print("legality columns hide it, because a foreign token is usually illegal")
        print("anyway and just looks like an ordinary miss.\n")
        print(f"shared graph: {multi['shared_nodes']:.0f} nodes; six separate graphs: "
              f"{multi['separate_nodes']:.0f} nodes (ratio {multi['node_ratio']:.2f}x)\n")
        print("| game | legal@1 shared | legal@1 alone | refusals shared | refusals alone | answered with another game's move |")
        print("|---|---:|---:|---:|---:|---:|")
        for name in multi["shared"]:
            sh, al = multi["shared"][name], multi["separate"][name]
            print(f"| `{name}` | {_fmt(sh['legal_at_1'])} | {_fmt(al['legal_at_1'])} | "
                  f"{_fmt(sh['refusals'],2)} | {_fmt(al['refusals'],2)} | "
                  f"{_fmt(multi.get('cross_talk', {}).get(name), 3)} |")

    twonrl = load("twonrl")
    if twonrl:
        section("TWONRL")
        print("### twonrl\n")
        print("Three arms from the *same* warm-started model, with the same number of")
        print("gradient passes over the same number of texts of the same length:")
        print("`2nrl` trains toward the refused moves then inverts, `positive` trains")
        print("toward the correct ones on the same prefixes and does not, and `local`")
        print("uses `invert_paths` instead of the global inversion. `base` is the warm")
        print("start all three began from.\n")
        print("| game | arm | legal@1 | legal@1 covered | refusals | teacher match | nodes |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for name, rows in twonrl["games"].items():
            for arm in ("base", "2nrl", "positive", "local"):
                r = rows.get(arm)
                if not r:
                    continue
                ci = f" ± {r['legal_at_1_ci']:.3f}" if r.get("legal_at_1_ci") else ""
                print(f"| `{name}` | {arm} | {_fmt(r['legal_at_1'])}{ci} | {_fmt(r['legal_at_1_covered'])} | "
                      f"{_fmt(r['refusals'],2)} | {_fmt(r['teacher_match'])} | {_fmt(r['nodes'],0)} |")



    if into_readme:
        _splice(sections)
        return
    for lines in sections.values():
        for line in lines:
            builtins.print(line)


def _splice(sections: dict) -> None:
    """Write each section between its ``<!--NAME-->`` markers in README.md."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for name, lines in sections.items():
        body = "\n".join(lines).strip()
        if not body:
            continue
        open_tag, close_tag = f"<!--{name}-->", f"<!--/{name}-->"
        block = f"{open_tag}\n\n{body}\n\n{close_tag}"
        pattern = re.compile(re.escape(open_tag) + r".*?" + re.escape(close_tag), re.S)
        if pattern.search(text):
            text = pattern.sub(lambda _m: block, text)
        elif open_tag in text:
            text = text.replace(open_tag, block)
        else:
            builtins.print(f"no marker {open_tag} in README.md", file=sys.stderr)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    builtins.print(f"wrote {len(sections)} sections into {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

ARMS = {
    "rules": (arm_rules, GAMES),
    "phi": (arm_phi, ["chess", "othello", "connect4", "tictactoe", "nim"]),
    "mix": (arm_mix, ["chess", "othello", "connect4", "tictactoe"]),
    "phase": (arm_phase, ["connect4", "nim", "othello", "chess"]),
    "layout": (arm_layout, ["chess", "othello", "tictactoe"]),
    "multi": (arm_multi, GAMES),
    "twonrl": (arm_twonrl, ["connect4", "nim", "othello"]),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=[*ARMS, "all", "summary"], help="which arm to run")
    p.add_argument("--quick", action="store_true", help="a tenth of the corpus and half the epochs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed; intervals need 2+")
    p.add_argument("--games", nargs="+", default=None, help="override the arm's game list")
    p.add_argument("--games-count", type=int, default=None, help="games per corpus")
    p.add_argument("--epochs", type=int, default=None, help="training epochs")
    p.add_argument("--into-readme", action="store_true",
                   help="with 'summary': splice the tables into README.md instead of printing them")
    p.add_argument("--out-dir", default=None,
                   help="where to read and write the JSON (default: results/)")
    args = p.parse_args(argv)

    if args.out_dir:
        global RESULTS  # noqa: PLW0603 - one setting, read by write() and summarise()
        RESULTS = os.path.abspath(args.out_dir)

    if args.arm == "summary":
        summarise(args.into_readme)
        return 0

    cfg = dict(QUICK if args.quick else FULL)
    if args.games_count:
        cfg["games"] = args.games_count
        cfg["chess_games"] = max(5, args.games_count // 3)
    if args.epochs:
        cfg["epochs"] = args.epochs

    names = [args.arm] if args.arm != "all" else list(ARMS)
    for name in names:
        fn, default_games = ARMS[name]
        games = args.games or default_games
        log(f"== {name} ({', '.join(games)}) ==")
        t0 = time.perf_counter()
        payload = fn(cfg, args.seeds, games)
        payload["seconds"] = time.perf_counter() - t0
        path = write(name, payload)
        log(f"== {name} done in {payload['seconds']:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
