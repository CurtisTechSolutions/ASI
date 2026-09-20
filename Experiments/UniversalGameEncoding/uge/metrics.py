"""What the network actually learned, measured against the rules.

This is the only module that imports ``radixnet``.  Everything it measures
comes from the same two facts:

* a prefix that ends on a token boundary is a question, and its answer is the
  next three characters;
* the decoder is the referee (:mod:`uge.codec`), so an answer is right or
  wrong without anybody's opinion.

Ranking, and why it is not a beam
---------------------------------
``RadixNet.predict(mode="beam")`` returns the ``k`` cheapest continuations,
which is the wrong tool for "how many moves would it propose before finding a
legal one": a beam of width ``B`` silently truncates the ranking and the number
that comes out is a property of ``B``.  :func:`rank_tokens` instead expands the
graph exhaustively to :data:`~uge.tape.W` emitted characters - cheap, because
the graph only has the edges it observed - so the ranking is *complete over
what the model knows*, and what the model does not know is accounted for
separately and analytically (:func:`expected_refusals`).

The position the model has never seen
-------------------------------------
``RadixNet._locate`` falls back to a linear scan of the trigram index when the
last trigram of the prefix is unknown, and returns the most-visited node that
shares two characters with it.  That is a reasonable thing for a text model to
do and a misleading thing to measure: it turns "I have never seen this
position" into a confident answer about a different one.  So every function
here checks ``graph.lookup`` first and reports ``coverage`` - the fraction of
questions the model was actually in a position to answer - beside every score.
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Any

from .codec import decode, encode_prefix
from .game import Game
from .tape import W, TapeSpec

__all__ = [
    "expected_refusals",
    "evaluate",
    "free_rollout",
    "greedy_rollout",
    "perplexity",
    "propose",
    "rank_tokens",
    "random_baseline",
]

_MAX_EXPANSIONS = 20_000


def _radixnet():
    """Imported lazily so the rest of :mod:`uge` stays dependency-free."""
    from radixnet.graph import END, FIRST, START  # noqa: PLC0415
    from radixnet.search import onward  # noqa: PLC0415

    return START, END, FIRST, onward


def rank_tokens(model, prefix: str, max_expansions: int = _MAX_EXPANSIONS) -> list[tuple[str, float]]:
    """Every :data:`~uge.tape.W`-character continuation the model knows, cheapest first.

    ``[]`` when the last trigram of ``prefix`` is not in the graph: the model
    has never stood here and has nothing to say, which is a different answer
    from a wrong one.  Costs are the graph's own ``-log softmax`` per edge, so
    ``exp(-cost)`` is the probability the model assigns the token, and a token
    reachable by two paths keeps the cheaper.
    """
    START, END, FIRST, onward = _radixnet()
    graph = model.graph
    loc = graph.lookup(prefix[-W:]) if len(prefix) >= W else None
    if loc is None:
        return []
    node, offset = loc
    lead = graph.labels[node][offset + W :]  # the compressed remainder is deterministic and free
    if len(lead) >= W:
        return [(lead[:W], 0.0)]

    best: dict[str, float] = {}
    heap: list[tuple[float, int, int, str]] = [(0.0, 0, node, lead)]
    counter = 1
    expansions = 0
    while heap and expansions < max_expansions:
        cost, _, current, emitted = heapq.heappop(heap)
        expansions += 1
        for child, _edge, step in onward(graph.child_costs(current)):
            if child == END or child < FIRST:
                continue  # sentinels emit nothing; END simply ends this branch
            text = emitted + graph.labels[child][W - 1 :]
            total = cost + step
            if len(text) >= W:
                token = text[:W]
                if total < best.get(token, math.inf):
                    best[token] = total
            else:
                heapq.heappush(heap, (total, counter, child, text))
                counter += 1
    return sorted(best.items(), key=lambda kv: (kv[1], kv[0]))


def expected_refusals(n_actions: int, n_legal: int) -> float:
    """Refusals before the first acceptance when the remaining actions are tried in random order.

    The expectation of the number of failures before the first success in a
    uniformly random permutation of ``n_actions`` items of which ``n_legal``
    are successes: ``(n - L) / (L + 1)``.  Used for the part of the action
    space the model has not ranked, and on its own as the guessing baseline -
    about 675 refusals a move for chess, where ~30 of 20480 tokens are legal.
    """
    if n_legal <= 0:
        return float(n_actions)
    return (n_actions - n_legal) / (n_legal + 1)


def random_baseline(game: Game, state: Any) -> float:
    """What a uniform guess over the whole action space costs in this position."""
    return expected_refusals(game.action_count, len(game.legal(state)))


def propose(model, game: Game, spec: TapeSpec, prefix: str, state: Any) -> dict:
    """Put one position to the model and grade the answer.

    Returns the top token, whether it was legal, how far down the ranking the
    first legal move sat, and - for the tail the model did not rank - the
    analytic expectation of what carrying on would cost.
    """
    ranked = rank_tokens(model, prefix)
    legal = game.legal(state)
    legal_set = set(legal)
    out = {
        "covered": bool(ranked),
        "ranked": len(ranked),
        "top": ranked[0][0] if ranked else "",
        "legal_at_1": False,
        "legal_at_3": False,
        "legal_at_5": False,
        "refusals": 0.0,
        "resolved": False,
        "syntax_refusals": 0,
        "action": None,
    }
    ranked_actions = set()
    for rank, (token, _cost) in enumerate(ranked):
        action = game.token_action(token, spec)
        if action is None:
            out["syntax_refusals"] += 1
            continue
        ranked_actions.add(action)
        if action in legal_set and not out["resolved"]:
            out["refusals"] = float(rank)
            out["resolved"] = True
            out["action"] = action
            out["legal_at_1"] = rank == 0
            out["legal_at_3"] = rank < 3
            out["legal_at_5"] = rank < 5
    if not out["resolved"]:
        # everything the model ranked was refused, so carrying on means guessing
        # from what is left; its cost is an expectation rather than a draw
        remaining = max(0, game.action_count - len(ranked_actions))
        remaining_legal = len(legal_set - ranked_actions)
        out["refusals"] = len(ranked) + expected_refusals(remaining, remaining_legal)
    out["baseline"] = random_baseline(game, state)
    out["legal_share"] = len(legal) / max(1, game.action_count)
    return out


# ---------------------------------------------------------------------------
# playing whole games
# ---------------------------------------------------------------------------


def greedy_rollout(
    model,
    game: Game,
    spec: TapeSpec,
    max_plies: int | None = None,
    assume_winner: int | None = None,
    fallback: bool = False,
    seed: int = 0,
) -> dict:
    """The model plays both sides from the opening; we keep the tape honest for it.

    The ``phi`` tokens are written by us from the real position, which is the
    difference between this and :func:`free_rollout`: here the model only has to
    choose a move, there it has to keep the whole tape consistent as well.

    Without ``fallback`` the model's **top** token is played - no second chance -
    so the rollout ends at the first refusal and ``plies`` is how long it can
    stay inside the rules unaided.  It is deterministic, so running it twice
    tells you nothing.

    With ``fallback`` it plays the way you actually would: down its own ranking
    to the first move the board accepts, and a seeded random legal move when the
    ranking runs out.  The game then always finishes, and what is measured is
    ``refusals_per_move`` **on the model's own trajectory** - which is not the
    same question as :func:`evaluate`'s, because by move ten the positions are
    the model's, not the teacher's, and the distribution has shifted out from
    under the corpus.
    """
    cap = max_plies or 200
    rng = random.Random(seed)
    state = game.initial()
    states, actions = [state], []
    break_kind = "none"
    refusals = 0.0
    while game.winner(state) is None and len(actions) < cap:
        prefix = encode_prefix(game, spec, states, actions, assume_winner)
        ranked = rank_tokens(model, prefix)
        legal = game.legal(state)
        if not fallback:
            if not ranked:
                break_kind = "unknown"
                break
            action = game.token_action(ranked[0][0], spec)
            if action is None:
                break_kind = "syntax"
                break
            if action not in legal:
                break_kind = "illegal"
                break
        else:
            legal_set = set(legal)
            action = None
            for rank, (token, _cost) in enumerate(ranked):
                candidate = game.token_action(token, spec)
                if candidate is not None and candidate in legal_set:
                    action, refusals = candidate, refusals + rank
                    break
            if action is None:
                tried = {game.token_action(t, spec) for t, _ in ranked}
                refusals += len(ranked) + expected_refusals(
                    max(0, game.action_count - len(tried)), len(legal_set - tried)
                )
                action = rng.choice(legal)
        state = game.apply(state, action)
        states.append(state)
        actions.append(action)
    return {
        "plies": len(actions),
        "break_kind": break_kind,
        "terminal": game.winner(state) is not None,
        "capped": len(actions) >= cap,
        "refusals_per_move": refusals / max(1, len(actions)),
    }


def free_rollout(
    model,
    game: Game,
    spec: TapeSpec,
    length: int = 240,
    mode: str = "sample",
    samples: int = 1,
    assume_winner: int | None = None,
) -> dict:
    """The model writes a whole tape from the header alone, and the rules read it back.

    The hardest question in this file, and the one the network fails.  Writing a
    tape means emitting the ``phi`` tokens too, and a hash is not invertible, so
    the model can only get them right where it has *observed* the transition
    ``phi(s) -> a -> phi(s')``.  Where the corpus ran out, ``break_kind = "phi"``
    is the tape telling on itself: every move legal, and the position it thinks
    it is in is not the position it is in.

    ``mode="dijkstra"`` asks for the **cheapest complete tape** and gets
    something worse than a desynchronised game - it gets a one-ply one.  Every
    tape ends ``... action, outcome``, so "an action may be followed by the end"
    is an edge the graph genuinely observed, out of the node for *that action*;
    and that node is shared by every occurrence of the move, first ply and last
    alike.  A first-order chain has nowhere to keep "but it is only move one",
    so the shortest path from the header to END is header, position, one move,
    result.  It is not a defect in the search and no encoding fixes it: it is
    what "conditioned on the last trigram" means, stated as a number.

    Which is why :func:`greedy_rollout` is the measurement that counts.  The
    network can *read* this tape - given the position, name the move - and it
    cannot *write* one, because writing needs the state and only the referee has
    it.  For playing, that is enough: the referee is always there.
    """
    prefix = spec.header()
    runs = []
    for _ in range(max(1, samples)):
        result = model.predict(prefix, length=length, mode=mode, max_length=length)
        decoded = decode(game, spec, prefix + result.text, assume_winner=assume_winner)
        runs.append(
            {
                "plies": decoded.plies,
                "break_kind": decoded.break_kind,
                "ok": decoded.ok,
                "complete": decoded.complete,
                "terminal": decoded.terminal,
                "chars": len(result.text),
                "trimmed": decoded.trimmed,
                "phi_checked": decoded.phi_checked,
            }
        )
    breaks: dict[str, int] = {}
    for r in runs:
        breaks[r["break_kind"]] = breaks.get(r["break_kind"], 0) + 1
    return {
        "plies": sum(r["plies"] for r in runs) / len(runs),
        "break_kind": max(breaks, key=breaks.get),
        "breaks": breaks,
        "complete": sum(r["complete"] for r in runs) / len(runs),
        "terminal": sum(r["terminal"] for r in runs) / len(runs),
        "phi_checked": runs[0]["phi_checked"],
        "samples": len(runs),
    }


def perplexity(model, tapes: list[str]) -> dict:
    """Mean ``-log P`` per token over held-out tapes, and how much of it was a shrug.

    ``RadixNet.score`` charges ``log(1e-6)`` for an unknown trigram or a
    missing edge, so a model that has seen nothing scores about 13.8 nats a
    transition.  ``unknown`` says what fraction of the bill was that charge
    rather than a real probability, and a perplexity quoted without it is
    meaningless.
    """
    total_lp = 0.0
    total_tokens = 0
    total_transitions = 0
    unknown = 0
    for tape in tapes:
        s = model.score(tape)
        total_lp += s["log_prob"]
        total_tokens += len(tape) // W
        total_transitions += s["transitions"]
        unknown += s["unknown_transitions"]
    if not total_tokens:
        return {"nats_per_token": 0.0, "unknown": 0.0, "tokens": 0}
    return {
        "nats_per_token": -total_lp / total_tokens,
        "perplexity_per_token": math.exp(min(-total_lp / total_tokens, 700.0)),
        "unknown": unknown / max(1, total_transitions),
        "tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# the whole battery
# ---------------------------------------------------------------------------


def evaluate(
    model,
    game: Game,
    spec: TapeSpec,
    eval_positions: list[tuple[Any, int]],
    test_tapes: list[str],
    rollouts: int = 1,
    rollout_plies: int | None = None,
    free_length: int = 240,
) -> dict:
    """Every number this experiment reports, for one model on one game.

    ``eval_positions`` are ``(play, ply)`` pairs from :mod:`uge.corpus`; the
    prefix is rebuilt from the play so that the question put to the model is
    exactly the one the encoder would have written.

    Three legality numbers, because one of them alone lies:

    ``legal_at_1``           the model's top token was legal, counting a
                             position it has never seen as a miss.
    ``legal_at_1_covered``   the same, over the positions it *had* seen.
    ``legal_at_1_baseline``  what a uniform guess over the whole action space
                             scores here - which for ``connect4`` is about
                             0.95, because all seven codes are actions and
                             nearly all of them are legal.  Quoting
                             ``legal_at_1`` for such a game without its
                             baseline would make a model that has learned the
                             rules look worse than one that has learned
                             nothing.

    ``legal_at_1_fallback`` is the honest single number: the model where it has
    something to say and a guess where it has not, which is how you would
    actually play with it.
    """
    n = len(eval_positions)
    agg = {
        "positions": n,
        "covered": 0,
        "legal_at_1": 0,
        "legal_at_3": 0,
        "legal_at_5": 0,
        "resolved": 0,
        "teacher_match": 0,
        "optimal_match": 0,
        "optimal_positions": 0,
        "refusals_sum": 0.0,
        "baseline_sum": 0.0,
        "legal_share_sum": 0.0,
        "ranked_sum": 0,
    }
    has_optimal = hasattr(game, "optimal")
    for play, ply in eval_positions:
        states, actions = play.states[: ply + 1], play.actions[:ply]
        state = states[-1]
        prefix = encode_prefix(game, spec, states, actions, play.winner)
        r = propose(model, game, spec, prefix, state)
        agg["covered"] += r["covered"]
        agg["legal_at_1"] += r["legal_at_1"]
        agg["legal_at_3"] += r["legal_at_3"]
        agg["legal_at_5"] += r["legal_at_5"]
        agg["resolved"] += r["resolved"]
        agg["refusals_sum"] += r["refusals"]
        agg["baseline_sum"] += r["baseline"]
        agg["legal_share_sum"] += r["legal_share"]
        agg["ranked_sum"] += r["ranked"]
        if r["top"] and r["top"] == game.action_token(play.actions[ply], spec):
            agg["teacher_match"] += 1
        if has_optimal and r["legal_at_1"]:
            agg["optimal_positions"] += 1
            if r["action"] in game.optimal(state):
                agg["optimal_match"] += 1
        elif has_optimal:
            agg["optimal_positions"] += 1

    d = max(1, n)
    coverage = agg["covered"] / d
    guess = agg["legal_share_sum"] / d
    covered_hit = agg["legal_at_1"] / max(1, agg["covered"])
    out = {
        "positions": n,
        "coverage": coverage,
        "legal_at_1": agg["legal_at_1"] / d,
        "legal_at_3": agg["legal_at_3"] / d,
        "legal_at_5": agg["legal_at_5"] / d,
        "legal_at_1_covered": covered_hit,
        "legal_at_1_baseline": guess,
        "legal_at_1_fallback": coverage * covered_hit + (1.0 - coverage) * guess,
        "resolved_in_model": agg["resolved"] / d,
        "teacher_match": agg["teacher_match"] / d,
        "refusals": agg["refusals_sum"] / d,
        "refusals_baseline": agg["baseline_sum"] / d,
        "ranked_mean": agg["ranked_sum"] / d,
    }
    if has_optimal:
        out["optimal_match"] = agg["optimal_match"] / max(1, agg["optimal_positions"])
    out.update({"ppl_" + k: v for k, v in perplexity(model, test_tapes).items()})

    cap = rollout_plies or 200
    greedy = greedy_rollout(model, game, spec, cap)  # deterministic: once is every time
    out["greedy_plies"] = greedy["plies"]
    out["greedy_break"] = greedy["break_kind"]
    out["greedy_terminal"] = float(greedy["terminal"])
    played = [
        greedy_rollout(model, game, spec, cap, fallback=True, seed=i)
        for i in range(max(1, rollouts))
    ]
    out["selfplay_plies"] = sum(p["plies"] for p in played) / len(played)
    out["selfplay_terminal"] = sum(p["terminal"] for p in played) / len(played)
    out["selfplay_refusals"] = sum(p["refusals_per_move"] for p in played) / len(played)
    free = free_rollout(model, game, spec, free_length, mode="sample", samples=max(3, rollouts))
    out["free_plies"] = free["plies"]
    out["free_break"] = free["break_kind"]
    out["free_breaks"] = free["breaks"]
    out["free_complete"] = free["complete"]
    cheapest = free_rollout(model, game, spec, free_length, mode="dijkstra", samples=1)
    out["free_best_plies"] = cheapest["plies"]
    out["free_best_break"] = cheapest["break_kind"]

    stats = model.graph
    out.update(
        {
            "nodes": stats.num_nodes(),
            "edges": stats.num_edges(),
            "trigrams": stats.num_trigrams(),
            "compression_ratio": stats.compression_ratio(),
        }
    )
    return out
