"""An SBNN learns chess from refusals and from losing to Stockfish - with and without 2NRL.

``Research/2NRL.md`` §10: *"Plainly: I have not measured 2NRL against a
baseline."*  §11 says what the measurement would take - matched compute,
negative sets of different entropy, two controls, several seeds, variance
reported.  This is that experiment, on chess instead of text, with a
self-building network as the learner and Stockfish as both opponent and judge.

The network is given the board and nothing else.  It is **not** given the legal
moves: it proposes one of 4096 from-to pairs, the board refuses or accepts, and
it keeps proposing down its own ranking until something is accepted.  So it has
two things to learn and they are the two halves of §6.5 - first the **rules**,
by being refused, and then the **payoffs**, by being graded.

The loop, one round at a time
-----------------------------
1. **Play.**  ``games`` games against a skill-limited Stockfish.  Each decision
   records how many refusals it took to find a legal move, and each *accepted*
   move is ranked by a full-strength Stockfish in a single ``MultiPV`` search -
   one engine call, identical for every arm, yielding the best move, the worst
   move, and what the played move gave away.
2. **Grade.**  A decision failed if the board refused anything or if the move
   it settled on gave away more than ``blunder_margin`` centipawns.  It is
   weighted by how blatant that was, ``w = min(boost, 1 + g)`` - §6.1's
   proportional boosting, "the worse the failure, the harder the network is
   pushed to reproduce it".
3. **Train.**  §3's three phases, spread over the run exactly as
   ``TwoNRL_CartPole`` spreads them.  The first ``neg_fraction`` of the rounds
   (30% by default) is
   **phase 1**: the network plays, fails, and is trained at the full ``neg_lr``
   *toward* the failures it produced - so it gets worse on purpose, and its
   games generate more failure to learn from.  Then **phase 2**, one inversion.
   Then **phase 3**: the remaining rounds fine-tune toward Stockfish's move at
   ``pos_lr = neg_lr / 5``, with the activation parameters at a tenth of that.

   ``--schedule per-round`` runs a negative block, an inversion and a positive
   block inside *every* round instead.  That is the shape of §9.3's perpetual
   self-improvement loop, and the README reports what it does here: it thrashes,
   because an inversion is global (§12 item 2) and a phase-1 block that has not
   fully turned the policy into a failure policy leaves the inversion negating
   the good parts along with the bad.
4. **Grow.**  If the training loss has stalled the hidden layer gains units -
   the self-building half, identical in every arm so it cannot explain a
   difference between them.

The arms
--------
=================  ====================================================
arm                the first block trains toward
=================  ====================================================
``2nrl``           its **own failure** - the move the board refused, or
                   the blunder it settled for - then **invert**
``2nrl-worst``     Stockfish's **worst legal move**, then **invert**
``2nrl-random``    a **uniformly random legal move**, then **invert**
``positive``       Stockfish's best move; **no inversion** (control D)
``repulsion``      **away from** its own failure; no inversion (arm E)
=================  ====================================================

``target_entropy`` measures ``H(q)`` for each of those negative sets rather than
assuming an ordering for them, which is what §5 turns on and what §7.2 wants
made measurable.

Every arm ends on the same positive fine-tune toward Stockfish's best move, so
``positive`` is exactly ``2nrl`` with the negatives replaced by positives and
the inversion removed.  P4 - inversion against repulsion on identical data - is
the load-bearing comparison.
"""

from __future__ import annotations

import os as _os

# One BLAS thread per worker.  Every matrix here is small, the parallelism that
# matters is across runs, and four workers each opening a machine-sized thread
# pool spend their time fighting each other.  Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import collections
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import chess
import numpy as np

import dataset
from agent import (Agent, Batch, apply_grad, cross_entropy_grad, nll, policy,
                   unlikelihood_grad)
from engine import Judge, Opponent, find_stockfish
from features import material
from moves import N_ACTIONS, dense, to_action, to_move
from sbnn import Adam

ARMS = ("2nrl", "2nrl-worst", "2nrl-random", "positive", "repulsion")
INVERTING = ("2nrl", "2nrl-worst", "2nrl-random")
CP_CLIP = 1000.0       # a mate is worth 10000; without a clip it is the whole mean


@dataclass
class Config:
    arm: str = "2nrl"
    hidden: tuple[int, ...] = (32, 32)
    rounds: int = 24
    games: int = 8                   # training games per round
    # 2NRL's rates: the research ratio is neg:pos = 5:1 and act_lr = lr/10.
    neg_lr: float = 0.01
    pos_lr: float = 0.002
    neg_updates: int = 40
    pos_updates: int = 40
    minibatch: int = 64
    tau: float = 1.0                 # softmax temperature over the score head
    explore: float = 0.5             # Gumbel temperature while playing training games
    width: int = 12                  # candidate moves kept per decision
    blunder_margin: float = 50.0     # centipawns; below this a move is not a failure
    refusal_margin: float = 4.0      # refusals; the rule-failure equivalent of the above
    boost: float = 3.0               # cap on §6.1's proportional boosting
    buffer_rounds: int = 5           # how many rounds of failures a block trains on
    schedule: str = "phased"         # "phased" = §3's three phases across the run;
                                     # "per-round" = a negative block, an inversion
                                     # and a positive block inside every round
    neg_fraction: float = 0.3        # in "phased", how much of the run is phase 1
    judge_depth: int = 6
    opp_skill: int = 0
    opp_depth: int = 4
    opp_elo: int | None = None
    opening_plies: int = 4
    max_plies: int = 80
    resign_cp: int = 1200
    growth: int = 8                  # units added when the loss stalls
    growth_patience: int = 3         # rounds without improvement before growing
    growth_min_delta: float = 1e-3
    max_hidden: int = 256
    eval_games: int = 10             # games against Stockfish at the end of the run
    track_positions: int = 120       # held-out positions scored every round
    heldout: str = dataset.DEFAULT_PATH
    checkpoints: str | None = None   # save the network at every phase boundary
    seed: int = 0
    stockfish: str | None = None


# ----------------------------------------------------------------- playing


def random_opening(board: chess.Board, rng: random.Random, plies: int) -> None:
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return
        board.push(rng.choice(moves))


def play_game(agent: Agent, judge: Judge, opponent: Opponent, cfg: Config,
              agent_white: bool, rng: random.Random, nprng: np.random.Generator,
              collect: bool, temperature: float) -> tuple[list[dict], dict]:
    """One game.  Returns the graded decisions and how it went."""
    board = chess.Board()
    random_opening(board, rng, cfg.opening_plies)
    pov = chess.WHITE if agent_white else chess.BLACK
    samples: list[dict] = []
    losses: list[float] = []
    refusals: list[int] = []
    gave_up = 0
    plies = 0
    verdict = "played out"
    final_cp = 0

    while not board.is_game_over(claim_draw=True) and plies < cfg.max_plies:
        if board.turn != pov:
            board.push(opponent.play(board))
            plies += 1
            continue

        proposal = agent.propose(board, nprng, temperature)
        refusals.append(proposal["n_refused"])
        gave_up += int(proposal["gave_up"])

        ranking = judge.rank(board)
        legal = list(board.legal_moves)
        cp = np.array([ranking[board.uci(m)] for m in legal], dtype=float)
        actions = [to_action(board, m, pov) for m in legal]
        best = actions[int(np.argmax(cp))]
        worst = actions[int(np.argmin(cp))]
        played_cp = float(ranking[board.uci(proposal["move"])])
        loss_cp = float(cp.max() - played_cp)
        losses.append(loss_cp)
        final_cp = int(cp.max())

        if collect:
            samples.append(_sample(proposal, actions, cp, best, worst, loss_cp, cfg, nprng))
        board.push(proposal["move"])
        plies += 1

        if cp.max() < -cfg.resign_cp:
            verdict = "resigned"
            break
        if cp.max() > cfg.resign_cp:
            verdict = "winning"
            break

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = 1.0 if outcome.winner == pov else (0.5 if outcome.winner is None else 0.0)
        verdict = "terminal"
    else:
        result = 1.0 if final_cp > 150 else (0.0 if final_cp < -150 else 0.5)
    stats = {"result": result, "plies": plies, "verdict": verdict,
             "acpl": float(np.mean(losses)) if losses else 0.0,
             "refusals": float(np.mean(refusals)) if refusals else 0.0,
             "top1_legal": float(np.mean([r == 0 for r in refusals])) if refusals else 0.0,
             "gave_up": gave_up,
             "blunders": int(sum(1 for l in losses if l > cfg.blunder_margin)),
             "decisions": len(losses),
             "material": material(board, pov), "final_cp": final_cp}
    return samples, stats


def _sample(proposal: dict, legal_actions: list[int], cp: np.ndarray, best: int,
            worst: int, loss_cp: float, cfg: Config, nprng: np.random.Generator) -> dict:
    """One decision, reduced to ``width`` candidate actions with every target in it.

    See :class:`agent.Batch` for what each slot holds.  The rows are built once
    here, so every arm trains on byte-identical candidate features.
    """
    order: list[int] = []
    slot: dict[int, int] = {}

    def add(action: int) -> int:
        if action not in slot:
            slot[action] = len(order)
            order.append(action)
        return slot[action]

    played_s = add(proposal["action"])
    best_s = add(best)
    worst_s = add(worst)
    rand_s = add(int(legal_actions[int(nprng.integers(len(legal_actions)))]))
    refused_s = add(proposal["refused"][0]) if proposal["refused"] else rand_s
    wild_s = add(int(nprng.integers(N_ACTIONS)))
    # The rest of the refusals are the *hard* negatives - illegal moves this
    # network itself ranked highly - and a second wild action is a cheap easy
    # one. Every arm gets them, so they cannot explain a difference; what they
    # do is give the softmax something to push down that is actually wrong.
    for action in proposal["refused"][1:]:
        if len(order) < cfg.width - 2:
            add(int(action))
    if len(order) < cfg.width - 1:
        add(int(nprng.integers(N_ACTIONS)))
    pool = [a for a in legal_actions if a not in slot]
    nprng.shuffle(pool)
    for action in pool[:max(0, cfg.width - len(order))]:
        add(int(action))

    is_rule = float(proposal["n_refused"] > 0)
    magnitude = (proposal["n_refused"] / cfg.refusal_margin if is_rule
                 else loss_cp / cfg.blunder_margin)
    return {"X": dense(proposal["board_x"], proposal["states"], np.array(order)),
            "played": played_s, "best": best_s, "worst": worst_s, "rand": rand_s,
            "refused": refused_s, "wild": wild_s, "loss_cp": loss_cp,
            # where the played move sits among the legal ones: 0 is Stockfish's
            # own choice, 1 its worst. This is what H(q) is measured over.
            "rank": float((cp > cp.max() - loss_cp).mean()),
            "weight": float(min(cfg.boost, 1.0 + magnitude)), "is_rule": is_rule}


def play_round(agent: Agent, judge: Judge, opponent: Opponent, cfg: Config,
               rng: random.Random, nprng: np.random.Generator, games: int,
               collect: bool = True, temperature: float | None = None) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    stats: list[dict] = []
    temp = cfg.explore if temperature is None else temperature
    for g in range(games):
        s, st = play_game(agent, judge, opponent, cfg, g % 2 == 0, rng, nprng, collect, temp)
        samples.extend(s)
        stats.append(st)
    decisions = max(1, int(np.sum([s["decisions"] for s in stats])))
    agg = {"score": float(np.mean([s["result"] for s in stats])),
           "acpl": float(np.mean([s["acpl"] for s in stats])),
           "refusals": float(np.mean([s["refusals"] for s in stats])),
           "top1_legal": float(np.mean([s["top1_legal"] for s in stats])),
           "gave_up": int(np.sum([s["gave_up"] for s in stats])),
           "plies": float(np.mean([s["plies"] for s in stats])),
           "blunder_rate": float(np.sum([s["blunders"] for s in stats]) / decisions),
           "material": float(np.mean([s["material"] for s in stats])),
           "games": games}
    return samples, agg


# ---------------------------------------------------------------- training


def negative_target(arm: str, batch: Batch) -> np.ndarray:
    """Which slot the first block aims at.

    For ``2nrl`` and ``repulsion`` the negative is the network's **own**
    failure, and it takes whichever kind the decision actually produced: the
    move the board refused when there was one, and the blunder it settled for
    otherwise.  A refusal comes first because a rule failure is the
    lower-entropy of the two - §5 - and because until the rules are learned
    there is nothing for a payoff to be learned *about*.
    """
    if arm == "2nrl-worst":
        return batch.worst
    if arm == "2nrl-random":
        return batch.rand
    return np.where(batch.is_rule > 0, batch.refused, batch.played)


def target_entropy(batch: Batch, target: np.ndarray, bins: int = 10) -> float:
    """``H(q)`` for the negative set, in nats, from where the target sits.

    §5 makes the whole claim turn on how *concentrated* the failure
    distribution is, and §7.2 notes that a standing model of failure would make
    that "measurable instead of assumed".  Here it is measured directly: bin
    each target by its rank among the legal moves (0 = Stockfish's best,
    1 = its worst; an illegal move counts as 1, the far end) and take the
    entropy of the histogram.  Always aiming at the same end is ``H = 0``; a
    uniformly random legal move approaches ``log(bins)``.
    """
    if len(batch) == 0:
        return 0.0
    ranks = np.where(target == batch.best, 0.0,
                     np.where(target == batch.worst, 1.0,
                              np.where(target == batch.refused, 1.0,
                                       np.where(target == batch.played, batch.rank, 0.5))))
    hist, _ = np.histogram(np.clip(ranks, 0, 1), bins=bins, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_block(net, opt, batch: Batch, target: np.ndarray, weights: np.ndarray,
              cfg: Config, updates: int, nprng: np.random.Generator,
              repel: bool = False) -> float:
    """``updates`` Adam steps toward (or, if ``repel``, away from) ``target``."""
    if len(batch) == 0:
        return float("nan")
    last = float("nan")
    for _ in range(updates):
        idx = nprng.choice(len(batch), size=min(cfg.minibatch, len(batch)), replace=False)
        mb = batch.subset(idx)
        p = policy(net, mb, cfg.tau, train=True)
        t, w = target[idx], weights[idx]
        grad = (unlikelihood_grad(p, t, w) if repel else cross_entropy_grad(p, t, w))
        apply_grad(net, opt, mb, grad, cfg.tau)
        last = nll(p, t)
    return last


def train_round(agent: Agent, cfg: Config, samples: list[dict],
                nprng: np.random.Generator, phase: str, invert_now: bool,
                probe=None) -> dict:
    """One round of training.

    ``phase`` is ``"negative"``, ``"positive"`` or ``"both"``.  The first two are
    §3's phases spread across the run; ``"both"`` is the per-round schedule,
    which runs a negative block, the inversion and a positive block inside the
    round.  Every arm gets the same number of updates at the same rates
    whichever is chosen - only the target and the inversion differ.

    ``probe`` is called immediately before and immediately after the sign flip,
    with nothing in between, so ``inversion_before`` and ``inversion_after``
    measure **the inversion alone** - no training, no play, one closed-form
    operation.  That is the number this whole experiment exists to produce.
    """
    if not samples:
        return {"n": 0, "negative_nll": float("nan"), "positive_nll": float("nan"),
                "inverted": False, "negative_entropy": 0.0, "mean_weight": 0.0,
                "rule_share": 0.0, "phase": phase}
    batch = Batch(samples, cfg.width)
    weights = batch.weight
    net, arm = agent.net, cfg.arm
    out = {"n": len(batch), "mean_weight": float(weights.mean()),
           "rule_share": float((batch.is_rule > 0).mean()),
           "mean_loss_cp": float(batch.loss_cp.mean()), "phase": phase,
           "negative_nll": float("nan"), "positive_nll": float("nan"),
           "inverted": False, "negative_entropy": 0.0}

    def negative_block(updates: int) -> None:
        opt = Adam(net, lr=cfg.neg_lr, act_lr=cfg.neg_lr / 10.0)
        if arm == "positive":
            target, repel = batch.best, False
        else:
            target, repel = negative_target(arm, batch), arm == "repulsion"
        out["negative_nll"] = run_block(net, opt, batch, target, weights, cfg,
                                        updates, nprng, repel)
        out["negative_entropy"] = target_entropy(batch, target) if arm != "positive" else 0.0

    def positive_block(updates: int) -> None:
        opt = Adam(net, lr=cfg.pos_lr, act_lr=cfg.pos_lr / 10.0)
        out["positive_nll"] = run_block(net, opt, batch, batch.best,
                                        np.ones(len(batch)), cfg, updates, nprng)

    def invert() -> None:
        rows = np.asarray(batch.X[:64], dtype=np.float64)
        before = net.forward(rows, train=False)
        if probe is not None:
            out["inversion_before"] = probe()
        net.invert()
        out["inversion_error"] = float(np.abs(net.forward(rows, train=False) + before).max())
        if probe is not None:
            out["inversion_after"] = probe()
        out["inverted"] = True

    updates = cfg.neg_updates + cfg.pos_updates
    if phase == "negative":
        negative_block(updates)
        if invert_now and arm in INVERTING:
            invert()
    elif phase == "positive":
        positive_block(updates)
    else:                                   # "both": the per-round schedule
        negative_block(cfg.neg_updates)
        if invert_now and arm in INVERTING:
            invert()
        positive_block(cfg.pos_updates)
    return out


# -------------------------------------------------------------- evaluation


def evaluate_heldout(agent: Agent, positions: list[dict]) -> dict:
    """The exam: propose into the same positions, and see what comes back.

    Costs no engine calls - every legal move in every position was graded once,
    when the set was built.  Three numbers, and the first two are the two halves
    of §6.5: ``top1_legal`` and ``refusals`` say whether the **rules** have been
    learned, ``cp_loss`` and ``agreement`` whether the **payoffs** have.  The
    loss is clipped at ``CP_CLIP`` because a missed mate scores 10000 and would
    otherwise *be* the mean.
    """
    agree = top1 = 0
    losses, refusals = [], []
    for item in positions:
        board = item["board"]
        scores, _, _ = agent.action_scores(board)
        order = np.argsort(-scores)
        n_refused = 0
        for action in order:
            move = to_move(board, int(action), board.turn)
            if board.is_legal(move):
                break
            n_refused += 1
        canonical = board.uci(move)
        idx = next((i for i, m in enumerate(item["moves"]) if board.uci(m) == canonical), 0)
        agree += int(canonical in item["best_moves"])
        top1 += int(n_refused == 0)
        refusals.append(n_refused)
        losses.append(min(CP_CLIP, item["best_cp"] - item["cp"][idx]))
    loss = np.array(losses, dtype=float)
    n = max(len(positions), 1)
    return {"agreement": agree / n, "top1_legal": top1 / n,
            "refusals": float(np.mean(refusals)), "cp_loss": float(loss.mean()),
            "cp_loss_median": float(np.median(loss)),
            "blunder_rate": float((loss > 200).mean()), "positions": len(positions)}


# --------------------------------------------------------------------- run


def run(cfg: Config, verbose: bool = True) -> tuple[dict, Agent]:
    started = time.time()
    path = find_stockfish(cfg.stockfish)
    judge = Judge(path, depth=cfg.judge_depth)
    opponent = Opponent(path, skill=cfg.opp_skill, depth=cfg.opp_depth, elo=cfg.opp_elo)
    rng = random.Random(cfg.seed)
    nprng = np.random.default_rng(cfg.seed + 977)
    agent = Agent.build(list(cfg.hidden), seed=cfg.seed + 1000, tau=cfg.tau)
    positions = dataset.load(cfg.heldout)
    tracked = positions[:cfg.track_positions]

    def checkpoint(name: str) -> None:
        """Keep the network as it was at a phase boundary.

        Phase 2 is a single operation and the whole argument turns on what it
        does, so `untrained`, `phase1_end`, `after_invert` and `final` are the
        four states worth being able to look at side by side.  `export_viz.py`
        scores the action space from them.
        """
        if not cfg.checkpoints:
            return
        _os.makedirs(cfg.checkpoints, exist_ok=True)
        agent.net.save(_os.path.join(cfg.checkpoints, f"{cfg.arm}_seed{cfg.seed}_{name}.npz"))

    checkpoint("untrained")
    history = [{"round": 0, **evaluate_heldout(agent, tracked),
                "hidden": list(agent.net.hidden_sizes)}]
    if verbose:
        h = history[0]
        print(f"  [{cfg.arm:12s} seed {cfg.seed}] round  0  legal@1 {h['top1_legal']:.3f}  "
              f"refusals {h['refusals']:6.1f}  cp loss {h['cp_loss']:6.1f}  (untrained)")

    neg_rounds = (max(1, int(round(cfg.rounds * cfg.neg_fraction)))
                  if cfg.schedule == "phased" else 0)
    best_seen, stalled, growth_events = float("inf"), 0, []
    last_phase = ""
    # A short replay of the last few rounds' failures. A block of updates on the
    # handful of decisions in one round would be noise. Identical in every arm.
    buffer: collections.deque = collections.deque(maxlen=max(1, cfg.buffer_rounds))
    for r in range(1, cfg.rounds + 1):
        samples, play = play_round(agent, judge, opponent, cfg, rng, nprng, cfg.games)
        buffer.append([s for s in samples
                       if s["is_rule"] > 0 or s["loss_cp"] > cfg.blunder_margin])
        failures = [s for group in buffer for s in group]
        if cfg.schedule == "phased":
            phase = "negative" if r <= neg_rounds else "positive"
            invert_now = r == neg_rounds
        else:
            phase, invert_now = "both", True
        if phase != last_phase:                    # a phase change resets the stall
            best_seen, stalled, last_phase = float("inf"), 0, phase
        if invert_now and cfg.arm in INVERTING:
            checkpoint("phase1_end")
        train = train_round(agent, cfg, failures, nprng, phase, invert_now,
                            probe=lambda: evaluate_heldout(agent, tracked))
        if train.get("inverted"):
            checkpoint("after_invert")
        ev = evaluate_heldout(agent, tracked)

        # ---- the self-building half: widen on a stall ----------------------
        score = (train["negative_nll"] if phase == "negative" else train["positive_nll"])
        if not math.isnan(score):
            if score < best_seen - cfg.growth_min_delta:
                best_seen, stalled = score, 0
            else:
                stalled += 1
        if stalled >= cfg.growth_patience and cfg.growth:
            added = agent.net.grow_hidden(cfg.growth, nprng, layer=0, max_hidden=cfg.max_hidden)
            if added:
                growth_events.append({"round": r, "added": added,
                                      "hidden": list(agent.net.hidden_sizes)})
            stalled, best_seen = 0, score

        history.append({"round": r, **ev, "hidden": list(agent.net.hidden_sizes),
                        "train": train, "play": play})
        if verbose:
            grew = (f"  grew->{agent.net.hidden_sizes}" if growth_events
                    and growth_events[-1]["round"] == r else "")
            print(f"  [{cfg.arm:12s} seed {cfg.seed}] round {r:2d}  "
                  f"legal@1 {ev['top1_legal']:.3f}  refusals {ev['refusals']:6.1f}  "
                  f"cp loss {ev['cp_loss']:6.1f}  failures {train['n']:4d} "
                  f"({train['rule_share']:.0%} rule)  {train['phase']:8s}"
                  f"{'  INVERTED' if train['inverted'] else ''}{grew}")

    checkpoint("final")
    final = play_round(agent, judge, opponent, cfg, rng, nprng, cfg.eval_games,
                       collect=False, temperature=0.0)[1]
    out = {"arm": cfg.arm, "seed": cfg.seed, "config": asdict(cfg),
           "heldout": evaluate_heldout(agent, positions),
           "heldout_tracked": evaluate_heldout(agent, tracked),
           "vs_stockfish": final, "opponent": opponent.describe(),
           "hidden": list(agent.net.hidden_sizes), "parameters": agent.net.n_params(),
           "growth": growth_events, "activation": agent.net.act_report(),
           "negative_rounds": neg_rounds,
           "engine_calls": judge.calls, "engine_cache_hits": judge.hits,
           "history": history, "seconds": round(time.time() - started, 1)}
    judge.close()
    opponent.close()
    if verbose:
        hv, sv = out["heldout"], out["vs_stockfish"]
        print(f"  [{cfg.arm:12s} seed {cfg.seed}] DONE  legal@1 {hv['top1_legal']:.3f}  "
              f"refusals {hv['refusals']:.1f}  cp loss {hv['cp_loss']:.1f}  |  "
              f"vs Stockfish: score {sv['score']:.2f} acpl {sv['acpl']:.0f} "
              f"refusals {sv['refusals']:.1f}  ({out['seconds']}s)")
    return out, agent


def _run_one(payload: tuple) -> dict:
    cfg, weights_dir = payload
    out, agent = run(cfg)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        path = os.path.join(weights_dir, f"{cfg.arm}_seed{cfg.seed}.npz")
        agent.net.save(path)
        out["weights"] = path
    return out


def summarise(runs: list[dict]) -> None:
    arms: dict[str, list[dict]] = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r)
    print()
    print(f"{'arm':<14}{'H(q)':>6}{'legal@1':>18}{'refusals':>16}"
          f"{'cp loss':>18}{'vs SF':>14}")
    for arm, rs in arms.items():
        def col(key):
            return np.array([r["heldout"][key] for r in rs])
        legal, refus, cpl = col("top1_legal"), col("refusals"), col("cp_loss")
        score = np.array([r["vs_stockfish"]["score"] for r in rs])
        ent = [h["train"]["negative_entropy"] for r in rs for h in r["history"][1:]
               if h["train"].get("phase") in ("negative", "both") and h["train"]["n"]]
        print(f"{arm:<14}{np.mean(ent) if ent else 0:6.2f}"
              f"{legal.mean():11.3f} +-{legal.std():5.3f}"
              f"{refus.mean():9.1f} +-{refus.std():5.1f}"
              f"{cpl.mean():11.1f} +-{cpl.std():5.1f}"
              f"{score.mean():9.2f} +-{score.std():4.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", nargs="+", default=["2nrl", "positive"], choices=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=Config.rounds)
    p.add_argument("--games", type=int, default=Config.games)
    p.add_argument("--width", type=int, default=Config.width)
    p.add_argument("--hidden", type=int, nargs="+", default=list(Config.hidden))
    p.add_argument("--neg-lr", type=float, default=Config.neg_lr)
    p.add_argument("--pos-lr", type=float, default=Config.pos_lr)
    p.add_argument("--neg-updates", type=int, default=Config.neg_updates)
    p.add_argument("--pos-updates", type=int, default=Config.pos_updates)
    p.add_argument("--opp-skill", type=int, default=Config.opp_skill)
    p.add_argument("--opp-depth", type=int, default=Config.opp_depth)
    p.add_argument("--judge-depth", type=int, default=Config.judge_depth)
    p.add_argument("--eval-games", type=int, default=Config.eval_games)
    p.add_argument("--buffer-rounds", type=int, default=Config.buffer_rounds)
    p.add_argument("--track-positions", type=int, default=Config.track_positions)
    p.add_argument("--schedule", choices=["phased", "per-round"], default=Config.schedule,
                   help="phased: §3's three phases over the run (default).  "
                        "per-round: a negative block, an inversion and a positive "
                        "block inside every round")
    p.add_argument("--neg-fraction", type=float, default=Config.neg_fraction,
                   help="in the phased schedule, the share of rounds spent failing")
    p.add_argument("--no-growth", action="store_true", help="freeze the architecture")
    p.add_argument("--heldout", default=Config.heldout)
    p.add_argument("--checkpoints", default=None,
                   help="directory to save the network at each phase boundary")
    p.add_argument("--stockfish", default=None)
    p.add_argument("--jobs", type=int, default=1, help="(arm, seed) jobs to run in parallel")
    p.add_argument("--weights", default="weights", help="directory for the trained networks")
    p.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()

    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            cfg = Config(arm=arm, seed=seed, rounds=args.rounds, games=args.games,
                         hidden=tuple(args.hidden), neg_lr=args.neg_lr, pos_lr=args.pos_lr,
                         neg_updates=args.neg_updates, pos_updates=args.pos_updates,
                         opp_skill=args.opp_skill, opp_depth=args.opp_depth,
                         judge_depth=args.judge_depth, eval_games=args.eval_games,
                         buffer_rounds=args.buffer_rounds, width=args.width,
                         track_positions=args.track_positions,
                         schedule=args.schedule, neg_fraction=args.neg_fraction,
                         growth=0 if args.no_growth else Config.growth,
                         heldout=args.heldout, stockfish=args.stockfish,
                         checkpoints=args.checkpoints)
            jobs.append((cfg, args.weights))

    print(f"=== {len(jobs)} runs: arms {args.arms}, seeds {args.seeds}, "
          f"{args.rounds} rounds x {args.games} games ===")
    started = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            runs = pool.map(_run_one, jobs)
    else:
        runs = [_run_one(job) for job in jobs]
    print(f"=== {len(runs)} runs in {time.time() - started:.0f}s ===")

    summarise(runs)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"runs": runs}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
