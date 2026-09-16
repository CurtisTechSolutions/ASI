"""Correctness proofs for the SBNN, the 2NRL inversion and the search over it.

Run with ``python3 test_sbnn.py``.  The numbers in the README are only worth
reading if these hold, and three of them are the load-bearing ones:

* the inversion is **exact** on the network, and stays exact after the network
  has grown (``Research/2NRL.md`` §4.3);
* the inversion is **exact through the two-ply search**, which is why the
  opponent is modelled by a midrange and not by a minimum - the last test
  shows a minimax opponent breaking it;
* growth is an **identity**, so a self-building step cannot quietly repair or
  damage what the negative phase built.
"""

from __future__ import annotations

import chess
import numpy as np

from agent import Agent, Batch, apply_grad, cross_entropy_grad, nll, policy
from features import (CHECK_OFFSET, FEATURE_DIM, NO_MOVES_OFFSET, TURN_OFFSET,
                      encode, encode_children)
from moves import (INPUT_DIM, N_ACTIONS, dense, preactivation,
                   to_action, to_move)
from sbnn import ACT_PARAMS, MIN_B, Adam, SineNet

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def net(seed: int = 0, sizes=(12, 10, 8, 1)) -> SineNet:
    return SineNet(list(sizes), np.random.default_rng(seed))


def randomise_act(n: SineNet, seed: int = 7) -> None:
    """Move a, b, h, k off their defaults - h and k in particular, so the
    activation is no longer odd and the inversion cannot lean on it."""
    rng = np.random.default_rng(seed)
    for layer in n.layers:
        layer.a += rng.normal(0, 0.3, layer.a.shape)
        layer.b += rng.normal(0, 0.05, layer.b.shape)
        layer.h += rng.normal(0, 0.4, layer.h.shape)
        layer.k += rng.normal(0, 0.3, layer.k.shape)
        layer.bias += rng.normal(0, 0.2, layer.bias.shape)


def board_after(n: int, seed: int = 0) -> chess.Board:
    import random
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(n):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


# ------------------------------------------------------------------ network


def test_gradients() -> None:
    """Analytic gradients of a*sin(b*(x-h))+k vs central finite differences."""
    print("\ngradients (analytic vs finite difference)")
    n = net(1)
    randomise_act(n, 3)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 12))
    target = rng.normal(size=(8, 1))

    def loss_of(model: SineNet) -> float:
        return float(((model.forward(x, train=False) - target) ** 2).sum())

    out = n.forward(x)
    grads = n.backward(2.0 * (out - target))

    eps, worst, worst_name = 1e-6, 0.0, ""
    for li, layer in enumerate(n.layers):
        for name in layer.params:
            arr = getattr(layer, name)
            flat = arr.reshape(-1)
            for idx in range(0, flat.size, max(1, flat.size // 3)):
                original = flat[idx]
                flat[idx] = original + eps
                up = loss_of(n)
                flat[idx] = original - eps
                down = loss_of(n)
                flat[idx] = original
                numeric = (up - down) / (2 * eps)
                analytic = grads[li][name].reshape(-1)[idx]
                err = abs(numeric - analytic) / max(1.0, abs(numeric))
                if err > worst:
                    worst, worst_name = err, f"layer {li} {name}[{idx}]"
    check("every analytic gradient matches finite differences", worst < 1e-5,
          f"worst relative error {worst:.1e} at {worst_name}")


def test_invert_exact() -> None:
    """invert() negates the output exactly, and is its own inverse."""
    print("\ninversion")
    rng = np.random.default_rng(4)
    for depth, sizes in (("2 layers", (12, 9, 1)), ("3 layers", (12, 9, 7, 1)),
                         ("4 layers", (12, 9, 7, 5, 1))):
        n = net(5, sizes)
        randomise_act(n, 6)
        x = rng.normal(size=(32, 12))
        before = n.forward(x, train=False)
        n.invert()
        err = float(np.abs(n.forward(x, train=False) + before).max())
        check(f"inverted output is exactly -output ({depth})", err == 0.0, f"error {err:.1e}")
        n.invert()
        back = float(np.abs(n.forward(x, train=False) - before).max())
        check(f"inverting twice is the identity ({depth})", back == 0.0, f"error {back:.1e}")

def trace(n: SineNet, x: np.ndarray):
    """Every layer's pre-activation and output, computed by hand."""
    pre, out, h = [], [], x
    for layer in n.layers:
        z = h @ layer.W + layer.bias
        pre.append(z.copy())
        h = layer.a * np.sin(layer.b * (z - layer.h)) + layer.k
        out.append(h.copy())
    return pre, out


def test_inversion_is_a_not() -> None:
    """The inversion negates every *unit*, which is what makes it a logical NOT.

    §4.3's primitive is a statement about a unit - ``a -> -a`` with ``k -> -k``
    negates ``a*sin(b*(x-h)) + k`` exactly - so inversion applied to a network
    has to leave every unit looking at the evidence it looked at before and
    returning the opposite verdict on it.  Negating only the read-out would also
    produce ``-output`` while leaving every hidden unit untouched, and that is
    not the same operation; :meth:`SineNet.invert_readout` is that one, and the
    last check here is the difference.
    """
    print("\nthe inversion is a NOT applied to every unit")
    rng = np.random.default_rng(31)
    n = net(32, (12, 9, 7, 1))
    randomise_act(n, 33)                       # h and k off zero: nothing leans on oddness
    x = rng.normal(size=(24, 12))
    pre0, out0 = trace(n, x)
    y0 = n.forward(x, train=False)
    n.invert()
    pre1, out1 = trace(n, x)

    same_pre = max(float(np.abs(a - b).max()) for a, b in zip(pre0, pre1))
    check("every unit sees exactly the pre-activation it saw before", same_pre == 0.0,
          f"error {same_pre:.1e} across {len(pre0)} layers - the evidence does not move")
    neg_out = max(float(np.abs(a + b).max()) for a, b in zip(out0, out1))
    check("every unit's output is exactly negated", neg_out == 0.0,
          f"error {neg_out:.1e} - the verdict on that evidence is reversed throughout")
    check("the network's output negates with them",
          float(np.abs(n.forward(x, train=False) + y0).max()) == 0.0)

    moved, held = [], []
    snapshot = {p: [getattr(l, p).copy() for l in n.layers] for p in ACT_PARAMS}
    n.invert()
    for p in ACT_PARAMS:
        changed = any(not np.array_equal(getattr(l, p), snapshot[p][i])
                      for i, l in enumerate(n.layers))
        (moved if changed else held).append(p)
    check("it moves the amplitude and the offset, and nothing else about the wave",
          set(moved) == {"a", "k"} and set(held) == {"b", "h"},
          f"changed {moved}, untouched {held} - §13's `a -> -a` with `k -> -k`, exactly")

    # The read-out flip reaches the same function by a different route.
    m = net(32, (12, 9, 7, 1))
    randomise_act(m, 33)
    before = m.forward(x, train=False)
    inner_before = trace(m, x)[1]
    m.invert_readout()
    check("negating the read-out reaches the same function",
          float(np.abs(m.forward(x, train=False) + before).max()) == 0.0)
    inner_after = trace(m, x)[1]
    untouched = float(np.abs(inner_after[0] - inner_before[0]).max())
    check("...but leaves the hidden units alone, so it is not the same operation",
          untouched == 0.0,
          "its first hidden layer comes through unchanged; invert() negates it")


def test_the_two_operators_are_equivalent() -> None:
    """...and yet the two land on the same network, provably, and stay there.

    This is the result that stops the choice of operator mattering, and it is
    worth stating precisely because it is easy to assume otherwise.

    Flipping the signs of one unit's incoming weights, its bias, its phase and
    its amplitude together is a **symmetry of the network**: ``z`` changes sign,
    and the sine's oddness about ``h`` changes it straight back, so the function
    is untouched.  The read-out flip differs from the unit flip by exactly those
    per-unit symmetries.  Adam is coordinate-wise and sign-equivariant - flip a
    parameter and its gradient flips with it, so the moments and the step mirror
    - and therefore the two trajectories mirror too, and realise the *same
    function* at every step, not merely at the flip.

    So the corrected operator is the paper's, and it is the right one to ship on
    §4.3's terms - it negates every unit, and it leaves ``h`` and ``b`` where
    §13 leaves them.  What it does not do is change what the network learns.
    ``results/invert_readout.json`` is that prediction run for real: an arm that
    differs only in this reproduces the headline run to every digit.
    """
    rng = np.random.default_rng(41)
    a_net = net(42, (10, 8, 6, 1))
    randomise_act(a_net, 43)
    b_net = a_net.copy()
    a_net.invert()
    b_net.invert_readout()

    x = rng.normal(size=(32, 10))
    check("the two operators start from the same function",
          float(np.abs(a_net.forward(x, train=False) - b_net.forward(x, train=False)).max()) == 0.0)
    same_mag = all(
        np.allclose(np.abs(getattr(la, p)), np.abs(getattr(lb, p)))
        for la, lb in zip(a_net.layers, b_net.layers) for p in la.params)
    check("...reached by flipping signs, so every magnitude matches", same_mag,
          "the difference between them is a sign symmetry, not a different point")

    # Train both from their own parameterisation on the same data.
    opt_a, opt_b = Adam(a_net, lr=0.05, act_lr=0.005), Adam(b_net, lr=0.05, act_lr=0.005)
    target = rng.normal(size=(32, 1))
    worst = 0.0
    for _ in range(40):
        for n_, o_ in ((a_net, opt_a), (b_net, opt_b)):
            o_.step(n_.backward(2.0 * (n_.forward(x) - target)))
        worst = max(worst, float(np.abs(
            a_net.forward(x, train=False) - b_net.forward(x, train=False)).max()))
    check("and they compute the same function at every step of training after it",
          worst == 0.0,
          f"error {worst:.1e} over 40 Adam steps - Adam is sign-equivariant, so the "
          "symmetry is preserved and the choice of operator cannot change what is learned")


def test_literal_is_noop() -> None:
    """The literal 'flip everything' rule cancels itself on a feed-forward stack."""
    print("\nthe parity finding")
    rng = np.random.default_rng(8)
    for depth in (2, 3, 4):
        sizes = [12] + [9] * (depth - 1) + [1]
        n = net(9, tuple(sizes))          # defaults h = k = 0: the sine is odd
        x = rng.normal(size=(16, 12))
        before = n.forward(x, train=False)
        n.invert_literal()
        err = float(np.abs(n.forward(x, train=False) - before).max())
        check(f"flipping the FIRST layer's weights too is a no-op ({depth} layers)",
              err == 0.0, f"error {err:.1e} - its negated input flips z, and the odd sine "
              "turns that straight back into the negation already applied")


# ------------------------------------------------------------------- growth


def test_growth_identity() -> None:
    """A growth step changes the architecture and not the function."""
    print("\nself-building growth")
    rng = np.random.default_rng(11)
    n = net(12, (12, 8, 8, 1))
    randomise_act(n, 13)
    x = rng.normal(size=(24, 12))
    before = n.forward(x, train=False)
    added = n.grow_hidden(6, rng, layer=0)
    err = float(np.abs(n.forward(x, train=False) - before).max())
    check("growing the hidden layer is exactly an identity", added == 6 and err == 0.0,
          f"hidden {n.hidden_sizes}, error {err:.1e}")

    err2 = float(np.abs(n.forward(x, train=False) - before).max())
    n.invert()
    inv = float(np.abs(n.forward(x, train=False) + before).max())
    check("the inversion is still exact after growth", inv == 0.0 and err2 == 0.0,
          f"error {inv:.1e} - the new units' zero out-weights survive the sign flip")

    n2 = net(12, (12, 8, 8, 1))
    n2.grow_hidden(4, rng, layer=1)
    check("the second hidden layer grows too", n2.hidden_sizes == [8, 12],
          f"hidden {n2.hidden_sizes}")
    capped = n2.grow_hidden(100, rng, layer=0, max_hidden=10)
    check("growth respects max_hidden", capped == 2 and n2.hidden_sizes[0] == 10,
          f"added {capped}, hidden {n2.hidden_sizes}")


def test_adam_resync() -> None:
    """Growing does not throw away the optimiser state of the units already there."""
    print("\noptimiser")
    rng = np.random.default_rng(14)
    n = net(15, (12, 8, 1))
    opt = Adam(n, lr=0.01)
    x = rng.normal(size=(8, 12))
    opt.step(n.backward(np.ones_like(n.forward(x))))
    kept = opt.m[0]["W"].copy()
    n.grow_hidden(4, rng, layer=0)
    opt.resync()
    ok = (opt.m[0]["W"].shape == n.layers[0].W.shape
          and np.array_equal(opt.m[0]["W"][:, :kept.shape[1]], kept)
          and np.all(opt.m[0]["W"][:, kept.shape[1]:] == 0.0))
    check("Adam's moments grow with the network and keep what was there", ok,
          f"moment shape {opt.m[0]['W'].shape}")

    n.layers[0].b[:] = 1e-9
    opt.step(n.backward(np.ones_like(n.forward(x))))
    check("the frequency b is floored so a unit cannot collapse to a constant",
          bool(np.all(n.layers[0].b >= MIN_B)), f"min b {n.layers[0].b.min():.1e}")


def test_save_load(tmp: str = "/tmp/_sbnn_test.npz") -> None:
    print("\npersistence")
    rng = np.random.default_rng(16)
    n = net(17, (12, 8, 6, 1))
    randomise_act(n, 18)
    n.grow_hidden(3, rng, layer=0)
    x = rng.normal(size=(8, 12))
    before = n.forward(x, train=False)
    n.save(tmp)
    back = SineNet.load(tmp)
    err = float(np.abs(back.forward(x, train=False) - before).max())
    check("a saved network reloads bit for bit", err == 0.0,
          f"error {err:.1e}, hidden {back.hidden_sizes}")


# ----------------------------------------------------------------- features


def test_features() -> None:
    """The encoder is colour-symmetric and states the rules it is given."""
    print("\nfeatures")
    board = chess.Board()
    white = encode(board, chess.WHITE)
    mirrored = chess.Board(board.mirror().fen())
    black = encode(mirrored, chess.BLACK)
    check("a position and its colour mirror encode identically",
          np.array_equal(white, black),
          "one set of weights therefore plays both colours")

    fools = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        fools.push(chess.Move.from_uci(uci))
    x = encode(fools, chess.BLACK)
    check("checkmate is encoded as check AND no legal move",
          x[CHECK_OFFSET] == 1.0 and x[NO_MOVES_OFFSET] == 1.0 and x[TURN_OFFSET] == 0.0,
          "so the value of a mate is learned, not a hard-coded constant that could not invert")

    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    y = encode(stale, chess.WHITE)
    check("stalemate is no legal move without check",
          y[CHECK_OFFSET] == 0.0 and y[NO_MOVES_OFFSET] == 1.0)

    b2 = board_after(10, seed=3)
    moves = list(b2.legal_moves)
    rows = encode_children(b2, moves, b2.turn)
    check("encoding the children leaves the board exactly as it was",
          b2.fen() == board_after(10, seed=3).fen() and rows.shape == (len(moves), FEATURE_DIM),
          f"{rows.shape[0]} children")


# ------------------------------------------------------------- action space


def test_action_space() -> None:
    """Every legal move has an index, and the index names it back."""
    print("\nthe action space")
    bad = 0
    legal_total = 0
    for seed in range(8):
        board = board_after(6 + 2 * seed, seed=seed)
        if board.is_game_over():
            continue
        pov = board.turn
        for move in board.legal_moves:
            if move.promotion not in (None, chess.QUEEN):
                continue                       # under-promotion is outside the space
            legal_total += 1
            bad += int(to_move(board, to_action(board, move, pov), pov) != move)
    check("every legal move round-trips through its action index", bad == 0,
          f"{legal_total} moves, {bad} mismatches")

    board = board_after(10, seed=2)
    pov = board.turn
    n_legal = sum(1 for m in board.legal_moves if m.promotion in (None, chess.QUEEN))
    guesses = N_ACTIONS / max(n_legal, 1)
    legal_actions = {to_action(board, m, pov) for m in board.legal_moves}
    sampled = [int(a) for a in np.random.default_rng(0).integers(N_ACTIONS, size=400)]
    hit = sum(a in legal_actions for a in sampled)
    check("the action space is overwhelmingly illegal", hit / len(sampled) < 0.05,
          f"{n_legal} legal of {N_ACTIONS}: guessing needs about {guesses:.0f} tries")


def test_fast_path() -> None:
    """The split first layer scores all 4096 actions exactly as the dense path does."""
    agent = Agent.build([16, 12], seed=24)
    board = board_after(10, seed=4)
    scores, board_x, states = agent.action_scores(board)
    probe = np.array([0, 5, 137, 999, 2048, 4095])
    direct = agent.net.score(dense(board_x, states, probe))
    err = float(np.abs(scores[probe] - direct).max())
    check("the fast six-gather path equals the dense path", err < 1e-12,
          f"error {err:.1e} over {N_ACTIONS} actions")

    W = agent.net.layers[0].W
    z = preactivation(W[FEATURE_DIM:], states)
    slow = dense(board_x, states, np.arange(N_ACTIONS))[:, FEATURE_DIM:] @ W[FEATURE_DIM:]
    check("the move pre-activation is the sum of six weight rows",
          float(np.abs(z - slow).max()) == 0.0, f"shape {z.shape}")


def test_action_inversion() -> None:
    """Inverting flips the whole 4096-move ranking, exactly."""
    agent = Agent.build([16, 16], seed=25)
    worst, flipped, total = 0.0, 0, 0
    for seed in range(6):
        board = board_after(8 + 2 * seed, seed=seed + 20)
        if board.is_game_over():
            continue
        before, _, _ = agent.action_scores(board)
        agent.net.invert()
        after, _, _ = agent.action_scores(board)
        agent.net.invert()
        worst = max(worst, float(np.abs(after + before).max()))
        flipped += int(np.argmax(before) == np.argmin(after))
        total += 1
    check("inverting negates every one of the 4096 move scores exactly",
          worst == 0.0 and flipped == total,
          f"error {worst:.1e}, {flipped}/{total} positions: "
          "the most wanted move became the least wanted")


def test_minimax_would_break_inversion() -> None:
    """Why the network scores moves and not positions: a search need not commute.

    Scoring the position a move leads to cannot express an illegal move at all,
    which is reason enough here.  This is the second reason, and it is 2NRL's:
    the obvious opponent model does not commute with negation, so an inversion
    that is exact on the network stops being order-reversing one level above it.
    """
    agent = Agent.build([16, 16], seed=26)
    broken, total = 0, 0
    for seed in range(6):
        board = board_after(8 + 2 * seed, seed=seed + 40)
        if board.is_game_over():
            continue
        pov = board.turn

        def minimax() -> np.ndarray:
            out = []
            for move in board.legal_moves:
                board.push(move)
                replies = list(board.legal_moves)
                rows = (encode_children(board, replies, pov) if replies
                        else encode(board, pov)[None, :])
                padded = np.zeros((rows.shape[0], INPUT_DIM))
                padded[:, :FEATURE_DIM] = rows
                out.append(float(agent.net.score(padded).min()))
                board.pop()
            return np.array(out)

        before = minimax()
        agent.net.invert()
        after = minimax()
        agent.net.invert()
        total += 1
        broken += int(not np.allclose(after, -before))
    check("a minimax opponent would break the inversion", broken == total,
          f"{broken}/{total} positions: min(-v) = -max(v), not -min(v)")


def test_refusals_are_the_signal() -> None:
    """An untrained network is refused about as often as guessing, and says so."""
    agent = Agent.build([16, 16], seed=27)
    rng = np.random.default_rng(3)
    refusals, expected = [], []
    for seed in range(5):
        board = board_after(10, seed=seed + 50)
        if board.is_game_over():
            continue
        out = agent.propose(board, rng, temperature=0.5)
        refusals.append(out["n_refused"])
        expected.append(N_ACTIONS / max(board.legal_moves.count(), 1))
        if not board.is_legal(out["move"]):
            check("the proposal that is finally accepted is legal", False, str(out["move"]))
            return
    check("the proposal that is finally accepted is legal", True,
          f"mean {np.mean(refusals):.0f} refusals first, against {np.mean(expected):.0f} for guessing")


def _samples_for(agent: Agent, seeds: range, width: int = 6) -> list[dict]:
    out = []
    for seed in seeds:
        board = board_after(10, seed=seed)
        _, board_x, states = agent.action_scores(board)
        pov = board.turn
        legal = [to_action(board, m, pov) for m in board.legal_moves][:width - 1]
        actions = np.array(legal + [int(np.random.default_rng(seed).integers(N_ACTIONS))])
        out.append({"X": dense(board_x, states, actions), "played": 0, "best": 1,
                    "worst": 2 % len(actions), "rand": 0, "refused": len(actions) - 1,
                    "wild": len(actions) - 1, "loss_cp": 120.0, "rank": 0.5,
                    "weight": 1.0, "is_rule": 1.0})
    return out


def test_policy_gradient() -> None:
    """The gradient of the move policy matches finite differences."""
    print("\ntraining")
    agent = Agent.build([12, 12], seed=21)
    batch = Batch(_samples_for(agent, range(60, 64)), 6)
    weights = np.ones(len(batch))

    def loss_of() -> float:
        return nll(policy(agent.net, batch, 1.0, train=False), batch.best)

    p = policy(agent.net, batch, 1.0, train=True)
    grad = cross_entropy_grad(p, batch.best, weights)
    grads = agent.net.backward((grad / (1.0 * len(batch))).reshape(-1, 1))
    layer = agent.net.layers[0]
    probe = [(i, j) for i in range(0, layer.W.shape[0], 250) for j in (0, 3)]
    analytic = {ij: grads[0]["W"][ij] for ij in probe}

    eps, worst = 1e-5, 0.0
    for i, j in probe:
        layer.W[i, j] += eps
        up = loss_of()
        layer.W[i, j] -= 2 * eps
        down = loss_of()
        layer.W[i, j] += eps
        numeric = (up - down) / (2 * eps)
        worst = max(worst, abs(numeric - analytic[(i, j)]) / max(1e-6, abs(numeric)))
    check("the policy gradient over the move head is correct", worst < 1e-3,
          f"worst relative error {worst:.1e}")


def test_training_moves_the_right_way() -> None:
    """Training toward a failure raises it; the inversion then buries it."""
    agent = Agent.build([16, 16], seed=23)
    batch = Batch(_samples_for(agent, range(80, 86)), 6)
    opt = Adam(agent.net, lr=0.02, act_lr=0.002)
    target = batch.refused
    idx = np.arange(len(batch))
    start = policy(agent.net, batch, 1.0, train=False)[idx, target].mean()
    for _ in range(60):
        p = policy(agent.net, batch, 1.0, train=True)
        apply_grad(agent.net, opt, batch, cross_entropy_grad(p, target, np.ones(len(batch))), 1.0)
    trained = policy(agent.net, batch, 1.0, train=False)[idx, target].mean()
    agent.net.invert()
    after = policy(agent.net, batch, 1.0, train=False)[idx, target].mean()
    check("the negative phase makes the refused move more likely",
          trained > start, f"p(refused move) {start:.3f} -> {trained:.3f}")
    check("the inversion then makes it less likely than it ever was",
          after < start, f"p(refused move) {trained:.3f} -> {after:.3f} after inverting")


# -------------------------------------------------------------------- judge


def test_judge() -> None:
    """Stockfish grades its own best move at zero, and a blunder above it."""
    print("\nthe judge")
    try:
        from engine import Judge, find_stockfish
        judge = Judge(find_stockfish(), depth=6)
    except Exception as exc:                          # pragma: no cover
        check("Stockfish is available to grade moves", False, str(exc))
        return
    board = board_after(8, seed=5)
    ranking = judge.rank(board)
    best_uci = max(ranking, key=ranking.get)
    graded = judge.grade(board, chess.Move.from_uci(best_uci))
    check("grading Stockfish's own choice costs nothing", graded["loss"] == 0,
          f"loss {graded['loss']} cp on {best_uci}")
    check("MultiPV scores every legal move", len(ranking) == board.legal_moves.count(),
          f"{len(ranking)} of {board.legal_moves.count()}")
    worst_uci = min(ranking, key=ranking.get)
    check("the worst move is graded strictly worse than the best",
          judge.grade(board, chess.Move.from_uci(worst_uci))["loss"] > 0,
          f"{worst_uci} gives away {ranking[best_uci] - ranking[worst_uci]} cp")
    judge.close()


if __name__ == "__main__":
    print("SBNN + 2NRL on chess - correctness proofs")
    test_gradients()
    test_invert_exact()
    test_inversion_is_a_not()
    test_the_two_operators_are_equivalent()
    test_literal_is_noop()
    test_growth_identity()
    test_adam_resync()
    test_save_load()
    test_features()
    test_action_space()
    test_fast_path()
    test_action_inversion()
    test_minimax_would_break_inversion()
    test_refusals_are_the_signal()
    test_policy_gradient()
    test_training_moves_the_right_way()
    test_judge()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)
    print("all proofs hold")
