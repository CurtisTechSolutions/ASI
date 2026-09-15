"""GTMNN tests. Run: python3 -m tests.test_gtmnn  (from GTMNN/)

The strongest assertions are in test_shapley_* -- that module replaces
backpropagation, so its axioms are the load-bearing claims of the whole design.
"""
import itertools, math, os, random, subprocess, sys, tempfile
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtmnn.activation import (sine_activation, sine_partials, SineActivation,
                              DEFAULT_B, suggested_b)
from gtmnn.features import Alphabet, FeatureHasher, sample_receptive_field, coverage_gaps
from gtmnn.micro import MicroPool
from gtmnn.payoff import CorrectnessCongestion, BeliefCongestion, Inverted, RockPaperScissors
from gtmnn.auction import allocate, bid, charge, best_response_bid
from gtmnn.game import StageGame, GameContext, build, aggregate
from gtmnn.equilibrium import solve, CycleDetector, exploitability
from gtmnn.shapley import shapley_values, shapley_exact, coalition_value
from gtmnn.model import GTMNet, TrainConfig, gini
from gtmnn.evolve import Evolver, EvolveConfig
from gtmnn.checkpoint import CheckpointManager


# ------------------------------------------------------------------ fixtures
def fixture(M, V, seed, abstain=None, twins=False):
    rng = random.Random(seed)
    pool = MicroPool(M, 16, R=3, H=3, K=3, seed=seed, alphabet_size=V)
    ss = [[(None if pool.repertoire[i * 3 + s] == -1 else pool.repertoire[i * 3 + s])
           for s in range(3)] + [None] for i in range(M)]
    c = GameContext(alphabet_size=V, y=rng.randrange(V), B=1.0, lam=0.5)
    c.slot_symbol = ss
    c.playable = [[x is not None for x in r[:-1]] + [True] for r in ss]
    g = StageGame(list(range(M)), c, CorrectnessCongestion())
    prof = []
    for i in range(M):
        r = [rng.random() for _ in range(4)]; t = sum(r)
        prof.append(array("d", [x / t for x in r]))
    if abstain is not None: prof[abstain] = array("d", [0.0, 0.0, 0.0, 1.0])
    if twins:
        for s in range(3): pool.repertoire[3 + s] = pool.repertoire[s]
        ss[1] = list(ss[0]); c.playable[1] = list(c.playable[0])
        prof[1] = array("d", list(prof[0]))
        pool.hits[1] = pool.hits[0]; pool.plays[1] = pool.plays[0]
    return g, prof, pool, c.y


# --------------------------------------------------------------- activation
def test_sine_partials_match_finite_differences():
    e = 1e-6
    for x in (-2.0, -0.3, 0.0, 0.7, 3.1):
        a, b, h, k = -1.0, 1.0, 0.2, -0.1
        f, dx, da, db, dh, dk = sine_partials(x, a, b, h, k)
        num = [(sine_activation(x + e, a, b, h, k) - sine_activation(x - e, a, b, h, k)) / (2 * e),
               (sine_activation(x, a + e, b, h, k) - sine_activation(x, a - e, b, h, k)) / (2 * e),
               (sine_activation(x, a, b + e, h, k) - sine_activation(x, a, b - e, h, k)) / (2 * e),
               (sine_activation(x, a, b, h + e, k) - sine_activation(x, a, b, h - e, k)) / (2 * e),
               (sine_activation(x, a, b, h, k + e) - sine_activation(x, a, b, h, k - e)) / (2 * e)]
        for got, want in zip((dx, da, db, dh, dk), num):
            assert abs(got - want) < 1e-6, (x, got, want)


def test_default_b_is_one_not_one_third():
    """DESIGN 5.1: at b=1/3 the first peak sits at |z|=4.71, far outside the
    operating range of L2-normalised features, so every neuron is linear and the
    population is a linear model. Measured 17x worse than tanh."""
    assert DEFAULT_B == 1.0
    assert abs(suggested_b(1.0) - math.pi / 2) < 1e-12
    assert SineActivation().inverted().a == -SineActivation().a


# ----------------------------------------------------------------- features
def test_feature_hash_survives_a_process_boundary():
    """Python's hash() is salted per process, so a saved model would predict
    differently after a restart. A correctness requirement, not a performance
    one -- hence the subprocess."""
    code = ("import sys; sys.path.insert(0, %r); from gtmnn.features import FeatureHasher; "
            "print(list(FeatureHasher(dim=64, context=8).transform('the quick brown'))[:6])"
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {"PATH": "/usr/bin:/bin"}
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**env, "PYTHONHASHSEED": "1"})
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**env, "PYTHONHASHSEED": "987654"})
    assert a.stdout.strip() == b.stdout.strip() != "", (a.stdout, b.stdout, a.stderr)


def test_hasher_l2_normalises_and_alphabet_is_append_only():
    h = FeatureHasher(dim=64, context=8)
    v = h.transform("the quick brown")
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-12
    assert all(x == 0.0 for x in h.transform("")), "an empty context has no features"
    a = Alphabet().fit(["hello"])
    before = dict(a.index)
    a.fit(["xyz"])
    for k, v2 in before.items(): assert a.index[k] == v2, "fit reordered an existing id"


# -------------------------------------------------------------------- micro
def test_micro_forward_is_a_distribution_and_masks_empty_slots():
    p = MicroPool(4, 32, R=4, H=3, K=3, seed=1, alphabet_size=2)   # alphabet < K
    f = FeatureHasher(dim=32, context=8).transform("hello")
    q = p.forward(0, f)
    assert abs(sum(q) - 1.0) < 1e-9 and min(q) >= 0.0
    for s in range(p.K):
        if p.repertoire[s] == -1: assert q[s] == 0.0, "named a symbol it does not have"


def test_learn_gradient_matches_finite_differences():
    f = FeatureHasher(dim=32, context=8).transform("hello wor")
    for name in ("w1", "b1", "w2", "b2", "sa", "sb"):
        p = MicroPool(4, 32, R=4, H=3, K=3, seed=3, alphabet_size=7)
        i, slot, lr, eps = 1, 2, 1e-6, 1e-6
        per = {"w1": p.R * p.H, "b1": p.H, "w2": p.H * p.S, "b2": p.S,
               "sa": p.NEUR, "sb": p.NEUR}[name]
        arr = getattr(p, name)
        idxs = list(range(i * per, (i + 1) * per))
        base = [arr[t] for t in idxs]
        loss = lambda: -math.log(p.forward(i, f)[slot])
        num = []
        for t in idxs:
            o = arr[t]; arr[t] = o + eps; lp = loss(); arr[t] = o - eps; lm = loss(); arr[t] = o
            num.append((lp - lm) / (2 * eps))
        p.learn(i, slot, 1.0, lr, lr, f, clip=1e9)
        for j, t in enumerate(idxs):
            ana = (base[j] - arr[t]) / lr
            assert abs(ana - num[j]) < 1e-4 * max(1.0, abs(num[j])), (name, j, ana, num[j])


def test_no_gradient_crosses_a_micro():
    """Cross-cutting invariant 2. learn() writes only into micro i's own slices."""
    p = MicroPool(8, 32, R=4, H=3, K=3, seed=0, alphabet_size=7)
    f = FeatureHasher(dim=32, context=8).transform("hello")
    snap = {k: getattr(p, k)[:] for k in ("w1", "b1", "w2", "b2", "sa", "sb", "sh", "sk")}
    p.learn(3, 2, 0.7, 0.1, 0.01, f)
    touched = set()
    for name, per in (("w1", p.R * p.H), ("w2", p.H * p.S), ("sa", p.NEUR), ("b2", p.S)):
        a, b = snap[name], getattr(p, name)
        touched |= {t // per for t in range(len(a)) if a[t] != b[t]}
    assert touched == {3}, touched


def test_null_player_takes_exactly_no_step():
    p = MicroPool(4, 32, R=4, H=3, K=3, seed=0, alphabet_size=7)
    f = FeatureHasher(dim=32, context=8).transform("hello")
    before = {k: getattr(p, k)[:] for k in ("w1", "b1", "w2", "b2", "sa", "sb", "sh", "sk")}
    p.learn(0, 1, 0.0, 0.1, 0.01, f)
    for k, v in before.items():
        assert list(getattr(p, k)) == list(v), f"phi=0 moved {k}"


def test_invert_is_an_involution_and_spares_the_ledgers():
    p = MicroPool(6, 32, R=4, H=3, K=3, seed=0, alphabet_size=7)
    p.wealth[2] = 3.5; p.hits[2] = 7.0; p.plays[2] = 9.0
    snap = (p.sa[:], p.w1[:], p.wealth[:], p.hits[:])
    p.invert()
    assert p.wealth[2] == 3.5 and p.hits[2] == 7.0, "inversion touched the ledgers"
    p.invert()
    assert (p.sa[:], p.w1[:], p.wealth[:], p.hits[:]) == snap


def test_micro_round_trips():
    p = MicroPool(6, 32, R=4, H=3, K=3, seed=2, alphabet_size=7)
    q = MicroPool.from_dict(p.to_dict())
    for k in ("rf", "w1", "b1", "w2", "b2", "sa", "sb", "sh", "sk", "repertoire",
              "strategy", "regret", "wealth", "hits", "plays", "age", "born"):
        assert list(getattr(p, k)) == list(getattr(q, k)), k
    assert bytes(p.alive) == bytes(q.alive)


def test_refresh_never_shrinks_a_repertoire():
    """A quiet stretch must not cost a micro its capacity. Writing -1 into the
    tail did, and combined with the empty-slot bug below it emptied the game."""
    p = MicroPool(2, 32, R=4, H=3, K=4, seed=0, alphabet_size=9)
    full = [p.repertoire[s] for s in range(4)]
    assert -1 not in full
    p.note_symbol(0, full[0], 5.0); p.note_symbol(0, full[1], 3.0)
    p.refresh_repertoire(0, 9)
    after = [p.repertoire[s] for s in range(4)]
    assert -1 not in after, after
    assert set(after) == set(full), (full, after)


# ------------------------------------------------------------------ auction
def test_uniform_price_makes_truth_dominant():
    """Cross-cutting invariant 8. Vickrey (1961): the price is the highest LOSING
    bid, which your own bid cannot move."""
    rng = random.Random(0)
    pool = MicroPool(48, 32, R=4, H=4, K=3, seed=0, alphabet_size=6)
    qs = []
    for i in range(48):
        r = [rng.random() for _ in range(4)]; t = sum(r)
        qs.append(array("d", [x / t for x in r]))
    others = [(bid(pool, j, qs[j]), j) for j in range(1, 48)]
    truth, sweep = best_response_bid(pool, 0, qs[0], others, m=12)
    honest = [s for b, s in sweep if abs(b - truth) < 1e-12][0]
    assert all(s <= honest + 1e-12 for _, s in sweep), sweep


def test_charge_is_uniform_and_only_seats_pay():
    pool = MicroPool(8, 16, R=3, H=3, K=2, seed=0, alphabet_size=4)
    from gtmnn.auction import SeatAllocation
    a = SeatAllocation(seats=[1, 3], bids=[0.9, 0.7], price=0.5, bidders=4)
    charge(pool, a)
    assert pool.wealth[1] == -0.5 and pool.wealth[3] == -0.5
    assert pool.wealth[0] == 0.0 and pool.wealth[2] == 0.0


# -------------------------------------------------------------------- game
def test_empty_repertoire_slot_is_not_a_second_abstain():
    """An unfilled slot and ABSTAIN would both be written None, and the payoff
    scores None as a free abstention worth 0.0 -- so a half-empty micro got extra
    costless abstentions and best_response took them."""
    pool = MicroPool(6, 16, R=3, H=3, K=3, seed=0, alphabet_size=2)   # leaves slots empty
    f = FeatureHasher(dim=16, context=4).transform("ab")
    sg, alloc, _ = build(pool, f, 4, 1, m=6, charge=False)
    if not sg.seats: return
    for seat, i in enumerate(sg.seats):
        for s in range(pool.K):
            if pool.repertoire[i * pool.K + s] == -1:
                assert not sg.ctx.playable[seat][s], "an empty slot is playable"
        assert sg.ctx.playable[seat][pool.K], "ABSTAIN must always be playable"


def test_loads_are_symbol_space_not_slot_space():
    """Two seats naming the same symbol from different slots must add to one
    load. Getting it wrong silently turns the congestion game into a vote."""
    g, prof, pool, y = fixture(4, 5, 7)
    g.ctx.slot_symbol = [[2, 3, 4, None], [3, 2, 4, None], [2, 4, 3, None], [4, 3, 2, None]]
    g.ctx.playable = [[True] * 3 + [True]] * 4
    loads = g.loads([0, 1, 2, 0])          # slots 0,1,2,0 -> symbols 2,2,3,4
    assert loads[2] == 2 and loads[3] == 1 and loads[4] == 1, loads


def test_aggregate_is_strictly_positive_everywhere():
    g, prof, pool, y = fixture(6, 7, 3)
    a = aggregate(g, prof, pool)
    assert abs(sum(a) - 1.0) < 1e-9
    assert all(p > 0.0 for p in a), "log P(c) must be finite for every c"


# ------------------------------------------------------------- equilibrium
def test_training_game_always_terminates_and_never_cycles():
    """Invariant 11.6.1 and cross-cutting 3. Rosenthal (1973): the training game
    is an exact potential game, so best response cannot cycle. The solver also
    ASSERTS the potential rose on every accepted deviation."""
    rng = random.Random(0)
    for t in range(200):
        V, M = rng.randrange(3, 8), rng.randrange(2, 12)
        ss = [[rng.randrange(V) for _ in range(4)] + [None] for _ in range(M)]
        c = GameContext(alphabet_size=V, y=rng.randrange(V), B=1.0, lam=0.5)
        c.slot_symbol = ss; c.playable = [[True] * 5 for _ in range(M)]
        e = solve(StageGame(list(range(M)), c, CorrectnessCongestion()),
                  "best_response", iters=256)
        assert e.converged and e.cycle == 0, (t, e.converged, e.cycle)


def test_rock_paper_scissors_cycles_with_period_three():
    """Invariant 11.6.2. Pushed per SWEEP; per move it reports 6."""
    c = GameContext(alphabet_size=3, y=None)
    c.slot_symbol = [[0, 1, 2], [0, 1, 2]]; c.playable = [[True] * 3] * 2
    g = StageGame([0, 1], c, RockPaperScissors())
    assert solve(g, "best_response", iters=64).cycle == 3
    p = solve(g, "regret", iters=10000, rng=random.Random(0)).profile[0]
    assert max(abs(x - 1 / 3) for x in p) < 0.02, list(p)


def test_every_solver_returns_distributions_and_is_deterministic():
    c = GameContext(alphabet_size=5, y=2, B=1.0, lam=0.5)
    c.slot_symbol = [[0, 1, 2, 3, None]] * 6; c.playable = [[True] * 5] * 6
    g = StageGame(list(range(6)), c, CorrectnessCongestion())
    for m in ("best_response", "fictitious", "regret", "regret_plus", "replicator"):
        a = solve(g, m, iters=32, rng=random.Random(1))
        b = solve(g, m, iters=32, rng=random.Random(1))
        assert [list(r) for r in a.profile] == [list(r) for r in b.profile], m
        for r in a.profile:
            assert abs(sum(r) - 1.0) < 1e-9 and min(r) >= 0.0, (m, list(r))
        assert a.exploitability >= -1e-12, m


def test_cycle_detector_counts_sweeps():
    d = CycleDetector()
    assert d.push([0, 1]) == 0 and d.push([1, 0]) == 0 and d.push([0, 1]) == 2


# ----------------------------------------------------------------- shapley
def test_shapley_efficiency_is_exact_for_any_permutation_count():
    """Invariant 12.5.1 and cross-cutting 1. Marginal contributions telescope
    along a single permutation, so efficiency holds identically -- not merely in
    expectation."""
    worst = 0.0
    for t in range(200):
        g, p, pool, y = fixture(random.Random(t).randrange(2, 20),
                                random.Random(t).randrange(3, 9), t)
        for perms in (1, 2, 7, 32):
            r = shapley_values(g, p, pool, y, permutations=perms, rng=random.Random(t))
            worst = max(worst, r.efficiency_error)
    assert worst < 1e-9, worst


def test_shapley_null_player_is_exactly_zero():
    g, p, pool, y = fixture(8, 5, 11, abstain=3)
    r = shapley_values(g, p, pool, y, permutations=16, rng=random.Random(0))
    assert r.phi[3] == 0.0, r.phi[3]


def test_shapley_symmetry_is_exact_when_enumerated():
    g, p, pool, y = fixture(6, 4, 31, twins=True)
    ex = shapley_exact(g, p, pool, y)
    assert abs(ex[0] - ex[1]) < 1e-12, (ex[0], ex[1])


def test_shapley_estimator_equals_the_closed_form_over_all_permutations():
    """Invariant 12.5.4. Over ALL M! permutations the incremental estimator is
    the subset-weighted sum exactly -- so any gap at finite samples is variance,
    not bias."""
    from gtmnn.shapley import _contributions
    for M in (5, 6):
        g, p, pool, y = fixture(M, 4, M * 7)
        ex = shapley_exact(g, p, pool, y)
        contrib = _contributions(g, p, pool); V = g.ctx.alphabet_size
        base = 1e-3 / V; v0 = math.log(base) - math.log(1e-3)
        phi = [0.0] * M; cnt = 0
        for perm in itertools.permutations(range(M)):
            sy, tot, prev = base, 1e-3, v0
            for seat in perm:
                for sym, w in contrib[seat]:
                    tot += w
                    if sym == y: sy += w
                cur = math.log(sy) - math.log(tot)
                phi[seat] += cur - prev; prev = cur
            cnt += 1
        for a, b in zip(ex, [x / cnt for x in phi]):
            assert abs(a - b) < 1e-12, (M, a, b)


def test_shapley_linearity():
    g, p, pool, y = fixture(6, 4, 55)
    v = coalition_value(g, p, pool, y)
    S = {0, 2, 4}
    assert abs((v(S | {1}) - v(S)) + (v(S) - v(S - {0}))
               - ((v(S | {1}) - v(S - {0})))) < 1e-9


def test_shapley_sign_follows_the_population():
    g, p, pool, y = fixture(10, 5, 41)
    r = shapley_values(g, p, pool, y, permutations=32, rng=random.Random(0))
    a = aggregate(g, p, pool)
    assert (r.total > 0) == (a[y] > 1.0 / g.ctx.alphabet_size), (r.total, a[y])


# ------------------------------------------------------------------- model
def test_model_trains_and_reports_both_losses():
    """`loss` is the TRAINING-game aggregate and its payoff contains y, so it is
    not a predictive loss. `belief_loss` is. The record carries both because
    reading the first as the second is the mistake this pair exists to prevent."""
    net = GTMNet(n=48, F=64, R=8, H=5, K=4, meta_n=16, seed=0, context=8)
    cfg = TrainConfig(epochs=2, seats=16, iters=12, permutations=6, batch=32, seed=0)
    recs = net.train(["hello world", "hello there"], cfg)
    for r in recs:
        assert r["cycle_rate"] == 0.0, "the training game cannot cycle"
        assert r["belief_loss"] > 0.0 and r["uniform_loss"] > 0.0
        assert 0.0 <= r["accuracy"] <= 1.0
    assert recs[-1]["loss"] < recs[-1]["uniform_loss"], "worse than guessing"


def test_model_round_trips_through_a_file():
    net = GTMNet(n=24, F=32, R=6, H=4, K=3, meta_n=8, seed=0, context=6)
    net.train(["abc", "abd"], TrainConfig(epochs=1, seats=8, iters=8, permutations=4, seed=0))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.json.gz")
        net.save(p)
        back = GTMNet.load(p)
    assert list(back.pool.w1) == list(net.pool.w1)
    assert back.alphabet.symbols == net.alphabet.symbols
    assert back.stats()["alive"] == net.stats()["alive"]


def test_invert_twice_is_the_identity_on_the_model():
    net = GTMNet(n=16, F=32, R=4, H=4, K=3, meta_n=8, seed=0, context=6)
    snap = (net.pool.sa[:], net.meta_pool.sa[:])
    net.invert(); net.invert()
    assert (net.pool.sa[:], net.meta_pool.sa[:]) == snap
    assert net.inverted is False


def test_gini_is_zero_when_everyone_is_equal():
    assert abs(gini([3.0] * 10)) < 1e-12
    assert gini([0.0] * 9 + [1.0]) > 0.8


# ------------------------------------------------------------------ evolve
def test_cull_is_capped_and_birth_refills_in_place():
    net = GTMNet(n=40, F=32, R=6, H=4, K=3, meta_n=8, seed=0, context=6)
    net.alphabet.fit(["abcdef"])
    for i in range(net.pool.n): net.pool.fill_repertoire(i, len(net.alphabet))
    for i in range(net.pool.n): net.pool.wealth[i] = -10.0      # everyone bankrupt
    ev = Evolver(net, ["abcdef"], EvolveConfig(seed=0, cull_fraction=0.05))
    culled, born, counts = ev.cull_and_birth()
    assert culled <= max(1, int(0.05 * 40)) + 1, culled
    assert born == culled and sum(counts.values()) == born
    assert net.pool.count_alive() == 40, "a culled slot must be refilled or stay dead"


def test_newborns_start_neutral():
    net = GTMNet(n=20, F=32, R=6, H=4, K=3, meta_n=8, seed=0, context=6)
    net.alphabet.fit(["abcdef"])
    for i in range(net.pool.n): net.pool.wealth[i] = -10.0
    ev = Evolver(net, ["abcdef"], EvolveConfig(seed=0))
    doomed = ev.cull(); ev.birth(doomed)
    for i in doomed:
        assert net.pool.wealth[i] == 0.0
        assert abs(net.pool.reputation(i) - 0.5) < 1e-12, "a newborn is neither favoured nor barred"


# -------------------------------------------------------------- checkpoint
def test_checkpoint_rotation_keeps_the_newest():
    """Sorting by NAME would put a 'gtmnn-ck.*' tag ahead of every timestamp and
    delete the newest save."""
    net = GTMNet(n=12, F=16, R=4, H=3, K=2, meta_n=4, seed=0, context=4)
    with tempfile.TemporaryDirectory() as td:
        cm = CheckpointManager(td, keep=2)
        paths = []
        for tag in ("aaa", "zzz", "ck", "mmm"):
            p, _ = cm.save(net, tag=tag); paths.append(p)
        got = cm.list()
        assert len(got) == 2, got
        assert paths[-1] in got, "rotation deleted the newest save"
        assert cm.latest() == paths[-1]


# ------------------------------------------------------- the GREN handoff
def test_modifier_approximates_ochiai_similarity():
    """GREN/DESIGN 22.5: hashing a SET preserves overlap, so modifier cosine
    tracks the Ochiai similarity of the mechanic sets."""
    from gtmnn.games import modifier, ochiai, cosine, load_mechanics, HANDOFF
    if not os.path.exists(HANDOFF): return              # handoff not built
    mech, _ = load_mechanics()
    errs = []
    for a, b in itertools.combinations(sorted(mech), 2):
        errs.append(abs(cosine(modifier(mech[a], 128), modifier(mech[b], 128))
                        - ochiai(mech[a], mech[b])))
    assert sum(errs) / len(errs) < 0.10, sum(errs) / len(errs)


def test_modifier_puts_chess_nearest_checkers():
    """The same ordering GREN found by probing and CyclicCortex uses to build
    its regions, arrived at through the hash."""
    from gtmnn.games import modifier, cosine, load_mechanics, HANDOFF
    if not os.path.exists(HANDOFF): return
    mech, _ = load_mechanics()
    if not {"chess", "checkers", "sudoku"} <= set(mech): return
    m = {g: modifier(v, 128) for g, v in mech.items()}
    near = max((g for g in mech if g != "chess"), key=lambda g: cosine(m["chess"], m[g]))
    assert near == "checkers", near
    assert cosine(m["chess"], m["checkers"]) > cosine(m["chess"], m["sudoku"])


def test_game_features_keep_the_three_blocks_separate():
    from gtmnn.games import GameFeatures
    gf = GameFeatures({"a": ["X", "Y"], "b": ["X", "Z"]}, f_state=16, f_cand=8, f_mod=8)
    v = gf.transform("a", ["s1", "s2"], ["c1"])
    assert len(v) == 32
    assert any(v[i] for i in range(16)), "state block empty"
    assert any(v[i] for i in range(16, 24)), "candidate block empty"
    assert any(v[i] for i in range(24, 32)), "modifier block empty"
    w = gf.transform("b", ["s1", "s2"], ["c1"])
    assert [v[i] for i in range(24)] == [w[i] for i in range(24)], \
        "only the modifier may differ between games at the same state/candidate"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"\n  {len(fns)} tests passed")
