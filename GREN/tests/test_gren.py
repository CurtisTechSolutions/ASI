"""GREN tests. Run: python3 -m tests.test_gren  (from GREN/)"""
import sys, os, math, random, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gren.verdict import Verdict, Outcome, normalise, signature
from gren.reasons import ReasonTaxonomy
from gren.radix import RadixGameTree
from gren.axis import Evidence, distance
from gren.oracle import build_all, CODES
from gren.explorer import Explorer
from gren import package as pkg

def test_normalisation_is_total_and_stable():
    assert normalise("error at /a/b/c.py:12:5 took 4ms") == "error at <path>:<pos> took <dur>"
    assert normalise("moved 3 squares") == "moved <n> squares"
    assert normalise(None) == "" and normalise("") == ""
    assert normalise("x") == "x"

def test_taxonomy_is_order_independent():
    """An agglomerative clusterer that depends on arrival order would make every
    downstream signature non-reproducible."""
    msgs = ["target occupied by your own piece", "path blocked at square 4",
            "leaves your king in check", "target occupied by your own piece",
            "not a legal move pattern for N"]
    parts = set()
    for seed in range(8):
        m = msgs[:]; random.Random(seed).shuffle(m)
        t = ReasonTaxonomy(); lab = {}
        for x in m: lab[x] = t.classify(Verdict(Outcome.ILLEGAL, reason=x))
        parts.add(frozenset(frozenset(k for k, v in lab.items() if v == c)
                            for c in set(lab.values())))
    assert len(parts) == 1, f"{len(parts)} different partitions across orderings"

def test_error_never_becomes_a_rule():
    """A timeout has not taught you a rule. Folding ERROR into ILLEGAL poisons
    the boundary with the oracle's own failures."""
    e = Evidence("x")
    e.observe(Verdict(Outcome.ERROR, reason="boom"))
    assert e.illegal == 0 and not e.codes
    v = Verdict(Outcome.ERROR)
    assert not v.informative and not v.legal

def test_radix_never_merges_a_terminal():
    """The one bug this structure is prone to: compression deleting the general
    classes that are the reason for using a trie."""
    t = RadixGameTree()
    t.insert(["board", "two-player", "perfect"], "board-game")
    t.insert(["board", "two-player", "perfect", "capture", "sliding"], "chess")
    t.insert(["board", "two-player", "perfect", "capture", "jump"], "checkers")
    t.insert(["puzzle", "one-player", "constraint"], "sudoku")
    node = t.lookup(["board", "two-player", "perfect"])
    assert node is not None and t.game[node] == "board-game"
    assert len(t.children[node]) == 1, "the general class has exactly one child"
    assert not t.merge_child(node), "a terminal must refuse to merge"
    before = {t.game[n] for n in t.terminals()}
    t.compress()
    assert {t.game[n] for n in t.terminals()} == before
    node = t.lookup(["board", "two-player", "perfect"])
    assert node is not None and t.game[node] == "board-game", "survives compression"

def test_radix_roundtrip_and_retrieval():
    t = RadixGameTree()
    sigs = {"a": ["p", "q", "r"], "b": ["p", "q", "s"], "c": ["z"]}
    for g, s in sigs.items(): t.insert(s, g)
    for g, s in sigs.items():
        n = t.lookup(s)
        assert n is not None and t.game[n] == g and list(t.path(n)) == s
    got = [g for _, g in t.retrieve({"p", "q"})]
    assert got[:2] == ["a", "b"] or got[:2] == ["b", "a"], got

def test_distance_is_a_metric():
    from gren.axis import distance as d
    sets = [frozenset("abc"), frozenset("abd"), frozenset("xyz"), frozenset("abcd")]
    for a in sets:
        assert d(a, a) == 0.0
        for b in sets:
            assert abs(d(a, b) - d(b, a)) < 1e-12
            for c in sets:
                assert d(a, c) <= d(a, b) + d(b, c) + 1e-12, "triangle inequality"

def test_oracles_explain_their_refusals():
    """A bare rejection is one bit; a code from R kinds is log2(R+1)."""
    rng = random.Random(0)
    for name, o in build_all().items():
        g = o.game; found = set(); legal_seen = False
        for _ in range(25):
            st = g.random_state(rng) if hasattr(g, "random_state") else g.new(seed=rng.randrange(10**6))
            for mv in o.candidates(st, rng, 8):
                v = o.probe(st, mv)
                if v.outcome == Outcome.ILLEGAL:
                    assert v.reason_code in CODES, (name, v.reason_code)
                    assert v.reason, "a refusal must say why"
                    found.add(v.reason_code)
                elif v.outcome == Outcome.LEGAL: legal_seen = True
        assert found, f"{name}: no refusal codes discovered"
        assert legal_seen, f"{name}: no legal move ever found"
        assert o.declared(), f"{name}: declared evidence should be free"

def test_better_probing_finds_more_rules():
    """The point of a probe policy: random probing only ever trips the common
    refusal, so the rare rules are never learned."""
    o = build_all()["chess"]
    found = {}
    for pn in ("random", "boundary"):
        ex = Explorer(o, policy=pn, seed=1); ex.run(budget=500, k=10)
        found[pn] = len(ex.evidence.codes)
    assert found["boundary"] >= found["random"], found

def test_discovered_similarity_matches_hand_written():
    """The acceptance test: GREN probes, is refused, and recovers the similarity
    structure CyclicCortex hard-codes."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "CyclicCortex"))
    from cortex.games import ALL
    ev = {}
    for name, o in build_all().items():
        ex = Explorer(o, policy="eig", seed=0); ex.run(budget=700, k=10)
        ev[name] = ex.evidence
    rows = []
    for x, y in itertools.combinations(ev, 2):
        dg = distance(ev[x].signature(), ev[y].signature())
        dh = 1.0 - len(ALL[x].mechanics & ALL[y].mechanics) / len(ALL[x].mechanics | ALL[y].mechanics)
        rows.append((dg, dh))
    def rk(v):
        o_ = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for p, i in enumerate(o_): r[i] = p
        return r
    a = [r[0] for r in rows]; b = [r[1] for r in rows]
    ra, rb = rk(a), rk(b); n = len(a)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = math.sqrt(sum((x-ma)**2 for x in ra)); db = math.sqrt(sum((x-mb)**2 for x in rb))
    rho = num/(da*db)
    assert rho >= 0.7, f"discovered similarity should track the hand-written sets, rho={rho:.3f}"
    # chess and checkers must be the closest pair under BOTH
    assert min(rows, key=lambda r: r[0]) == min(rows, key=lambda r: r[1])

def test_package_gates_on_confidence():
    """'Understand the game, then play it' enforced mechanically."""
    o = build_all()["sudoku"]
    ex = Explorer(o, policy="eig", seed=0); ex.run(budget=400, k=10)
    p = pkg.build(ex.evidence, {})
    assert p.mechanics and p.probes == 400
    assert 0.0 <= p.confidence <= 1.0
    thin = Evidence("mystery"); thin.declare({"category": "board"})
    assert not pkg.build(thin, {}).accepted(), "no probes means no confidence"

def test_sbnn_all_three_growths_are_identities():
    """Growth must be free: the network computes exactly what it computed
    before, the instant after growing."""
    from gren.sbnn import GrowingSBNN
    rng = random.Random(0)
    n = GrowingSBNN(nh=8, seed=1)
    feats = [{"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1)} for _ in range(120)]
    tgt = lambda f: "LEGAL" if f["a"] > 0 else "OCCUPIED_TARGET"
    for _ in range(60):
        for f in feats: n.step(f, tgt(f))
    acc = sum(1 for f in feats if max(n.predict(f), key=n.predict(f).get) == tgt(f)) / len(feats)
    assert acc > 0.9, f"a separable task must be learnable, got {acc}"
    before = [n.predict(f) for f in feats]
    for label, fn in (("hidden", lambda: n.grow_hidden(6)),
                      ("inputs", lambda: n.feature_index("unseen")),
                      ("outputs", lambda: n.label_index("SUICIDE"))):
        fn()
        after = [n.predict(f) for f in feats]
        drift = max(abs(a[k] - b[k]) for a, b in zip(before, after) for k in a)
        assert drift < 1e-12, f"{label} growth disturbed the output by {drift}"

def test_sbnn_new_output_enters_negligible_and_becomes_learnable():
    """A softmax cares only about RELATIVE logits: a fixed large negative bias is
    not enough, because trained logits can sit below any constant and the new
    class then becomes the maximum."""
    from gren.sbnn import GrowingSBNN
    rng = random.Random(0)
    n = GrowingSBNN(nh=8, seed=1)
    feats = [{"a": rng.uniform(-1, 1)} for _ in range(80)]
    for _ in range(80):
        for f in feats: n.step(f, "LEGAL" if f["a"] > 0 else "SUICIDE")
    n.label_index("REPETITION")
    share = max(n.predict(f)["REPETITION"] for f in feats)
    assert share < 1e-6, f"a new class must enter negligible, got {share}"
    tgt = lambda f: "REPETITION" if f["a"] > 0.6 else ("LEGAL" if f["a"] > 0 else "SUICIDE")
    for _ in range(80):
        for f in feats: n.step(f, tgt(f))
    acc = sum(1 for f in feats if max(n.predict(f), key=n.predict(f).get) == tgt(f)) / len(feats)
    assert acc > 0.8, f"the grown-in class must be learnable, got {acc}"

def test_sbnn_is_not_symmetry_locked_at_init():
    """Zero-weight rows AND columns make growth free and make INITIALISATION
    impossible: ah = tanh(0) = 0 zeroes every gradient and only the biases move."""
    from gren.sbnn import GrowingSBNN
    rng = random.Random(0)
    n = GrowingSBNN(nh=6, seed=2)
    feats = [{"a": rng.uniform(-1, 1)} for _ in range(60)]
    for _ in range(30):
        for f in feats: n.step(f, "LEGAL" if f["a"] > 0 else "SUICIDE")
    assert any(abs(v) > 1e-9 for row in n.W1 for v in row), "W1 never moved"
    assert any(abs(v) > 1e-9 for row in n.W2 for v in row), "W2 never moved"
    assert n.loss < 0.6, f"loss stuck at {n.loss}"

def test_sbnn_round_trip():
    from gren.sbnn import GrowingSBNN
    rng = random.Random(0)
    n = GrowingSBNN(nh=6, seed=3)
    feats = [{"a": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)} for _ in range(40)]
    for _ in range(20):
        for f in feats: n.step(f, "LEGAL" if f["a"] > 0 else "OCCUPIED_TARGET")
    m = GrowingSBNN.from_dict(n.to_dict())
    for f in feats:
        a, b = n.predict(f), m.predict(f)
        assert all(abs(a[k] - b[k]) < 1e-15 for k in a)

def test_one_network_grows_across_every_game():
    """The demonstration: a single net starts with no inputs and no outputs and
    grows to cover four games, discovering both vocabularies by probing."""
    from gren.sbnn import GrowingSBNN
    net = GrowingSBNN(nh=16, seed=0)
    assert net.ni == 0 and net.no == 0, "it starts knowing nothing"
    seen = []
    for name, o in build_all().items():
        ex = Explorer(o, policy="learned", seed=0, net=net)
        r = ex.run(budget=400, k=8)
        seen.append((net.ni, net.no))
    for (i1, o1), (i2, o2) in zip(seen, seen[1:]):
        assert i2 > i1, "each new game brings new features"
        assert o2 >= o1, "output count never shrinks"
    assert net.ni > 30 and net.no > 8, net.shape()
    assert all(c in net.outputs for c in ("LEGAL", "OCCUPIED_TARGET")), net.outputs

def test_auc_is_symmetric_and_calibrated():
    from gren.vocabulary import _auc, _signed
    assert abs(_auc([1,2,3],[1,2,3]) - 0.5) < 1e-9, "identical samples must be 0.5"
    assert _auc([4,5,6],[1,2,3]) == 1.0
    assert _auc([1,2,3],[4,5,6]) == 0.0
    assert abs(_auc([1,1,1],[1,1,1]) - 0.5) < 1e-9, "all ties must be 0.5"
    assert _signed([4,5,6],[1,2,3]) == 1.0 and _signed([1,2,3],[4,5,6]) == -1.0

def test_legality_comes_from_is_legal_not_from_why():
    """`why` is a DIAGNOSIS -- it assumes refusal and falls through to a default
    code, so reading legality off it labels every move illegal. That mistake gave
    chess a zero-size legal class and every feature a separation of 0.000."""
    import random
    from gren.oracle import build_all
    from gren import vocabulary as V
    o = build_all()["chess"]
    rows, blocks = V.sample(o, o.game, random.Random(1), states=20, k=8)
    assert rows and blocks
    legal = sum(1 for c, _ in rows if c is None)
    assert legal > 0.1 * len(rows), f"only {legal}/{len(rows)} legal -- why() misread"
    for mv_code, _ in rows: assert mv_code is None or mv_code in CODES

def test_derived_vocabulary_beats_its_own_noise_floor():
    """Every assignment must clear the separation the SAME column reaches on
    shuffled labels. Without the floor, finite samples give every column a score
    above zero and the layout is noise with a threshold on it."""
    import random
    from gren.oracle import build_all
    from gren import vocabulary as V
    o = build_all()["sudoku"]
    rows, blocks = V.sample(o, o.game, random.Random(0), states=40, k=10)
    assert len(rows) > 200
    assoc = V.associate(rows, V.columns(rows, blocks), seed=0)
    detail = V.explains(assoc)
    for key, (code, sep, floor) in detail.items():
        if code is not None: assert abs(sep) - floor >= V.MARGIN, (key, code, sep, floor)
    # sudoku's three uniqueness features must separate to three DIFFERENT codes:
    # the block I wrote as one 3-wide unit is really row, column and box.
    got = {code: k for k, (code, _, _) in detail.items()
           if k[0] == "CONSTRAINT_UNIQUE" and code}
    assert set(got) == {"CONSTRAINT_ROW", "CONSTRAINT_COL", "CONSTRAINT_BOX"}, got
    assert len({k[1] for k in got.values()}) == 3, "must be three distinct indices"

def test_alignment_never_puts_two_dims_of_one_game_in_a_slot():
    """A slot is one quantity. Two dimensions of the SAME game are by
    construction not the same quantity, so a slot holding both would be an
    aliasing bug of exactly the kind alignment exists to prevent."""
    from gren.oracle import build_all, CODES
    from gren import vocabulary as V
    lay, slots, priv, _ = V.build_aligned(build_all(), CODES, seed=0, states=60, k=12)
    assert slots, "nothing matched at all"
    for sl in slots:
        assert len(sl["dims"]) >= 2, sl
        assert len(set(sl["dims"])) == len(sl["dims"])
        for g, (b, i, sg) in sl["dims"].items(): assert sg in (1, -1), sl
    for g, keys in priv.items():
        placed = {k for sl in slots for gg, (b, i, _) in sl["dims"].items()
                  if gg == g for k in [(b, i)]}
        assert not (set(keys) & placed), f"{g}: a dim is both shared and private"

def test_export_round_trips_the_vocabulary():
    import tempfile, os
    from gren import package as pkg
    from gren.oracle import build_all, CODES
    from gren import vocabulary as V
    oc = build_all()
    # 60/12 is what the CLI exports at. Below it checkers produces fewer than
    # min_class OCCUPIED_TARGET refusals and NOTHING matches -- the floor
    # declining to align on too little evidence, which is correct behaviour.
    lay, slots, _, _ = V.build_aligned({k: oc[k] for k in ("chess", "checkers")},
                                       CODES, seed=0, states=60, k=12)
    lay = {n: {c: [list(t) for t in d] for c, d in l.items()} for n, l in lay.items()}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "h.json")
        pkg.export([], p, vocabulary=lay, slots=slots)
        doc = pkg.load(p)
    assert doc["slots"] and doc["vocabulary"]["chess"]
    for sl in doc["slots"]:
        assert len(sl["games"]) >= 2
        for g in sl["games"]: assert len(doc["vocabulary"][g][sl["name"]]) == 1
    for code, dims in doc["vocabulary"]["chess"].items():
        for b, i, sg in dims:
            assert isinstance(b, str) and isinstance(i, int) and sg in (1, -1)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"\n  {len(fns)} tests passed")
