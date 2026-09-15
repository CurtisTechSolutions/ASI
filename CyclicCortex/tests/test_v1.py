"""V1 tests. Run: python3 -m tests.test_v1  (from CyclicCortex/)"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortex.sbnn import SBNN
from cortex.vocabulary import Vocabulary
from cortex.graph import SimilarityGraph, distance
from cortex.cortex import Cortex
from cortex.games import ALL

def test_growth_identity():
    """The invariant the whole design rests on: growth changes nothing."""
    rng = random.Random(0)
    n = SBNN(6, 8, seed=1)
    xs = [[rng.uniform(-1, 1) for _ in range(6)] for _ in range(300)]
    for _ in range(40):
        for x in xs: n.step(x, valid=1.0 if x[0] > 0 else 0.0, grade=x[1])
    before = [n.predict(x) for x in xs]
    n.grow_hidden(6)
    assert all(a == b for a, b in zip(before, [n.predict(x) for x in xs])), "hidden growth"
    n.grow_inputs(4)
    after = [n.predict(x + [0.0]*4) for x in xs]
    assert all(a == b for a, b in zip(before, after)), "input growth"

def test_vocabulary_slots_are_permanent():
    v = Vocabulary()
    v.extend({"A": 3, "B": 2})
    a0 = v.slot["A"]; b0 = v.slot["B"]
    v.extend({"C": 4, "A": 9})            # A already present: must not move or resize
    assert v.slot["A"] == a0 and v.slot["B"] == b0
    assert v.width == 9 and v.order == ["A", "B", "C"]
    e = v.encode({"A": [1, 2, 3], "ZZZ": [9]})   # unknown mechanic ignored, not an error
    assert len(e) == 9 and list(e[:3]) == [1, 2, 3]

def test_metric():
    g = SimilarityGraph()
    for n, gm in ALL.items(): g.add(n, gm.mechanics)
    assert g.check_metric() == 0, "Jaccard distance must satisfy the triangle inequality"
    assert g.dist(g.index("chess"), g.index("chess")) == 0.0

def test_sudoku_is_nearer_go_than_chess():
    """A prediction of the mechanic sets, checked rather than assumed."""
    S, G, C = ALL["sudoku"].mechanics, ALL["go"].mechanics, ALL["chess"].mechanics
    assert distance(S, G) < distance(S, C)

def test_regions_and_isolation():
    c = Cortex()
    for n in ("chess", "checkers", "go", "sudoku"): c.add_game(ALL[n])
    rid = {g.name: r.id for r in c.regions for g in r.games}
    assert rid["chess"] == rid["checkers"], "movement games share a region"
    assert rid["sudoku"] != rid["chess"], "sudoku is not a movement game"
    assert c.stats()["metric_violations"] == 0
    # regions share no weights
    a, b = c.regions[0], c.regions[-1]
    assert a.net is not b.net and a.vocab is not b.vocab

def test_rules_are_correct():
    rng = random.Random(0)
    for n, g in ALL.items():
        s = g.random_state(rng) if hasattr(g, "random_state") else g.new(seed=1)
        for mv in g.legal_moves(s)[:20]:
            assert g.is_legal(s, mv), f"{n}: legal_moves returned an illegal move"

def test_learns_above_baseline():
    c = Cortex(); c.add_game(ALL["sudoku"])
    c.train(ALL["sudoku"], episodes=300, rng=random.Random(0), k=10)
    e = c.evaluate(ALL["sudoku"], n=30, rng=random.Random(1), k=10)
    assert e["legality_acc"] > e["majority_baseline"] + 0.2, e

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"\n  {len(fns)} tests passed")
