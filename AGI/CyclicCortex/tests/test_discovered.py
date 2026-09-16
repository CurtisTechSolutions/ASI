"""GREN handoff tests. Run: python3 -m tests.test_discovered  (from CyclicCortex/)

These pin the one property that makes the swap worth making: the cortex built
from what GREN MEASURED must land in the same place as the cortex built from
what I wrote down. If it does not, either the measurement or the hand-written
sets are wrong, and the disagreement is the finding.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortex.cortex import Cortex
from cortex.graph import distance
from cortex.games import ALL
from cortex import discovered as D

ORDER = ["chess", "sudoku", "checkers", "go"]

def _layout(reg, tau_new=0.60):
    c = Cortex(tau_new=tau_new)
    for n in ORDER: c.add_game(reg[n])
    return c, [sorted(g.name for g in r.games) for r in c.regions]

def test_handoff_is_present_and_complete():
    assert D.available(), "run: cd GREN && python3 -m gren.cli package --out ..."
    doc = D.load()
    assert doc["format"] == 1
    for n in ORDER:
        assert n in doc["packages"], n
        p = doc["packages"][n]
        assert p["signature"] and p["mechanics"], n
        assert p["probes"] >= 200, (n, p["probes"])

def test_every_game_is_placeable():
    """The gate is characterisation, not identification confidence. Sudoku scores
    0.35 confidence because it is 0.80 from everything GREN has seen -- which is
    confidently NOVEL, not uncertain, and novelty is a reason to found a region
    rather than a reason to refuse one."""
    games, report = D.registry(ALL)
    for n, status, ch in report:
        assert status == "discovered", (n, status, ch)
        assert isinstance(games[n], D.Discovered)

def test_discovered_mechanics_reproduce_the_regions():
    _, hand = _layout(ALL)
    _, mech = _layout(D.registry(ALL, vocabulary=False)[0])
    _, voc  = _layout(D.registry(ALL)[0])
    assert hand == mech == voc, (hand, mech, voc)
    assert ["checkers", "chess"] in hand, hand      # the shared region survives
    assert ["sudoku"] in hand and ["go"] in hand    # both outliers keep their own

def test_discovered_distance_is_still_a_metric():
    for vocab in (False, True):
        c, _ = _layout(D.registry(ALL, vocabulary=vocab)[0])
        assert c.graph.check_metric() == 0

def test_discovered_agrees_on_the_ordering_of_neighbours():
    """The absolute distances move -- GREN's signature has different cardinality
    -- but chess must still be nearest checkers, and sudoku must still be nearer
    go than either board game."""
    d = D.registry(ALL)[0]
    m = {n: d[n].mechanics for n in ORDER}
    near = lambda a: min((x for x in ORDER if x != a), key=lambda b: distance(m[a], m[b]))
    assert near("chess") == "checkers", near("chess")
    assert near("checkers") == "chess", near("checkers")
    assert near("sudoku") == "go", near("sudoku")

def test_the_wrapper_delegates_play_untouched():
    """The signature is discovered; the RULES are not negotiable."""
    d = D.registry(ALL)[0]
    rng = random.Random(0)
    for n in ORDER:
        g, w = ALL[n], d[n]
        s = g.new(seed=3)
        assert w.name == n
        assert [str(m) for m in w.legal_moves(s)] == [str(m) for m in g.legal_moves(s)]
        for mv in g.candidates(s, random.Random(1), 8):
            assert w.is_legal(s, mv) == g.is_legal(s, mv), (n, mv)

def test_shared_slots_hold_one_quantity():
    """A slot matched across games is ONE input wide, so two games can never
    disagree about its width and no game's tail is silently truncated."""
    doc = D.load()
    assert doc["slots"], "handoff carries no aligned slots"
    d = D.registry(ALL)[0]
    for sl in doc["slots"]:
        assert len(sl["games"]) >= 2, sl
        for g in sl["games"]:
            assert d[g].spec[sl["name"]] == 1, (g, sl["name"])
    for n in ORDER:
        rng = random.Random(0); s = ALL[n].new(seed=1)
        for mv in ALL[n].candidates(s, rng, 6):
            f = d[n].generalise(s, mv)
            for code, vec in f.items(): assert len(vec) == d[n].spec[code], (n, code)

def test_shared_slots_are_signed_consistently():
    """The alignment claims two dimensions are the same quantity up to sign. If
    that holds, the values written into one slot by two games must correlate --
    a slot whose members anti-correlate is the aliasing bug this replaced."""
    doc = D.load(); d = D.registry(ALL)[0]
    lay = doc["vocabulary"]
    for sl in doc["slots"]:
        for g in sl["games"]:
            b, i, sg = lay[g][sl["name"]][0]
            assert sg in (1, -1) and isinstance(i, int)
            assert b in ALL[g].spec, (g, b)
            assert i < ALL[g].spec[b], (g, b, i)

def test_encoding_routes_through_the_admitted_adapter():
    """Passing the hand-written adapter to a discovered cortex must NOT silently
    encode an all-zero input."""
    d = D.registry(ALL)[0]
    c = Cortex(); c.add_game(d["chess"])
    r = c.region_for(d["chess"])
    s = ALL["chess"].new(seed=1)
    mv = ALL["chess"].legal_moves(s)[0]
    x_wrapped = c.encode(r, d["chess"], s, mv)
    x_raw     = c.encode(r, ALL["chess"], s, mv)
    assert list(x_wrapped) == list(x_raw)
    assert any(v != 0.0 for v in x_raw), "encoded to all zeros"

def test_a_discovered_cortex_still_learns():
    d = D.registry(ALL)[0]
    c = Cortex(); c.add_game(d["sudoku"])
    c.train(d["sudoku"], episodes=300, rng=random.Random(0), k=10)
    e = c.evaluate(d["sudoku"], n=30, rng=random.Random(1), k=10)
    assert e["legality_acc"] > e["majority_baseline"] + 0.2, e

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"\n  {len(fns)} tests passed")
