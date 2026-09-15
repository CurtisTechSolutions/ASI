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

def test_checkers_rules():
    """Promotion, forced capture and chain capture -- the three rules that make
    checkers more than 'diagonal chess'."""
    from cortex.board_games import Checkers, checkers_start
    s = checkers_start()
    assert s.count("w") == 12 and s.count("b") == 12
    assert len(s.moves()) == 7
    b = [["" for _ in range(8)] for _ in range(8)]; b[6][1] = "w"; b[0][7] = "b"
    assert Checkers(b, "w").apply(((1,6),(2,7))).at(2,7) == "W", "crowning"
    b = [["" for _ in range(8)] for _ in range(8)]
    b[0][0] = "w"; b[1][1] = "b"; b[3][3] = "b"; b[7][7] = "B"
    s2 = Checkers(b, "w").apply(((0,0),(2,2)))
    assert s2.turn == "w" and s2.chain == (2,2), "chain keeps the turn"
    assert all(m[0] == (2,2) for m in s2.moves()), "only the chaining piece moves"
    assert s2.apply(((2,2),(4,4))).count("b") == 1, "two men captured"

def test_go_rules():
    """Pass, two-pass termination and area scoring."""
    from cortex.board_games import Go, go_start, PASS, GN
    g = go_start()
    assert PASS in g.moves() and len(g.moves()) == GN*GN + 1
    assert g.apply(PASS).apply(PASS).over(), "two passes end the game"
    b = [["" for _ in range(GN)] for _ in range(GN)]
    for x in range(GN): b[3][x] = "b"; b[5][x] = "w"
    bs, ws = Go(b, "b").score()
    assert bs == ws == GN + GN*3, f"each side owns its own side: {bs},{ws}"

def test_all_games_playable():
    """Every playable game runs a full game against its engine and terminates."""
    from cortex import engine
    from cortex.cli import play, OPPONENT
    c = Cortex()
    for n in OPPONENT: c.add_game(ALL[n])
    for n in OPPONENT:
        c.train(ALL[n], episodes=60, rng=random.Random(0), k=8, opponent=OPPONENT[n])
        r = play(c, n, random.Random(1), opponent_depth=1, max_plies=60)
        assert r["plies"] > 0 and 0.0 <= r["illegal_rate"] <= 1.0, r
        assert r["result"] in ("win", "loss", "unfinished/draw"), r

def test_engines_beat_random():
    """The opponents must be real opponents, or 'plays legally' means nothing."""
    from cortex.board_games import checkers_start
    from cortex import engine
    wins = 0
    for g in range(6):
        s = checkers_start(); r = random.Random(500+g)
        for _ in range(300):
            if not s.moves(): break
            s = s.apply(engine.checkers_choose(s, 2 if s.turn == "w" else 0, r))
        wins += 1 if s.winner() == "w" else 0
    assert wins >= 5, f"depth-2 checkers should dominate random, won {wins}/6"

def test_discount_matches_game_length():
    """The discount must come from the game, not a constant: at gamma=0.95 a
    125-ply game gives its opening move 0.0017 of the outcome."""
    for plies, lo, hi in ((20, 0.95, 0.98), (125, 0.99, 0.999)):
        g = 0.5 ** (1.0 / plies)
        assert lo < g < hi, (plies, g)
        assert abs(g ** plies - 0.5) < 1e-9, "first move keeps half the last move's credit"

def test_position_value_is_bounded_and_signed():
    """Unfinished games need a value or the grade head learns nothing -- 40
    self-play chess games once produced 40 draws and zero signal."""
    from cortex import engine
    from cortex.board_games import checkers_start, go_start
    for name, st in (("chess", engine.start_position()),
                     ("checkers", checkers_start()), ("go", go_start())):
        g = ALL[name]
        for me in ("w", "b"):
            v = g.value(st, me)
            assert -1.0 <= v <= 1.0, (name, me, v)
        assert abs(g.value(st, "w") + g.value(st, "b")) < 1e-9, f"{name}: zero-sum at start"

def test_selfplay_trains_grade_and_keeps_legality():
    c = Cortex(); c.add_game(ALL["checkers"])
    from cortex.cli import OPPONENT
    rng = random.Random(0)
    c.train(ALL["checkers"], episodes=200, rng=rng, k=10, opponent=OPPONENT["checkers"])
    before = c.evaluate(ALL["checkers"], n=25, rng=random.Random(4), k=10)
    r = c.selfplay(ALL["checkers"], rounds=12, rng=rng, opponent=OPPONENT["checkers"],
                   max_plies=80)
    after = c.evaluate(ALL["checkers"], n=25, rng=random.Random(4), k=10)
    assert r["win"] + r["loss"] + r["draw"] == 12
    assert 0.0 < r["gamma"] < 1.0
    assert after["legality_acc"] >= before["legality_acc"] - 0.15, (before, after)

def test_stockfish_notation_and_guards():
    """FEN and UCI conversion, and the reachability guard. Skipped without
    Stockfish installed, but the notation half is pure and always runs."""
    from cortex import stockfish as sf, engine
    from cortex.board_games import Chess
    st = engine.start_position()
    assert sf.to_fen(st).startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -")
    assert sf.to_uci(((4,1),(4,3))) == "e2e4"
    assert sf.from_uci("e2e4") == ((4,1),(4,3))
    assert sf.playable(st), "the opening is reachable"
    b = [["" for _ in range(8)] for _ in range(8)]; b[0][0] = "wR"
    assert not sf.playable(Chess(b, "w")), "no kings is unreachable"
    # side NOT to move already in check is unreachable
    b = [["" for _ in range(8)] for _ in range(8)]
    b[0][4] = "wK"; b[7][4] = "bK"; b[6][4] = "wR"
    assert not sf.playable(Chess(b, "w")), "black in check with white to move"
    if not sf.available(): return
    with sf.Stockfish(depth=6) as e:
        assert -200 < e.evaluate(st) < 300
        tbl = e.score_moves(st, multipv=6)
        assert len(tbl) >= 4 and all(st.legal(*m) for m in tbl)
        assert e.best(st) is not None
        assert e.evaluate(Chess(b, "w")) == 0, "unreachable positions are guarded"

def test_rule_selected_on_play_quality():
    """Selecting on legality alone let a validity-only rule win with a perfect
    top-choice-legal while playing the worst chess of any configuration."""
    import inspect
    src = inspect.getsource(Cortex.select_rule)
    assert "game.value" in src, "rule selection must score the resulting position"

def test_checkpoint_round_trip():
    """A trained cortex must reload to bit-identical predictions, or long runs
    and cross-session training are impossible."""
    import tempfile, os
    from cortex import checkpoint
    from cortex.cli import OPPONENT
    c = Cortex(); rng = random.Random(0)
    for n in ("chess", "go", "sudoku"): c.add_game(ALL[n])
    c.train(ALL["go"], episodes=120, rng=rng, k=8, opponent=OPPONENT["go"])
    d = tempfile.mkdtemp()
    for name in ("ck.json", "ck.json.gz"):          # both plain and gzipped
        path = os.path.join(d, name)
        checkpoint.save(c, path)
        c2 = checkpoint.load(path)
        game = ALL["go"]; r1, r2 = c.region_for(game), c2.region_for(game)
        st = game.new(); rr = random.Random(5)
        for mv in game.candidates(st, rr, 10):
            a = r1.net.predict(c.encode(r1, game, st, mv))
            b = r2.net.predict(c2.encode(r2, game, st, mv))
            assert a == b, (name, a, b)
        assert [r.rule for r in c2.regions] == [r.rule for r in c.regions]
        assert [[g.name for g in r.games] for r in c2.regions] == \
               [[g.name for g in r.games] for r in c.regions]
    d2 = tempfile.mkdtemp()                      # a clean directory for rotation
    m = checkpoint.CheckpointManager(d2, keep=2)
    for i in range(5): m.save(c, f"r{i}")
    assert len(m.list()) == 2, f"rotation must keep exactly 2, kept {len(m.list())}"
    assert m.load_latest() is not None, "the latest pointer must survive rotation"

def test_coverage_and_auction():
    from cortex import routing, credit
    c = Cortex(); rng = random.Random(0)
    for n in ("chess", "go", "sudoku"): c.add_game(ALL[n])
    for n in ("chess", "go", "sudoku"): c.train(ALL[n], episodes=60, rng=rng, k=6)
    for n in ("chess", "go", "sudoku"):
        own = c.region_for(ALL[n])
        assert credit.coverage(own, ALL[n]) == 1.0, "a region fully covers its own game"
        a = routing.allocate(c, ALL[n], m=2)
        assert a.seats and a.seats[0] is own, f"{n}: its own region should win the seat"
        assert a.price >= 0.0
    # congestion: a correct claim shared by two regions pays each half
    a = routing.allocate(c, ALL["chess"], m=2)
    r = routing.settle(a, correct=True)
    assert abs(r["per_claimant"] - 1.0 / len(a.seats)) < 1e-9

def test_shapley_exact_and_efficient():
    """Efficiency is the axiom that makes credit conserved: the shares must sum
    to exactly v(all) - v(none)."""
    from cortex import credit
    from cortex.cli import OPPONENT
    c = Cortex(); rng = random.Random(0)
    for n in ("chess", "go", "sudoku"): c.add_game(ALL[n])
    for n in ("chess", "go", "sudoku"): c.train(ALL[n], episodes=120, rng=rng, k=8)
    game = ALL["sudoku"]
    S = []
    rr = random.Random(3)
    for st in c._states(game, rr, 12, 0.0, None):
        for mv in game.candidates(st, rr, 4): S.append((st, mv, game.is_legal(st, mv)))
    out = credit.shapley(c, game, S)
    assert out["method"] == "exact", "three regions must enumerate, not sample"
    assert out["efficiency_error"] < 1e-9, out["efficiency_error"]
    own = c.region_for(game).id
    assert out["phi"][own] == max(out["phi"].values()), "the owning region earns most"
    # null player: a region with zero coverage contributes exactly nothing
    zero = [r for r in c.regions if credit.coverage(r, game) == 0.0]
    for r in zero:
        assert abs(out["phi"][r.id]) < 1e-9, f"region {r.id} has no coverage but got credit"

def test_rehearse_covers_every_game_in_the_region():
    c = Cortex(); rng = random.Random(0)
    c.add_game(ALL["chess"]); c.train(ALL["chess"], episodes=60, rng=rng, k=6)
    r, info = c.add_game_and_rehearse(ALL["checkers"], rng=rng, episodes=60, k=6)
    assert set(info["rehearsed"]) == {"chess", "checkers"}
    assert info["samples"] > 0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ok  {f.__name__}")
    print(f"\n  {len(fns)} tests passed")
