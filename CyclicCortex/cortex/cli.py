"""CyclicCortex V1.

  python3 -m cortex.cli demo        build the cortex, train, play chess, add sudoku
  python3 -m cortex.cli map         the similarity graph and its regions
  python3 -m cortex.cli play        play chess against the engine
  python3 -m cortex.cli sudoku      solve a puzzle with the sudoku region
"""
import argparse, random, sys
from cortex.cortex import Cortex
from cortex.games import ALL
from cortex.graph import distance
from cortex import engine

def build(names, tau_new=0.60, seed=0, nh=24):
    c = Cortex(tau_new=tau_new, seed=seed, nh=nh)
    for n in names: c.add_game(ALL[n])
    return c

def cmd_map(a):
    c = build(a.games)
    print(f"\nSimilarity graph -- distance IS similarity (Jaccard over mechanics)\n")
    names = c.graph.names
    print("       " + "".join(f"{n[:8]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:6]:>6}" + "".join(f"{c.graph.dist(i,j):>10.3f}" for j in range(len(names))))
    print(f"\n  triangle-inequality violations: {c.graph.check_metric()}")
    print(f"\nRegions\n")
    for r in c.regions:
        print(f"  region {r.id}: {[g.name for g in r.games]}")
        print(f"    vocabulary {r.vocab.width} inputs over {len(r.vocab.order)} mechanics")
        print(f"    network    {r.shape()}")
    print("\nRouting log")
    for e in c.log: print("   ", e)

def play_chess(c, rng, opponent_depth=1, max_plies=60, verbose=False):
    """The cortex plays white. Its own ranking picks the move -- if that move is
    illegal it is COUNTED and the best legal alternative is substituted, so the
    game continues and the illegal rate is measured rather than hidden."""
    g = ALL["chess"]; s = engine.start_position()
    illegal = tried = 0
    for ply in range(max_plies):
        if not engine.legal_moves(s): break
        if s.turn == "w":
            mv = c.choose(g, s, rng, k=20); tried += 1
            if mv is None or not s.legal(*mv):
                illegal += 1
                r = c.region_for(g); best, bs = None, -1e9
                for m in engine.legal_moves(s):
                    pv, gr = r.net.predict(c.encode(r, g, s, m))
                    sc = pv * (0.5 + 0.5*gr)
                    if sc > bs: best, bs = m, sc
                mv = best
        else:
            mv = engine.choose(s, depth=opponent_depth, rng=rng)
        if mv is None: break
        s = engine.apply_move(s, mv)
        if verbose and ply < 6: print(f"    ply {ply}: {mv}")
    mat = engine.material(s, "w")
    return {"plies": ply+1, "illegal_rate": illegal/max(1,tried),
            "material": mat, "in_check": s.in_check()}

def cmd_play(a):
    c = build(["chess"]); rng = random.Random(a.seed)
    print("\n  training the movement region on chess...")
    print("   ", c.train(ALL["chess"], episodes=a.episodes, rng=rng, k=12))
    print("   ", c.evaluate(ALL["chess"], n=50, rng=random.Random(7), k=12))
    print(f"\n  playing {a.games_n} games vs the engine (depth {a.depth})\n")
    for i in range(a.games_n):
        print("   ", play_chess(c, random.Random(100+i), a.depth, a.plies))

def cmd_sudoku(a):
    c = build(["sudoku"]); rng = random.Random(a.seed)
    print("\n  training the placement region on sudoku...")
    print("   ", c.train(ALL["sudoku"], episodes=a.episodes, rng=rng, k=12))
    print("   ", c.evaluate(ALL["sudoku"], n=40, rng=random.Random(7), k=12))
    g = ALL["sudoku"]
    print("\n  solving puzzles by the network's own ranking\n")
    for i in range(a.games_n):
        s = g.new(seed=200+i); start = s.empties(); bad = 0; placed = 0
        for _ in range(start):
            mv = c.choose(g, s, rng, k=20)
            if mv is None: break
            if not g.is_legal(s, mv):
                bad += 1
                legal = g.legal_moves(s)
                if not legal: break
                r = c.region_for(g)
                mv = max(legal, key=lambda m: (lambda t: t[0]*(0.5+0.5*t[1]))(
                    r.net.predict(c.encode(r, g, s, m))))
            s = g.apply(mv); placed += 1
        print(f"    puzzle {i}: filled {placed}/{start}, solved={s.solved()}, "
              f"illegal picks {bad}")

def cmd_demo(a):
    rng = random.Random(0)
    print("\n=== 1. chess alone ===")
    c = build(["chess"])
    print("   ", c.train(ALL["chess"], episodes=a.episodes, rng=rng, k=12))
    print("   ", c.evaluate(ALL["chess"], n=50, rng=random.Random(7), k=12))
    print("\n    vs engine depth 0 (random control):", play_chess(c, random.Random(1), 0))
    print("    vs engine depth 1                 :", play_chess(c, random.Random(1), 1))

    print("\n=== 2. add sudoku ===")
    c.add_game(ALL["sudoku"])
    for e in c.log[-1:]: print("   ", e)
    print("   ", c.train(ALL["sudoku"], episodes=a.episodes, rng=rng, k=12))
    print("   ", c.evaluate(ALL["sudoku"], n=40, rng=random.Random(7), k=12))
    print("\n    chess after sudoku joined:",
          c.evaluate(ALL["chess"], n=50, rng=random.Random(7), k=12))

    print("\n=== 3. the whole cortex ===")
    c2 = build(["chess", "checkers", "go", "sudoku"])
    for e in c2.log: print("   ", e)
    st = c2.stats()
    print(f"\n    regions {st['regions']}, metric violations {st['metric_violations']}")
    for d in st["detail"]: print("    ", d)

def main(argv=None):
    p = argparse.ArgumentParser(prog="cortex")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("demo", cmd_demo), ("map", cmd_map),
                     ("play", cmd_play), ("sudoku", cmd_sudoku)):
        q = sub.add_parser(name); q.set_defaults(fn=fn)
        q.add_argument("--episodes", type=int, default=400)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--games-n", type=int, default=3)
        q.add_argument("--depth", type=int, default=1)
        q.add_argument("--plies", type=int, default=60)
        q.add_argument("--games", nargs="*", default=["chess","checkers","go","sudoku"])
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__": main()
