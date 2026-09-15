"""CyclicCortex V1.

  python3 -m cortex.cli demo        build the cortex, train, play chess, add sudoku
  python3 -m cortex.cli map         the similarity graph and its regions
  python3 -m cortex.cli play        play chess against the engine
  python3 -m cortex.cli sudoku      solve a puzzle with the sudoku region
"""
import argparse, random, sys
from cortex import checkpoint
from cortex.cortex import Cortex
from cortex.games import ALL
from cortex.graph import distance
from cortex import engine

def registry(a=None):
    """The game adapters to build from.

    `--discovered` swaps in GREN's measured signature for the hand-written
    `mechanics` frozenset, and `--discovered-vocab` also swaps the input
    vocabulary for the one GREN derived. Default is off so every earlier number
    in this repo stays reproducible.
    """
    if a is None or not getattr(a, "discovered", False): return ALL, []
    from cortex import discovered as disc
    return disc.registry(ALL, vocabulary=getattr(a, "discovered_vocab", False))

def build(names, tau_new=0.60, seed=0, nh=24, a=None):
    reg, _ = registry(a)
    c = Cortex(tau_new=tau_new, seed=seed, nh=nh)
    for n in names: c.add_game(reg[n])
    return c

def cmd_map(a):
    c = build(a.games, a=a)
    reg, rep = registry(a)
    if rep:
        print("\nMechanics from GREN's handoff (cortex/discovered.py)\n")
        for n, st, ch in rep:
            print(f"  {n:>9}: {st}" + (f"   characterisation {ch:.3f}" if ch else ""))
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

_SF = {}
def stockfish_opponent(skill=5, depth=6):
    """Stockfish as the chess opponent, throttled by Skill Level so the cortex
    has something it can measure progress against. Falls back to the built-in
    engine when Stockfish is not installed."""
    from cortex import stockfish as sfmod
    if not sfmod.available(): return None
    if "sf" not in _SF:
        _SF["sf"] = sfmod.Stockfish(depth=depth)
        _SF["sf"].set_skill(skill)
    sf = _SF["sf"]
    return lambda s, d, r: sf.best(s, depth=max(1, d))

OPPONENT = {
    "chess":    lambda s, d, r: engine.choose(s, depth=d, rng=r),
    "checkers": lambda s, d, r: engine.checkers_choose(s, d, r),
    "go":       lambda s, d, r: engine.go_choose(s, d, r),
}
CORTEX_SIDE = {"chess": "w", "checkers": "w", "go": "b"}

def play(c, name, rng, opponent_depth=2, max_plies=300):
    """One full game: the cortex on one side, the engine on the other.

    The cortex's own ranking picks the move. If that move is ILLEGAL it is
    counted and the best legal alternative is substituted, so the game continues
    and the illegal rate is measured rather than hidden -- which is the number
    that says what the network learned, more than the result does.
    """
    g = ALL[name]; s = g.new(); me = CORTEX_SIDE[name]
    illegal = tried = 0; plies = 0
    for plies in range(1, max_plies+1):
        if g.terminal(s): break
        legal = g.legal_moves(s)
        if not legal: break
        if s.turn == me:
            mv = c.choose(g, s, rng, k=24); tried += 1
            if mv is None or not g.is_legal(s, mv):
                illegal += 1
                r = c.region_for(g)
                mv = max(legal, key=lambda m: r.net.predict(c.encode(r, g, s, m))[1])
        else:
            mv = OPPONENT[name](s, opponent_depth, rng)
            if mv is None: break
        s = g.apply(s, mv)
    w = g.winner(s) if hasattr(g, "winner") else None
    out = {"game": name, "plies": plies, "illegal_rate": round(illegal/max(1,tried), 3),
           "result": "win" if w == me else ("loss" if w else "unfinished/draw")}
    if name == "go": out["score"] = s.score()
    if name == "chess": out["material"] = engine.material(s, me)
    if name == "checkers": out["pieces"] = (s.count("w"), s.count("b"))
    return out

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
    """Train each playable game's region, then play it against its engine."""
    names = [n for n in a.games if n in OPPONENT]
    c = checkpoint.load(a.load) if a.load else build(names)
    rng = random.Random(a.seed)
    for n in names:
        print(f"\n  === {n} ===")
        opp = OPPONENT[n]
        print("    train   ", c.train(ALL[n], episodes=a.episodes, rng=rng, k=12,
                                      opponent=opp))
        print("    selfplay", c.selfplay(ALL[n], rounds=a.rounds, rng=rng,
                                         opponent=opp, opponent_depth=1,
                                         max_plies=a.plies))
        print("    eval    ", c.evaluate(ALL[n], n=40, rng=random.Random(7), k=12,
                                         opponent=opp))
        for i in range(a.games_n):
            print("    vs engine", play(c, n, random.Random(100+i), a.depth, a.plies))
    if a.save: print("\n  saved:", checkpoint.save(c, a.save), "bytes ->", a.save)

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
                mv = max(legal, key=lambda m: r.net.predict(c.encode(r, g, s, m))[1])
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

def cmd_transfer(a):
    """Does one region's learning reach another's game? Regions share no
    weights, so an ensemble is the only possible channel."""
    from cortex import routing, credit
    c = build(a.games); rng = random.Random(a.seed)
    for n in a.games: c.train(ALL[n], episodes=a.episodes, rng=rng, k=12,
                              opponent=OPPONENT.get(n))
    def samples(game, n=40):
        rr = random.Random(7); out = []
        for st in c._states(game, rr, n, 0.5, OPPONENT.get(game.name)):
            for mv in game.candidates(st, rr, 6): out.append((st, mv, game.is_legal(st, mv)))
        return out
    def acc(game, regions, S):
        return sum(1 for st, mv, l in S
                   if (credit.ensemble(regions, c, game, st, mv)[0] >= 0.5) == l) / len(S)
    print(f"\n  {'game':>9} {'own region':>11} {'auction top-2':>14} {'all regions':>12} {'baseline':>9}")
    for n in a.games:
        game = ALL[n]; S = samples(game)
        base = max(sum(1 for _,_,l in S if l), sum(1 for _,_,l in S if not l))/len(S)
        own = [c.region_for(game)]
        top2 = routing.allocate(c, game, m=2).seats
        alln = [r for r in c.regions if r.net is not None]
        print(f"  {n:>9} {acc(game,own,S):>11.3f} {acc(game,top2,S):>14.3f} "
              f"{acc(game,alln,S):>12.3f} {base:>9.3f}")
    if a.save: print("\n  saved:", checkpoint.save(c, a.save), "bytes ->", a.save)

def cmd_credit(a):
    """Shapley credit over regions, exact where the region count allows."""
    from cortex import credit, routing
    c = checkpoint.load(a.load) if a.load else build(a.games)
    rng = random.Random(a.seed)
    if not a.load:
        for n in a.games: c.train(ALL[n], episodes=a.episodes, rng=rng, k=12,
                                  opponent=OPPONENT.get(n))
    print(f"\n  coverage (which regions can encode which game)\n")
    print("  " + " "*28 + "".join(f"{g:>10}" for g in a.games))
    for r in c.regions:
        nm = ",".join(g.name for g in r.games)
        print(f"  {nm:>28}" + "".join(f"{credit.coverage(r, c.adapt(ALL[g])):>10.2f}" for g in a.games))
    print("\n  shapley\n")
    for n in a.games:
        game = ALL[n]; rr = random.Random(3); S = []
        for st in c._states(game, rr, 20, 0.5, OPPONENT.get(n)):
            for mv in game.candidates(st, rr, 4): S.append((st, mv, game.is_legal(st, mv)))
        out = credit.shapley(c, game, S)
        phis = " ".join(f"R{k}={v:+.3f}" for k, v in sorted(out["phi"].items()))
        print(f"  {n:>9}  {phis}   total={out['total']:+.3f}  "
              f"eff.err={out['efficiency_error']:.1e}  {out['method']}")
        print(f"  {'':>9}  auction: {routing.allocate(c, game, m=2).to_dict()}")

def cmd_discovered(a):
    """Compare the cortex built from GREN's measurement against the one built
    from the hand-written mechanics, and measure what the shared vocabulary is
    actually worth."""
    import statistics as stat
    from cortex import discovered as disc
    from cortex.graph import distance

    games, report = disc.registry(ALL)
    mech_only, _ = disc.registry(ALL, vocabulary=False)
    doc = disc.load()
    print("\nGREN handoff\n")
    for n, st, ch in report:
        p = doc["packages"][n]
        print(f"  {n:>9}: {st:<12} {p['probes']} probes, {len(p['signature'])} tokens, "
              f"characterise {ch:.3f}, identify {p['confidence']:.3f}")

    print("\nRegions\n")
    print(f"  {'regime':<22}{'layout':<44}{'metric violations':>18}")
    for label, reg in (("hand-written", ALL), ("discovered mechanics", mech_only),
                       ("+ derived vocabulary", games)):
        c = Cortex(tau_new=0.60)
        for n in a.games: c.add_game(reg[n])
        lay = " | ".join("+".join(sorted(g.name for g in r.games)) for r in c.regions)
        print(f"  {label:<22}{lay:<44}{c.graph.check_metric():>18}")

    print("\nInput slots matched across games by legality signature\n")
    for sl in doc["slots"]:
        print(f"  {sl['name']:<20} strength {sl['strength']:.2f}  "
              f"{', '.join(sl['games'])}")
        for g in sl["games"]:
            b, i, sg = doc["vocabulary"][g][sl["name"]][0]
            print(f"      {g:>9}  {'+' if sg > 0 else '-'}{b}[{i}]")

    if a.games_n < 1: return
    print("\nWhat the shared vocabulary is worth: train chess, evaluate checkers\n")
    print("  The untrained network is the WRONG baseline -- training chess alone moves")
    print("  the region's shared hidden layer and bias, and that takes checkers to its")
    print("  majority class with no transfer at all. The baseline is that majority.\n")
    print(f"  {'regime':<24}{'shared':>8}{'majority':>10}{'after chess':>13}{'gain':>9}")
    for label, reg in (("hand-written", ALL), ("GREN, aligned slots", games)):
        acc, nsl, maj = [], 0, 0.0
        for seed in range(a.games_n):
            c = Cortex(tau_new=0.60, seed=seed)
            c.add_game(reg["chess"]); c.add_game(reg["checkers"])
            r = c.regions[0]
            nsl = sum(r.vocab.slot[k][1]
                      for k in set(reg["chess"].spec) & set(reg["checkers"].spec))
            c.train(reg["chess"], episodes=a.episodes, rng=random.Random(seed), k=12)
            e = c.evaluate(reg["checkers"], n=60, rng=random.Random(99), k=12)
            acc.append(e["legality_acc"]); maj = e["majority_baseline"]
        m = stat.mean(acc)
        print(f"  {label:<24}{nsl:>8}{maj:>10.3f}{m:>9.3f}±{stat.pstdev(acc):.3f}"
              f"{m - maj:>+9.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cortex")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("demo", cmd_demo), ("map", cmd_map),
                     ("play", cmd_play), ("sudoku", cmd_sudoku),
                     ("transfer", cmd_transfer), ("credit", cmd_credit),
                     ("discovered", cmd_discovered)):
        q = sub.add_parser(name); q.set_defaults(fn=fn)
        q.add_argument("--episodes", type=int, default=400)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--games-n", type=int, default=3)
        q.add_argument("--depth", type=int, default=1)
        q.add_argument("--plies", type=int, default=60)
        q.add_argument("--rounds", type=int, default=40)
        q.add_argument("--save", default=None, help="write a checkpoint here")
        q.add_argument("--load", default=None, help="resume from a checkpoint")
        q.add_argument("--discovered", action="store_true",
                       help="use GREN's measured signature as the mechanics")
        q.add_argument("--discovered-vocab", action="store_true",
                       help="also use GREN's derived input vocabulary")
        q.add_argument("--games", nargs="*", default=["chess","checkers","go","sudoku"])
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__": main()
