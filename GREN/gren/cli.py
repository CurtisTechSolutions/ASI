"""GREN.

  python3 -m gren.cli explore   probe every game and report what it refuses
  python3 -m gren.cli similar   discovered similarity vs CyclicCortex's hand-written sets
  python3 -m gren.cli policies  failure rate and rule coverage by probe policy
  python3 -m gren.cli tree      the radix tree over discovered signatures
  python3 -m gren.cli package   the GamePackage handed to the player network
"""
import argparse, itertools, math, random, sys
from gren.oracle import build_all
from gren.explorer import Explorer
from gren.axis import distance
from gren.probe import POLICIES
from gren.radix import RadixGameTree, goal_first
from gren import package as pkg

def _explore(budget, policy="eig", seed=0):
    ev = {}
    for name, o in build_all().items():
        ex = Explorer(o, policy=policy, seed=seed)
        ex.run(budget=budget, k=10)
        ev[name] = (ex.evidence, ex.report())
    return ev

def cmd_explore(a):
    print("\n  Probing. The failure rate target is 0.50 and is DERIVED (§4), not set.\n")
    print(f"  {'game':>9} {'probes':>7} {'fail':>6} {'codes':>6} {'legality':>9} {'entropy':>8}")
    for n, (e, r) in _explore(a.budget, a.policy, a.seed).items():
        print(f"  {n:>9} {r['probes']:>7} {r['failure_rate']:>6.3f} {r['codes']:>6} "
              f"{r['legality_density']:>9.3f} {r['reason_entropy']:>8.2f}")

def cmd_similar(a):
    sys.path.insert(0, pkg.__file__.rsplit("/GREN/", 1)[0] + "/CyclicCortex")
    from cortex.games import ALL
    ev = _explore(a.budget, a.policy, a.seed)
    rows = []
    for x, y in itertools.combinations(ev, 2):
        dg = distance(ev[x][0].signature(), ev[y][0].signature())
        dh = 1.0 - len(ALL[x].mechanics & ALL[y].mechanics) / len(ALL[x].mechanics | ALL[y].mechanics)
        rows.append((dg, dh, x, y))
    print(f"\n  {'pair':>22} {'GREN discovered':>16} {'hand-written':>14}")
    for dg, dh, x, y in sorted(rows):
        print(f"  {x+' / '+y:>22} {dg:>16.3f} {dh:>14.3f}")
    print(f"\n  Spearman = {_spearman([r[0] for r in rows], [r[1] for r in rows]):.3f}")

def cmd_policies(a):
    print("\n  Failure rate and rule coverage by policy.\n")
    print(f"  {'game':>9} {'policy':>9} {'failure rate':>13} {'codes found':>12}")
    for name, o in build_all().items():
        for pn in POLICIES:
            ex = Explorer(o, policy=pn, seed=a.seed); ex.run(budget=a.budget, k=10)
            print(f"  {name:>9} {pn:>9} {ex.policy.failure_rate:>13.3f} "
                  f"{len(ex.evidence.codes):>12}")
        print()

def cmd_tree(a):
    ev = _explore(a.budget, a.policy, a.seed)
    t = RadixGameTree()
    for n, (e, _) in ev.items(): t.insert(goal_first(e.signature()), n)
    merged = t.compress()
    print(f"\n  radix tree over discovered signatures: {len(t.terminals())} terminals, "
          f"{merged} unary chains compressed\n")
    for n in t.terminals():
        print(f"    {t.game[n]:>9}  depth {len(t.path(n))}")
    print("\n  partial evidence retrieval:")
    for known in ({"category=board"}, {"refuses=CONSTRAINT_ROW"}, {"players=2"}):
        got = [g for _, g in t.retrieve(known)][:3]
        print(f"    {str(sorted(known)):>34} -> {got}")

def cmd_grow(a):
    """One Self-Building network across every game: inputs grow per new feature
    name, outputs per newly discovered refusal code."""
    from gren.sbnn import GrowingSBNN
    net = GrowingSBNN(nh=a.hidden, seed=a.seed)
    print(f"\n  one network, starting with {net.ni} inputs and {net.no} outputs\n")
    print(f"  {'after':>10} {'inputs':>7} {'outputs':>8} {'hidden':>7} {'params':>8} "
          f"{'predict acc':>12} {'fail rate':>10}")
    for name, o in build_all().items():
        ex = Explorer(o, policy="learned", seed=a.seed, net=net)
        r = ex.run(budget=a.budget, k=10)
        sh = r["model"]
        print(f"  {name:>10} {sh['ni']:>7} {sh['no']:>8} {sh['nh']:>7} {sh['params']:>8} "
              f"{r['predict_acc']:>12.3f} {r['failure_rate']:>10.3f}")
    print(f"\n  discovered refusal codes: {sorted(k for k in net.outputs if k != 'LEGAL')}")
    grew = [e for e in net.log if e["grew"] == "hidden"]
    print(f"  hidden grew {len(grew)} time(s) on a loss plateau")

def cmd_package(a):
    """Build a GamePackage per game, and optionally write the handoff file that
    CyclicCortex consumes in place of its hard-coded mechanics."""
    from gren.oracle import build_all as _oracles
    ev = _explore(a.budget, a.policy, a.seed)
    corpus = {n: e.signature() for n, (e, _) in ev.items()}
    built = []
    print()
    for n, (e, _) in ev.items():
        p = pkg.build(e, corpus); built.append(p)
        print(f"  {n:>9}: {len(p.signature)} tokens, {len(p.mechanics)} codes | "
              f"identify {p.confidence:.3f} | characterise {p.characterisation:.3f} "
              f"| placeable={p.placeable()}")
        print(f"             nearest: {[(round(d,3), g) for d, g in p.candidates[:2]]}")
    if a.out:
        rng = random.Random(a.seed)
        feats = {}
        for n, o in _oracles().items():
            g = o.game
            names = set()
            for _ in range(12):
                st = (g.random_state(rng) if hasattr(g, "random_state")
                      else g.new(seed=rng.randrange(10**6)))
                for mv in o.candidates(st, rng, 6):
                    try: names |= set(o.features(st, mv))
                    except Exception: pass
            feats[n] = sorted(names)
        from gren import vocabulary as voc
        from gren.oracle import CODES
        lay, slots, priv, det = voc.build_aligned(_oracles(), CODES, seed=a.seed)
        print("\n  input slots matched ACROSS games by legality signature:")
        for sl in slots:
            print(f"    {sl['name']:<20} {sl['strength']:.2f}  " + ", ".join(
                f"{g}:{'+' if sg > 0 else '-'}{b}[{i}]"
                for g, (b, i, sg) in sorted(sl["dims"].items())))
        for n in sorted(det):
            expl = sum(1 for c, _, _ in det[n].values() if c)
            print(f"    {n:>9}: {expl} of {len(det[n])} dims explain a refusal, "
                  f"{len(priv[n])} stay private")
        lay = {n: {c: [list(t) for t in d] for c, d in l.items()}
               for n, l in lay.items()}
        size = pkg.export(built, a.out, feats, vocabulary=lay, slots=slots)
        print(f"\n  wrote {size} bytes -> {a.out}")
        print(f"  feature languages: " +
              ", ".join(f"{k}={len(v)}" for k, v in feats.items()))

def cmd_regress(a):
    """Goal regression: work backwards from the end goal to subgoals."""
    import time, statistics as stat
    from gren.oracle import build_all as _oracles
    from gren.regress import blocker_histogram, most_constrained, selectivity
    from cortex import sudoku as sud
    from cortex.board_games import random_chess
    from cortex import engine

    oc = _oracles()
    print("\nGoal regression asks what would have to be TRUE for the goal to hold,")
    print("and keeps asking until the answer is something already true. Its usual")
    print("cost is writing every operator's preconditions by hand -- which is where")
    print("domain knowledge smuggles itself in. GREN does not need them written:\n")
    print("    A REFUSAL CODE IS A PRECONDITION VIOLATION.\n")
    print("`why()` already answers 'what blocks this action?', which is the")
    print("regression step. The preconditions are measured, not authored.\n")

    o = oc["sudoku"]
    s = sud.generate(clues=30, seed=3)
    allm = [(x, y, v) for y in range(9) for x in range(9) for v in range(1, 10)]
    print(f"SUDOKU -- goal: every cell filled.  ({s.empties()} empty)\n")
    print("  what stands between here and every conceivable move:")
    for code, n in blocker_histogram(o, s, allm).items():
        print(f"    {code:<20} {n:>5}")
    print("\n  the most constrained subgoals, attacked first:")
    for cell, mv in most_constrained(o, s, allm, key=lambda m: (m[0], m[1]))[:4]:
        print(f"    cell {cell}: {len(mv)} value(s) clear every precondition "
              f"-> {[m[2] for m in mv]}")
    print("\n  A cell with one clearing value is forced. Nobody coded 'naked")
    print("  single' -- that heuristic IS constraint propagation, regressed.\n")

    def solve(s0, guided, budget=400000):
        work = [0]; nodes = [0]
        def rec(st):
            nodes[0] += 1
            if st.solved(): return st
            if work[0] >= budget: return None
            cells = [(x, y) for y in range(9) for x in range(9) if not st.at(x, y)]
            if guided:
                best = None
                for (x, y) in cells:
                    c = st.candidates(x, y); work[0] += 1
                    if not c: return None
                    if best is None or len(c) < len(best[2]): best = (x, y, c)
                x, y, cand = best
            else:
                x, y = cells[0]; cand = st.candidates(x, y); work[0] += 1
                if not cand: return None
            for v in cand:
                r = rec(st.apply((x, y, v)))
                if r is not None: return r
            return None
        return rec(s0), work[0], nodes[0]

    print(f"  {'clues':>6}{'guided nodes':>14}{'control nodes':>15}{'x':>9}"
          f"{'guided calls':>14}{'control calls':>15}{'x':>8}")
    for clues in (30, 26, 22):
        gn, un, gc, uc = [], [], [], []
        for seed in range(a.trials):
            p = sud.generate(clues=clues, seed=seed)
            _, wa, na = solve(p, True); _, wb, nb = solve(p, False)
            gn.append(na); un.append(nb); gc.append(wa); uc.append(wb)
        g, u = stat.mean(gn), stat.mean(un)
        gg, uu = stat.mean(gc), stat.mean(uc)
        print(f"  {clues:>6}{g:>14.0f}{u:>15.0f}{u/max(1,g):>8.1f}x"
              f"{gg:>14.0f}{uu:>15.0f}{uu/max(1,gg):>7.1f}x")
    print("\n  By NODES -- what regression reduces -- guidance wins everywhere.")
    print("  By raw work it loses on easy puzzles: scanning 51 cells to save a few")
    print("  nodes is not worth it when almost any order solves the puzzle.\n")

    print("CHESS -- goal: mate.  mate <== check AND no-escape.\n")
    CALLS = {"escape": 0}
    def search(st, regress):
        for mv in engine.legal_moves(st):
            nxt = engine.apply_move(st, mv)
            if regress:
                if not nxt.in_check(): continue
                CALLS["escape"] += 1
                if not engine.legal_moves(nxt): return mv
            else:
                CALLS["escape"] += 1
                if engine.legal_moves(nxt): continue
                if nxt.in_check(): return mv
        return None

    def played(n, seed=3):
        rng = random.Random(seed); out = []
        while len(out) < n:
            st = engine.start_position()
            for _ in range(rng.randrange(10, 60)):
                ms = engine.legal_moves(st)
                if not ms: break
                st = engine.apply_move(st, engine.choose(st, depth=1, rng=rng) or rng.choice(ms))
            if engine.legal_moves(st): out.append(st)
        return out

    rng = random.Random(7)
    scattered = []
    while len(scattered) < a.trials * 4:
        st = random_chess(rng)
        if engine.legal_moves(st): scattered.append(st)
    real = played(a.trials * 4)
    print(f"  {'positions':<20}{'% moves giving check':>22}{'expensive tests saved':>23}{'speedup':>10}")
    for lab, states in (("pieces scattered", scattered), ("from actual play", real)):
        sel = selectivity(o, states, engine.legal_moves,
                          lambda st, mv: engine.apply_move(st, mv).in_check())
        res = {}
        for reg in (True, False):
            CALLS["escape"] = 0; t0 = time.time()
            for st in states: search(st, reg)
            res[reg] = (CALLS["escape"], time.time() - t0)
        saved = 100 * (1 - res[True][0] / max(1, res[False][0]))
        print(f"  {lab:<20}{100*sel:>21.1f}%{saved:>22.1f}%{res[False][1]/max(1e-9,res[True][1]):>9.1f}x")
    print("\n  A subgoal prunes exactly what it excludes. That is the whole law:")
    print("  regression pays in proportion to how SELECTIVE the cheap subgoal is.\n")
    print("  Worth naming: the first attempt at this measured 1.0x, because the")
    print("  baseline had regressed all along -- is_mate() is `in_check() and not")
    print("  legal_moves()`, and `and` short-circuits. Goal regression over a")
    print("  conjunctive goal IS a short-circuiting `and` with the cheap, selective")
    print("  conjunct written first, which is why it is easy to have done already.")


def cmd_transfer(a):
    """Do the RULES transfer? Train a growing code-predictor on one game's
    (shared-vocabulary features -> refusal code), test it per code on another."""
    import statistics as stat
    from gren.oracle import build_all as _oracles
    from gren.sbnn import GrowingSBNN
    from cortex.games import ALL
    G = ["chess", "checkers", "go", "sudoku"]
    OR = _oracles()

    def probes(name, rng, states, k=10):
        o = OR[name]; g = o.game; rows = []
        for _ in range(states):
            s = (g.random_state(rng) if hasattr(g, "random_state")
                 else g.new(seed=rng.randrange(10 ** 6)))
            for mv in o.candidates(s, rng, k):
                try:
                    f = g.generalise(s, mv)
                    lab = "LEGAL" if g.is_legal(s, mv) else o.why(s, mv)[0]
                except Exception:
                    continue
                rows.append(({f"{b}[{i}]": v for b, vec in f.items() for i, v in enumerate(vec)}, lab))
        return rows

    def balance(rows, rng, cap=120):
        by = {}
        for f, l in rows: by.setdefault(l, []).append((f, l))
        out = []
        for l, rs in by.items(): rng.shuffle(rs); out += rs[:cap]
        rng.shuffle(out); return out

    def surface(x, y):
        # SIDE is whose turn it is: one slot, no rule content. A net given only
        # that emits its prior, and a prior is not transfer.
        k = set(ALL[x].spec) & set(ALL[y].spec) - {"SIDE"}
        return sum(ALL[x].spec[b] for b in k)

    print("\nLegality is one bit. A RULE is which refusal fires, and under what")
    print("condition. Train on A, test per code on B, through the vocabulary they")
    print("share -- the only channel a rule can cross.\n")
    test = {b: balance(probes(b, random.Random(500 + i), 50), random.Random(1)) for i, b in enumerate(G)}
    R = {}
    for x in G:
        for y in G:
            accs, conf = {}, {}
            for seed in range(a.trials):
                tr = balance(probes(x, random.Random(seed * 31 + 7), 60), random.Random(seed))
                net = GrowingSBNN(nh=24, seed=seed)
                for _ in range(6):
                    for f, l in tr: net.step(f, l, lr=0.05)
                hit, tot = {}, {}
                for f, l in test[y]:
                    p = net.predict(f); pred = max(p, key=p.get) if p else "?"
                    tot[l] = tot.get(l, 0) + 1
                    if pred == l: hit[l] = hit.get(l, 0) + 1
                    else: conf.setdefault(l, {}); conf[l][pred] = conf[l].get(pred, 0) + 1
                for l in tot: accs.setdefault(l, []).append(hit.get(l, 0) / tot[l])
            R[(x, y)] = ({l: stat.mean(v) for l, v in accs.items()}, tot, conf)

    print(f"  {'A -> B':<22}{'rule-bearing slots':>20}   per-code recall on B")
    for x in G:
        for y in G:
            if x == y: continue
            pc, tot, _ = R[(x, y)]
            shared = [c for c in sorted(pc) if c in R[(x, x)][1] and tot.get(c, 0) >= 10]
            sf = surface(x, y)
            body = "  ".join(f"{c}={pc[c]:.2f}" for c in shared) if sf else "(no rule-bearing surface: any score is a prior)"
            print(f"  {x + ' -> ' + y:<22}{sf:>20}   {body}")

    print("\n  The two facts that decide it:\n")
    pc, tot, conf = R[("go", "sudoku")]
    print(f"  go -> sudoku through GRID_PLACE:  LEGAL {pc.get('LEGAL', 0):.2f}  "
          f"OCCUPIED_TARGET {pc.get('OCCUPIED_TARGET', 0):.2f}")
    print("     the one rule they share means the same thing, and it transfers whole.")
    pc, tot, conf = R[("chess", "checkers")]
    c = conf.get("LEGAL", {}); t = sum(c.values()) or 1
    print(f"  chess -> checkers on checkers' LEGAL moves: recall {pc.get('LEGAL', 0):.2f}; "
          f"called BLOCKED_PATH {c.get('BLOCKED_PATH', 0) / t:.0%} of the time it was wrong")
    print("     a checkers jump has an occupied midpoint -- the piece being taken. To")
    print("     chess that is a blocked bishop. Same feature, opposite rule.")


def _spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for p, i in enumerate(o): r[i] = p
        return r
    ra, rb = rk(a), rk(b); n = len(a)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = math.sqrt(sum((x-ma)**2 for x in ra)); db = math.sqrt(sum((x-mb)**2 for x in rb))
    return num/(da*db) if da and db else 0.0

def main(argv=None):
    p = argparse.ArgumentParser(prog="gren")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("explore", cmd_explore), ("similar", cmd_similar),
                     ("policies", cmd_policies), ("tree", cmd_tree),
                     ("package", cmd_package), ("grow", cmd_grow),
                     ("regress", cmd_regress), ("transfer", cmd_transfer)):
        q = sub.add_parser(name); q.set_defaults(fn=fn)
        q.add_argument("--budget", type=int, default=1200)
        q.add_argument("--policy", default="eig", choices=list(POLICIES))
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--hidden", type=int, default=24)
        q.add_argument("--out", default=None, help="write the handoff file here")
        q.add_argument("--trials", type=int, default=6)
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__": main()
