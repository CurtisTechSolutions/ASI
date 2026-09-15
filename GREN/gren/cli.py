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
from gren.radix import RadixGameTree
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
    for n, (e, _) in ev.items(): t.insert(sorted(e.signature()), n)
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
    ev = _explore(a.budget, a.policy, a.seed)
    corpus = {n: e.signature() for n, (e, _) in ev.items()}
    print()
    for n, (e, _) in ev.items():
        p = pkg.build(e, corpus)
        print(f"  {n}: {p.to_dict()}")
        print(f"     accepted by the player network: {p.accepted()}\n")

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
                     ("package", cmd_package), ("grow", cmd_grow)):
        q = sub.add_parser(name); q.set_defaults(fn=fn)
        q.add_argument("--budget", type=int, default=1200)
        q.add_argument("--policy", default="eig", choices=list(POLICIES))
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--hidden", type=int, default=24)
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__": main()
