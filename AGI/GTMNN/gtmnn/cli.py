"""GTMNN command line.

  python3 -m gtmnn.cli demo       train a small population and show every readout
  python3 -m gtmnn.cli game       one stage game, inspected, nobody charged
  python3 -m gtmnn.cli shapley    the credit axioms, checked numerically
  python3 -m gtmnn.cli cycles     rock-paper-scissors: the cycle and the meta-player
  python3 -m gtmnn.cli auction    is truthful bidding really dominant?
  python3 -m gtmnn.cli trie       the game-theory trie, and every solver compared
  python3 -m gtmnn.cli transfer   GREN modifier: does chess transfer to checkers?
  python3 -m gtmnn.cli modifier   modifier cosine against Ochiai similarity
  python3 -m gtmnn.cli evolve     cull, birth and evolutionary stability
"""
import argparse, itertools, math, random, sys

from gtmnn.model import GTMNet, TrainConfig, gini
from gtmnn.evolve import Evolver, EvolveConfig

CORPUS = ["the cat sat on the mat", "the dog sat on the log",
          "the cat ran to the mat", "the dog ran to the log"]


def _net(a):
    return GTMNet(n=a.micros, F=a.features, R=a.rf, H=6, K=4, meta_n=max(16, a.micros // 4),
                  seed=a.seed, context=10)


def _cfg(a, **kw):
    d = dict(seats=a.seats, iters=a.iters, permutations=a.permutations,
             batch=64, lr=a.lr, seed=a.seed, epochs=a.epochs)
    d.update(kw)
    return TrainConfig(**d)


def cmd_demo(a):
    net = _net(a)
    cfg = _cfg(a)
    print("\nTraining a population of micros. Two losses are reported and they are")
    print("NOT the same thing:")
    print("  loss          the training-game aggregate. Its payoff CONTAINS y, so the")
    print("                seats are paid for naming the answer they were shown. It is")
    print("                the right partner for shapley_total and is not a prediction.")
    print("  belief_loss   the population's own beliefs, no game solved. This is what")
    print("                the model would actually predict.\n")
    print(f"{'ep':>3}{'loss':>9}{'belief':>9}{'uniform':>9}{'acc':>7}{'gini':>7}"
          f"{'div':>7}{'cycle':>7}{'meta':>7}{'shap':>8}{'s':>7}")
    for r in net.train(CORPUS, cfg):
        print(f"{r['epoch']:>3}{r['loss']:>9.3f}{r['belief_loss']:>9.3f}"
              f"{r['uniform_loss']:>9.3f}{r['accuracy']:>7.3f}{r['gini']:>7.3f}"
              f"{r['diversity']:>7.3f}{r['cycle_rate']:>7.3f}{r['meta_rate']:>7.3f}"
              f"{r['shapley_total']:>+8.3f}{r['seconds']:>7.1f}")
    print("\ncycle_rate is 0 during training and must be: the training game is an exact")
    print("potential game (Rosenthal 1973), so best-response dynamics cannot cycle.\n")
    st = net.stats()
    print("stats:", {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in st.items() if k != "backend"})


def cmd_game(a):
    from gtmnn import game as gm
    net = _net(a); cfg = _cfg(a)
    net.train(CORPUS, cfg)
    r = net.step("the c", None, cfg, learn=False, charge=False)
    print(f"\ncontext 'the c' -- inference, nobody charged\n")
    print(f"  bidders {r.alloc.bidders}  seated {len(r.game.seats)}  "
          f"clearing price {r.alloc.price:.4f}")
    print(f"  solver {r.eq.solver}  iterations {r.eq.iterations}  "
          f"converged {r.eq.converged}  cycle {r.eq.cycle}")
    print(f"  exploitability {r.eq.exploitability:.6f}   (0 exactly at Nash)")
    print(f"  coalitions {len(r.coalitions)}")
    if r.meta: print(f"  meta fired on '{r.meta.triggered}' -> {r.meta.policy_name}")
    print("\n  top:", [(s, round(p, 4)) for s, p in gm.top_k(r.agg, net.alphabet, 6)])
    loads = r.game.loads(r.eq.actions)
    named = {net.alphabet.decode(c): n for c, n in enumerate(loads[:-1]) if n}
    print(f"  loads {named}   abstentions {loads[-1]}/{len(r.eq.actions)}")


def cmd_shapley(a):
    from array import array
    from gtmnn.game import StageGame, GameContext, aggregate
    from gtmnn.payoff import CorrectnessCongestion
    from gtmnn.micro import MicroPool
    from gtmnn.shapley import shapley_values, shapley_exact

    def fixture(M, V, seed, abstain=None):
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
        return g, prof, pool, c.y

    print("\nThe four axioms, checked numerically.\n")
    worst = 0.0
    for t in range(200):
        g, p, pool, y = fixture(random.Random(t).randrange(2, 20),
                                random.Random(t).randrange(3, 9), t)
        for perms in (1, 2, 7, 32):
            worst = max(worst, shapley_values(g, p, pool, y, permutations=perms,
                                              rng=random.Random(t)).efficiency_error)
    print(f"  efficiency   |sum(phi) - (v(N)-v(0))| <= {worst:.3e} over 200 games x 4 "
          f"permutation counts")
    print("               exact per permutation, because marginal contributions telescope")
    g, p, pool, y = fixture(8, 5, 11, abstain=3)
    ph = shapley_values(g, p, pool, y, permutations=16, rng=random.Random(0)).phi
    print(f"  null player  a seat wholly on ABSTAIN gets phi = {ph[3]!r}")
    g, p, pool, y = fixture(6, 4, 31)
    for s in range(3): pool.repertoire[1 * 3 + s] = pool.repertoire[0 * 3 + s]
    g.ctx.slot_symbol[1] = list(g.ctx.slot_symbol[0]); p[1] = array("d", list(p[0]))
    pool.hits[1] = pool.hits[0]; pool.plays[1] = pool.plays[0]
    ex = shapley_exact(g, p, pool, y)
    print(f"  symmetry     two interchangeable seats: {ex[0]:.12f} vs {ex[1]:.12f} "
          f"(diff {abs(ex[0]-ex[1]):.1e})")
    g, p, pool, y = fixture(6, 4, 21)
    ex = shapley_exact(g, p, pool, y)
    allp = shapley_values(g, p, pool, y, permutations=10000, rng=random.Random(0))
    print(f"  exactness    closed form vs 10k permutations: max diff "
          f"{max(abs(x-z) for x, z in zip(ex, allp.phi)):.3e}")


def cmd_cycles(a):
    from gtmnn.game import StageGame, GameContext
    from gtmnn.payoff import RockPaperScissors
    from gtmnn.equilibrium import solve
    ctx = GameContext(alphabet_size=3, y=None); ctx.slot_symbol = [[0, 1, 2], [0, 1, 2]]
    ctx.playable = [[True] * 3, [True] * 3]
    g = StageGame([0, 1], ctx, RockPaperScissors())
    print("\nRock-paper-scissors: no pure Nash, so best response MUST cycle.\n")
    eq = solve(g, "best_response", iters=64)
    print(f"  best_response  cycle length {eq.cycle} (sweeps)  converged {eq.converged}")
    print("                 pushed per sweep, not per move: per move it reports 6")
    eq2 = solve(g, "regret", iters=10000, rng=random.Random(0))
    print(f"  regret         time-average {[round(x,4) for x in eq2.profile[0]]}  "
          f"max deviation from uniform {max(abs(x-1/3) for x in eq2.profile[0]):.4f}")
    print("\n  A cycle is not the solver failing. Milchtaich (1996): the inference game")
    print("  is player-specific, a pure equilibrium exists but the path may loop.")
    print("  Iterating longer does not help. Leaving does -- that is the meta-player.")


def cmd_auction(a):
    from gtmnn.auction import best_response_bid, bid, allocate
    from gtmnn.micro import MicroPool
    from array import array
    rng = random.Random(a.seed)
    pool = MicroPool(64, 32, R=4, H=4, K=3, seed=a.seed, alphabet_size=6)
    qs = []
    for i in range(64):
        r = [rng.random() for _ in range(4)]; t = sum(r)
        qs.append(array("d", [x / t for x in r]))
    others = [(bid(pool, j, qs[j]), j) for j in range(1, 64)]
    truth, sweep = best_response_bid(pool, 0, qs[0], others, m=16)
    print(f"\nUniform-price auction. Micro 0's true value for a seat: {truth:.4f}\n")
    print(f"{'bid':>10}{'surplus':>10}   ")
    best = max(sweep, key=lambda t: t[1])
    for b, s in sweep:
        mark = "  <- truthful" if abs(b - truth) < 1e-12 else ""
        print(f"{b:>10.4f}{s:>10.4f}{mark}")
    print(f"\n  best surplus {best[1]:.4f} at bid {best[0]:.4f}; truthful earns {dict(sweep).get(truth, 0.0):.4f}")
    print("  No misreport beats the truth, because the price is the highest LOSING bid")
    print("  and raising your own bid cannot change it (Vickrey 1961).")


def cmd_modifier(a):
    from gtmnn.games import load_mechanics, modifier, ochiai, cosine
    mech, _ = load_mechanics()
    print("\nThe game modifier: the mechanic set GREN discovered, hashed.\n")
    print(f"{'pair':<24}{'Ochiai':>9}{'cos@64':>9}{'cos@128':>9}{'cos@256':>9}")
    errs = {64: [], 128: [], 256: []}
    for x, y in itertools.combinations(sorted(mech), 2):
        o = ochiai(mech[x], mech[y]); row = f"{x}/{y}"; cs = {}
        for d in (64, 128, 256):
            c = cosine(modifier(mech[x], d), modifier(mech[y], d))
            cs[d] = c; errs[d].append(abs(c - o))
        print(f"{row:<24}{o:>9.3f}{cs[64]:>9.3f}{cs[128]:>9.3f}{cs[256]:>9.3f}")
    print()
    for d in (64, 128, 256):
        print(f"  mean |cos - Ochiai| at dim {d}: {sum(errs[d])/len(errs[d]):.4f}")
    print("\n  Hashing a SET preserves overlap, so similar games land at nearby points")
    print("  automatically. Transfer is a consequence of that, not a mechanism.")


def cmd_transfer(a):
    import statistics as st
    from gtmnn.play import GamePlayer, adapters, samples, evaluate, train_on
    from gtmnn.games import load_mechanics
    G = adapters(("chess", "checkers")); mech, _ = load_mechanics()
    cfg = TrainConfig(seats=a.seats, iters=a.iters, permutations=a.permutations,
                      batch=48, lr=a.lr, seed=a.seed)
    res = {}
    for seed in range(a.trials):
        test = {n: samples(G[n], n, random.Random(4000 + seed * 7 + k), 14, 8)
                for k, n in enumerate(G)}
        for label, mod_on, tr in (("untrained", True, False),
                                  ("chess-trained, modifier ON", True, True),
                                  ("chess-trained, modifier OFF", False, True)):
            p = GamePlayer(mech, n=a.micros, R=12, H=6, seed=seed, modifier_on=mod_on)
            if tr: train_on(p, G, ["chess"], cfg, rounds=3, positions=14, per_position=8, seed=seed)
            res.setdefault(label, []).append(
                (evaluate(p, G["chess"], "chess", test["chess"], cfg)["acc"],
                 evaluate(p, G["checkers"], "checkers", test["checkers"], cfg)["acc"]))
    print(f"\nTrain on CHESS only; evaluate CHECKERS with no further training.")
    print(f"The control zeroes the modifier block, so a gap is the modifier's doing.\n")
    print(f"{'condition':<30}{'chess acc':>16}{'checkers acc':>16}")
    for label, v in res.items():
        c = [x[0] for x in v]; k = [x[1] for x in v]
        print(f"{label:<30}{st.mean(c):>10.3f}+-{st.pstdev(c):.3f}"
              f"{st.mean(k):>10.3f}+-{st.pstdev(k):.3f}")


def cmd_trie(a):
    """The trie over game-theoretic structure, and the tournament that checks it."""
    from gtmnn.trie import build
    from gtmnn.tournament import run, verify, SOLVERS
    t, placed = build(seats=a.seats)
    print("\nA TRIE OVER WHAT IS PROVABLE. The root asserts nothing; each level adds")
    print("one fact, so a node's guarantees are a function of its PATH and nothing else.\n")

    def show(node=0, indent=""):
        n = t.nodes[node]
        lab = "." if node == 0 else "=".join(n.token)
        sv, _ = t.solver(node)
        g = t.guarantees(node)
        games = ("  <- " + ", ".join(n.games)) if n.games else ""
        print(f"{indent}{lab:<30}{sv:<15}{len(g)} thm{games}")
        for c in n.children.values(): show(c, indent + "  ")
    show()

    print("\nWhat each theorem needs, and where it starts to hold:\n")
    for name, (node, f) in sorted(placed.items()):
        seen = []
        for nd in t.walk(f):
            for thm, _ in t.guarantees(nd):
                if thm not in seen:
                    seen.append(thm)
                    d = t.nodes[nd].depth
                    tok = "." if nd == 0 else "=".join(t.nodes[nd].token)
                    print(f"  {name:<32} d{d} {tok:<28} {thm}")
        print()

    from gtmnn.trie import TrieNetworks
    print("\nA NETWORK AT EVERY NODE -- reaching a node is how you find the one to ask\n")
    tn = TrieNetworks(t, F=64, n=32, R=8, H=6, K=4, seed=a.seed)
    facts = placed["correctness_congestion"][1]
    pth = t.walk(facts)
    for nd in (pth[3], pth[-1]): tn.attach(nd)
    tn.note(pth[3])
    for f, lab in ((facts, "correctness_congestion"),):
        nd, _, backed = tn.resolve(f)
        print(f"  {lab}: walks to d{t.nodes[pth[-1]].depth}, that node is cold, so it "
              f"backs off to d{t.nodes[nd].depth}")
        print(f"     {'/'.join(f'{k}={v}' for k, v in t.facts(nd).items())}")
    tn.note(pth[-1])
    nd, _, backed = tn.resolve(facts)
    print(f"  once the leaf has played: answers at d{t.nodes[nd].depth}, "
          f"backed_off={backed}")
    print("\n  Routing and backoff are verified. What backoff is WORTH here is not:")
    print("  over 5 seeds it beats a cold leaf by +0.006 with a per-seed swing of")
    print("  +-0.15 -- nothing. The class it backs off to has barely learned either,")
    print("  which is the same weak-credit-path result the README opens with.")


    if a.trials < 1: return
    print(f"EVERY SOLVER AGAINST EVERY OTHER  ({a.trials} games per class)\n")
    res = run(trials=a.trials, iters=a.iters, seed=a.seed)
    print("mixed exploitability -- 0 means the mixed profile is unexploitable\n")
    print(f"{'game class':<26}" + "".join(f"{x[:13]:>14}" for x in SOLVERS))
    for cls, per in res.items():
        best = min(per, key=lambda x: per[x]["mean_mexpl"])
        print(f"{cls:<26}" + "".join(
            f"{(format(per[x]['mean_mexpl'], '.4f') + ('*' if x == best else '')):>14}"
            for x in SOLVERS))
    print("\n  * = best in class.  The PURE measure is meaningless where the equilibrium")
    print("  is mixed: on rock-paper-scissors every pure profile is exploitable by 2.0,")
    print("  which scores all five identically at their worst.\n")
    print(f"{'game class':<26}" + "".join(f"{x[:13]:>14}" for x in SOLVERS) + "   ms/game")
    for cls, per in res.items():
        print(f"{cls:<26}" + "".join(f"{per[x]['ms']:>14.2f}" for x in SOLVERS))

    print("\nDOES THE TRIE'S RECOMMENDATION WIN IN ITS OWN CLASS?\n")
    rows = verify(res, t, placed)
    print(f"{'class':<26}{'recommended':<16}{'measured best':<16}{'rank':>5}   verdict")
    for r in rows:
        v = "vindicated" if r["vindicated"] else f"conservative (see below)"
        print(f"{r['class']:<26}{r['recommended']:<16}{r['best_measured']:<16}"
              f"{r['rank_of_recommended']:>5}   {v}")
    bad = [r for r in rows if not r["vindicated"]]
    for r in bad:
        print(f"\n  {r['class']}: the trie says {r['recommended']} because "
              f"{r['why']}.")
        print(f"  Measured, {r['best_measured']} wins ({r['best_expl']:.4f} against "
              f"{r['rec_expl']:.4f}). Both are right:")
        print("  the theorem is a WORST-CASE guarantee and the tournament is an average")
        print("  case. The trie's job is to say which is which, not to predict the mean.")


def cmd_evolve(a):
    net = _net(a)
    cfg = _cfg(a, epochs=1)
    net.config = cfg
    net.train(CORPUS, cfg)
    ev = Evolver(net, CORPUS, EvolveConfig(seed=a.seed, games_per_generation=200))
    print(f"\n{'gen':>4}{'culled':>8}{'born':>6}{'imm':>5}{'cross':>7}{'clone':>7}"
          f"{'alive':>7}{'gini':>8}{'ess':>7}{'belief':>9}")
    for g in range(a.generations):
        r = ev.run_generation()
        print(f"{r['generation']:>4}{r['culled']:>8}{r['born']:>6}{r['immigrants']:>5}"
              f"{r['crossovers']:>7}{r['clones']:>7}{r['alive']:>7}{r['gini']:>8.3f}"
              f"{r['ess_score']:>7.3f}{(r['belief_loss'] or 0):>9.3f}")
    print("\n  ess_score near 1.0 means no small mutation pays to enter -- a converged")
    print("  population and, past a point, a stagnant one. Read the trajectory.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gtmnn")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("demo", cmd_demo), ("game", cmd_game), ("shapley", cmd_shapley),
                     ("cycles", cmd_cycles), ("auction", cmd_auction),
                     ("modifier", cmd_modifier), ("transfer", cmd_transfer),
                     ("trie", cmd_trie), ("evolve", cmd_evolve)):
        q = sub.add_parser(name); q.set_defaults(fn=fn)
        q.add_argument("--micros", type=int, default=192)
        q.add_argument("--features", type=int, default=128)
        q.add_argument("--rf", type=int, default=12)
        q.add_argument("--seats", type=int, default=24)
        q.add_argument("--iters", type=int, default=16)
        q.add_argument("--permutations", type=int, default=8)
        q.add_argument("--epochs", type=int, default=6)
        q.add_argument("--lr", type=float, default=0.05)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--trials", type=int, default=3)
        q.add_argument("--generations", type=int, default=3)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__": main()
