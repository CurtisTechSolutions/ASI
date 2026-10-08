"""Every solver against every other, on every class in the trie.

The trie states which solver is GUARANTEED to work in each class. That is a
falsifiable prediction, and this is what falsifies it: run all five on games
drawn from every class and see whether the recommended one actually wins where
the theorem says it must.

The scoreboard is EXPLOITABILITY -- max over seats of (best-response utility -
realised utility), which is zero exactly at a Nash equilibrium. "The solver ran
64 iterations" says nothing; "exploitability 0.004" says the profile is within
0.004 of nobody wanting to move. Everything else reported here is secondary.
"""
import random, time
from array import array

from gtmnn import payoff as pay
from gtmnn.equilibrium import solve, mixed_exploitability
from gtmnn.game import StageGame, GameContext

SOLVERS = ("best_response", "fictitious", "regret", "regret_plus", "replicator")


def congestion_game(rng, kind, M=8, V=6, K=4):
    """A random stage game of the given payoff class."""
    ss, playable = [], []
    for _ in range(M):
        row = [rng.randrange(V) for _ in range(K)]
        ss.append(row + [None]); playable.append([True] * (K + 1))
    q = []
    for _ in range(M):
        r = [rng.random() for _ in range(K + 1)]; t = sum(r)
        q.append(array("d", [x / t for x in r]))
    y = rng.randrange(V)
    ctx = GameContext(alphabet_size=V, y=(y if kind != "belief" else None),
                      B=1.0, lam=0.5, q=q)
    ctx.slot_symbol = ss; ctx.playable = playable
    if kind == "correctness": p = pay.CorrectnessCongestion()
    elif kind == "belief": p = pay.BeliefCongestion()
    elif kind == "inverted": p = pay.Inverted(pay.CorrectnessCongestion())
    else: raise ValueError(kind)
    return StageGame(list(range(M)), ctx, p)


def matrix_game(rng, cls, n_actions=3):
    p = cls()
    k = len(p.table)
    ctx = GameContext(alphabet_size=k, y=None)
    ctx.slot_symbol = [list(range(k)), list(range(k))]
    ctx.playable = [[True] * k, [True] * k]
    return StageGame([0, 1], ctx, p)


CLASSES = {
    "correctness_congestion": lambda r: congestion_game(r, "correctness"),
    "belief_congestion":      lambda r: congestion_game(r, "belief"),
    "inverted_correctness":   lambda r: congestion_game(r, "inverted"),
    "rock_paper_scissors":    lambda r: matrix_game(r, pay.RockPaperScissors),
    "stag_hunt":              lambda r: matrix_game(r, pay.StagHunt),
    "prisoners_dilemma":      lambda r: matrix_game(r, pay.PrisonersDilemma),
}


def run(trials=40, iters=64, seed=0, classes=None, solvers=SOLVERS):
    """{class: {solver: stats}}. Every solver sees the SAME games, seed for seed."""
    out = {}
    for name, make in (classes or CLASSES).items():
        per = {s: {"expl": [], "mexpl": [], "conv": 0, "cycle": 0, "sec": 0.0,
                   "iters": []} for s in solvers}
        for t in range(trials):
            g = make(random.Random(seed * 1000 + t))
            for s in solvers:
                t0 = time.time()
                try:
                    e = solve(g, s, iters=iters, rng=random.Random(seed * 1000 + t))
                except AssertionError as exc:
                    # the potential assertion firing is a REAL result, not a crash
                    per[s]["expl"].append(float("inf")); per[s]["broke"] = str(exc)[:80]
                    continue
                per[s]["sec"] += time.time() - t0
                per[s]["expl"].append(e.exploitability)
                per[s]["mexpl"].append(mixed_exploitability(
                    g, e.profile, samples=192, rng=random.Random(7)))
                per[s]["conv"] += 1 if e.converged else 0
                per[s]["cycle"] += 1 if e.cycle else 0
                per[s]["iters"].append(e.iterations)
        for s in solvers:
            d = per[s]
            ok = [x for x in d["expl"] if x != float("inf")]
            mk = [x for x in d["mexpl"] if x != float("inf")]
            d["mean_mexpl"] = sum(mk) / len(mk) if mk else float("inf")
            d["mean_expl"] = sum(ok) / len(ok) if ok else float("inf")
            d["worst_expl"] = max(ok) if ok else float("inf")
            d["nash_frac"] = sum(1 for x in ok if x < 1e-9) / max(1, len(ok))
            d["conv_frac"] = d["conv"] / max(1, trials)
            d["cycle_frac"] = d["cycle"] / max(1, trials)
            d["mean_iters"] = sum(d["iters"]) / max(1, len(d["iters"]))
            d["ms"] = 1000.0 * d["sec"] / max(1, trials)
        out[name] = per
    return out


def verify(results, trie, placed):
    """Does the solver the trie RECOMMENDS actually win in its own class?"""
    rows = []
    alias = {"inverted_correctness": "inverted_correctness_congestion"}
    for cls, per in results.items():
        key = alias.get(cls, cls)
        if key not in placed: continue
        node = trie.walk(placed[key][1])[-1]
        rec, why = trie.solver(node)
        ranked = sorted(per.items(), key=lambda kv: kv[1]["mean_mexpl"])
        best = ranked[0][0]
        rec_rank = [s for s, _ in ranked].index(rec) + 1
        rows.append({"class": cls, "recommended": rec, "best_measured": best,
                     "rank_of_recommended": rec_rank,
                     "rec_expl": per[rec]["mean_mexpl"], "best_expl": per[best]["mean_mexpl"],
                     "vindicated": rec_rank == 1 or
                                   abs(per[rec]["mean_mexpl"] - per[best]["mean_mexpl"]) < 1e-6,
                     "why": why})
    return rows
