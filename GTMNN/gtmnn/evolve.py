"""Culling, birth, and evolutionary stability.

Bankruptcy, not error. Wealth is the only survival criterion and it already
integrates everything the design cares about: a micro that bids honestly and is
often right accumulates B/n_a on rare calls; one that bids high and is wrong pays
the clearing price every game and loses lambda on top; one that always abstains
earns nothing and slowly bleeds out through wealth_decay. Being useless is fatal
on a longer timescale than being wrong, which is the right ordering.
"""
import math, random, time
from dataclasses import dataclass, asdict

from gtmnn.features import coverage, coverage_gaps, sample_receptive_field
from gtmnn.model import gini


@dataclass
class EvolveConfig:
    generations: int = 0
    games_per_generation: int = 2048
    cull_fraction: float = 0.05
    bankrupt: float = -1.0
    wealth_decay: float = 0.99
    mutation_sigma: float = 0.05
    crossover_rate: float = 0.5
    immigrant_rate: float = 0.10
    tournament: int = 4
    ess_samples: int = 16
    ess_iters: int = 256
    checkpoint_every: int = 0
    seed: int = 0

    def to_dict(self): return asdict(self)


class Evolver:
    def __init__(self, net, corpus, config=None):
        self.net = net
        self.corpus = list(corpus)
        self.cfg = config or EvolveConfig()
        self.rng = random.Random(self.cfg.seed)
        self.generation = net.generation
        self.history = []

    # ----------------------------------------------------------------- cull
    def cull(self):
        """At most cull_fraction of the living die per generation, poorest first.

        The cap matters more than it looks: without it a corpus shift can
        bankrupt most of the population in one generation and take the feature
        coverage with it. With it, the survivors' receptive fields keep the space
        covered while newborns arrive."""
        pool = self.cfg, self.net.pool
        p = self.net.pool
        alive = p.living()
        for i in alive: p.wealth[i] *= self.cfg.wealth_decay
        broke = sorted((i for i in alive if p.wealth[i] < self.cfg.bankrupt),
                       key=lambda i: p.wealth[i])
        cap = int(self.cfg.cull_fraction * len(alive))
        doomed = broke[:cap]
        for i in doomed: p.kill(i)
        return doomed

    # ---------------------------------------------------------------- birth
    def _tournament(self, alive):
        k = min(self.cfg.tournament, len(alive))
        cand = self.rng.sample(alive, k)
        return max(cand, key=lambda i: self.net.pool.wealth[i])

    def birth(self, slots):
        """Immigrants aim at blind spots; crossover mixes two tournament winners;
        otherwise a clone. Every newborn is mutated and starts at wealth 0 and
        reputation 0.5 -- neither advantaged nor protected."""
        p = self.net.pool
        cfg = self.cfg
        rng = self.rng
        alive = p.living()
        V = len(self.net.alphabet)
        counts = {"immigrants": 0, "crossovers": 0, "clones": 0}
        if not alive:
            for i in slots: p.rebirth(i, V, self.generation)
            counts["immigrants"] = len(slots)
            return counts
        cov = coverage(p)
        cold = sorted(range(p.F), key=lambda f: cov[f])
        for i in slots:
            r = rng.random()
            if r < cfg.immigrant_rate:
                p.rebirth(i, V, self.generation)
                field = sorted(rng.sample(cold[:max(p.R * 4, 16)], p.R))
                for t in range(p.R): p.rf[i * p.R + t] = field[t]
                counts["immigrants"] += 1
            elif r < cfg.immigrant_rate + cfg.crossover_rate and len(alive) >= 2:
                a, b = self._tournament(alive), self._tournament(alive)
                fit = a if p.wealth[a] >= p.wealth[b] else b
                self._cross(i, a, b, fit)
                counts["crossovers"] += 1
            else:
                a = self._tournament(alive)
                self._cross(i, a, a, a)
                counts["clones"] += 1
            self._mutate(i)
            p.wealth[i] = 0.0; p.hits[i] = 0.0; p.plays[i] = 0.0
            p.age[i] = 0; p.born[i] = self.generation; p.alive[i] = 1
            p._tally[i] = {}
        p.pop_version += 1
        return counts

    def _cross(self, i, a, b, fit):
        p, rng = self.net.pool, self.rng
        R, H, S, K = p.R, p.H, p.S, p.K
        fa = {p.rf[a * R + t] for t in range(R)} | {p.rf[b * R + t] for t in range(R)}
        field = sorted(rng.sample(sorted(fa), min(R, len(fa))))
        while len(field) < R: field.append(rng.randrange(p.F))
        for t in range(R): p.rf[i * R + t] = field[t]
        for t in range(R * H):
            src = a if rng.random() < 0.5 else b
            p.w1[i * R * H + t] = p.w1[src * R * H + t]
        for t in range(H):
            src = a if rng.random() < 0.5 else b
            p.b1[i * H + t] = p.b1[src * H + t]
        for t in range(H * S):
            src = a if rng.random() < 0.5 else b
            p.w2[i * H * S + t] = p.w2[src * H * S + t]
        for t in range(S):
            src = a if rng.random() < 0.5 else b
            p.b2[i * S + t] = p.b2[src * S + t]
        for t in range(p.NEUR):                       # sines from the fitter parent
            p.sa[i * p.NEUR + t] = p.sa[fit * p.NEUR + t]
            p.sb[i * p.NEUR + t] = p.sb[fit * p.NEUR + t]
            p.sh[i * p.NEUR + t] = p.sh[fit * p.NEUR + t]
            p.sk[i * p.NEUR + t] = p.sk[fit * p.NEUR + t]
        for t in range(K): p.repertoire[i * K + t] = p.repertoire[fit * K + t]

    def _mutate(self, i):
        p, rng, s = self.net.pool, self.rng, self.cfg.mutation_sigma
        R, H, S = p.R, p.H, p.S
        for t in range(R * H): p.w1[i * R * H + t] += rng.gauss(0.0, s)
        for t in range(H): p.b1[i * H + t] += rng.gauss(0.0, s)
        for t in range(H * S): p.w2[i * H * S + t] += rng.gauss(0.0, s)
        for t in range(S): p.b2[i * S + t] += rng.gauss(0.0, s)
        for t in range(p.NEUR):
            p.sa[i * p.NEUR + t] += rng.gauss(0.0, s / 10.0)
            p.sb[i * p.NEUR + t] += rng.gauss(0.0, s / 10.0)

    def cull_and_birth(self):
        doomed = self.cull()
        counts = self.birth(doomed) if doomed else {"immigrants": 0, "crossovers": 0, "clones": 0}
        return len(doomed), sum(counts.values()), counts

    # ------------------------------------------------------------------- ESS
    def ess_score(self, samples=None, iters=None):
        """Can a small mutation pay to enter?

        Near 1.0 means evolutionarily stable -- a converged population and, past
        a point, a stagnant one. Near 0.0 means anything can invade and the
        population has not found structure worth defending. The useful reading is
        the TRAJECTORY. Reported, plotted, never optimised against directly."""
        from gtmnn import game as gm, equilibrium as eq_mod
        cfg = self.cfg
        samples = samples or cfg.ess_samples
        p = self.net.pool
        alive = p.living()
        if len(alive) < 4 or not self.corpus:
            return {"ess_score": 1.0, "invaders": [], "mean_final_share": 0.0}
        failed = 0; shares = []; invaders = []
        for _ in range(samples):
            host = self.rng.choice(alive)
            clone = p.clone()
            victim = self.rng.choice([i for i in alive if i != host])
            self._copy_into(clone, victim, host)
            old_sigma = cfg.mutation_sigma
            self.cfg.mutation_sigma = old_sigma * 4.0
            saved = self.net.pool
            self.net.pool = clone
            self._mutate(victim)
            self.cfg.mutation_sigma = old_sigma
            text = self.rng.choice(self.corpus)
            pos = self.rng.randrange(len(text) + 1) if text else 0
            ctx = text[max(0, pos - self.net.hasher.context):pos]
            feat = self.net.hasher.transform(ctx)
            sg, alloc, _ = gm.build(clone, feat, len(self.net.alphabet), None,
                                    m=self.net.config.seats, charge=False)
            if not sg.seats:
                failed += 1; shares.append(0.0); continue
            e = eq_mod.solve(sg, "replicator", iters=min(iters or cfg.ess_iters, 64),
                             rng=self.rng)
            share = 0.0
            if victim in sg.seats:
                k = sg.seats.index(victim)
                share = 1.0 - e.profile[k][len(e.profile[k]) - 1]
            base = 1.0 / max(1, len(sg.seats))
            shares.append(share)
            if share <= base: failed += 1
            else: invaders.append(victim)
            self.net.pool = saved
        return {"ess_score": failed / max(1, samples), "invaders": invaders,
                "mean_final_share": sum(shares) / max(1, len(shares))}

    @staticmethod
    def _copy_into(pool, dst, src):
        R, H, S, K, N = pool.R, pool.H, pool.S, pool.K, pool.NEUR
        for t in range(R): pool.rf[dst * R + t] = pool.rf[src * R + t]
        for t in range(R * H): pool.w1[dst * R * H + t] = pool.w1[src * R * H + t]
        for t in range(H): pool.b1[dst * H + t] = pool.b1[src * H + t]
        for t in range(H * S): pool.w2[dst * H * S + t] = pool.w2[src * H * S + t]
        for t in range(S): pool.b2[dst * S + t] = pool.b2[src * S + t]
        for t in range(N):
            pool.sa[dst * N + t] = pool.sa[src * N + t]
            pool.sb[dst * N + t] = pool.sb[src * N + t]
        for t in range(K): pool.repertoire[dst * K + t] = pool.repertoire[src * K + t]

    # ------------------------------------------------------------- generation
    def run_generation(self, stop_event=None):
        t0 = time.time()
        cfg = self.cfg
        net = self.net
        games = 0
        rng = self.rng
        chars = sum(len(t) + 1 for t in self.corpus) or 1
        epochs = max(1, cfg.games_per_generation // chars)
        recs = net.train(self.corpus, __import__("dataclasses").replace(
            net.config, epochs=epochs), stop_event=stop_event)
        games = sum(r["games"] for r in recs)
        culled, born, counts = self.cull_and_birth()
        self.generation += 1
        net.generation = self.generation
        p = net.pool
        alive = p.living()
        w = [p.wealth[i] for i in alive] or [0.0]
        ess = self.ess_score()
        last = recs[-1] if recs else {}
        sample = ""
        try:
            sample = net.predict("", length=40, mode="sample", to_end=False).text
        except Exception:
            pass
        rec = {"generation": self.generation, "games": games,
               "seconds": round(time.time() - t0, 3),
               "culled": culled, "born": born, **counts,
               "alive": len(alive), "mean_wealth": sum(w) / len(w),
               "median_wealth": sorted(w)[len(w) // 2], "gini": gini(w),
               "mean_reputation": sum(p.reputation(i) for i in alive) / max(1, len(alive)),
               "mean_age": sum(p.age[i] for i in alive) / max(1, len(alive)),
               "loss": last.get("loss"), "belief_loss": last.get("belief_loss"),
               "uniform_loss": last.get("uniform_loss"),
               "accuracy": last.get("accuracy"), "diversity": last.get("diversity"),
               "coverage_gaps": coverage_gaps(p), "ess_score": ess["ess_score"],
               "cycle_rate": last.get("cycle_rate"), "meta_rate": last.get("meta_rate"),
               "mean_price": last.get("mean_price"), "sample": sample}
        self.history.append(rec)
        return rec

    def run(self, generations=0, stop_event=None, progress=None):
        out = []
        g = 0
        while generations == 0 or g < generations:
            if stop_event is not None and stop_event.is_set(): break
            rec = self.run_generation(stop_event)
            out.append(rec)
            if progress: progress(rec)
            g += 1
            if generations == 0 and g > 10 ** 9: break
        return out
