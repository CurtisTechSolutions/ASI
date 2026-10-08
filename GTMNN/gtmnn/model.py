"""GTMNet -- the whole system. Every other module exists to be called from step().
"""
import gzip, json, math, os, random, time
from array import array
from dataclasses import dataclass, asdict, field

from gtmnn import auction, coalition as coal_mod, game as game_mod
from gtmnn import equilibrium as eq_mod, meta as meta_mod, shapley as shap_mod
from gtmnn.backend import get_backend
from gtmnn.features import Alphabet, FeatureHasher, coverage_gaps
from gtmnn.micro import MicroPool
from gtmnn.payoff import CorrectnessCongestion, BeliefCongestion, Inverted
from gtmnn.trie import build as build_trie, classify as classify_game

FORMAT_VERSION = 1


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 0.35                 # NOT 0.05. phi is a marginal contribution and
                                     # a seat's mean |phi| is ~0.15, so at lr 0.05
                                     # the credit path learned ~7x slower than the
                                     # same micros under a unit supervised signal.
                                     # Measured (real inference loss, 60 epochs):
                                     # 2.781 at 0.05 against 2.353 at 0.35, with
                                     # uniform at 2.708. See README.
    act_lr: float = 0.035            # an order of magnitude slower: the activation
                                     # is shared structure and should move slower
                                     # than the weights that use it
    seats: int = 64
    iters: int = 64
    solver: str = "auto"        # "auto" asks gtmnn/trie.py what is PROVABLE about
                                # the payoff in hand and takes that solver
    permutations: int = 32
    B: float = 1.0
    lam: float = 0.5
    reserve: float = 0.01
    batch: int = 64
    clip: float = 5.0
    meta: bool = True
    max_depth: int = 2
    margin_tau: float = 0.05
    expl_tau: float = 0.05
    evolve_every: int = 0
    repertoire_every: int = 512
    shuffle: bool = True
    seed: int = 0
    coalitions: bool = True
    credit: str = "shapley"    # "shapley": advantage = phi; a null player takes no step
                               # "regret":  advantage = phi - marginal(best alternative),
                               #            which DOES teach abstention -- and collapses
                               #            the inference game to uniform. Abstainers are
                               #            77% of seats and take 76% of the gradient;
                               #            q_abstain rises, bids fall under the reserve,
                               #            nobody is seated, the aggregate is the eps
                               #            smoothing. Kept so the collapse is reproducible.

    def to_dict(self): return asdict(self)


@dataclass
class StepResult:
    agg: array
    eq: object
    alloc: object
    coalitions: list = field(default_factory=list)
    meta: object = None
    phi: object = None
    game: object = None


@dataclass
class CharStep:
    symbol: str; p: float; entropy: float
    seats: int; price: float; bidders: int
    exploitability: float; cycle: int; converged: bool; iterations: int
    meta: object = None; coalitions: int = 0
    top: list = field(default_factory=list)

    def to_dict(self):
        d = {"symbol": self.symbol, "p": round(self.p, 6),
             "entropy": round(self.entropy, 6), "seats": self.seats,
             "price": round(self.price, 6), "bidders": self.bidders,
             "exploitability": round(self.exploitability, 6), "cycle": self.cycle,
             "converged": self.converged, "iterations": self.iterations,
             "coalitions": self.coalitions,
             "top": [(s, round(p, 6)) for s, p in self.top]}
        if self.meta: d["meta"] = self.meta
        return d


@dataclass
class Prediction:
    prefix: str; text: str; full_text: str
    steps: list
    logprob: float; perplexity: float

    def to_dict(self):
        return {"prefix": self.prefix, "text": self.text, "full_text": self.full_text,
                "logprob": round(self.logprob, 6), "perplexity": round(self.perplexity, 6),
                "steps": [s.to_dict() for s in self.steps]}


_TRIE = None


def solver_for(payoff, seats, requested="auto"):
    """Which solver to run, derived rather than configured.

    `auto` classifies the payoff on the trie's axes and takes the deepest rule
    whose premises the path satisfies -- best_response where a potential function
    proves the finite improvement property, fictitious on a two-player zero-sum
    game, regret matching where nothing stronger is provable. Anything else is
    passed through unchanged, so a caller can still force one.

    This is what makes the trie load-bearing instead of a diagram: the solver is
    a consequence of what is provable about the game in hand, and a new payoff
    gets the right solver by answering the same questions rather than by someone
    remembering to add a branch here."""
    if requested and requested != "auto": return requested
    global _TRIE
    if _TRIE is None: _TRIE = build_trie()[0]
    _, facts = classify_game(payoff, seats=seats)
    node = _TRIE.walk(facts)[-1]
    return _TRIE.solver(node)[0]


def gini(xs):
    """The specialisation readout. A population where everyone earns the same is
    a population that has not differentiated."""
    v = sorted(xs)
    n = len(v)
    if n == 0: return 0.0
    lo = min(v)
    if lo < 0.0: v = [x - lo for x in v]        # Gini needs non-negatives
    s = sum(v)
    if s <= 0.0: return 0.0
    cum = 0.0
    for i, x in enumerate(v, 1): cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


class GTMNet:
    def __init__(self, n=4096, F=256, R=8, H=6, K=4, meta_n=256, seed=0,
                 backend="auto", device=None, context=12, ngrams=(1, 2, 3)):
        self.alphabet = Alphabet()
        self.hasher = FeatureHasher(dim=F, context=context, ngrams=ngrams, seed=seed)
        self.pool = MicroPool(n, F, R, H, K, seed=seed, alphabet_size=len(self.alphabet))
        self.meta_pool = MicroPool(meta_n, meta_mod.META_F, R=6, H=6, K=4,
                                   seed=seed + 1, alphabet_size=len(meta_mod.POLICY_NAMES))
        self.backend = get_backend(backend, device)
        self.history = []
        self.generation = 0
        self.inverted = False
        self.config = TrainConfig(seed=seed)
        self.rng = random.Random(seed)
        self._games = 0
        self._pending = []
        self._meta_pending = []

    # ------------------------------------------------------------------ step
    def step(self, context, y, cfg, learn=True, charge=True):
        """One character, end to end. The whole system in one function."""
        pool, V = self.pool, len(self.alphabet)
        feat = self.hasher.transform(context)
        ids, q = self.backend.forward_population(pool, feat)
        payoff = None
        if self.inverted:
            payoff = Inverted(CorrectnessCongestion() if y is not None else BeliefCongestion())
        sg, alloc, _ = game_mod.build(pool, feat, V, y, q=q, m=cfg.seats,
                                      reserve=cfg.reserve, charge=charge and learn,
                                      B=cfg.B, lam=cfg.lam, ids=ids, payoff=payoff)
        if not sg.seats:
            agg = array("d", [1.0 / V]) * V
            return StepResult(agg, eq_mod.Equilibrium([], [], 0, True), alloc, [], None, None, sg)

        method = solver_for(sg.payoff, len(sg.seats), cfg.solver)
        eq = eq_mod.solve(sg, method, iters=cfg.iters, rng=self.rng, pool=pool)
        coals = (coal_mod.propose(pool, sg.seats, eq.profile, sg.ctx, rng=self.rng)
                 if cfg.coalitions else [])
        agg = game_mod.aggregate(sg, eq.profile, pool)

        mres = None
        if cfg.meta:
            t = meta_mod.triggered(eq, alloc, agg, cfg.seats, cfg.margin_tau, cfg.expl_tau)
            if t:
                mres = meta_mod.resolve(eq, alloc, sg, agg, pool, self.meta_pool,
                                        0, cfg.max_depth, self.rng, t, cfg.iters,
                                        coalitions=coals, solver=method)
                agg = mres.distribution

        phi = None
        if learn and y is not None:
            phi = shap_mod.shapley_values(sg, eq.profile, pool, y,
                                          permutations=cfg.permutations, rng=self.rng)
            adv = (shap_mod.regret_advantages(sg, eq.profile, pool, y, phi.phi)
                   if cfg.credit == "regret" else phi.phi)
            us = sg.utilities(eq.actions)
            pool.discount_reputation()
            for seat, i in enumerate(sg.seats):
                pool.wealth[i] += us[seat]
                slot = eq.actions[seat]
                sym = sg.ctx.slot_symbol[seat][slot]
                if sym is not None:
                    pool.plays[i] += 1.0
                    if sym == y: pool.hits[i] += 1.0
                pool.age[i] += 1
                if phi.phi[seat] > 0.0: pool.note_symbol(i, y, phi.phi[seat])
                self._pending.append((i, list(eq.profile[seat]), adv[seat], feat))
            self._games += 1
        return StepResult(agg, eq, alloc, coals, mres, phi, sg)

    # ----------------------------------------------------------------- train
    def train(self, texts, config=None, progress=None, stop_event=None):
        cfg = config or self.config
        self.config = cfg
        before = len(self.alphabet)
        self.alphabet.fit(texts)
        if len(self.alphabet) != before:
            for i in range(self.pool.n):
                if self.pool.alive[i]: self.pool.fill_repertoire(i, len(self.alphabet))
        records = []
        rng = random.Random(cfg.seed)
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            order = list(texts)
            if cfg.shuffle: rng.shuffle(order)
            nll = 0.0; chars = 0; correct = 0; games = 0
            expl = 0.0; conv = 0; cyc = 0; metas = 0
            prices = 0.0; bidders = 0.0; shap_tot = 0.0
            policies = {}
            diversity = 0.0
            for text in order:
                if stop_event is not None and stop_event.is_set(): break
                for p in range(len(text) + 1):
                    ctx = text[max(0, p - self.hasher.context):p]
                    y = self.alphabet.encode(text[p]) if p < len(text) else 1   # END
                    r = self.step(ctx, y, cfg, learn=True)
                    pv = r.agg[y] if y < len(r.agg) else 1e-12
                    nll -= math.log(max(pv, 1e-12)); chars += 1
                    if max(range(len(r.agg)), key=lambda c: r.agg[c]) == y: correct += 1
                    games += 1
                    expl += r.eq.exploitability
                    conv += 1 if r.eq.converged else 0
                    cyc += 1 if r.eq.cycle else 0
                    prices += r.alloc.price; bidders += r.alloc.bidders
                    if r.phi is not None: shap_tot += r.phi.total
                    if r.meta is not None:
                        metas += 1
                        policies[r.meta.policy_name] = policies.get(r.meta.policy_name, 0) + 1
                    if r.game is not None and r.game.seats:
                        loads = r.game.loads(r.eq.actions)
                        diversity += sum(1 for c in loads[:-1] if c > 0) / len(r.game.seats)
                    if len(self._pending) >= cfg.batch: self._flush(cfg)
                    if cfg.repertoire_every and self._games % cfg.repertoire_every == 0:
                        for i in self.pool.living():
                            self.pool.refresh_repertoire(i, len(self.alphabet))
            self._flush(cfg)
            g = max(1, games)
            rec = {"epoch": epoch, "loss": nll / max(1, chars),
                   "perplexity": math.exp(min(50.0, nll / max(1, chars))),
                   "accuracy": correct / max(1, chars), "chars": chars, "games": games,
                   "seconds": round(time.time() - t0, 3),
                   "alive": self.pool.count_alive(), "culled": 0, "born": 0,
                   "generation": self.generation,
                   "mean_wealth": sum(self.pool.wealth) / self.pool.n,
                   "gini": gini([self.pool.wealth[i] for i in self.pool.living()]),
                   "mean_price": prices / g, "mean_bidders": bidders / g,
                   "mean_exploitability": expl / g, "converged_frac": conv / g,
                   "cycle_rate": cyc / g, "meta_rate": metas / g,
                   "meta_policies": policies, "diversity": diversity / g,
                   "shapley_total": shap_tot / g,
                   "mean_reputation": sum(self.pool.reputation(i) for i in self.pool.living())
                                      / max(1, self.pool.count_alive()),
                   "coverage_gaps": coverage_gaps(self.pool),
                   # `loss` above is the TRAINING-game aggregate, whose payoff
                   # contains y. These two are the honest pair.
                   "belief_loss": self.belief_loss(order, stride=8),
                   "uniform_loss": self.uniform_loss()}
            records.append(rec); self.history.append(rec)
            if progress: progress(rec)
        return records

    def _flush(self, cfg):
        if not self._pending: return
        self.backend.learn_batch(self.pool, self._pending, cfg.lr, cfg.act_lr, cfg.clip)
        self._pending = []

    # ------------------------------------------------------- honest metrics
    def belief_loss(self, texts, stride=1):
        """Mean -log P(y) under the population's OWN beliefs, with no game solved
        at all: reputation-weighted q, aggregated exactly as game.aggregate does.

        This exists because the epoch record's `loss` CANNOT be read as a
        predictive loss. That number is the aggregate of a CorrectnessCongestion
        equilibrium, and that payoff function contains y -- the seats are being
        paid for naming the answer they were shown. It is the right quantity to
        pair with shapley_total (they are two views of v(N) - v(0)) and the wrong
        quantity to call accuracy. Measured on a 2-text corpus: `loss` 0.59 while
        the population's actual predictive loss was 4.2, against a uniform
        baseline of 2.77."""
        V = len(self.alphabet)
        if V < 2: return 0.0
        nll = 0.0; n = 0
        for text in texts:
            for p in range(0, len(text) + 1, stride):
                ctx = text[max(0, p - self.hasher.context):p]
                y = self.alphabet.encode(text[p]) if p < len(text) else 1
                feat = self.hasher.transform(ctx)
                ids, qs = self.backend.forward_population(self.pool, feat)
                sc = array("d", [1e-3 / V]) * V; tot = 1e-3
                for i, q in zip(ids, qs):
                    rho = self.pool.reputation(i)
                    for sl, sym in enumerate(self.pool.slots(i)[:-1]):
                        if sym == -1 or sym >= V: continue
                        w = rho * q[sl]; sc[sym] += w; tot += w
                nll -= math.log(max(sc[y] / tot, 1e-12)); n += 1
        return nll / max(1, n)

    def uniform_loss(self):
        return math.log(len(self.alphabet)) if len(self.alphabet) > 1 else 0.0

    # ------------------------------------------------------------ prediction
    def _char_step(self, ctx, cfg):
        r = self.step(ctx, None, cfg, learn=False, charge=False)
        return r

    def predict(self, prefix, length=None, mode="argmax", temperature=1.0, top_k=5,
                to_end=True, cfg=None, rng=None):
        cfg = cfg or self.config
        rng = rng or self.rng
        if length is None and not to_end:
            raise ValueError("predict needs a length or to_end=True, or it cannot stop")
        ctx, out, steps, lp = prefix, [], [], 0.0
        limit = length if length is not None else 512
        for _ in range(limit):
            r = self._char_step(ctx[-self.hasher.context:], cfg)
            agg = r.agg
            if mode == "argmax":
                c = max(range(len(agg)), key=lambda i: agg[i])
            else:
                w = [a ** (1.0 / max(1e-6, temperature)) for a in agg]
                t = sum(w); x = rng.random() * t; acc = 0.0; c = len(w) - 1
                for i, v in enumerate(w):
                    acc += v
                    if x <= acc: c = i; break
            p = agg[c]; lp += math.log(max(p, 1e-12))
            steps.append(CharStep(
                symbol=self.alphabet.decode(c), p=p, entropy=game_mod.entropy(agg),
                seats=len(r.game.seats) if r.game else 0, price=r.alloc.price,
                bidders=r.alloc.bidders, exploitability=r.eq.exploitability,
                cycle=r.eq.cycle, converged=r.eq.converged, iterations=r.eq.iterations,
                meta=r.meta.to_dict() if r.meta else None, coalitions=len(r.coalitions),
                top=game_mod.top_k(agg, self.alphabet, top_k)))
            if c == 1 and to_end: break                    # END
            ch = self.alphabet.decode(c)
            out.append(ch); ctx += ch
        text = "".join(out)
        n = max(1, len(steps))
        return Prediction(prefix, text, prefix + text, steps, lp,
                          math.exp(min(50.0, -lp / n)))

    def generate(self, count=5, max_length=60, prefix="", mode="sample",
                 temperature=1.0, cfg=None, seed=None):
        rng = random.Random(seed if seed is not None else self.config.seed)
        return [self.predict(prefix, max_length, mode, temperature, cfg=cfg, rng=rng)
                for _ in range(count)]

    def score(self, text, cfg=None):
        """No learning, no wealth movement, no auction charges: scoring must not
        perturb the model."""
        cfg = cfg or self.config
        lp = 0.0; n = 0; expl = 0.0; cyc = 0
        for p in range(len(text) + 1):
            ctx = text[max(0, p - self.hasher.context):p]
            y = self.alphabet.encode(text[p]) if p < len(text) else 1
            r = self.step(ctx, None, cfg, learn=False, charge=False)
            lp += math.log(max(r.agg[y] if y < len(r.agg) else 1e-12, 1e-12))
            expl += r.eq.exploitability; cyc += 1 if r.eq.cycle else 0
            n += 1
        n = max(1, n)
        return {"text": text, "logprob": lp, "per_char": lp / n,
                "perplexity": math.exp(min(50.0, -lp / n)), "chars": n,
                "mean_exploitability": expl / n, "cycle_rate": cyc / n}

    # ----------------------------------------------------------------- 2NRL
    def invert(self):
        self.pool.invert(); self.meta_pool.invert()
        self.inverted = not self.inverted

    def two_nrl(self, bad, good, neg_epochs=1, pos_epochs=1, neg_lr=0.05, pos_lr=0.01,
                cfg=None, progress=None, stop_event=None):
        """Learn the garbage properly, invert, then fine-tune on truth more slowly.

        Inverting a POPULATION is cheaper and better behaved than inverting a
        graph: a -> -a on every sine reflects every micro's response about its own
        operating point without touching a single weight, so who reads which
        features, who is coalitioned with whom and who is solvent all survive
        intact."""
        import copy
        base = cfg or self.config
        neg = copy.replace(base, epochs=neg_epochs, lr=neg_lr) if hasattr(copy, "replace") \
              else TrainConfig(**{**base.to_dict(), "epochs": neg_epochs, "lr": neg_lr})
        pos = TrainConfig(**{**base.to_dict(), "epochs": pos_epochs, "lr": pos_lr})
        r1 = self.train(bad, neg, progress, stop_event)
        self.invert()
        r2 = self.train(good, pos, progress, stop_event)
        return {"negative": r1, "positive": r2, "inverted": self.inverted}

    # ----------------------------------------------------------------- stats
    def stats(self):
        alive = self.pool.living()
        return {"alive": len(alive), "capacity": self.pool.n, "generation": self.generation,
                "alphabet": len(self.alphabet), "features": self.pool.F,
                "R": self.pool.R, "H": self.pool.H, "K": self.pool.K,
                "seats": self.config.seats, "inverted": self.inverted,
                "mean_wealth": sum(self.pool.wealth) / self.pool.n,
                "gini": gini([self.pool.wealth[i] for i in alive]),
                "mean_reputation": sum(self.pool.reputation(i) for i in alive) / max(1, len(alive)),
                "coverage_gaps": coverage_gaps(self.pool),
                "meta_alive": self.meta_pool.count_alive(),
                "backend": self.backend.describe(), "history_len": len(self.history)}

    # --------------------------------------------------------- serialisation
    def to_dict(self):
        return {"format_version": FORMAT_VERSION, "alphabet": self.alphabet.to_dict(),
                "hasher": self.hasher.to_dict(), "pool": self.pool.to_dict(),
                "meta_pool": self.meta_pool.to_dict(), "history": self.history,
                "generation": self.generation, "inverted": self.inverted,
                "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, d, backend="auto", device=None):
        fv = d.get("format_version")
        if fv != FORMAT_VERSION:
            raise ValueError(f"model format {fv}, this build reads {FORMAT_VERSION}")
        p = MicroPool.from_dict(d["pool"])
        net = cls.__new__(cls)
        net.alphabet = Alphabet.from_dict(d["alphabet"])
        net.hasher = FeatureHasher.from_dict(d["hasher"])
        net.pool = p
        net.meta_pool = MicroPool.from_dict(d["meta_pool"])
        net.backend = get_backend(backend, device)
        net.history = list(d["history"]); net.generation = d["generation"]
        net.inverted = d["inverted"]; net.config = TrainConfig(**d["config"])
        net.rng = random.Random(net.config.seed)
        net._games = 0; net._pending = []; net._meta_pending = []
        return net

    def save(self, path, gzip_it=None):
        blob = json.dumps(self.to_dict()).encode("utf-8")
        use_gz = path.endswith(".gz") if gzip_it is None else gzip_it
        d = os.path.dirname(os.path.abspath(path))
        if d: os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        if use_gz:
            with gzip.open(tmp, "wb") as f: f.write(blob)
        else:
            with open(tmp, "wb") as f: f.write(blob)
        os.replace(tmp, path)
        return os.path.getsize(path)

    @classmethod
    def load(cls, path, backend="auto", device=None):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f: d = json.loads(f.read().decode("utf-8"))
        return cls.from_dict(d, backend, device)


def new_model(**kw): return GTMNet(**kw)
def load_model(path, **kw): return GTMNet.load(path, **kw)
