"""Playing CyclicCortex's games with a GTMNN population.

The population learns ONE question -- is this candidate legal? -- across every
game at once, because the game modifier (games.py) puts the game's identity in
the input rather than in a different network. This is GREN/DESIGN.md 22.6's
validity head: the cheap, abundant, transferable signal, trained first and
separately from grade.
"""
import math, os, random, sys
from array import array

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CC = os.path.join(_REPO, "CyclicCortex")
if _CC not in sys.path: sys.path.insert(0, _CC)

from gtmnn import equilibrium as eq_mod, game as game_mod, shapley as shap_mod
from gtmnn.backend import get_backend
from gtmnn.games import GameFeatures, load_mechanics, state_tokens, cand_tokens, ILLEGAL, LEGAL
from gtmnn.micro import MicroPool
from gtmnn.model import TrainConfig, gini, solver_for


def adapters(names=("chess", "checkers", "go", "sudoku")):
    from cortex.games import ALL
    return {n: ALL[n] for n in names if n in ALL}


class GamePlayer:
    """A micro population that scores one candidate at a time, in any game."""

    def __init__(self, mechanics=None, n=256, R=12, H=6, seed=0,
                 f_state=128, f_cand=64, f_mod=64, modifier_on=True):
        self.mech = mechanics if mechanics is not None else load_mechanics()[0]
        self.feats = GameFeatures(self.mech, f_state, f_cand, f_mod, seed=seed)
        self.modifier_on = modifier_on
        # |V| = 2: every micro's repertoire is exactly {ILLEGAL, LEGAL}, so the
        # action set is that plus ABSTAIN and the congestion game is unchanged.
        self.pool = MicroPool(n, self.feats.dim, R=R, H=H, K=2, seed=seed, alphabet_size=2)
        for i in range(n):
            self.pool.repertoire[i * 2 + 0] = ILLEGAL
            self.pool.repertoire[i * 2 + 1] = LEGAL
        self.backend = get_backend("python")
        self.rng = random.Random(seed)
        self._pending = []

    def _feat(self, name, s, mv):
        f = self.feats.transform(name, state_tokens(name, s), cand_tokens(name, s, mv))
        if not self.modifier_on:
            base = self.feats.f_state + self.feats.f_cand
            for i in range(base, self.feats.dim): f[i] = 0.0
        return f

    def score(self, name, s, mv, cfg):
        """P(legal) for one candidate, from the equilibrium aggregate."""
        feat = self._feat(name, s, mv)
        ids, q = self.backend.forward_population(self.pool, feat)
        sg, alloc, _ = game_mod.build(self.pool, feat, 2, None, q=q, m=cfg.seats,
                                      reserve=cfg.reserve, charge=False,
                                      B=cfg.B, lam=cfg.lam, ids=ids)
        if not sg.seats: return 0.5, None, None
        eq = eq_mod.solve(sg, solver_for(sg.payoff, len(sg.seats), cfg.solver),
                          iters=cfg.iters, rng=self.rng, pool=self.pool)
        agg = game_mod.aggregate(sg, eq.profile, self.pool)
        return agg[LEGAL], sg, eq

    def learn_one(self, name, s, mv, y, cfg):
        feat = self._feat(name, s, mv)
        ids, q = self.backend.forward_population(self.pool, feat)
        sg, alloc, _ = game_mod.build(self.pool, feat, 2, y, q=q, m=cfg.seats,
                                      reserve=cfg.reserve, charge=True,
                                      B=cfg.B, lam=cfg.lam, ids=ids)
        if not sg.seats: return None
        eq = eq_mod.solve(sg, solver_for(sg.payoff, len(sg.seats), cfg.solver),
                          iters=cfg.iters, rng=self.rng, pool=self.pool)
        phi = shap_mod.shapley_values(sg, eq.profile, self.pool, y,
                                      permutations=cfg.permutations, rng=self.rng)
        us = sg.utilities(eq.actions)
        self.pool.discount_reputation()
        for seat, i in enumerate(sg.seats):
            self.pool.wealth[i] += us[seat]
            sym = sg.ctx.slot_symbol[seat][eq.actions[seat]]
            if sym is not None:
                self.pool.plays[i] += 1.0
                if sym == y: self.pool.hits[i] += 1.0
            self.pool.age[i] += 1
            self._pending.append((i, list(eq.profile[seat]), phi.phi[seat], feat))
        if len(self._pending) >= cfg.batch: self.flush(cfg)
        return phi

    def flush(self, cfg):
        if not self._pending: return
        self.backend.learn_batch(self.pool, self._pending, cfg.lr, cfg.act_lr, cfg.clip)
        self._pending = []


def samples(game, name, rng, positions=20, per_position=8):
    """Balanced legal / illegal candidates. Training on the natural imbalance
    teaches the prior, not the rule."""
    out = []
    for _ in range(positions):
        s = (game.random_state(rng) if hasattr(game, "random_state")
             else game.new(seed=rng.randrange(10 ** 6)))
        cands = game.candidates(s, rng, per_position)
        good = [m for m in cands if game.is_legal(s, m)]
        bad = [m for m in cands if m not in good]
        k = min(len(good), len(bad), max(1, per_position // 2))
        if k == 0: continue
        rng.shuffle(good); rng.shuffle(bad)
        for m in good[:k]: out.append((s, m, LEGAL))
        for m in bad[:k]: out.append((s, m, ILLEGAL))
    rng.shuffle(out)
    return out


def evaluate(player, game, name, rows, cfg):
    hit = 0; nll = 0.0
    for s, mv, y in rows:
        p, _, _ = player.score(name, s, mv, cfg)
        if (p >= 0.5) == (y == LEGAL): hit += 1
        nll -= math.log(max(p if y == LEGAL else 1.0 - p, 1e-12))
    n = max(1, len(rows))
    return {"acc": hit / n, "nll": nll / n, "n": n}


def train_on(player, games, names, cfg, rounds=3, positions=20, per_position=8, seed=0):
    rng = random.Random(seed)
    for r in range(rounds):
        for name in names:
            for s, mv, y in samples(games[name], name, rng, positions, per_position):
                player.learn_one(name, s, mv, y, cfg)
    player.flush(cfg)
    return player
