"""The cortex: a cyclic similarity graph partitioned into regions, each region
owning ONE Self-Building Neural Network over the common denominators of its
games. A game is routed to a region and played by that region's network.
"""
import random
from cortex.graph import SimilarityGraph, distance
from cortex.vocabulary import Vocabulary
from cortex.sbnn import SBNN

class Region:
    def __init__(self, rid, seed=0, nh=24):
        self.id = rid; self.games = []
        self.vocab = Vocabulary(); self.net = None
        self.seed = seed; self.nh = nh
        self.mech = frozenset()
        self.plays = 0.0; self.hits = 0.0

    def centroid(self):
        """Majority mechanics, not the intersection: an intersection shrinks
        toward empty as a region grows and stops being a usable position."""
        if not self.games: return frozenset()
        cnt = {}
        for g in self.games:
            for m in g.mechanics: cnt[m] = cnt.get(m, 0) + 1
        half = len(self.games) / 2.0
        return frozenset(m for m, c in cnt.items() if c >= half)

    def admit(self, game):
        """Add a game: extend the vocabulary, grow the network's inputs to match.
        Growth is an identity, so nothing already learned is disturbed."""
        self.games.append(game)
        old, new = self.vocab.extend(game.spec)
        self.mech = self.mech | game.mechanics
        if self.net is None:
            self.net = SBNN(new, nh=self.nh, seed=self.seed)
        elif new > old:
            self.net.grow_inputs(new - old)
        return new - old

    def reputation(self): return (self.hits + 1.0) / (self.plays + 2.0)
    def shape(self): return self.net.shape() if self.net else {}


class Cortex:
    def __init__(self, tau=0.75, tau_new=0.60, seed=0, nh=24):
        self.graph = SimilarityGraph(tau=tau)
        self.regions = []; self.route_of = {}
        self.tau_new, self.seed, self.nh = tau_new, seed, nh
        self.log = []

    def add_game(self, game):
        """Route to the nearest region, or found a new one if nothing is close
        enough. This is how the cortex grows areas: genuinely novel structure
        gets its own place rather than being forced into an existing one."""
        self.graph.add(game.name, game.mechanics)
        best, bd = None, 1e9
        for r in self.regions:
            d = distance(game.mechanics, r.centroid())
            if d < bd: best, bd = r, d
        if best is None or bd > self.tau_new:
            best = Region(len(self.regions), seed=self.seed + len(self.regions), nh=self.nh)
            self.regions.append(best)
            self.log.append({"game": game.name, "action": "founded", "region": best.id,
                             "nearest": round(bd, 3) if bd < 1e9 else None})
        else:
            self.log.append({"game": game.name, "action": "joined", "region": best.id,
                             "distance": round(bd, 3)})
        grew = best.admit(game)
        self.route_of[game.name] = best.id
        self.log[-1]["inputs_grown"] = grew
        return best

    def region_for(self, game): return self.regions[self.route_of[game.name]]

    def encode(self, region, game, state, mv):
        return region.vocab.encode(game.generalise(state, mv))

    # ----------------------------------------------------------------- train
    @staticmethod
    def _balance(game, s, cands, rng):
        """Equal legal and illegal. Candidate sets are naturally imbalanced (chess
        is 36% legal), and training on that teaches the prior rather than the rule."""
        good = [m for m in cands if game.is_legal(s, m)]
        bad  = [m for m in cands if m not in good]
        n = min(len(good), len(bad))
        if n == 0: return cands
        rng.shuffle(good); rng.shuffle(bad)
        out = good[:n] + bad[:n]; rng.shuffle(out)
        return out

    def train(self, game, episodes=200, rng=None, lr=0.05, k=12, balance=True):
        """Validity from the rules; grade from the game's reward. Two signals,
        two heads, trained in the same pass but never conflated."""
        rng = rng or random.Random(0)
        r = self.region_for(game)
        seen = correct = 0; grew = 0
        for ep in range(episodes):
            s = (game.random_state(rng) if hasattr(game, 'random_state')
                 else game.new(seed=rng.randrange(10**6)))
            cands = game.candidates(s, rng, k)
            if balance: cands = self._balance(game, s, cands, rng)
            for mv in cands:
                x = self.encode(r, game, s, mv)
                legal = 1.0 if game.is_legal(s, mv) else 0.0
                grade = game.reward(s, mv) if legal else 0.0
                r.net.step(x, valid=legal, grade=grade, lr=lr)
                pv, _ = r.net.predict(x)
                seen += 1; correct += 1 if (pv >= 0.5) == (legal >= 0.5) else 0
            grew += r.net.maybe_grow()
        return {"game": game.name, "region": r.id, "samples": seen,
                "train_acc": correct / max(1, seen), "hidden_grown": grew,
                "shape": r.net.shape()}

    def evaluate(self, game, n=60, rng=None, k=12):
        rng = rng or random.Random(99)
        r = self.region_for(game)
        seen = correct = 0; top_legal = 0; rounds = 0; legal_n = 0
        for _ in range(n):
            s = (game.random_state(rng) if hasattr(game, 'random_state')
                 else game.new(seed=rng.randrange(10**6)))
            cands = game.candidates(s, rng, k)
            if not cands: continue
            scored = []
            for mv in cands:
                x = self.encode(r, game, s, mv)
                pv, gr = r.net.predict(x)
                legal = game.is_legal(s, mv)
                seen += 1; legal_n += 1 if legal else 0
                correct += 1 if (pv >= 0.5) == legal else 0
                scored.append((pv * (0.5 + 0.5 * gr), mv, legal))
            scored.sort(key=lambda t: -t[0])
            rounds += 1; top_legal += 1 if scored[0][2] else 0
        base = legal_n / max(1, seen)
        return {"game": game.name,
                "legality_acc": round(correct / max(1, seen), 3),
                "majority_baseline": round(max(base, 1.0 - base), 3),
                "top_choice_legal": round(top_legal / max(1, rounds), 3),
                "random_pick_baseline": round(base, 3)}

    def choose(self, game, state, rng, k=24):
        """Rank candidates by p_valid * grade. The network's own pick -- NOT
        filtered by the rules, so `top_choice_legal` measures what it learned."""
        r = self.region_for(game)
        best, bs = None, -1e9
        for mv in game.candidates(state, rng, k):
            pv, gr = r.net.predict(self.encode(r, game, state, mv))
            sc = pv * (0.5 + 0.5 * gr)
            if sc > bs: best, bs = mv, sc
        return best

    def stats(self):
        return {"regions": len(self.regions),
                "metric_violations": self.graph.check_metric(),
                "detail": [{"id": r.id, "games": [g.name for g in r.games],
                            "vocab_width": r.vocab.width, "shape": r.shape()}
                           for r in self.regions]}
