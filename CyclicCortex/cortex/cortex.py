"""The cortex: a cyclic similarity graph partitioned into regions, each region
owning ONE Self-Building Neural Network over the common denominators of its
games. A game is routed to a region and played by that region's network.
"""
import math, random
from cortex.graph import SimilarityGraph, distance
from cortex.vocabulary import Vocabulary
from cortex.sbnn import SBNN

def _gated(scored, tau=0.5):
    """Among moves believed legal, take the best grade. If none clears the
    threshold, fall back to the most-likely-legal move."""
    believed = [t for t in scored if t[0] >= tau]
    if believed: return max(believed, key=lambda t: t[1])[2]
    return max(scored, key=lambda t: t[0])[2]


class Region:
    def __init__(self, rid, seed=0, nh=24):
        self.id = rid; self.games = []
        self.vocab = Vocabulary(); self.net = None
        self.seed = seed; self.nh = nh
        self.mech = frozenset()
        self.rule = "pv"                 # chosen by measurement in select_rule
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

    @staticmethod
    def _states(game, rng, n, selfplay=0.5, opponent=None, max_plies=120):
        """Training positions. Random states alone are NOT enough: a network
        trained on random midgames and then asked to play from the opening is
        answering a different distribution, and its illegal rate in real play is
        far worse than its evaluation suggests. Half the positions therefore come
        from actual games."""
        out = []
        while len(out) < n:
            if opponent is not None and rng.random() < selfplay:
                s = game.new(seed=rng.randrange(10**6)); plies = 0
                while plies < max_plies and not game.terminal(s):
                    out.append(s)
                    mvs = game.legal_moves(s)
                    if not mvs: break
                    mv = opponent(s, 1 if rng.random() < 0.5 else 0, rng)
                    if mv is None or not game.is_legal(s, mv): mv = rng.choice(mvs)
                    s = game.apply(s, mv); plies += 1
                    if len(out) >= n: break
            elif hasattr(game, "random_state"):
                out.append(game.random_state(rng))
            else:
                out.append(game.new(seed=rng.randrange(10**6)))
        return out[:n]

    def train(self, game, episodes=200, rng=None, lr=0.05, k=12, balance=True,
              opponent=None, selfplay=0.5):
        """Validity from the rules; grade from the game's reward. Two signals,
        two heads, trained in the same pass but never conflated."""
        rng = rng or random.Random(0)
        r = self.region_for(game)
        seen = correct = 0; grew = 0
        states = self._states(game, rng, episodes, selfplay, opponent)
        for s in states:
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
        r.rule = self.select_rule(game, rng, opponent, selfplay)
        return {"game": game.name, "region": r.id, "samples": seen,
                "train_acc": round(correct / max(1, seen), 3), "hidden_grown": grew,
                "rule": r.rule, "shape": r.net.shape()}

    def select_rule(self, game, rng, opponent=None, selfplay=0.5, n=40, k=12):
        """Pick the region's decision rule by how well it PLAYS, not merely by
        how often its pick is legal.

        Selecting on legality alone was a real defect: after distillation the
        validity-only rule scored a perfect 1.000 top-choice-legal, was therefore
        selected, and discarded the grade head entirely -- so the material result
        was the WORST of every configuration despite the cleanest legality. A
        rule that always plays a legal blunder is not a good rule.

        The objective is the value of the position the move leads to, with an
        illegal pick scoring the worst value available, so legality is enforced by
        consequence rather than by being the whole target.
        """
        r = self.region_for(game)
        # Select on the distribution the rule will FACE. Using the training
        # mixture instead let checkers pick a rule scoring 0.425 in evaluation and
        # 0.04 in actual play -- the rule was chosen for the wrong distribution.
        states = self._states(game, random.Random(12345), n,
                              1.0 if opponent else 0.0, opponent)
        cache = []
        for s in states:
            cand = game.candidates(s, rng, k)
            if not cand: continue
            cache.append((s, [(*r.net.predict(self.encode(r, game, s, mv)), mv)
                              for mv in cand]))
        me = "b" if game.name == "go" else "w"
        has_value = hasattr(game, "value")
        best, bestname = -1e9, "pv"
        for name in self.RULES:
            total = 0.0
            for st, sc in cache:
                mv = self.rank(sc, name)
                if not game.is_legal(st, mv):
                    total -= 1.0                       # an illegal pick is the worst case
                elif has_value:
                    turn = getattr(st, "turn", me)
                    v = game.value(game.apply(st, mv), turn)
                    total += v
                else:
                    total += 1.0
            score = total / max(1, len(cache))
            if score > best: best, bestname = score, name
        return bestname

    # ----------------------------------------------------------------- replay
    def rehearse(self, region, rng=None, episodes=200, lr=0.05, k=12, opponent=None):
        """Interleave every game in a region when one of them is new.

        Growth is an identity (SBNN §8.2), so admitting a game changes nothing at
        the moment it happens. What is NOT protected is the training that
        follows: train only on the newcomer and the incumbents are overwritten by
        ordinary catastrophic forgetting. Rehearsal interleaves them so the
        region keeps what it had.

        NeuralCompression/FINDINGS.md §12 measured replay as the better of the two
        consolidation routes -- it transfers the function rather than the
        parameters -- with a small margin at matched compute. It is used here for
        RETENTION, which is what it is actually good at.
        """
        rng = rng or random.Random(0)
        if not region.games: return {"rehearsed": [], "samples": 0}
        per = max(1, episodes // len(region.games))
        seen = 0
        order = []
        for _ in range(per):
            for g in region.games: order.append(g)
        rng.shuffle(order)
        for g in order:
            st = (g.random_state(rng) if hasattr(g, "random_state")
                  else g.new(seed=rng.randrange(10**6)))
            cands = self._balance(g, st, g.candidates(st, rng, k), rng)
            for mv in cands:
                legal = 1.0 if g.is_legal(st, mv) else 0.0
                region.net.step(self.encode(region, g, st, mv), valid=legal,
                                grade=g.reward(st, mv) if legal else 0.0, lr=lr)
                seen += 1
        return {"rehearsed": [g.name for g in region.games], "samples": seen}

    def add_game_and_rehearse(self, game, rng=None, episodes=200, **kw):
        """Admit a game and immediately rehearse the region it joined."""
        r = self.add_game(game)
        if len(r.games) > 1 and r.net is not None:
            return r, self.rehearse(r, rng, episodes, **kw)
        return r, {"rehearsed": [], "samples": 0}

    # ------------------------------------------------------------ distillation
    def distill(self, game, teacher, positions=150, rng=None, lr=0.05,
                depth=8, multipv=24, scale=400.0, opponent=None, selfplay=0.5):
        """Train GRADE from a teacher's per-move evaluation.

        Self-play gives ONE number per game, spread over ~100 moves by a
        discount. A teacher scores EVERY legal move in EVERY position, so one
        position yields ~24 supervised targets instead of a share of one. That is
        the difference between sparse and dense supervision, and it is the whole
        reason to use an engine as a teacher rather than only as an opponent.

        Centipawns are squashed to [-1,1] by tanh(cp/scale): grade must stay
        bounded and comparable across games, and a linear map would let a mate
        score dominate every ordinary move.
        """
        rng = rng or random.Random(0)
        r = self.region_for(game)
        states = self._states(game, rng, positions, selfplay, opponent)
        scored = skipped = 0
        for st in states:
            try:
                tbl = teacher.score_moves(st, depth=depth, multipv=multipv)
            except Exception:
                skipped += 1; continue
            if not tbl: skipped += 1; continue
            for mv, cp in tbl.items():
                if not game.is_legal(st, mv): continue      # teacher outranged us
                x = self.encode(r, game, st, mv)
                # GRADE ONLY. Training valid=1.0 here floods the validity head
                # with positives -- the teacher only ever scores legal moves --
                # and measurably wrecked it: legality 0.925 -> 0.748, and playing
                # strength with it. Two heads, two signals, never conflated.
                r.net.step(x, grade=math.tanh(cp / scale), lr=lr)
                scored += 1
        return {"game": game.name, "positions": len(states), "moves_scored": scored,
                "skipped": skipped, "shape": r.net.shape()}

    # --------------------------------------------------------- outcome credit
    def selfplay(self, game, rounds=40, rng=None, opponent=None, side=None,
                 gamma=None, lr=0.05, eps=0.25, max_plies=200, k=16,
                 opponent_depth=1, train_valid=True):
        """Train the GRADE head from what actually happened.

        V1's grade was a hand-written heuristic -- material delta, real captures,
        enabled captures -- which is why the go region could play 125 legal plies
        and still lose 0-81: legal and GOOD are different questions and only the
        first was trained. Here the cortex plays, the game ends, and every move it
        made is credited with the discounted outcome.

        This also closes the distribution gap V1 measured. Evaluation sampled
        positions from ENGINE games while play reached positions the cortex's own
        weak moves lead to; self-play trains on exactly the latter.

        Moves are chosen from LEGAL moves only: grade is a question about legal
        moves, and letting the cortex play illegally would produce trajectories
        no real game could reach. Validity keeps training from the candidate sets
        alongside, where the labels are free.

        `gamma=None` sets the discount FROM THE GAME'S OWN LENGTH, so the first
        move of a game retains half the credit of the last:

            gamma = 0.5 ** (1 / plies)

        A fixed discount cannot serve games of different lengths. At gamma=0.95 a
        125-ply go game gives its opening move 0.0017 of the outcome -- credit
        that has effectively vanished -- and go stayed at 0 wins and -80 points.
        Setting it from the length took the same network to 3 wins in 6 and -14.
        Checkers and chess run to different lengths again, so the discount is
        measured per game rather than chosen.
        """
        rng = rng or random.Random(0)
        r = self.region_for(game)
        me = side or ("b" if game.name == "go" else "w")
        results = {"win": 0, "loss": 0, "draw": 0}
        for _ in range(rounds):
            s = game.new(seed=rng.randrange(10**6))
            traj = []
            for _ply in range(max_plies):
                if game.terminal(s): break
                legal = game.legal_moves(s)
                if not legal: break
                turn = getattr(s, "turn", me)
                if turn == me:
                    if rng.random() < eps:
                        mv = rng.choice(legal)
                    else:
                        scored = [(*r.net.predict(self.encode(r, game, s, m)), m)
                                  for m in legal]
                        mv = max(scored, key=lambda t: t[1])[2]   # best grade among legal
                    traj.append((s, mv))
                    if train_valid:
                        for c in game.candidates(s, rng, k):
                            r.net.step(self.encode(r, game, s, c),
                                       valid=1.0 if game.is_legal(s, c) else 0.0, lr=lr)
                else:
                    mv = opponent(s, opponent_depth, rng) if opponent else rng.choice(legal)
                    if mv is None or not game.is_legal(s, mv): mv = rng.choice(legal)
                s = game.apply(s, mv)
            w = game.winner(s) if hasattr(game, "winner") else None
            if w is not None:
                z = 1.0 if w == me else -1.0
            elif hasattr(game, "value"):
                z = game.value(s, me)      # unfinished: bootstrap from the position
            else:
                z = 0.0
            results["win" if z > 0.05 else ("loss" if z < -0.05 else "draw")] += 1
            g = gamma if gamma is not None else 0.5 ** (1.0 / max(1, len(traj)))
            for i, (st, mv) in enumerate(reversed(traj)):
                r.net.step(self.encode(r, game, st, mv), grade=z * (g ** i), lr=lr)
        r.plays += rounds; r.hits += results["win"]
        return {"game": game.name, "rounds": rounds, **results,
                "gamma": round(gamma if gamma is not None
                               else 0.5 ** (1.0 / max(1, len(traj))), 4),
                "shape": r.net.shape()}

    def evaluate(self, game, n=60, rng=None, k=12, opponent=None, selfplay=0.5):
        """Evaluated on the SAME mixture it will face in play, not on random
        positions only -- otherwise the metric flatters the model."""
        rng = rng or random.Random(99)
        r = self.region_for(game)
        seen = correct = 0; top_legal = 0; rounds = 0; legal_n = 0
        for s in self._states(game, rng, n, selfplay, opponent):
            cands = game.candidates(s, rng, k)
            if not cands: continue
            scored = []
            for mv in cands:
                x = self.encode(r, game, s, mv)
                pv, gr = r.net.predict(x)
                legal = game.is_legal(s, mv)
                seen += 1; legal_n += 1 if legal else 0
                correct += 1 if (pv >= 0.5) == legal else 0
                scored.append((pv, gr, mv, legal))
            pick = self.rank([(a, b, (c, d)) for a, b, c, d in scored], r.rule)
            rounds += 1; top_legal += 1 if pick[1] else 0
        base = legal_n / max(1, seen)
        return {"game": game.name,
                "legality_acc": round(correct / max(1, seen), 3),
                "majority_baseline": round(max(base, 1.0 - base), 3),
                "top_choice_legal": round(top_legal / max(1, rounds), 3),
                "random_pick_baseline": round(base, 3)}

    # How a region turns (p_valid, grade) into a choice. No single rule wins on
    # every game, which was measured rather than assumed:
    #
    #                        chess  checkers  go
    #   pv only              1.000     0.250  1.000
    #   gate pv>=0.5, max g  0.850     0.700  1.000
    #
    # Chess and go want validity to decide outright. Checkers does not, because
    # forced capture makes legality a property of the MOVE SET rather than the
    # move -- the same (dx, dy, occupancy) is legal or illegal depending on
    # whether a jump exists elsewhere -- so its validity head is miscalibrated in
    # a way the others are not, and grade (a jump scores 1.0) carries the signal
    # instead. Each region therefore SELECTS its rule by measurement (§select).
    RULES = {
        "pv":        lambda sc: max(sc, key=lambda t: t[0])[2],
        "penalised": lambda sc: max(sc, key=lambda t: t[0]*t[1] - 2.0*(1.0-t[0]))[2],
        "gated":     lambda sc: _gated(sc),
    }

    @staticmethod
    def rank(scored, rule="pv"):
        return Cortex.RULES[rule](scored) if scored else None

    def choose(self, game, state, rng, k=24):
        """The network's own pick -- NOT filtered by the rules, so
        `top_choice_legal` measures what it actually learned."""
        r = self.region_for(game)
        scored = []
        for mv in game.candidates(state, rng, k):
            pv, gr = r.net.predict(self.encode(r, game, state, mv))
            scored.append((pv, gr, mv))
        return self.rank(scored, self.region_for(game).rule)

    def stats(self):
        return {"regions": len(self.regions),
                "metric_violations": self.graph.check_metric(),
                "detail": [{"id": r.id, "games": [g.name for g in r.games],
                            "vocab_width": r.vocab.width, "shape": r.shape()}
                           for r in self.regions]}
