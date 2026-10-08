"""MicroPool -- the population, stored as a struct of flat arrays.

There are no Micro objects in the hot path. Micro `i` owns the slice
[i*stride, (i+1)*stride) of each array. 4096 Python objects each holding six
lists would cost more in pointer chasing than the arithmetic they exist to do.

A micro is an R -> H -> (K+1) network: R=8 inputs drawn at birth (its receptive
field, its patch of cortex), H=6 hidden units, K=4 symbols it may name plus
ABSTAIN. ~133 floats each. It never sees the input -- only its 8 features, which
is what makes the population a game and not a vote.
"""
import math, random
from array import array

from gtmnn.activation import (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K,
                              sine_activation, sine_derivative, sine_partials)
from gtmnn.features import sample_receptive_field

ABSTAIN = -1                    # the action id meaning "I decline to play"
GAMMA_REP = 0.999               # reputation discount, applied every stage game


class MicroPool:
    def __init__(self, n, F, R=8, H=6, K=4, seed=0, alphabet_size=2, inverted=False):
        self.n, self.F, self.R, self.H, self.K = int(n), int(F), int(R), int(H), int(K)
        self.rng = random.Random(seed)
        self.inverted = bool(inverted)
        self.version = 0
        self.pop_version = 0
        S = K + 1                                   # action slots, including ABSTAIN
        self.S = S
        self.NEUR = H + S                           # sine parameters per micro

        self.rf = array("i", [0]) * (n * R)
        self.w1 = array("d", [0.0]) * (n * R * H)
        self.b1 = array("d", [0.0]) * (n * H)
        self.w2 = array("d", [0.0]) * (n * H * S)
        self.b2 = array("d", [0.0]) * (n * S)
        a0 = -DEFAULT_A if inverted else DEFAULT_A
        self.sa = array("d", [a0]) * (n * self.NEUR)
        self.sb = array("d", [DEFAULT_B]) * (n * self.NEUR)
        self.sh = array("d", [DEFAULT_H]) * (n * self.NEUR)
        self.sk = array("d", [DEFAULT_K]) * (n * self.NEUR)
        self.repertoire = array("i", [-1]) * (n * K)
        self.strategy = array("d", [1.0 / S]) * (n * S)
        self.regret = array("d", [0.0]) * (n * S)
        self.wealth = array("d", [0.0]) * n
        self.hits = array("d", [0.0]) * n
        self.plays = array("d", [0.0]) * n
        self.age = array("i", [0]) * n
        self.born = array("i", [0]) * n
        self.alive = bytearray([1]) * n
        self._tally = [dict() for _ in range(n)]     # symbol -> credit, for repertoire upkeep

        for i in range(n):
            self._init_micro(i, alphabet_size)

    # ------------------------------------------------------------------ birth
    def _init_micro(self, i, alphabet_size, generation=0):
        rng = self.rng
        R, H, K, S = self.R, self.H, self.K, self.S
        rfv = sample_receptive_field(self.F, R, rng)
        for r in range(R): self.rf[i * R + r] = rfv[r]
        base1 = i * R * H
        for t in range(R * H): self.w1[base1 + t] = rng.uniform(-0.5, 0.5)
        for h in range(H): self.b1[i * H + h] = 0.0
        base2 = i * H * S
        for t in range(H * S): self.w2[base2 + t] = rng.uniform(-0.5, 0.5)
        for o in range(S): self.b2[i * S + o] = 0.0
        a0 = -DEFAULT_A if self.inverted else DEFAULT_A
        nb = i * self.NEUR
        for t in range(self.NEUR):
            self.sa[nb + t] = a0; self.sb[nb + t] = DEFAULT_B
            self.sh[nb + t] = DEFAULT_H; self.sk[nb + t] = DEFAULT_K
        self.fill_repertoire(i, alphabet_size)
        for o in range(S): self.strategy[i * S + o] = 1.0 / S; self.regret[i * S + o] = 0.0
        self.wealth[i] = 0.0; self.hits[i] = 0.0; self.plays[i] = 0.0
        self.age[i] = 0; self.born[i] = generation; self.alive[i] = 1
        self._tally[i] = {}

    def fill_repertoire(self, i, alphabet_size):
        """Fill empty slots with distinct symbol ids. Ids 0 (UNK) and 1 (END)
        are real targets, so they are drawn from too; a micro that can only ever
        name END is a valid, if narrow, specialist."""
        K = self.K
        have = {self.repertoire[i * K + s] for s in range(K)} - {-1}
        pool = [c for c in range(alphabet_size) if c not in have]
        if not pool: return 0
        filled = 0
        for s in range(K):
            if self.repertoire[i * K + s] != -1: continue
            if not pool: break
            self.repertoire[i * K + s] = pool.pop(self.rng.randrange(len(pool)))
            filled += 1
        if filled: self.pop_version += 1
        return filled

    # --------------------------------------------------------------- derived
    def reputation(self, i):
        """Laplace-smoothed, so a micro that has never played sits at exactly
        0.5 -- enough to win a seat on an input the incumbents are unsure about
        and not enough to take one they are confident about. That is the
        exploration schedule, and nobody had to write it down as a schedule."""
        return (self.hits[i] + 1.0) / (self.plays[i] + 2.0)

    def living(self):
        return [i for i in range(self.n) if self.alive[i]]

    def count_alive(self):
        return sum(self.alive)

    def slots(self, i):
        """slot -> global symbol id; ABSTAIN for slot K."""
        K = self.K
        return [self.repertoire[i * K + s] for s in range(K)] + [ABSTAIN]

    # --------------------------------------------------------------- forward
    def forward(self, i, feat, out=None, cache=None):
        """out receives q_i, the micro's own probabilities over its K+1 slots.

        Slots whose repertoire entry is -1 are masked BEFORE the softmax, so a
        micro cannot name a symbol it does not have."""
        R, H, K, S = self.R, self.H, self.K, self.S
        if out is None: out = array("d", [0.0]) * S
        rf, w1, b1, w2, b2 = self.rf, self.w1, self.b1, self.w2, self.b2
        sa, sb, sh, sk = self.sa, self.sb, self.sh, self.sk
        rb, wb1, bb1, wb2, bb2, nb = i * R, i * R * H, i * H, i * H * S, i * S, i * self.NEUR

        ah = [0.0] * H; zh = [0.0] * H
        for h in range(H):
            z = b1[bb1 + h]
            for r in range(R):
                z += w1[wb1 + r * H + h] * feat[rf[rb + r]]
            zh[h] = z
            ah[h] = sine_activation(z, sa[nb + h], sb[nb + h], sh[nb + h], sk[nb + h])

        zo = [0.0] * S; ao = [0.0] * S
        for o in range(S):
            z = b2[bb2 + o]
            for h in range(H):
                z += w2[wb2 + h * S + o] * ah[h]
            zo[o] = z
            j = nb + H + o
            ao[o] = sine_activation(z, sa[j], sb[j], sh[j], sk[j])

        mask = [(o == K or self.repertoire[i * K + o] != -1) for o in range(S)]
        m = max(ao[o] for o in range(S) if mask[o])
        tot = 0.0
        for o in range(S):
            if mask[o]:
                e = math.exp(ao[o] - m); out[o] = e; tot += e
            else:
                out[o] = 0.0
        for o in range(S): out[o] /= tot
        if cache is not None:
            cache[:] = [ah, zh, zo, ao, list(out), mask]
        return out

    def forward_many(self, ids, feat):
        return [self.forward(i, feat) for i in ids]

    # ------------------------------------------------------------------ learn
    def learn(self, i, target, phi, lr, act_lr, feat, clip=5.0):
        """One REINFORCE step on micro i's OWN weights, with the Shapley value
        as the advantage. Nothing here touches any other micro.

        `target` is the distribution the credit was earned on -- the seat's mixed
        profile row -- or an int slot, which is read as the one-hot case and
        gives exactly the original rule.

        It must be the PROFILE and not the profile's argmax. phi prices seat i's
        whole contribution vector (rho_i times its profile row); the argmax is a
        summary of that vector, not the thing that was priced. Using it, a seat
        that earned positive credit through its symbol mass but whose single
        likeliest action was ABSTAIN got pushed toward abstaining HARDER. Measured
        before this was separated: 258 of 324 chosen actions were ABSTAIN, and
        belief loss sat above the uniform baseline however long it trained.

        Shapley is the right coefficient, not merely a convenient one.
        Efficiency makes the population-wide sum of learning signals exactly the
        log-likelihood gained over a uniform guess; the null-player axiom makes
        a micro that changed nothing take EXACTLY no step, not a small one; and
        symmetry means nothing arbitrary breaks a tie between interchangeable
        micros. No hand-tuned baseline has all three."""
        if phi == 0.0: return                # null player: exactly no step
        R, H, K, S = self.R, self.H, self.K, self.S
        if isinstance(target, int):
            t = [0.0] * S; t[target] = 1.0
        else:
            t = list(target) + [0.0] * (S - len(target))
        cache = []
        self.forward(i, feat, cache=cache)
        ah, zh, zo, ao, q, mask = cache
        sa, sb, sh, sk = self.sa, self.sb, self.sh, self.sk
        nb = i * self.NEUR

        def cl(g): return clip if g > clip else (-clip if g < -clip else g)

        # L = -phi * sum_s t[s] log q(s);   dL/d(ao[o]) = -phi * (t[o] - q[o])
        dao = [0.0] * S
        for o in range(S):
            if not mask[o]: continue
            dao[o] = -phi * (t[o] - q[o])

        dah = [0.0] * H
        wb2, bb2 = i * H * S, i * S
        for o in range(S):
            if dao[o] == 0.0 and mask[o] is False: continue
            j = nb + H + o
            f, dfdx, dfda, dfdb, dfdh, dfdk = sine_partials(zo[o], sa[j], sb[j], sh[j], sk[j])
            dz = dao[o] * dfdx
            for h in range(H):
                dah[h] += dz * self.w2[wb2 + h * S + o]
                self.w2[wb2 + h * S + o] -= lr * cl(dz * ah[h])
            self.b2[bb2 + o] -= lr * cl(dz)
            sa[j] -= act_lr * cl(dao[o] * dfda); sb[j] -= act_lr * cl(dao[o] * dfdb)
            sh[j] -= act_lr * cl(dao[o] * dfdh); sk[j] -= act_lr * cl(dao[o] * dfdk)

        rb, wb1, bb1 = i * R, i * R * H, i * H
        for h in range(H):
            j = nb + h
            f, dfdx, dfda, dfdb, dfdh, dfdk = sine_partials(zh[h], sa[j], sb[j], sh[j], sk[j])
            dz = dah[h] * dfdx
            for r in range(R):
                self.w1[wb1 + r * H + h] -= lr * cl(dz * feat[self.rf[rb + r]])
            self.b1[bb1 + h] -= lr * cl(dz)
            sa[j] -= act_lr * cl(dah[h] * dfda); sb[j] -= act_lr * cl(dah[h] * dfdb)
            sh[j] -= act_lr * cl(dah[h] * dfdh); sk[j] -= act_lr * cl(dah[h] * dfdk)
        self.version += 1

    # ------------------------------------------------------- bookkeeping
    def discount_reputation(self):
        g = GAMMA_REP
        for i in range(self.n):
            self.hits[i] *= g; self.plays[i] *= g

    def note_symbol(self, i, symbol, credit):
        t = self._tally[i]
        t[symbol] = t.get(symbol, 0.0) + credit

    def refresh_repertoire(self, i, alphabet_size):
        """The K highest-credit symbols become the repertoire. A symbol entering
        or leaving resets that slot's w2 column and b2 entry -- it is a different
        action now, and its old weights mean nothing."""
        K, S, H = self.K, self.S, self.H
        t = self._tally[i]
        if not t: return 0
        want = [s for s, _ in sorted(t.items(), key=lambda kv: -kv[1])[:K]]
        cur = [self.repertoire[i * K + s] for s in range(K)]
        # Keep the incumbent wherever the tally is short. Writing -1 into the
        # tail would SHRINK the micro's action set every time it happened to be
        # credited for fewer than K distinct symbols -- a capacity loss caused by
        # a quiet stretch rather than by anything the micro did wrong.
        keep = [c for c in cur if c != -1 and c not in want]
        while len(want) < K and keep: want.append(keep.pop(0))
        if want == [c for c in cur if c != -1]: return 0
        changed = 0
        for s in range(K):
            new = want[s] if s < len(want) else cur[s]
            if new == cur[s]: continue
            self.repertoire[i * K + s] = new
            for h in range(H):
                self.w2[i * H * S + h * S + s] = self.rng.uniform(-0.5, 0.5)
            self.b2[i * S + s] = 0.0
            changed += 1
        if changed:
            self.pop_version += 1
            self._tally[i] = {}
        return changed

    # ------------------------------------------------------------ population
    def invert(self):
        """a -> -a on every sine. Reflects every micro's response about its own
        operating point without touching a single weight, so who reads which
        features, who is coalitioned with whom and who is solvent all survive.
        Wealth and reputation deliberately do NOT invert: a micro that was
        trustworthy before the flip is still the one you want seated after it."""
        for t in range(len(self.sa)): self.sa[t] = -self.sa[t]
        self.inverted = not self.inverted
        self.version += 1

    def kill(self, i):
        self.alive[i] = 0; self.pop_version += 1

    def rebirth(self, i, alphabet_size, generation):
        self._init_micro(i, alphabet_size, generation)
        self.pop_version += 1

    def clone(self):
        import copy
        p = MicroPool.__new__(MicroPool)
        for k, v in self.__dict__.items():
            if isinstance(v, (array, bytearray)): setattr(p, k, v[:])
            elif k == "_tally": setattr(p, k, [dict(d) for d in v])
            elif k == "rng":
                r = random.Random(); r.setstate(v.getstate()); setattr(p, k, r)
            else: setattr(p, k, copy.copy(v))
        return p

    # --------------------------------------------------------- serialisation
    def to_dict(self):
        return {"n": self.n, "F": self.F, "R": self.R, "H": self.H, "K": self.K,
                "inverted": self.inverted, "version": self.version,
                "pop_version": self.pop_version,
                "rf": list(self.rf), "w1": list(self.w1), "b1": list(self.b1),
                "w2": list(self.w2), "b2": list(self.b2),
                "sa": list(self.sa), "sb": list(self.sb),
                "sh": list(self.sh), "sk": list(self.sk),
                "repertoire": list(self.repertoire), "strategy": list(self.strategy),
                "regret": list(self.regret), "wealth": list(self.wealth),
                "hits": list(self.hits), "plays": list(self.plays),
                "age": list(self.age), "born": list(self.born),
                "alive": list(self.alive),
                "tally": [{str(k): v for k, v in d.items()} for d in self._tally],
                "rng": _state_to_list(self.rng.getstate())}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["n"], d["F"], d["R"], d["H"], d["K"], seed=0)
        p.inverted = d["inverted"]; p.version = d["version"]; p.pop_version = d["pop_version"]
        for name in ("rf", "repertoire", "age", "born"):
            setattr(p, name, array("i", d[name]))
        for name in ("w1", "b1", "w2", "b2", "sa", "sb", "sh", "sk",
                     "strategy", "regret", "wealth", "hits", "plays"):
            setattr(p, name, array("d", d[name]))
        p.alive = bytearray(d["alive"])
        p._tally = [{int(k): v for k, v in t.items()} for t in d["tally"]]
        p.rng.setstate(_list_to_state(d["rng"]))
        return p


def _state_to_list(s):
    return [s[0], list(s[1]), s[2]]


def _list_to_state(l):
    return (l[0], tuple(l[1]), l[2])
