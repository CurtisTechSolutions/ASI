"""Coalitions: the only mechanism by which a micro's effective receptive field
can grow.

A micro sees 8 of 256 features. Two micros that share hidden vectors see 16
between them, at the cost of one trust relationship and no new parameters.
Sharing has to be EARNED from a partner rather than allocated, which is what
makes it a cooperative game rather than a wider layer.
"""
import math, random
from array import array
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Coalition:
    members: list
    formed: int = 0
    joint_action: object = None        # symbol id, or None for ABSTAIN
    payoff_share: array = field(default_factory=lambda: array("d"))
    trust: dict = field(default_factory=dict)   # (a,b) -> b's last move toward a

    def to_dict(self):
        return {"members": list(self.members), "formed": self.formed,
                "joint_action": self.joint_action,
                "payoff_share": [round(x, 6) for x in self.payoff_share],
                "trust": {f"{a}:{b}": v for (a, b), v in self.trust.items()}}


def complementarity(pool, i, j):
    """1 - |rf_i ∩ rf_j| / R. Two micros reading disjoint features score 1.0."""
    R = pool.R
    a = {pool.rf[i * R + r] for r in range(R)}
    b = {pool.rf[j * R + r] for r in range(R)}
    return 1.0 - len(a & b) / float(R)


def _agreement(profile, ss, a, b):
    """How much two seats' profiles overlap on a symbol."""
    sa, sb = {}, {}
    for s, sym in enumerate(ss[a]):
        if sym is not None: sa[sym] = sa.get(sym, 0.0) + profile[a][s]
    for s, sym in enumerate(ss[b]):
        if sym is not None: sb[sym] = sb.get(sym, 0.0) + profile[b][s]
    return sum(min(v, sb.get(k, 0.0)) for k, v in sa.items())


def propose(pool, seats, profile, ctx, max_size=4, min_complementarity=0.5, rng=None):
    """Greedy: seed on the highest-reputation unattached seat, add the partner
    maximising complementarity * agreement, stop at max_size or when nothing
    clears min_complementarity.

    max_size is 4 because core_deficit is exhaustive over sub-coalitions and
    2^4 - 1 = 15 subsets is the largest that stays free."""
    rng = rng or random.Random(0)
    n = len(seats)
    free = sorted(range(n), key=lambda s: -pool.reputation(seats[s]))
    taken = set()
    out = []
    for seed in free:
        if seed in taken: continue
        members = [seed]; taken.add(seed)
        while len(members) < max_size:
            best, bs = None, 0.0
            for c in range(n):
                if c in taken: continue
                comp = min(complementarity(pool, seats[m], seats[c]) for m in members)
                if comp < min_complementarity: continue
                score = comp * max(_agreement(profile, ctx.slot_symbol, m, c) for m in members)
                if score > bs: best, bs = c, score
            if best is None: break
            members.append(best); taken.add(best)
        if len(members) > 1:
            out.append(Coalition(members=[seats[m] for m in members],
                                 trust={(seats[a], seats[b]): 1
                                        for a in members for b in members if a != b}))
    return out


def joint_action(coalition, seats, profile, ss):
    """The symbol the coalition's members most agree on."""
    idx = {s: k for k, s in enumerate(seats)}
    tally = {}
    for m in coalition.members:
        k = idx.get(m)
        if k is None: continue
        for s, sym in enumerate(ss[k]):
            if sym is None: continue
            tally[sym] = tally.get(sym, 0.0) + profile[k][s]
    if not tally: return None
    return max(tally.items(), key=lambda kv: kv[1])[0]


def should_commit(coalition, pool, ctx, stag_gain=4.0, hare_gain=3.0):
    """The stag hunt. Commit only when the payoff-dominant equilibrium is ALSO
    risk-dominant for this coalition:

        p_partners_commit ** (len(members) - 1) * stag_gain  >  hare_gain

    estimated from mutual trust. Small high-trust coalitions commit; large or new
    ones do not -- which is the correct asymmetry. A five-member coalition needs
    four partners to show up and is exactly the case where hare is risk-dominant."""
    k = len(coalition.members)
    if k < 2: return False
    tv = list(coalition.trust.values())
    p = (sum(tv) / len(tv)) if tv else 0.5
    p = min(max(p, 0.0), 1.0)
    return (p ** (k - 1)) * stag_gain > hare_gain


def share_step(coalition, pool):
    """One round of the repeated prisoner's dilemma, tit-for-tat: share with a
    partner iff the partner shared last round (first round: share). Withholding
    costs nothing this round and costs the partner's cooperation next round."""
    moves = {}
    for (a, b) in list(coalition.trust.keys()):
        moves[(a, b)] = coalition.trust.get((b, a), 1)
    coalition.trust.update(moves)
    return moves


def payoff_division(coalition, phi):
    """The members' Shapley values WITHIN the coalition, normalised to sum to 1.
    A member whose marginal contribution to its own coalition is zero receives
    none of its payoff, by the null-player axiom."""
    vals = [max(0.0, p) for p in phi]
    t = sum(vals)
    k = len(vals)
    if t <= 0.0: return array("d", [1.0 / k] * k) if k else array("d")
    return array("d", [v / t for v in vals])


def core_deficit(coalition, v):
    """max over sub-coalitions S of (v(S) - sum of their shares of v(members)),
    clipped at 0.

    For v(S) = log P_S(y) the core is frequently EMPTY, so the Shapley division
    is not always core-stable. That is a real property of a game where a small
    expert subset can outscore the whole group -- adding a confidently wrong
    micro LOWERS v, so super-additivity fails. The design's answer is that
    coalitions are re-proposed every stage game rather than standing: a
    sub-coalition that would rather walk out simply forms on its own next time.
    Stability by churn, not by a division nobody can object to.

    Reported, never acted on."""
    n = len(coalition.members)
    if n == 0 or n > 4: return 0.0
    full = v(set(range(n)))
    share = list(coalition.payoff_share) or [1.0 / n] * n
    worst = 0.0
    for k in range(1, n):
        for S in combinations(range(n), k):
            d = v(set(S)) - sum(share[i] * full for i in S)
            if d > worst: worst = d
    return worst
