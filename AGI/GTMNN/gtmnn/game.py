"""StageGame -- seating, actions, loads, payoff evaluation, and the aggregate.

Every prediction is a stage game: micros bid for a seat, seated micros choose an
action, payoff depends on what everyone else chose, and the output is the
equilibrium.
"""
import math
from array import array
from dataclasses import dataclass, field

from gtmnn import auction
from gtmnn.micro import ABSTAIN
from gtmnn.payoff import CorrectnessCongestion, BeliefCongestion


@dataclass
class GameContext:
    alphabet_size: int
    y: object = None                # true symbol id, or None at inference
    B: float = 1.0
    lam: float = 0.5
    q: list = field(default_factory=list)            # per-seat beliefs over K+1 slots
    slot_symbol: list = field(default_factory=list)  # [seat][slot] -> symbol id, None for ABSTAIN
    playable: list = field(default_factory=list)     # [seat][slot] -> may this slot be chosen?
    profile_actions: list = field(default_factory=list)   # matrix fixtures only


class StageGame:
    def __init__(self, seats, ctx, payoff):
        self.seats, self.ctx, self.payoff = list(seats), ctx, payoff

    # ------------------------------------------------------------------ loads
    def loads(self, actions):
        """Indexed by symbol id, plus one final entry counting abstentions.

        Two seats naming the same symbol from different SLOTS add to the same
        load. This is the only place slot-space and symbol-space meet.

        A matrix fixture asks for the profile itself instead (payoff.summary)."""
        if getattr(self.payoff, "summary", "loads") == "profile":
            return list(actions)
        out = [0] * (self.ctx.alphabet_size + 1)
        for seat, slot in enumerate(actions):
            sym = self.ctx.slot_symbol[seat][slot]
            if sym is None: out[-1] += 1
            else: out[sym] += 1
        return out

    def utilities(self, actions):
        loads = self.loads(actions)
        return [self.payoff.utility(s, a, loads, self.ctx) for s, a in enumerate(actions)]

    def potential(self, actions):
        return self.payoff.potential(self.loads(actions), self.ctx)

    # --------------------------------------------------------- best response
    def slot_utilities(self, seat, actions, loads=None):
        """This seat's utility for EVERY slot, holding the others fixed.

        The single place the incremental load bookkeeping lives -- remove this
        seat's current action, try each slot, put it back. best_response, regret
        matching and the replicator all read through here, so a summary form
        that needs different bookkeeping (a matrix fixture's profile) is handled
        once rather than in three loops that can drift apart."""
        ctx = self.ctx
        nslots = len(ctx.slot_symbol[seat])
        if getattr(self.payoff, "summary", "loads") == "profile":
            prof = list(actions)
            out = []
            for s in range(nslots):
                prof[seat] = s
                out.append(self.payoff.utility(seat, s, prof, ctx))
            return out
        if loads is None: loads = self.loads(actions)
        cur_sym = ctx.slot_symbol[seat][actions[seat]]
        if cur_sym is None: loads[-1] -= 1
        else: loads[cur_sym] -= 1
        out = []
        for s in range(nslots):
            sym = ctx.slot_symbol[seat][s]
            if sym is None: loads[-1] += 1
            else: loads[sym] += 1
            out.append(self.payoff.utility(seat, s, loads, ctx))
            if sym is None: loads[-1] -= 1
            else: loads[sym] -= 1
        if cur_sym is None: loads[-1] += 1
        else: loads[cur_sym] += 1
        return out

    def best_response(self, seat, actions, loads=None):
        """The slot maximising this seat's utility holding the others fixed;
        ties go to the lowest slot.

        Loads are updated INCREMENTALLY -- this seat's current action removed,
        each candidate tried -- never recomputed from scratch. This is the
        innermost loop of the whole system."""
        us = self.slot_utilities(seat, actions, loads)
        ok = self.playable(seat)
        best_slot, best_u = len(us) - 1, -1e18
        for s, u in enumerate(us):
            if ok[s] and u > best_u + 1e-15: best_slot, best_u = s, u
        return best_slot, best_u

    def playable(self, seat):
        """Which slots this seat may actually choose. A slot whose repertoire
        entry is empty is not an action the micro has."""
        pl = self.ctx.playable
        if pl: return pl[seat]
        return [True] * len(self.ctx.slot_symbol[seat])


def build(pool, feat, alphabet_size, y, q=None, m=64, payoff=None,
          reserve=auction.RESERVE, charge=True, B=1.0, lam=0.5, ids=None):
    """Seat the auction and hand back the stage game.

    `q` is normally passed in by the caller: it is the ONE full-population
    forward pass per prediction and must never happen twice. build() computes it
    only when called directly, as the CLI's `game` command does.

    charge=False is the read-only path -- inspect a stage game without moving
    anyone's wealth."""
    ids = ids if ids is not None else pool.living()
    if q is None: q = pool.forward_many(ids, feat)
    alloc = auction.allocate(pool, ids, q, m, reserve)
    if charge: auction.charge(pool, alloc)

    by_id = {i: qi for i, qi in zip(ids, q)}
    slot_symbol, playable = [], []
    for i in alloc.seats:
        raw = pool.slots(i)                     # K repertoire ids, then ABSTAIN (-1)
        # An UNFILLED repertoire slot and ABSTAIN are different things. Both would
        # be written as None here, and the payoff scores None as a free
        # abstention worth 0.0 -- so a micro with a half-empty repertoire got
        # extra costless abstentions and best_response took them. Measured before
        # this was separated: 20 of 24 seats "abstaining", half of them onto dead
        # slots, and the aggregate decided by the four seats left.
        slot_symbol.append([(None if x == -1 else x) for x in raw[:-1]] + [None])
        playable.append([raw[s] != -1 for s in range(pool.K)] + [True])
    ctx = GameContext(alphabet_size=alphabet_size, y=y, B=B, lam=lam,
                      q=[by_id[i] for i in alloc.seats], slot_symbol=slot_symbol,
                      playable=playable)
    pf = payoff if payoff is not None else (
        CorrectnessCongestion() if y is not None else BeliefCongestion())
    return StageGame(alloc.seats, ctx, pf), alloc, q


def aggregate(game, profile, pool, eps=1e-3, out=None):
    """The equilibrium, turned into a distribution over the FULL alphabet.

        P(c) ~ eps/|V| + sum over seats, over slots naming c, of rho_i * profile[i][slot]

    ABSTAIN contributes to nothing. The eps smoothing guarantees P(c) > 0 for
    every c, which is what makes log P(c) finite and therefore makes v(S) finite
    for every coalition including the empty one.

    Reputation weighting here is deliberate double-counting: reputation already
    decided who got a seat, and it decides again how loudly they speak once
    seated. A micro right 90% of the time and one right 55% should not be equal
    voices merely because both cleared the auction."""
    V = game.ctx.alphabet_size
    if out is None: out = array("d", [0.0]) * V
    base = eps / V
    for c in range(V): out[c] = base
    total = eps
    for seat, i in enumerate(game.seats):
        rho = pool.reputation(i)
        row = profile[seat]
        syms = game.ctx.slot_symbol[seat]
        for s, sym in enumerate(syms):
            if sym is None: continue
            w = rho * row[s]
            if w: out[sym] += w; total += w
    for c in range(V): out[c] /= total
    return out


def entropy(dist):
    h = 0.0
    for p in dist:
        if p > 0.0: h -= p * math.log(p)
    return h


def top_k(dist, alphabet, k=5):
    order = sorted(range(len(dist)), key=lambda c: -dist[c])[:k]
    return [(alphabet.decode(c), dist[c]) for c in order]
