"""The uniform-price seat auction: who gets to play this stage game.

Only M=64 of N=4096 micros play any given stage game. Choosing them by "highest
confidence" would be a scoring rule the micros could game by overstating.
Selling the seats cannot be gamed, because of Vickrey (1961): a seated micro
pays the highest LOSING bid, a quantity it cannot influence by changing its own
bid. Under unit demand truthful bidding is therefore dominant, and calibration
becomes a property of the mechanism rather than something a loss function has to
be talked into.

A micro that inflates its bid wins seats on inputs it cannot answer and pays the
price on every one of them until it goes bankrupt and is culled. Calibration is
enforced by bankruptcy, not by a loss term.
"""
from dataclasses import dataclass, field

RESERVE = 0.01


@dataclass
class SeatAllocation:
    seats: list                 # micro ids, high bid first
    bids: list                  # each seated micro's bid
    price: float                # the highest LOSING bid; 0.0 when demand is short
    bidders: int                # how many micros bid above the reserve

    def to_dict(self):
        return {"seats": list(self.seats), "bids": [round(b, 6) for b in self.bids],
                "price": self.price, "bidders": self.bidders}


def bid(pool, i, q):
    """rho_i * (best non-abstain belief - belief in abstaining), floored at 0.

    "How much better than abstaining do I think my best call is, discounted by
    how often I am actually right." """
    K = pool.K
    best = 0.0
    for s in range(K):
        if pool.repertoire[i * K + s] == -1: continue
        if q[s] > best: best = q[s]
    v = best - q[K]
    return pool.reputation(i) * v if v > 0.0 else 0.0


def allocate(pool, ids, qs, m=64, reserve=RESERVE):
    """Top `m` bidders above the reserve are seated; everyone pays the highest
    losing bid. Ties break by reputation then by micro id, so the allocation is
    deterministic.

    When fewer than `m` clear the reserve the price is 0.0 and the game is played
    short-handed -- the correct behaviour for an input nobody in the population
    understands, and `bidders` falling below `m` is itself a strong signal,
    forwarded to the meta-player as a feature."""
    entries = []
    for i, q in zip(ids, qs):
        b = bid(pool, i, q)
        if b > reserve: entries.append((b, pool.reputation(i), -i, i))
    entries.sort(reverse=True)
    seated = entries[:m]
    price = entries[m][0] if len(entries) > m else 0.0
    return SeatAllocation(seats=[e[3] for e in seated], bids=[e[0] for e in seated],
                          price=price, bidders=len(entries))


def charge(pool, alloc):
    """Payment happens at seating, win or lose."""
    if alloc.price <= 0.0: return
    for i in alloc.seats: pool.wealth[i] -= alloc.price


def best_response_bid(pool, i, q, others, m=64, reserve=RESERVE, sweep=None):
    """What a micro would earn by misreporting, over a sweep of alternative bids.

    Exists so the truthfulness claim is CHECKED rather than cited: under a
    uniform price no misreport can beat the truthful bid, because raising your
    bid can only change whether you are seated, never what you pay."""
    truth = bid(pool, i, q)
    sweep = sweep or [truth * f for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)]
    out = []
    for b in sweep:
        entries = sorted([(bb, -j, j) for bb, j in others if bb > reserve] +
                         ([(b, -i, i)] if b > reserve else []), reverse=True)
        seated = [e[2] for e in entries[:m]]
        price = entries[m][0] if len(entries) > m else 0.0
        # value of a seat is the truthful valuation; surplus is value - price
        out.append((b, (truth - price) if i in seated else 0.0))
    return truth, out
