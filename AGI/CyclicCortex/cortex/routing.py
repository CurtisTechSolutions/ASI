"""The region-level auction and congestion game.

GTMNN's mechanism at a coarser grain: the players are REGIONS, not micros.

  auction     regions bid reputation x coverage; the price is the highest LOSING
              bid, so bidding true value is dominant (Vickrey) and a region
              cannot buy games it cannot handle without paying for them.
  congestion  the reward for handling a game is SPLIT among the regions that
              claimed it, so a region that uniquely covers an unusual game earns
              more than one of five that all claim chess. That is the same split
              reward that made GTMNN's micros specialise, applied to regions.
"""
from cortex.credit import coverage

RESERVE = 0.05

class Allocation:
    def __init__(self, seats, bids, price, bidders):
        self.seats, self.bids, self.price, self.bidders = seats, bids, price, bidders
    def to_dict(self):
        return {"seats": [r.id for r in self.seats], "bids": [round(b,3) for b in self.bids],
                "price": round(self.price, 3), "bidders": self.bidders}

def bid(region, game):
    """What a region thinks a seat is worth: how much of the game it can encode,
    discounted by how often it has been right."""
    return region.reputation() * coverage(region, game)

def allocate(cortex, game, m=2, reserve=RESERVE):
    """Sell up to `m` seats at a uniform price (the highest losing bid)."""
    scored = [(bid(r, game), r) for r in cortex.regions if r.net is not None]
    scored = [(b, r) for b, r in scored if b > reserve]
    scored.sort(key=lambda t: (-t[0], t[1].id))
    seats = [r for _, r in scored[:m]]
    price = scored[m][0] if len(scored) > m else 0.0
    for r in seats: r.wealth = getattr(r, "wealth", 0.0) - price
    return Allocation(seats, [b for b, _ in scored[:m]], price, len(scored))

def settle(alloc, correct, base=1.0, penalty=0.5):
    """Congestion: a correct claim pays `base` SPLIT among the claimants."""
    n = max(1, len(alloc.seats))
    for r in alloc.seats:
        r.wealth = getattr(r, "wealth", 0.0) + (base / n if correct else -penalty)
        r.plays += 1.0; r.hits += 1.0 if correct else 0.0
    return {"claimants": n, "per_claimant": (base / n) if correct else -penalty}
