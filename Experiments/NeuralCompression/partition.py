"""Experiment 5: when games are ADDED, how much of what was learned is destroyed?

Phase 1 trains 4 games. Phase 2 trains ONLY the 4 new ones. Then re-measure the
original 4: any degradation is displacement caused by the layout, not by lack of
training. (Experiment 4 retrained everything in phase 2 and so measured nothing.)

  one_over_N   4 bands of 1/4 -> 8 bands of 1/8. Every centre moves, every width halves.
  static_pow2  8 slots reserved, 4 filled. Nothing moves. Costs 50% dust while half-full.
  adaptive     coords by similarity; a new game is inserted at the MIDPOINT between its
               two nearest neighbours. Centres never move, bands tile [0,1] exactly
               (no dust), and only the two adjacent games are disturbed at all.
"""
import sys, random
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from experiment import MLP, make_games, target, data

def bands_from_coords(coords):
    """Voronoi intervals around sorted coords; boundaries at midpoints, 0 and 1 at the ends."""
    ks = sorted(coords, key=coords.get)
    out = {}
    for i, k in enumerate(ks):
        lo = 0.0 if i == 0 else (coords[ks[i-1]] + coords[k]) / 2
        hi = 1.0 if i == len(ks)-1 else (coords[k] + coords[ks[i+1]]) / 2
        out[k] = (lo, hi)
    return out

def train(net, games, TR, coords, bands, epochs, lr, rng):
    for _ in range(epochs):
        order = [(k, p) for k in games for p in TR[k]]; rng.shuffle(order)
        for k, (x0, x1, y) in order:
            lo, hi = bands[k]; w = hi - lo
            net.step([x0, x1, coords[k]], lo + w*y, lr, 1.0/w)   # loss in game units

def ev(net, games, TE, coords, bands):
    out = []
    for k in games:
        lo, hi = bands[k]; w = hi - lo; se = 0.0
        for x0, x1, y in TE[k]:
            o = net.forward([x0, x1, coords[k]])[2]
            se += ((o - lo)/w - y) ** 2
        out.append(se/len(TE[k]))
    return sum(out)/len(out)

def run(scheme, nh=16, epochs=400, lr=0.05, seed=0, act="tanh", b=1.0/3.0):
    rng = random.Random(seed)
    G = make_games(8, rng); TR = data(G, 48, rng); TE = data(G, 48, rng)
    old, new = [0,2,4,6], [1,3,5,7]
    net = MLP(3, nh, rng, act, b)
    if scheme == "one_over_N":
        c1 = {k: (i+0.5)/4 for i, k in enumerate(old)}
        c2 = {k: (k+0.5)/8 for k in range(8)}
    elif scheme == "static_pow2":
        c1 = {k: (k+0.5)/8 for k in old}
        c2 = {k: (k+0.5)/8 for k in range(8)}
    else:  # adaptive: insert at the midpoint between nearest neighbours
        c1 = {k: (i+0.5)/4 for i, k in enumerate(old)}
        c2 = dict(c1)
        srt = sorted(old, key=c1.get)
        for j, k in enumerate(new):
            lo = c1[srt[j]]; hi = c1[srt[j+1]] if j+1 < len(srt) else 1.0
            c2[k] = (lo + hi) / 2                      # existing coords untouched
    b1 = bands_from_coords(c1); b2 = bands_from_coords(c2)
    train(net, old, TR, c1, b1, epochs, lr, rng)
    before = ev(net, old, TE, c1, b1)
    train(net, new, TR, c2, b2, epochs, lr, rng)        # ONLY the new games
    return before, ev(net, old, TE, c2, b2), ev(net, new, TE, c2, b2)

def dust(slots, games, nh=16, epochs=400, lr=0.05, seed=0, act="tanh", b=1/3):
    """Experiment 6: fixed-width bands with some left EMPTY, isolating amplification."""
    rng = random.Random(seed)
    G = make_games(8, rng); TR = data(G, 48, rng); TE = data(G, 48, rng)
    net = MLP(3, nh, rng, act, b); idx = {k: i for i, k in enumerate(games)}
    for _ in range(epochs):
        order = [(k, p) for k in games for p in TR[k]]; rng.shuffle(order)
        for k, (x0, x1, y) in order:
            bd = idx[k]; net.step([x0, x1, (bd+0.5)/slots], (bd+y)/slots, lr, slots)
    tot = []
    for k in games:
        bd = idx[k]; se = 0.0
        for x0, x1, y in TE[k]:
            se += (net.forward([x0, x1, (bd+0.5)/slots])[2]*slots - bd - y)**2
        tot.append(se/len(TE[k]))
    return sum(tot)/len(tot)


if __name__ == "__main__":
    print("EXPERIMENT 6 - the cost of DUST (4 games, fixed-width bands, seeds 0-3)\n")
    print(f"  {'layout':>26} {'band width':>11} {'amplification':>14} {'mean MSE':>10}")
    for slots, lbl in ((4,"4 bands, 0% dust"),(8,"8 bands, 50% dust"),(16,"16 bands, 75% dust")):
        v=[dust(slots,[0,2,4,6],seed=s) for s in range(4)]
        print(f"  {lbl:>26} {1/slots:>11.4f} {'x'+str(slots):>14} {sum(v)/len(v):>10.5f}")

    print("\n\nEXPERIMENT 5 - damage to already-learned games when 4 more are added (seeds 0-3)\n")
    print("  'damage' is the mean of the four per-seed ratios, and those ratios range from")
    print("  0.6x to 3.7x -- so read the LAYOUT ORDER, which is stable, not the exact number.\n")
    print(f"  {'act':>9} {'layout':>12} {'old before':>11} {'old after':>10} {'new':>8} {'damage':>9} {'worst':>7} {'dust':>6}")
    dust_pct = {"one_over_N":"0%", "static_pow2":"50%", "adaptive":"0%"}
    for act, bb, lbl in (("tanh", 1/3, "tanh"), ("sine", 1.0, "sine b=1")):
        for s in ("one_over_N", "static_pow2", "adaptive"):
            r = [run(s, act=act, b=bb, seed=sd) for sd in range(4)]
            b0 = sum(x[0] for x in r)/len(r); a = sum(x[1] for x in r)/len(r)
            n = sum(x[2] for x in r)/len(r)
            ratios = [x[1]/x[0] for x in r]
            dmg = sum(ratios)/len(ratios)
            print(f"  {lbl:>9} {s:>12} {b0:>11.5f} {a:>10.5f} {n:>8.5f} {dmg:>8.2f}x "
                  f"{max(ratios):>6.2f}x {dust_pct[s]:>6}")
        print()
