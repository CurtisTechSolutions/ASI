"""Does a generalised action vocabulary let one network learn chess and checkers?

The claim under test: "for chess, a piece moving one space would count here --
this allows us to train the same network on checkers as well."

Two 6x6 move-legality games sharing structure but not surface form:
  king      legal iff target empty and max(|dx|,|dy|) == 1     (any direction)
  checkers  legal iff target empty and |dx| == 1 and dy == +1  (diagonal, forward)

Two encodings of the same move:
  raw          one-hot(from) + one-hot(to), 72 inputs. Board-specific; a square
               index means nothing outside its own game.
  generalised  the COMMON DENOMINATORS: |dx|, |dy|, forwardness, Chebyshev and
               Manhattan distance, occupancy, edge proximity. 8 inputs, and the
               same eight mean the same thing in both games.

Measured: train on king, then fine-tune on k checkers examples. Against training
on those k examples from scratch. If the generalised vocabulary is doing the work
the claim says, transfer should hold at small k where raw cannot.

Pure standard library.  Run:  python3 common_denominator.py
"""
import math, random

N = 6

def legal_king(fr, to, occ):
    dx, dy = to[0]-fr[0], to[1]-fr[1]
    return max(abs(dx), abs(dy)) == 1 and to not in occ

def legal_checkers(fr, to, occ):
    dx, dy = to[0]-fr[0], to[1]-fr[1]
    return abs(dx) == 1 and dy == 1 and to not in occ

def raw_enc(fr, to, occ):
    v = [0.0]*(2*N*N)
    v[fr[1]*N + fr[0]] = 1.0
    v[N*N + to[1]*N + to[0]] = 1.0
    return v

def gen_enc(fr, to, occ):
    """The common denominators. Every feature means the same thing in every game
    played on a grid, which is the whole point: `moved one space diagonally` is
    the same fact in chess and in checkers."""
    dx, dy = to[0]-fr[0], to[1]-fr[1]
    return [
        abs(dx)/2.0,                                   # horizontal extent
        abs(dy)/2.0,                                   # vertical extent
        1.0 if dy > 0 else (0.5 if dy == 0 else 0.0),  # forwardness
        max(abs(dx), abs(dy))/2.0,                     # Chebyshev distance
        (abs(dx)+abs(dy))/4.0,                         # Manhattan distance
        1.0 if abs(dx) == abs(dy) and dx != 0 else 0.0,# is diagonal
        0.0 if to in occ else 1.0,                     # target empty
        1.0 if (0 <= to[0] < N and 0 <= to[1] < N) else 0.0,  # on board
    ]

def sample(rule, enc, n, rng):
    D = []
    while len(D) < n:
        fr = (rng.randrange(N), rng.randrange(N))
        occ = {(rng.randrange(N), rng.randrange(N)) for _ in range(4)}
        # half the samples near-legal, so the boundary is actually probed
        if rng.random() < 0.5:
            to = (fr[0]+rng.choice([-1,0,1]), fr[1]+rng.choice([-1,0,1]))
        else:
            to = (rng.randrange(N), rng.randrange(N))
        if not (0 <= to[0] < N and 0 <= to[1] < N) or to == fr: continue
        D.append((enc(fr, to, occ), 1.0 if rule(fr, to, occ) else 0.0))
    return D

class Net:
    def __init__(self, ni, nh, rng):
        s = 1.0/math.sqrt(ni)
        self.ni, self.nh = ni, nh
        self.W1 = [[rng.uniform(-s,s) for _ in range(nh)] for _ in range(ni)]
        self.b1 = [0.0]*nh
        self.W2 = [rng.uniform(-1,1)/math.sqrt(nh) for _ in range(nh)]
        self.b2 = 0.0
    def fwd(self, x):
        zh = list(self.b1)
        for i, xi in enumerate(x):
            if xi == 0.0: continue
            Wi = self.W1[i]
            for h in range(self.nh): zh[h] += Wi[h]*xi
        ah = [math.tanh(z) for z in zh]
        z = self.b2 + sum(self.W2[h]*ah[h] for h in range(self.nh))
        p = 1.0/(1.0+math.exp(-max(-50.0, min(50.0, z))))
        return zh, ah, p
    def step(self, x, y, lr):
        zh, ah, p = self.fwd(x)
        d = p - y                                   # sigmoid + cross-entropy
        for h in range(self.nh):
            dh = d*self.W2[h]*(1.0-ah[h]*ah[h])
            self.W2[h] -= lr*d*ah[h]
            self.b1[h] -= lr*dh
            for i, xi in enumerate(x):
                if xi != 0.0: self.W1[i][h] -= lr*dh*xi
        self.b2 -= lr*d
    def acc(self, D):
        return sum(1 for x, y in D if (self.fwd(x)[2] >= 0.5) == (y >= 0.5))/len(D)

def train(net, D, epochs, lr, rng):
    for _ in range(epochs):
        rng.shuffle(D)
        for x, y in D: net.step(x, y, lr)

def experiment(enc, ni, k, seed, nh=12, lr=0.1):
    rng = random.Random(seed)
    pre   = sample(legal_king,     enc, 600, rng)
    few   = sample(legal_checkers, enc, k,   rng)
    test  = sample(legal_checkers, enc, 400, rng)
    scratch = Net(ni, nh, random.Random(seed+1)); train(scratch, few, 300, lr, rng)
    xfer    = Net(ni, nh, random.Random(seed+1))
    train(xfer, pre, 120, lr, rng)
    zero = xfer.acc(test)
    train(xfer, few, 300, lr, rng)
    return scratch.acc(test), xfer.acc(test), zero

if __name__ == "__main__":
    print("Train on KING moves, then fine-tune on k CHECKERS examples.")
    print("Accuracy on held-out checkers moves, 5 seeds.\n")
    print(f"  {'encoding':>12} {'k':>4} {'scratch':>9} {'transfer':>9} {'zero-shot':>10} {'gain':>7}")
    for name, enc, ni in (("raw", raw_enc, 2*N*N), ("generalised", gen_enc, 8)):
        for k in (5, 10, 25, 100):
            r = [experiment(enc, ni, k, s) for s in range(5)]
            sc = sum(a for a,_,_ in r)/5; tf = sum(b for _,b,_ in r)/5; zs = sum(c for _,_,c in r)/5
            print(f"  {name:>12} {k:>4} {sc:>9.3f} {tf:>9.3f} {zs:>10.3f} {tf-sc:>+7.3f}")
        print()
