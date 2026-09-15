"""Mechanism 2: dual network -- train granularly, then consolidate into a partition.

A SPECIALIST network learns one game alone, at full output range, with its own
capacity and no interference from other games. Its learning is then moved into
one band of the shared QUERY network, which is the only thing ever queried.
Fine-tuning follows. The analogy is systems consolidation in sleep: the fast
learner rehearses, the slow store integrates.

Three transfer routes, all measured against training the query network directly:

  joint     no specialists. Train the query network on every game at once.
            The baseline the other routes have to beat.
  graft     copy each specialist's weights into its own slice of the query
            network's hidden layer, zero the selector weights, then fine-tune.
            The literal "slot the weights in".
  replay    each specialist GENERATES samples; the query network trains on them
            in its band. Nothing is copied. This is the REM reading: the fast
            learner replays experience, the slow store learns from the replay.

Pure standard library. Run:  python3 consolidate.py
"""
import math, random, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import MLP, make_games, target, data

def train_specialist(g, nh, epochs, lr, rng, act, b, n=64):
    """One game, full [0,1] output, no partition, no interference."""
    net = MLP(2, nh, rng, act, b)
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    pts = [(x0, x1, target(g, x0, x1)) for x0, x1 in pts]
    for _ in range(epochs):
        rng.shuffle(pts)
        for x0, x1, y in pts: net.step([x0, x1], y, lr)
    return net

def graft(spec, q, k, N, hs):
    """Copy specialist k into hidden slice [k*hs, (k+1)*hs) of the query network.

    The selector row is zeroed: a grafted unit fires for every game, so it must
    LEARN to gate on s during fine-tuning. Output weights are scaled by 1/N
    because the band is 1/N as wide as the specialist's full range.
    """
    for i in range(2):
        for h in range(hs):
            q.W1[i][k*hs + h] = spec.W1[i][h]
    for h in range(hs):
        q.W1[2][k*hs + h] = 0.0                 # selector weight: learned, not copied
        q.b1[k*hs + h] = spec.b1[h]
        q.W2[k*hs + h] = spec.W2[h] / N
    # The band offset k/N is NOT grafted: it is game-dependent and the grafted units
    # are blind to the selector. Since s = (k+0.5)/N, the offset k/N = s - 0.5/N is
    # LINEAR in s, so fine-tuning can recover it through the selector row. Whether it
    # actually does is what this experiment measures.

def band_train(net, items, N, epochs, lr, rng):
    for _ in range(epochs):
        rng.shuffle(items)
        for k, x0, x1, y in items:
            net.step([x0, x1, (k+0.5)/N], (k+y)/N, lr, N)

def ev(net, N, TE):
    out = []
    for k in range(N):
        se = 0.0
        for x0, x1, y in TE[k]:
            se += (net.forward([x0, x1, (k+0.5)/N])[2]*N - k - y)**2
        out.append(se/len(TE[k]))
    return sum(out)/len(out)

def run(route, N=4, hs=4, epochs=400, ft=200, lr=0.05, seed=0, act="tanh", b=1/3):
    rng = random.Random(seed)
    G = make_games(N, rng); TR = data(G, 48, rng); TE = data(G, 48, rng)
    hq = N*hs
    q = MLP(3, hq, rng, act, b)
    items = [(k, x0, x1, y) for k in range(N) for x0, x1, y in TR[k]]
    if route == "joint":
        band_train(q, items, N, epochs + ft, lr, rng)
        return ev(q, N, TE)
    specs = [train_specialist(G[k], hs, epochs, lr, rng, act, b) for k in range(N)]
    solo = sum(sum((s.forward([x0,x1])[2]-y)**2 for x0,x1,y in TE[k])/len(TE[k])
               for k, s in enumerate(specs))/N
    if route == "graft":
        for k, s in enumerate(specs): graft(s, q, k, N, hs)
        band_train(q, items, N, ft, lr, rng)                     # fine-tune only
    elif route == "replay":
        gen = []
        for k, s in enumerate(specs):                             # the specialist dreams
            for _ in range(96):
                x0, x1 = rng.random(), rng.random()
                gen.append((k, x0, x1, max(0.0, min(1.0, s.forward([x0, x1])[2]))))
        band_train(q, gen, N, epochs, lr, rng)                    # learn from the replay
        band_train(q, items, N, ft, lr, rng)                      # then fine-tune on real data
    return ev(q, N, TE), solo

if __name__ == "__main__":
    print("Dual network: specialists learn alone, then consolidate into the query net.\n")
    print(f"  {'activation':>12} {'route':>10} {'query MSE':>10} {'specialist MSE':>15} {'vs joint':>9}")
    for act, b, lbl in (("tanh", 1/3, "tanh"), ("sine", 1.0, "sine b=1")):
        base = sum(run("joint", act=act, b=b, seed=s) for s in range(3))/3
        print(f"  {lbl:>12} {'joint':>10} {base:>10.5f} {'—':>15} {'—':>9}")
        for route in ("graft", "replay"):
            vals = [run(route, act=act, b=b, seed=s) for s in range(3)]
            m = sum(v[0] for v in vals)/3; solo = sum(v[1] for v in vals)/3
            print(f"  {lbl:>12} {route:>10} {m:>10.5f} {solo:>15.5f} {base/m:>8.2f}x")
        print()
