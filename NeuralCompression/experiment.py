"""Neural compression: can one network hold many games?

Tests the split-and-selector scheme -- each game occupying a band of the output
range, with a selector choosing the band -- against conditioning baselines.

Run:  python3 experiment.py            all three experiments
      python3 experiment.py sweep      the sine-frequency sweep only

Pure standard library, no dependencies. Deterministic under the seeds given.

Two methodological points that decide whether the numbers mean anything:

  1. A partitioned output has targets in a band of width 1/N, so its raw MSE is
     automatically N^2 smaller. Every error below is RESCALED back into the
     game's own [0,1] (yhat = o*N - band) before comparison.

  2. That same 1/N band makes the gradient N times smaller, which would train
     the partition N times slower and report a false negative. The loss is
     therefore measured in game units for every condition (gain=N), with
     gradient clipping at 5.0 per GTMNN/DESIGN.md 7.4 -- without the clip the
     amplified gradient diverges at N=16.
"""

import math, random

def make_games(N, rng):
    """Game k sits at t=k/(N-1) on a smooth curve, so index order IS similarity order."""
    G=[]
    for k in range(N):
        t = k/(N-1) if N>1 else 0.5
        a = 1.0 + 5.0*t; b = 4.0 - 3.0*t; c = 2.0*math.pi*t
        G.append((a,b,c))
    return G

def target(g,x0,x1):
    a,b,c=g
    return 0.5 + 0.4*math.sin(a*x0 + b*x1 + c)

def data(G,n,rng):
    D=[]
    for k,g in enumerate(G):
        pts=[]
        for _ in range(n):
            x0,x1=rng.random(),rng.random()
            pts.append((x0,x1,target(g,x0,x1)))
        D.append(pts)
    return D

class MLP:
    def __init__(self,ni,nh,rng,act="tanh",b=1.0/3.0):
        self.ni,self.nh,self.act,self.b=ni,nh,act,b
        s=1.0/math.sqrt(ni)
        self.W1=[[rng.uniform(-s,s) for _ in range(nh)] for _ in range(ni)]
        self.b1=[0.0]*nh
        self.W2=[rng.uniform(-1,1)/math.sqrt(nh) for _ in range(nh)]
        self.b2=0.0
    def f(self,z):
        return math.tanh(z) if self.act=="tanh" else -math.sin(self.b*z)
    def df(self,z,a):
        return 1.0-a*a if self.act=="tanh" else -math.cos(self.b*z)*self.b
    def forward(self,x):
        zh=[0.0]*self.nh; ah=[0.0]*self.nh
        for h in range(self.nh):
            z=self.b1[h]
            for i in range(self.ni): z+=self.W1[i][h]*x[i]
            zh[h]=z; ah[h]=self.f(z)
        o=self.b2
        for h in range(self.nh): o+=self.W2[h]*ah[h]
        return zh,ah,o
    def step(self,x,t,lr,gain=1.0):
        zh,ah,o=self.forward(x)
        d=(o-t)*gain      # gain=N for partitioned outputs: loss is measured in GAME units,
                          # so the band's 1/N width does not shrink the gradient
        if d >  5.0: d= 5.0      # gradient clipping, per GTMNN/DESIGN.md 7.4 (clip=5.0).
        elif d < -5.0: d=-5.0    # Without it the gain=N amplification diverges at N=16.
        for h in range(self.nh):
            gW2=d*ah[h]
            dh=d*self.W2[h]*self.df(zh[h],ah[h])
            self.W2[h]-=lr*gW2
            self.b1[h]-=lr*dh
            for i in range(self.ni): self.W1[i][h]-=lr*dh*x[i]
        self.b2-=lr*d
        return d*d
    def params(self): return self.ni*self.nh+self.nh+self.nh+1

def run(cond,N,nh=16,ntr=48,nte=48,epochs=400,lr=0.05,seed=0,act="tanh",b=1.0/3.0):
    rng=random.Random(seed)
    G=make_games(N,rng); TR=data(G,ntr,rng); TE=data(G,nte,rng)
    band=list(range(N))
    if cond=="partition_shuffled":
        rng2=random.Random(seed+99); rng2.shuffle(band)
    onehot = "onehot" in cond
    ni = 2+N if onehot else 3
    if cond=="separate":
        nets=[MLP(2,nh,random.Random(seed*100+k),act,b) for k in range(N)]
        for ep in range(epochs):
            for k in range(N):
                pts=TR[k][:]; rng.shuffle(pts)
                for x0,x1,y in pts: nets[k].step([x0,x1],y,lr)
        err=[]; 
        for k in range(N):
            se=sum((nets[k].forward([x0,x1])[2]-y)**2 for x0,x1,y in TE[k])/len(TE[k]); err.append(se)
        return sum(err)/N, max(err), nets[0].params()*N
    net=MLP(ni,nh,rng,act,b)
    def enc(k,x0,x1):
        if onehot:
            v=[x0,x1]+[1.0 if j==k else 0.0 for j in range(N)]; return v
        s=(band[k]+0.5)/N
        return [x0,x1,s]
    part = cond.startswith("partition")
    for ep in range(epochs):
        order=[(k,p) for k in range(N) for p in TR[k]]; rng.shuffle(order)
        for k,(x0,x1,y) in order:
            t=(band[k]+y)/N if part else y
            net.step(enc(k,x0,x1),t,lr,N if part else 1.0)
    err=[]
    for k in range(N):
        se=0.0
        for x0,x1,y in TE[k]:
            o=net.forward(enc(k,x0,x1))[2]
            yhat = o*N-band[k] if part else o      # RESCALE to the game's own [0,1]
            se+=(yhat-y)**2
        err.append(se/len(TE[k]))
    return sum(err)/N, max(err), net.params()


def run_holdout(cond, N=16, nh=16, ntr=48, nte=48, epochs=400, lr=0.05,
                seed=0, act="tanh", b=1.0/3.0):
    """Experiment 2: train on even-indexed games, evaluate on the held-out odd ones."""
    rng = random.Random(seed)
    G = make_games(N, rng); TR = data(G, ntr, rng); TE = data(G, nte, rng)
    train_k = [k for k in range(N) if k % 2 == 0]
    held    = [k for k in range(N) if k % 2 == 1]
    onehot = "onehot" in cond; part = cond.startswith("partition")
    net = MLP(2 + N if onehot else 3, nh, rng, act, b)
    def enc(k, x0, x1):
        if onehot: return [x0, x1] + [1.0 if j == k else 0.0 for j in range(N)]
        return [x0, x1, (k + 0.5) / N]          # the continuous "plane" coordinate
    for _ in range(epochs):
        order = [(k, p) for k in train_k for p in TR[k]]; rng.shuffle(order)
        for k, (x0, x1, y) in order:
            net.step(enc(k, x0, x1), (k + y) / N if part else y, lr, N if part else 1.0)
    def mse(ks):
        out = []
        for k in ks:
            se = 0.0
            for x0, x1, y in TE[k]:
                o = net.forward(enc(k, x0, x1))[2]
                se += ((o * N - k if part else o) - y) ** 2
            out.append(se / len(TE[k]))
        return sum(out) / len(out)
    return mse(train_k), mse(held)


def exp1():
    print("EXPERIMENT 1 - capacity vs number of games (test MSE, rescaled to each game's [0,1])\n")
    for act, b, lbl in (("tanh", 1/3, "tanh"), ("sine", 1.0, "sine b=1")):
        print(f"  activation: {lbl}")
        print(f"  {'N':>3} {'condition':>20} {'meanMSE':>9} {'worstMSE':>9} {'params':>7}")
        for N in (2, 4, 8, 16):
            for c in ("partition_ordered", "partition_shuffled", "partition_onehot",
                      "scalar_full", "onehot_full", "separate"):
                m, w, p = run(c, N, act=act, b=b)
                print(f"  {N:>3} {c:>20} {m:>9.5f} {w:>9.5f} {p:>7}")
            print()

def exp2():
    print("EXPERIMENT 2 - generalisation to UNSEEN games (train even k, test odd k), N=16\n")
    print(f"  {'activation':>10} {'condition':>20} {'seen':>9} {'UNSEEN':>9}")
    for act, b, lbl in (("tanh", 1/3, "tanh"), ("sine", 1.0, "sine b=1")):
        for c in ("partition_ordered", "scalar_full", "onehot_full"):
            a, h = run_holdout(c, act=act, b=b)
            print(f"  {lbl:>10} {c:>20} {a:>9.5f} {h:>9.5f}")
        print()

def sweep():
    print("EXPERIMENT 3 - sine frequency b (N=1; the target IS a sine of a linear")
    print("combination, so a sine unit should fit it exactly)\n")
    print(f"  {'b':>8} {'|z| to 1st peak':>16} {'sine MSE':>12}")
    for b in (1/3, 1.0, 2.0, 3.0, 5.0):
        m, _, _ = run("scalar_full", 1, lr=0.3, act="sine", b=b)
        print(f"  {b:>8.3f} {math.pi/(2*b):>16.2f} {m:>12.5f}")
    print(f"\n  {'tanh reference':>26} {run('scalar_full',1,lr=0.3,act='tanh')[0]:>12.5f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep": sweep()
    else:
        exp1(); print(); exp2(); print(); sweep()
