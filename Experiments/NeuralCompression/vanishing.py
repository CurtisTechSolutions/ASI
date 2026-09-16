"""Mechanism 1: accept the vanishing gradient, then invert out of it.

The vanishing gradient is treated as a feature: granular training is allowed to
stall, the stall is DETECTED, and the update sign is flipped to climb back out
before descent resumes.

Two triggers are implemented, because they fire in different regimes:

  grad    mean |dL/dw| in the first layer falls below `grad_tau`.
          The literal vanishing-gradient condition.
  weight  max |w| in the first layer stays below `weight_tau`.
          The author's stated condition -- it detects a layer that never moved
          off its initialisation, which is what vanishing gradients leave behind.

Inversion runs for `steps` updates (gradient ASCENT), then descent resumes.

Pure standard library. Run:  python3 vanishing.py
"""
import math, random

class DeepMLP:
    """ni -> nh -> ... -> 1, `depth` hidden layers. Deep enough to vanish."""
    def __init__(self, ni, nh, depth, rng, act="sine", b=1.0/3.0, a=-1.0, h=0.0, k=0.0):
        self.ni, self.nh, self.depth, self.act, self.b = ni, nh, depth, act, b
        self.a, self.h, self.k = a, h, k
        dims = [ni] + [nh]*depth
        self.W = [[[rng.uniform(-1,1)/math.sqrt(dims[l]) for _ in range(dims[l+1])]
                   for _ in range(dims[l])] for l in range(depth)]
        self.B = [[0.0]*dims[l+1] for l in range(depth)]
        self.Wo = [rng.uniform(-1,1)/math.sqrt(nh) for _ in range(nh)]
        self.bo = 0.0

    def f(self, z):
        """``f(z) = a*sin(b*(z-h)) + k`` -- Research/SineWaveActivationFunction.md 4.

        The author's formula in full. At the defaults ``a=-1, h=0, k=0`` it is
        ``-sin(b*z)``, which every number below was measured on; ``b`` is what
        this file sweeps (1/3 against 1)."""
        if self.act == "tanh": return math.tanh(z)
        return self.a*math.sin(self.b*(z-self.h))+self.k
    def df(self, z, fz):
        """``df/dz = a*b*cos(b*(z-h))``; ``fz`` is ``f(z)``, which only tanh reuses."""
        if self.act == "tanh": return 1.0-fz*fz
        return self.a*self.b*math.cos(self.b*(z-self.h))

    def forward(self, x):
        zs, as_ = [], [x]
        cur = x
        for l in range(self.depth):
            nz = len(self.B[l]); z = list(self.B[l])
            for i, xi in enumerate(cur):
                if xi == 0.0: continue
                Wi = self.W[l][i]
                for j in range(nz): z[j] += Wi[j]*xi
            a = [self.f(v) for v in z]
            zs.append(z); as_.append(a); cur = a
        o = self.bo + sum(self.Wo[h]*cur[h] for h in range(self.nh))
        return zs, as_, o

    def negate_layer(self, l):
        """w -> -w for hidden layer l. With an ODD activation this is a symmetry
        operation, not noise: -sin(b*(-z)) = -(-sin(b*z)), so the layer's output
        negates exactly and the function computed stays structurally related.
        f(z) = a*sin(b*(z-h)) + k is odd only while h = k = 0, which is the
        default here; a run that moves either off zero loses this symmetry."""
        for row in self.W[l]:
            for j in range(len(row)): row[j] = -row[j]
        for j in range(len(self.B[l])): self.B[l][j] = -self.B[l][j]

    def step(self, x, t, lr, sign=1.0, clip=5.0, signs=None):
        """sign=-1.0 performs gradient ASCENT (the inversion). `signs` overrides it
        per hidden layer (list of length `depth`), for rotating inversion.
        Returns (sq_err, |g| layer 0)."""
        zs, as_, o = self.forward(x)
        d = o - t
        if d > clip: d = clip
        elif d < -clip: d = -clip
        g = [d*self.Wo[h]*self.df(zs[-1][h], as_[-1][h]) for h in range(self.nh)]
        for h in range(self.nh): self.Wo[h] -= sign*lr*d*as_[-1][h]
        self.bo -= sign*lr*d
        g0mag = 0.0
        for l in range(self.depth-1, -1, -1):
            sgn = sign if signs is None else signs[l]
            prev = as_[l]; nz = len(g)
            if l == 0:
                g0mag = sum(abs(gj) for gj in g)/max(1, nz)
            nxt = [0.0]*len(prev)
            for i, xi in enumerate(prev):
                Wi = self.W[l][i]; acc = 0.0
                for j in range(nz):
                    acc += Wi[j]*g[j]
                    Wi[j] -= sgn*lr*g[j]*xi
                nxt[i] = acc
            for j in range(nz): self.B[l][j] -= sgn*lr*g[j]
            if l > 0:
                g = [nxt[i]*self.df(zs[l-1][i], as_[l][i]) for i in range(len(prev))]
        return d*d, g0mag

    def layer_grad_scale(self):
        """Product of |f'| at z=0 across layers: the multiplicative gradient decay."""
        d0 = 1.0 if self.act == "tanh" else self.b
        return d0 ** self.depth

    def max_abs_w0(self):
        return max(abs(w) for row in self.W[0] for w in row)


class Inversion:
    """Flip the update sign when training has stalled.

    `loss_tau` is the repair. A small gradient means EITHER stuck on a plateau OR
    converged, and the magnitude cannot tell those apart -- so the ungated version
    fires near the optimum and kicks a good solution out. Gating on "loss is still
    bad" is what makes the trigger mean `stuck` rather than merely `flat`.
    """
    def __init__(self, trigger="grad", grad_tau=2e-3, weight_tau=0.6,
                 loss_tau=None, steps=40, cooldown=400,
                 warmup=2000, window=1500, min_improve=0.02):
        self.trigger, self.grad_tau, self.weight_tau = trigger, grad_tau, weight_tau
        self.loss_tau = loss_tau
        self.steps, self.cooldown = steps, cooldown
        self.warmup, self.window, self.min_improve = warmup, window, min_improve
        self.left = 0; self.cool = 0; self.fired = 0; self.loss = 1.0
        self.seen = 0; self.best = None; self.since = 0
    def note_loss(self, sq):
        self.loss = 0.99*self.loss + 0.01*sq          # running mean
        self.seen += 1
        if self.best is None or self.loss < self.best*(1.0-self.min_improve):
            self.best = self.loss; self.since = 0     # real improvement: reset
        else:
            self.since += 1
    def plateaued(self):
        """Stalled = past warmup AND the running loss has not improved for `window`
        updates. This is the signal the gradient magnitude could not provide: it
        separates `stuck` from `still starting` and from `converged and flat`."""
        return self.seen > self.warmup and self.since > self.window
    def sign(self, net, g0mag):
        if self.left > 0:
            self.left -= 1; return -1.0
        if self.cool > 0:
            self.cool -= 1; return 1.0
        if self.trigger == "plateau":
            hit = self.plateaued()
        else:
            hit = (g0mag < self.grad_tau) if self.trigger == "grad" \
                  else (net.max_abs_w0() < self.weight_tau)
        if hit and self.loss_tau is not None and self.loss < self.loss_tau:
            hit = False                                # flat AND good = converged, not stuck
        if hit:
            self.left = self.steps - 1; self.cool = self.cooldown; self.fired += 1
            self.since = 0; self.best = self.loss
            return -1.0
        return 1.0


class Rotating:
    """Method 3: invert alternating layers each cycle.

    Depth 5 -> layers {0,2,4} invert, then {1,3}, then {0,2,4}, and so on
    (0-indexed; the author's 1,3,5 / 2,4).

    mode="grad"   the active parity performs gradient ASCENT for that cycle while
                  the other parity descends. An adversarial split within one net.
    mode="weight" the active parity has w -> -w at each cycle boundary. With an odd
                  activation this walks the network's sign-symmetry orbit, and the
                  parity alternation has PERIOD 4: (odd)(even)(odd)(even) returns
                  every layer to its original sign. Escape without unlearning.
    """
    def __init__(self, depth, mode="weight", period=25):
        self.depth, self.mode, self.period = depth, mode, period
        self.parity = 0; self.cycles = 0
    def layers(self, parity):
        return [l for l in range(self.depth) if l % 2 == parity]
    def on_cycle(self, net):
        """Call at each cycle boundary. Returns the signs for the coming cycle."""
        if self.mode == "weight":
            for l in self.layers(self.parity): net.negate_layer(l)
        self.cycles += 1
        active = self.layers(self.parity)
        self.parity ^= 1
        if self.mode == "grad":
            return [-1.0 if l in active else 1.0 for l in range(self.depth)]
        return None


def task(n, rng, kind="smooth"):
    """smooth: one low-frequency sine -- essentially no local minima.
    rugged: a high-frequency product -- many basins, so a symmetry hop has
            something to escape FROM. Testing an escape mechanism on a task with
            nothing to escape measures nothing."""
    D = []
    for _ in range(n):
        x0, x1 = rng.random(), rng.random()
        if kind == "smooth":
            y = 0.5 + 0.4*math.sin(3.0*x0 + 2.0*x1 + 0.7)
        else:
            y = 0.5 + 0.25*math.sin(9.0*x0)*math.cos(7.0*x1) + 0.15*math.sin(5.0*x1)
        D.append(([x0, x1], y))
    return D

def run(depth, act, b, inv, epochs=300, nh=8, lr=0.05, seed=0, ntr=64, nte=64, rot=None, kind="smooth"):
    rng = random.Random(seed)
    net = DeepMLP(2, nh, depth, rng, act, b)
    TR, TE = task(ntr, rng, kind), task(nte, rng, kind)
    pol = Inversion(**inv) if inv else None
    rotor = Rotating(depth, **rot) if rot else None
    signs = None
    gmags = []; g0 = 1.0
    for ep in range(epochs):
        if rotor and ep % rotor.period == 0:
            signs = rotor.on_cycle(net)
        rng.shuffle(TR)
        for x, t in TR:
            # decide the sign from the PREVIOUS step's gradient, so every condition
            # takes exactly one update per sample and the comparison stays fair
            sgn = pol.sign(net, g0) if pol else 1.0
            sq, g0 = net.step(x, t, lr, sgn, signs=signs)
            if pol: pol.note_loss(sq)
            gmags.append(g0)
    mse = sum((net.forward(x)[2]-t)**2 for x, t in TE)/len(TE)
    tail = gmags[-len(TR):]
    fired = pol.fired if pol else (rotor.cycles if rotor else 0)
    return mse, sum(tail)/len(tail), fired

if __name__ == "__main__":
    print("A. Is the vanishing gradient caused by b? Predicted decay = |f'(0)|^depth\n")
    print(f"  {'activation':>12} {'depth':>6} {'predicted decay':>16} {'measured |g| layer 0':>21} {'test MSE':>10}")
    for act, b, lbl in (("sine", 1/3, "sine b=1/3"), ("sine", 1.0, "sine b=1"), ("tanh", 1/3, "tanh")):
        for depth in (1, 2, 4):
            m, g, _ = run(depth, act, b, None)
            pred = (b if act == "sine" else 1.0) ** depth
            print(f"  {lbl:>12} {depth:>6} {pred:>16.5f} {g:>21.2e} {m:>10.5f}")
        print()

    print("\nB. Inversion as an escape (depth 4, where the gradient has vanished)\n")
    print("  'fires' is the TOTAL over the three seeds, and 'worst' is the worst single")
    print("  seed -- a mean hides the fact that ONE inversion can destroy a healthy net.\n")
    print(f"  {'activation':>12} {'policy':>18} {'test MSE':>10} {'worst seed':>11} {'fires':>6}")
    for act, b, lbl in (("sine", 1/3, "sine b=1/3"), ("sine", 1.0, "sine b=1"), ("tanh", 1/3, "tanh")):
        base = None
        for name, inv in (("none", None),
                          ("invert on grad", {"trigger": "grad"}),
                          ("invert on weight", {"trigger": "weight"}),
                          ("invert on plateau", {"trigger": "plateau"})):
            vals = [run(4, act, b, inv, seed=s) for s in range(3)]
            m = sum(v[0] for v in vals)/len(vals)
            fired = sum(v[2] for v in vals); worst = max(v[0] for v in vals)
            if base is None: base = m
            mark = "" if name == "none" else f"   ({base/m:.2f}x)"
            print(f"  {lbl:>12} {name:>18} {m:>10.5f} {worst:>11.5f} {fired:>6}{mark}")
        print()
