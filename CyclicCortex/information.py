"""What information does move legality actually require?

Balanced legal/illegal samples from real rules (board_games.py), with feature
groups added cumulatively. Each row answers: how far does this much information
get you, on each game?

  P  point      dx, dy, |dx|, |dy|, target empty, on board
                  -- the vocabulary CyclicCortex/common_denominator.py validated
  I  identity   moving piece type, target occupant (own / enemy / empty), direction
  A  path       path clear, path length, blockers between from and to
  G  group      liberties after the move, stones captured, adjacent group sizes
  S  move-set   is a capture available ELSEWHERE (legality depends on other moves)
  B  board      does this leave my own king attacked (needs the board AFTER the move)
  H  history    does this recreate the previous position (ko)

Pure standard library.  Run:  python3 information.py
"""
import math, random
from cortex.board_games import (Chess, Checkers, Go, random_chess, random_checkers,
                                random_go, on, GN)

KINDS = "PNBRQK"

# ------------------------------------------------------------ feature groups
def f_point(g, mv, kind):
    if kind == "go":
        x, y = mv
        return [x/(GN-1), y/(GN-1), 0.0, 0.0, 1.0 if not g.at(x,y) else 0.0, 1.0]
    (fx,fy),(tx,ty) = mv; dx,dy = tx-fx, ty-fy
    return [dx/7.0, dy/7.0, abs(dx)/7.0, abs(dy)/7.0,
            0.0 if g.at(tx,ty) else 1.0, 1.0 if on(tx,ty) else 0.0]

def f_identity(g, mv, kind):
    v = [0.0]*(len(KINDS)+4)
    if kind == "go":
        v[len(KINDS)] = 1.0 if g.turn == "b" else 0.0
        return v
    (fx,fy),(tx,ty) = mv
    p, t = g.at(fx,fy), g.at(tx,ty)
    if kind == "chess":
        if p and p[1] in KINDS: v[KINDS.index(p[1])] = 1.0
        v[len(KINDS)]   = 1.0 if p and p[0] == g.turn else 0.0
        v[len(KINDS)+1] = 1.0 if t and t[0] == g.turn else 0.0        # own piece on target
        v[len(KINDS)+2] = 1.0 if t and t[0] != g.turn else 0.0        # enemy on target
        v[len(KINDS)+3] = 1.0 if p and p[0] == "w" else 0.0           # direction of travel
    else:
        v[len(KINDS)]   = 1.0 if p and p.lower() == g.turn else 0.0
        v[len(KINDS)+1] = 1.0 if t else 0.0
        v[len(KINDS)+2] = 1.0 if p and p.isupper() else 0.0           # crowned
        v[len(KINDS)+3] = 1.0 if g.turn == "w" else 0.0
    return v

def f_path(g, mv, kind):
    if kind == "go": return [0.0, 0.0, 0.0]
    (fx,fy),(tx,ty) = mv; dx,dy = tx-fx, ty-fy
    sx = (dx>0)-(dx<0); sy = (dy>0)-(dy<0)
    n = max(abs(dx), abs(dy))
    if n == 0 or (dx and dy and abs(dx) != abs(dy)) or (dx and dy == 0 and False):
        return [0.0, n/7.0, 0.0]
    blockers = 0; x,y = fx+sx, fy+sy; steps = 0
    while (x,y) != (tx,ty) and on(x,y) and steps < 8:
        if g.at(x,y): blockers += 1
        x += sx; y += sy; steps += 1
    return [1.0 if blockers == 0 else 0.0, n/7.0, min(blockers,3)/3.0]

def f_group(g, mv, kind):
    if kind != "go": return [0.0, 0.0, 0.0, 0.0]
    x, y = mv
    if g.at(x,y): return [0.0, 0.0, 0.0, 1.0]
    b2 = [r[:] for r in g.b]; b2[y][x] = g.turn
    opp = "w" if g.turn == "b" else "b"
    caps = 0
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny = x+dx, y+dy
        if 0<=nx<GN and 0<=ny<GN and b2[ny][nx] == opp:
            grp, libs = g.group(nx,ny,b2)
            if not libs: caps += len(grp)
    own, libs = g.group(x,y,b2)
    return [min(len(libs),4)/4.0, min(caps,4)/4.0, min(len(own),6)/6.0, 0.0]

def f_moveset(g, mv, kind):
    """Does a capture exist ELSEWHERE? Checkers legality depends on it."""
    if kind != "checkers": return [0.0]
    return [1.0 if g.jumps() else 0.0]

def f_board(g, mv, kind):
    """Does this leave my own king attacked? Needs the board AFTER the move."""
    if kind != "chess": return [0.0, 0.0]
    (fx,fy),(tx,ty) = mv
    b2 = [r[:] for r in g.b]; b2[ty][tx] = b2[fy][fx]; b2[fy][fx] = ""
    nxt = Chess(b2, g.turn); ks = nxt.king_sq(g.turn)
    leaves = 1.0 if (ks and nxt.attacked(ks, "b" if g.turn=="w" else "w")) else 0.0
    return [leaves, 1.0 if g.in_check() else 0.0]

def f_history(g, mv, kind):
    if kind != "go": return [0.0]
    x,y = mv
    if g.at(x,y) or g.prev is None: return [0.0]
    b2 = [r[:] for r in g.b]; b2[y][x] = g.turn
    return [1.0 if b2 == g.prev else 0.0]

GROUPS = [("P", f_point), ("I", f_identity), ("A", f_path),
          ("G", f_group), ("S", f_moveset), ("B", f_board), ("H", f_history)]

def encode(g, mv, kind, upto):
    v = []
    for name, fn in GROUPS[:upto]: v += fn(g, mv, kind)
    return v

# --------------------------------------------------------------- sampling
def legal_targets_chess(g, fr):
    return [(x,y) for y in range(8) for x in range(8) if g.legal(fr,(x,y))]

def balanced(kind, n, rng):
    """Half legal, half illegal-but-plausible. Uniform sampling gives 0.4% legal
    for chess, where 'always illegal' scores 99.6% and the measurement is void."""
    out = []
    while len(out) < n:
        if kind == "chess":
            g = random_chess(rng)
            own = [(x,y) for y in range(8) for x in range(8)
                   if g.at(x,y) and g.at(x,y)[0] == g.turn]
            if not own: continue
            fr = rng.choice(own); good = legal_targets_chess(g, fr)
            if not good: continue
            bad = [(x,y) for y in range(8) for x in range(8)
                   if (x,y) not in good and (x,y) != fr]
            out.append((g, (fr, rng.choice(good)), 1.0))
            out.append((g, (fr, rng.choice(bad)), 0.0))
        elif kind == "checkers":
            g = random_checkers(rng)
            good = g.jumps() or g.simple_moves()
            if not good: continue
            mv = rng.choice(good)
            allsq = [(x,y) for y in range(8) for x in range(8)]
            for _ in range(40):
                bad = (rng.choice(allsq), rng.choice(allsq))
                if not g.legal(*bad): break
            out.append((g, mv, 1.0)); out.append((g, bad, 0.0))
        else:
            g = random_go(rng)
            pts = [(x,y) for y in range(GN) for x in range(GN)]
            good = [p for p in pts if g.legal(*p)]
            bad  = [p for p in pts if not g.legal(*p)]
            if not good or not bad: continue
            out.append((g, rng.choice(good), 1.0)); out.append((g, rng.choice(bad), 0.0))
    return out[:n]

# ------------------------------------------------------------------ model
class Net:
    def __init__(self, ni, nh, rng):
        s = 1.0/math.sqrt(max(1,ni)); self.ni, self.nh = ni, nh
        self.W1=[[rng.uniform(-s,s) for _ in range(nh)] for _ in range(ni)]
        self.b1=[0.0]*nh
        self.W2=[rng.uniform(-1,1)/math.sqrt(nh) for _ in range(nh)]; self.b2=0.0
    def fwd(self,x):
        zh=list(self.b1)
        for i,xi in enumerate(x):
            if xi==0.0: continue
            Wi=self.W1[i]
            for h in range(self.nh): zh[h]+=Wi[h]*xi
        ah=[math.tanh(z) for z in zh]
        z=self.b2+sum(self.W2[h]*ah[h] for h in range(self.nh))
        return ah, 1.0/(1.0+math.exp(-max(-50.0,min(50.0,z))))
    def step(self,x,y,lr):
        ah,p=self.fwd(x); d=p-y
        for h in range(self.nh):
            dh=d*self.W2[h]*(1.0-ah[h]*ah[h]); self.W2[h]-=lr*d*ah[h]; self.b1[h]-=lr*dh
            for i,xi in enumerate(x):
                if xi!=0.0: self.W1[i][h]-=lr*dh*xi
        self.b2-=lr*d

def evaluate(kind, upto, tr, te, seed, nh=20, lr=0.1, epochs=120):
    rng=random.Random(seed)
    X=[(encode(g,m,kind,upto),y) for g,m,y in tr]
    T=[(encode(g,m,kind,upto),y) for g,m,y in te]
    net=Net(len(X[0][0]),nh,rng)
    for _ in range(epochs):
        rng.shuffle(X)
        for x,y in X: net.step(x,y,lr)
    return sum(1 for x,y in T if (net.fwd(x)[1]>=0.5)==(y>=0.5))/len(T)

if __name__=="__main__":
    rng=random.Random(7)
    print("Move-legality accuracy as information is added. Balanced 50/50 samples,")
    print("so 0.500 is chance. 3 seeds.\n")
    data={}
    for kind,n in (("chess",900),("checkers",900),("go",900)):
        data[kind]=(balanced(kind,n,rng), balanced(kind,400,rng))
    names=["+P point","+I identity","+A path","+G group","+S move-set","+B board","+H history"]
    print(f"  {'information':>14} {'chess':>8} {'checkers':>9} {'go':>8}")
    prev={}
    for i,nm in enumerate(names,1):
        row=[]
        for kind in ("chess","checkers","go"):
            tr,te=data[kind]
            a=sum(evaluate(kind,i,tr,te,s) for s in range(3))/3
            row.append(a)
        marks=[]
        for kind,a in zip(("chess","checkers","go"),row):
            d=a-prev.get(kind,0.5); prev[kind]=a
            marks.append(f"{a:.3f}" + (f" (+{d:.3f})" if d>=0.005 else "")) 
        print(f"  {nm:>14} {marks[0]:>16} {marks[1]:>16} {marks[2]:>16}")
