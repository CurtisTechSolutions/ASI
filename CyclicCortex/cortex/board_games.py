"""Ground-truth rules for chess, checkers and go, to measure what information
move legality actually requires. Small but correct; self-checked at import.

Deliberately covers the parts that DIFFER in the kind of state they need:
  chess     piece types, sliding paths, capture-vs-block, and CHECK (a global
            predicate: legality depends on the board AFTER the move)
  checkers  men and kings, jumps, and FORCED CAPTURE (a set predicate: this
            move's legality depends on what OTHER moves exist)
  go        liberties and groups (a connectivity predicate), suicide, and KO
            (a history predicate: depends on a previous position)
"""
import random

# ---------------------------------------------------------------- chess
EMPTY = "."
SLIDE = {"B": [(1,1),(1,-1),(-1,1),(-1,-1)],
         "R": [(1,0),(-1,0),(0,1),(0,-1)]}
SLIDE["Q"] = SLIDE["B"] + SLIDE["R"]
STEP = {"N": [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)],
        "K": [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}

def on(x, y): return 0 <= x < 8 and 0 <= y < 8

class Chess:
    """board[y][x] = '' or colour+kind, e.g. 'wQ'. White moves up (+y)."""
    def __init__(self, board, turn="w"): self.b, self.turn = board, turn

    def at(self, x, y): return self.b[y][x]

    def pseudo_legal(self, fr, to):
        """Movement and capture rules, path clearance -- everything EXCEPT check."""
        (fx, fy), (tx, ty) = fr, to
        if not (on(fx,fy) and on(tx,ty)) or fr == to: return False
        p = self.at(fx,fy)
        if not p or p[0] != self.turn: return False
        t = self.at(tx,ty)
        if t and t[0] == self.turn: return False          # own piece blocks
        k, dx, dy = p[1], tx-fx, ty-fy
        if k == "P":
            d = 1 if p[0] == "w" else -1
            if dx == 0 and dy == d and not t: return True
            if dx == 0 and dy == 2*d and fy == (1 if p[0]=="w" else 6) \
               and not t and not self.at(fx, fy+d): return True
            return abs(dx) == 1 and dy == d and bool(t)   # capture only diagonally
        if k in STEP: return (dx,dy) in STEP[k]
        if k in SLIDE:
            # direction must match and the path must be clear
            sx = (dx > 0) - (dx < 0); sy = (dy > 0) - (dy < 0)
            if (sx,sy) not in SLIDE[k]: return False
            if dx and dy and abs(dx) != abs(dy): return False
            if (dx == 0) != (sx == 0) or (dy == 0) != (sy == 0): return False
            x, y = fx+sx, fy+sy
            while (x,y) != (tx,ty):
                if self.at(x,y): return False              # path blocked
                x += sx; y += sy
            return True
        return False

    def attacked(self, sq, by):
        save = self.turn; self.turn = by
        try:
            for y in range(8):
                for x in range(8):
                    p = self.at(x,y)
                    if p and p[0] == by and self.pseudo_legal((x,y), sq): return True
            return False
        finally: self.turn = save

    def king_sq(self, col):
        for y in range(8):
            for x in range(8):
                if self.at(x,y) == col+"K": return (x,y)
        return None

    def legal(self, fr, to):
        """Full legality: pseudo-legal AND does not leave own king attacked."""
        if not self.pseudo_legal(fr, to): return False
        b2 = [row[:] for row in self.b]
        b2[to[1]][to[0]] = b2[fr[1]][fr[0]]; b2[fr[1]][fr[0]] = ""
        nxt = Chess(b2, self.turn)
        ks = nxt.king_sq(self.turn)
        return ks is None or not nxt.attacked(ks, "b" if self.turn == "w" else "w")

    def in_check(self):
        ks = self.king_sq(self.turn)
        return ks is not None and self.attacked(ks, "b" if self.turn=="w" else "w")

def random_chess(rng):
    b = [["" for _ in range(8)] for _ in range(8)]
    sqs = rng.sample([(x,y) for x in range(8) for y in range(8)], rng.randint(6, 14))
    b[sqs[0][1]][sqs[0][0]] = "wK"; b[sqs[1][1]][sqs[1][0]] = "bK"
    for x,y in sqs[2:]:
        b[y][x] = rng.choice("wb") + rng.choice("PNBRQ")
    return Chess(b, rng.choice("wb"))

# ------------------------------------------------------------- checkers
class Checkers:
    """board[y][x] = '' | 'w' | 'b' | 'W' | 'B' (upper = king). w moves +y."""
    def __init__(self, board, turn="w"): self.b, self.turn = board, turn
    def at(self,x,y): return self.b[y][x]
    def dirs(self, p):
        if p.isupper(): return [(1,1),(1,-1),(-1,1),(-1,-1)]
        return [(1,1),(-1,1)] if p == "w" else [(1,-1),(-1,-1)]
    def simple_moves(self):
        out = []
        for y in range(8):
            for x in range(8):
                p = self.at(x,y)
                if p and p.lower() == self.turn:
                    for dx,dy in self.dirs(p):
                        nx,ny = x+dx, y+dy
                        if on(nx,ny) and not self.at(nx,ny): out.append(((x,y),(nx,ny)))
        return out
    def jumps(self):
        out = []
        for y in range(8):
            for x in range(8):
                p = self.at(x,y)
                if p and p.lower() == self.turn:
                    for dx,dy in self.dirs(p):
                        mx,my,nx,ny = x+dx, y+dy, x+2*dx, y+2*dy
                        if on(nx,ny) and not self.at(nx,ny):
                            m = self.at(mx,my)
                            if m and m.lower() != self.turn: out.append(((x,y),(nx,ny)))
        return out
    def legal(self, fr, to):
        """FORCED CAPTURE: if any jump exists, only jumps are legal. This move's
        legality depends on moves it is not part of."""
        js = self.jumps()
        return (fr,to) in js if js else (fr,to) in self.simple_moves()

def random_checkers(rng):
    b = [["" for _ in range(8)] for _ in range(8)]
    for x,y in rng.sample([(x,y) for x in range(8) for y in range(8) if (x+y)%2==0],
                          rng.randint(6, 16)):
        b[y][x] = rng.choice(["w","b","W","B"])
    return Checkers(b, rng.choice("wb"))

# ------------------------------------------------------------------ go
GN = 9
class Go:
    """board[y][x] = '' | 'b' | 'w'. `prev` is the position before the last move (ko)."""
    def __init__(self, board, turn="b", prev=None):
        self.b, self.turn, self.prev = board, turn, prev
    def at(self,x,y): return self.b[y][x]
    def group(self, x, y, b=None):
        b = b or self.b; col = b[y][x]
        seen, stack, libs = {(x,y)}, [(x,y)], set()
        while stack:
            cx,cy = stack.pop()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny = cx+dx, cy+dy
                if not (0<=nx<GN and 0<=ny<GN): continue
                v = b[ny][nx]
                if not v: libs.add((nx,ny))
                elif v == col and (nx,ny) not in seen:
                    seen.add((nx,ny)); stack.append((nx,ny))
        return seen, libs
    def legal(self, x, y):
        """Illegal if occupied, if SUICIDE (no liberties and captures nothing),
        or if KO (recreates the previous position)."""
        if self.at(x,y): return False
        b2 = [r[:] for r in self.b]; b2[y][x] = self.turn
        opp = "w" if self.turn=="b" else "b"
        captured = False
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0<=nx<GN and 0<=ny<GN and b2[ny][nx] == opp:
                g,l = self.group(nx,ny,b2)
                if not l:
                    captured = True
                    for gx,gy in g: b2[gy][gx] = ""
        if not captured:
            _, libs = self.group(x,y,b2)
            if not libs: return False                     # suicide
        if self.prev is not None and b2 == self.prev: return False   # ko
        return True

def random_go(rng):
    b = [["" for _ in range(GN)] for _ in range(GN)]
    for x,y in rng.sample([(x,y) for x in range(GN) for y in range(GN)], rng.randint(10, 45)):
        b[y][x] = rng.choice("bw")
    return Go(b, rng.choice("bw"))

# ------------------------------------------------------------- sanity
def _selftest():
    b = [["" for _ in range(8)] for _ in range(8)]
    b[1][4] = "wK"; b[7][4] = "bK"; b[0][0] = "wR"     # king OFF the first rank
    c = Chess(b, "w")
    assert c.pseudo_legal((0,0),(5,0)), "rook should slide along an empty rank"
    assert not c.pseudo_legal((0,0),(5,5)), "rook may not move diagonally"
    b[0][3] = "wP"
    assert not Chess(b,"w").pseudo_legal((0,0),(5,0)), "rook must be blocked by a pawn"
    # a pinned piece may not move: white rook on e2 pinned by black rook on e8
    b2 = [["" for _ in range(8)] for _ in range(8)]
    b2[0][4] = "wK"; b2[1][4] = "wR"; b2[7][4] = "bR"; b2[7][0] = "bK"
    assert Chess(b2,"w").pseudo_legal((4,1),(0,1)), "pseudo-legal ignores the pin"
    assert not Chess(b2,"w").legal((4,1),(0,1)), "legal must reject the pinned move"
    # checkers: a jump forces
    cb = [["" for _ in range(8)] for _ in range(8)]
    cb[2][2] = "w"; cb[3][3] = "b"; cb[0][0] = "w"
    ck = Checkers(cb,"w")
    assert ck.jumps(), "a jump is available"
    assert ck.legal((2,2),(4,4)), "the jump is legal"
    assert not ck.legal((0,0),(1,1)), "a simple move is illegal while a jump exists"
    # go: suicide and capture
    gb = [["" for _ in range(GN)] for _ in range(GN)]
    gb[0][1] = "w"; gb[1][0] = "w"
    assert not Go(gb,"b").legal(0,0), "filling one's own last liberty is suicide"
    gb2 = [["" for _ in range(GN)] for _ in range(GN)]
    gb2[0][0] = "w"; gb2[0][1] = "b"                     # white on (0,0), last liberty (0,1)
    assert Go(gb2,"b").legal(0,1), "playing the last liberty captures, so it is legal"
    assert not Go(gb2,"b").legal(0,0), "occupied squares are never legal"
    print("board_games self-test: OK")

_selftest()
