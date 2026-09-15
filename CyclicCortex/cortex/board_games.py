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
    """board[y][x] = '' | 'w' | 'b' | 'W' | 'B' (upper = king). w moves +y.

    `chain` is the square a multi-jump must continue from. While it is set, the
    turn has NOT passed and only further jumps by that piece are legal -- which
    is a real rule and changes the legal-move set, not just the bookkeeping."""
    def __init__(self, board, turn="w", chain=None):
        self.b, self.turn, self.chain = board, turn, chain
    def at(self,x,y): return self.b[y][x]
    def dirs(self, p):
        if p.isupper(): return [(1,1),(1,-1),(-1,1),(-1,-1)]
        return [(1,1),(-1,1)] if p == "w" else [(1,-1),(-1,-1)]
    def simple_moves(self):
        if self.chain: return []          # mid-chain: only jumps are available
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
        cells = [self.chain] if self.chain else [(x,y) for y in range(8) for x in range(8)]
        for (x, y) in cells:
            p = self.at(x,y)
            if True:
                if p and p.lower() == self.turn:
                    for dx,dy in self.dirs(p):
                        mx,my,nx,ny = x+dx, y+dy, x+2*dx, y+2*dy
                        if on(nx,ny) and not self.at(nx,ny):
                            m = self.at(mx,my)
                            if m and m.lower() != self.turn: out.append(((x,y),(nx,ny)))
        return out

    def moves(self):
        """The legal move set: jumps if any exist (forced capture), else simples."""
        return self.jumps() or self.simple_moves()

    def apply(self, mv):
        """Applies promotion and chain capture. If the jumping piece can jump
        again it keeps the turn, with `chain` pinning it to that square."""
        (fx,fy),(tx,ty) = mv
        b = [r[:] for r in self.b]
        p = b[fy][fx]
        b[ty][tx] = p; b[fy][fx] = ""
        jumped = abs(tx-fx) == 2
        if jumped: b[(fy+ty)//2][(fx+tx)//2] = ""
        crowned = False
        if p == "w" and ty == 7: b[ty][tx] = "W"; crowned = True
        if p == "b" and ty == 0: b[ty][tx] = "B"; crowned = True
        if jumped and not crowned:
            # crowning ENDS the turn even mid-chain; otherwise continue if able
            nxt = Checkers(b, self.turn, (tx,ty))
            if nxt.jumps(): return nxt
        return Checkers(b, "b" if self.turn == "w" else "w", None)

    def winner(self):
        """The side to move with no legal moves loses -- no stalemate in checkers."""
        if self.moves(): return None
        return "b" if self.turn == "w" else "w"

    def count(self, col):
        return sum(1 for y in range(8) for x in range(8)
                   if self.at(x,y) and self.at(x,y).lower() == col)
    def legal(self, fr, to):
        """FORCED CAPTURE: if any jump exists, only jumps are legal. This move's
        legality depends on moves it is not part of."""
        return (fr,to) in self.moves()

def checkers_start():
    """Standard opening: twelve men each on the dark squares of the first and
    last three ranks."""
    b = [["" for _ in range(8)] for _ in range(8)]
    for y in range(3):
        for x in range(8):
            if (x+y) % 2 == 0: b[y][x] = "w"
    for y in range(5, 8):
        for x in range(8):
            if (x+y) % 2 == 0: b[y][x] = "b"
    return Checkers(b, "w")

def random_checkers(rng):
    b = [["" for _ in range(8)] for _ in range(8)]
    for x,y in rng.sample([(x,y) for x in range(8) for y in range(8) if (x+y)%2==0],
                          rng.randint(6, 16)):
        b[y][x] = rng.choice(["w","b","W","B"])
    return Checkers(b, rng.choice("wb"))

# ------------------------------------------------------------------ go
GN = 9
PASS = "pass"

class Go:
    """board[y][x] = '' | 'b' | 'w'. `prev` is the position before the last move (ko).
    `passes` counts CONSECUTIVE passes; two in a row end the game."""
    def __init__(self, board, turn="b", prev=None, passes=0):
        self.b, self.turn, self.prev, self.passes = board, turn, prev, passes
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

    def moves(self):
        """Every legal placement, plus PASS -- which is always legal and is the
        only way the game can end."""
        return [(x,y) for y in range(GN) for x in range(GN) if self.legal(x,y)] + [PASS]

    def apply(self, mv):
        opp = "w" if self.turn == "b" else "b"
        if mv == PASS:
            return Go([r[:] for r in self.b], opp, self.prev, self.passes + 1)
        x, y = mv
        b2 = [r[:] for r in self.b]; b2[y][x] = self.turn
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0<=nx<GN and 0<=ny<GN and b2[ny][nx] == opp:
                g,l = self.group(nx,ny,b2)
                if not l:
                    for gx,gy in g: b2[gy][gx] = ""
        return Go(b2, opp, [r[:] for r in self.b], 0)

    def over(self): return self.passes >= 2

    def score(self):
        """Area scoring: stones on the board plus empty regions reaching only one
        colour. Returns (black, white)."""
        seen = set(); sc = {"b": 0, "w": 0}
        for y in range(GN):
            for x in range(GN):
                v = self.b[y][x]
                if v: sc[v] += 1
        for y in range(GN):
            for x in range(GN):
                if self.b[y][x] or (x,y) in seen: continue
                region, border, stack = set(), set(), [(x,y)]
                while stack:
                    cx,cy = stack.pop()
                    if (cx,cy) in region: continue
                    region.add((cx,cy)); seen.add((cx,cy))
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny = cx+dx, cy+dy
                        if not (0<=nx<GN and 0<=ny<GN): continue
                        v = self.b[ny][nx]
                        if v: border.add(v)
                        elif (nx,ny) not in region: stack.append((nx,ny))
                if len(border) == 1: sc[border.pop()] += len(region)
        return sc["b"], sc["w"]

    def winner(self):
        bs, ws = self.score()
        return "b" if bs > ws else ("w" if ws > bs else None)

def go_start(): return Go([["" for _ in range(GN)] for _ in range(GN)], "b")

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
