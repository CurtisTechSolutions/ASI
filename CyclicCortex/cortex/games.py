"""Game adapters. Each exposes the same interface so the cortex never knows
which game it is holding, and a `generalise` that returns MECHANIC -> features:
the common denominators, keyed by the same mechanic names that decide similarity.
"""
import random
from cortex.board_games import (Chess, Checkers, Go, on, GN, PASS,
                                checkers_start, go_start)
from cortex import engine, sudoku as sud

KINDS = "PNBRQK"

class ChessGame:
    name = "chess"
    mechanics = frozenset({"ALTERNATE_TURNS","GRID_BOARD","PERFECT_INFO","ZERO_SUM",
        "TWO_PLAYER","PIECE_OWNERSHIP","DISPLACEMENT_CAPTURE","PIECE_TYPES",
        "SLIDING_MOVE","STEP_MOVE","ROYAL_PIECE","CHECK_CONSTRAINT","BLOCKED_BY_OCCUPANT"})
    spec = {"GRID_MOVE":4, "OCCUPANCY":3, "PIECE_TYPES":6, "SLIDING_MOVE":3,
            "ROYAL_PIECE":2, "SIDE":1}

    def new(self, seed=0): return engine.start_position()
    def legal_moves(self, s): return engine.legal_moves(s)
    def is_legal(self, s, mv): return s.legal(*mv)
    def apply(self, s, mv): return engine.apply_move(s, mv)
    def terminal(self, s): return not engine.legal_moves(s)

    def candidates(self, s, rng, k=24):
        """Legal moves plus plausible illegal ones: what the network must rank.

        Enumerating EVERY legal move costs ~65k predicate evaluations per
        position, because each legality test scans the board for check. Sampling
        a few movers and enumerating only their targets costs a fraction of that
        and is unbiased for training."""
        own = [(x,y) for y in range(8) for x in range(8)
               if s.at(x,y) and s.at(x,y)[0] == s.turn]
        if not own: return []
        rng.shuffle(own)
        good, bad = [], []
        for fr in own[:4]:
            for ty in range(8):
                for tx in range(8):
                    if (tx,ty) == fr: continue
                    (good if s.legal(fr,(tx,ty)) else bad).append((fr,(tx,ty)))
            if len(good) >= k: break
        rng.shuffle(good); rng.shuffle(bad)
        return good[:k] + bad[:k]

    def random_state(self, rng):
        """Training positions: random midgames give far more variety than the
        opening, and legality is what is being learned, not openings."""
        from cortex.board_games import random_chess
        return random_chess(rng)

    def generalise(self, s, mv):
        (fx,fy),(tx,ty) = mv; dx,dy = tx-fx, ty-fy
        p, t = s.at(fx,fy), s.at(tx,ty)
        pt = [0.0]*6
        if p and p[1] in KINDS: pt[KINDS.index(p[1])] = 1.0
        sx = (dx>0)-(dx<0); sy = (dy>0)-(dy<0)
        n = max(abs(dx), abs(dy)); blockers = 0
        if n and (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
            x,y = fx+sx, fy+sy
            while (x,y) != (tx,ty) and on(x,y):
                if s.at(x,y): blockers += 1
                x += sx; y += sy
        b2 = [r[:] for r in s.b]
        if on(tx,ty) and p: b2[ty][tx] = b2[fy][fx]; b2[fy][fx] = ""
        nxt = Chess(b2, s.turn); ks = nxt.king_sq(s.turn)
        exposed = 1.0 if (ks and nxt.attacked(ks, "b" if s.turn=="w" else "w")) else 0.0
        return {
            "GRID_MOVE":  [dx/7.0, dy/7.0, abs(dx)/7.0, abs(dy)/7.0],
            "OCCUPANCY":  [0.0 if t else 1.0,
                           1.0 if (t and t[0]==s.turn) else 0.0,
                           1.0 if (t and t[0]!=s.turn) else 0.0],
            "PIECE_TYPES": pt,
            "SLIDING_MOVE":[1.0 if blockers==0 else 0.0, n/7.0, min(blockers,3)/3.0],
            "ROYAL_PIECE": [exposed, 1.0 if s.in_check() else 0.0],
            "SIDE":        [1.0 if s.turn=="w" else 0.0],
        }

    def value(self, s, me):
        """Position value in [-1,1] when a game did not finish. Without it a
        160-ply chess game that reaches the cap is scored a draw, every move gets
        zero credit, and the grade head learns NOTHING -- which is exactly what
        40 self-play games of 40 draws produced."""
        return max(-1.0, min(1.0, engine.material(s, me) / 4000.0))

    def reward(self, s, mv):
        """Grade signal: material swing from the mover's point of view."""
        before = engine.material(s, s.turn)
        after  = engine.material(self.apply(s, mv), s.turn)
        return max(-1.0, min(1.0, (after-before)/900.0))


class SudokuGame:
    name = "sudoku"
    mechanics = frozenset({"GRID_BOARD","PERFECT_INFO","SOLITAIRE","PLACEMENT_MOVE",
        "NO_MOVEMENT","DETERMINISTIC","CONSTRAINT_UNIQUE","FILL_TO_COMPLETE",
        "FREE_ORDER","SYMBOL_SET"})
    spec = {"GRID_PLACE":3, "SYMBOL_SET":9, "CONSTRAINT_UNIQUE":3, "FILL_TO_COMPLETE":1}

    def new(self, seed=0): return sud.generate(clues=34, seed=seed)
    def legal_moves(self, s): return s.legal_moves()
    def is_legal(self, s, mv): return s.legal(*mv)
    def apply(self, s, mv): return s.apply(mv)
    def terminal(self, s): return s.solved() or not s.legal_moves()

    def candidates(self, s, rng, k=24):
        good = s.legal_moves(); rng.shuffle(good)
        bad = []
        for _ in range(k*3):
            x,y,v = rng.randrange(9), rng.randrange(9), rng.randrange(1,10)
            if not s.legal(x,y,v): bad.append((x,y,v))
            if len(bad) >= k: break
        return good[:k] + bad[:k]

    def generalise(self, s, mv):
        x, y, v = mv
        sym = [0.0]*9
        if 1 <= v <= 9: sym[v-1] = 1.0
        in_row = 1.0 if any(s.g[y][i]==v for i in range(9)) else 0.0
        in_col = 1.0 if any(s.g[i][x]==v for i in range(9)) else 0.0
        bx,by = (x//3)*3, (y//3)*3
        in_box = 1.0 if any(s.g[by+j][bx+i]==v for j in range(3) for i in range(3)) else 0.0
        return {
            "GRID_PLACE":       [x/8.0, y/8.0, 0.0 if s.g[y][x] else 1.0],
            "SYMBOL_SET":       sym,
            "CONSTRAINT_UNIQUE":[in_row, in_col, in_box],
            "FILL_TO_COMPLETE": [1.0 - s.empties()/81.0],
        }

    def reward(self, s, mv):
        """Grade: prefer the most constrained cell -- fewer alternatives means a
        safer commitment. Standard sudoku strategy, supplied as the outcome signal."""
        x, y, _ = mv
        c = len(s.candidates(x, y))
        return 1.0 if c <= 1 else max(-1.0, 1.0 - (c-1)/4.0)


class CheckersGame:
    name = "checkers"
    mechanics = frozenset({"ALTERNATE_TURNS","GRID_BOARD","PERFECT_INFO","ZERO_SUM",
        "TWO_PLAYER","PIECE_OWNERSHIP","JUMP_CAPTURE","STEP_MOVE","DIAGONAL_ONLY",
        "FORCED_CAPTURE","BLOCKED_BY_OCCUPANT","PROMOTE_ON_RANK"})
    spec = {"GRID_MOVE":4, "OCCUPANCY":3, "JUMP_CAPTURE":3, "FORCED_CAPTURE":1,
            "PROMOTE_ON_RANK":2, "SIDE":1}
    def new(self, seed=0): return checkers_start()
    def random_state(self, rng):
        from cortex.board_games import random_checkers
        return random_checkers(rng)
    def legal_moves(self, s): return s.moves()
    def is_legal(self, s, mv): return s.legal(*mv)
    def apply(self, s, mv): return s.apply(mv)        # promotion + chain capture
    def terminal(self, s): return s.winner() is not None
    def winner(self, s): return s.winner()
    def candidates(self, s, rng, k=24):
        good=self.legal_moves(s); rng.shuffle(good)
        bad=[]
        allsq=[(x,y) for y in range(8) for x in range(8)]
        for _ in range(k*3):
            mv=(rng.choice(allsq), rng.choice(allsq))
            if not s.legal(*mv): bad.append(mv)
            if len(bad)>=k: break
        return good[:k]+bad[:k]
    def generalise(self, s, mv):
        (fx,fy),(tx,ty)=mv; dx,dy=tx-fx,ty-fy; t=s.at(tx,ty) if on(tx,ty) else ""
        p = s.at(fx,fy) if on(fx,fy) else ""
        reaches_back = 1.0 if ((p == "w" and ty == 7) or (p == "b" and ty == 0)) else 0.0
        # The MIDPOINT. A jump is legal only if the square BETWEEN from and to
        # holds an enemy piece -- exactly the path predicate the chess ablation
        # found mattered most. Without it the network cannot tell a legal jump
        # from an illegal jump-shaped move, and both score maximum grade.
        is_jump = 1.0 if (abs(dx) == 2 and abs(dy) == 2) else 0.0
        mid_enemy = mid_own = 0.0
        if is_jump:
            m = s.at((fx+tx)//2, (fy+ty)//2)
            if m:
                if m.lower() != s.turn: mid_enemy = 1.0
                else: mid_own = 1.0
        return {"JUMP_CAPTURE":[is_jump, mid_enemy, mid_own],
                "GRID_MOVE":[dx/7.0,dy/7.0,abs(dx)/7.0,abs(dy)/7.0],
                "OCCUPANCY":[0.0 if t else 1.0, 1.0 if (t and t.lower()==s.turn) else 0.0,
                             1.0 if (t and t.lower()!=s.turn) else 0.0],
                "FORCED_CAPTURE":[1.0 if s.jumps() else 0.0],
                "PROMOTE_ON_RANK":[1.0 if p.isupper() else 0.0, reaches_back],
                "SIDE":[1.0 if s.turn=="w" else 0.0]}
    def value(self, s, me):
        opp = "b" if me == "w" else "w"
        n, m = s.count(me), s.count(opp)
        return 0.0 if n + m == 0 else max(-1.0, min(1.0, (n - m) / 12.0))

    def reward(self, s, mv):
        """Only a REAL capture is worth anything. Rewarding jump SHAPE taught the
        grade head that any two-square move is good, which is precisely what the
        ranking then selected for."""
        (fx,fy),(tx,ty) = mv
        if abs(tx-fx) != 2: return 0.0
        m = s.at((fx+tx)//2, (fy+ty)//2)
        return 1.0 if (m and m.lower() != s.turn) else -1.0


class GoGame:
    name = "go"
    mechanics = frozenset({"ALTERNATE_TURNS","GRID_BOARD","PERFECT_INFO","ZERO_SUM",
        "TWO_PLAYER","PIECE_OWNERSHIP","PLACEMENT_MOVE","GROUP_LIBERTY",
        "SURROUND_CAPTURE","KO_REPETITION","NO_MOVEMENT","PASS_ALLOWED"})
    spec = {"GRID_PLACE":3, "GROUP_LIBERTY":3, "INFLUENCE":5, "CONNECTION":3,
            "PASS_ALLOWED":1, "SIDE":1}
    def new(self, seed=0): return go_start()
    def random_state(self, rng):
        from cortex.board_games import random_go
        return random_go(rng)
    def legal_moves(self, s): return s.moves()          # placements plus PASS
    def is_legal(self, s, mv): return mv == PASS or s.legal(*mv)
    def apply(self, s, mv): return s.apply(mv)          # capture, ko, pass counting
    def terminal(self, s): return s.over()
    def winner(self, s): return s.winner()
    def candidates(self, s, rng, k=24):
        pts=[(x,y) for y in range(GN) for x in range(GN)]
        good=[p for p in pts if s.legal(*p)]; bad=[p for p in pts if not s.legal(*p)]
        rng.shuffle(good); rng.shuffle(bad)
        return good[:k]+bad[:k]+[PASS]
    def generalise(self, s, mv):
        if mv == PASS:
            return {"GRID_PLACE":[0.0,0.0,0.0], "GROUP_LIBERTY":[0.0,0.0,0.0],
                    "INFLUENCE":[0.0]*5, "CONNECTION":[0.0,0.0,0.0],
                    "PASS_ALLOWED":[1.0], "SIDE":[1.0 if s.turn=="b" else 0.0]}
        x,y=mv
        if s.at(x,y): libs,caps,own = 0.0,0.0,0.0
        else:
            b2=[r[:] for r in s.b]; b2[y][x]=s.turn
            opp="w" if s.turn=="b" else "b"; caps=0
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=x+dx,y+dy
                if 0<=nx<GN and 0<=ny<GN and b2[ny][nx]==opp:
                    g,l=s.group(nx,ny,b2)
                    if not l: caps+=len(g)
            grp,lb=s.group(x,y,b2)
            libs,own=min(len(lb),4)/4.0, min(len(grp),6)/6.0; caps=min(caps,4)/4.0
        # INFLUENCE and CONNECTION: the board CONTEXT of the placement. Without
        # them the vocabulary can express legality and nothing about territory,
        # which is why the go region played 125 legal plies and lost 0-81. The
        # feature vocabulary sets the ceiling -- the same lesson as the checkers
        # midpoint, one level up.
        opp = "w" if s.turn == "b" else "b"
        near_own = near_opp = 0.0; d_own = d_opp = 1.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x+dx, y+dy
                if not (0 <= nx < GN and 0 <= ny < GN) or (dx == 0 and dy == 0): continue
                v = s.at(nx, ny)
                if not v: continue
                d = max(abs(dx), abs(dy)); wgt = 1.0/(d*d)
                if v == s.turn: near_own += wgt; d_own = min(d_own, d/3.0)
                else:           near_opp += wgt; d_opp = min(d_opp, d/3.0)
        edge = min(x, y, GN-1-x, GN-1-y)/4.0
        # does it join two distinct friendly groups?
        roots = set()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0<=nx<GN and 0<=ny<GN and s.at(nx,ny) == s.turn:
                grp,_ = s.group(nx,ny); roots.add(min(grp))
        adj_opp = sum(1 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))
                      if 0<=x+dx<GN and 0<=y+dy<GN and s.at(x+dx,y+dy) == opp)
        return {"GRID_PLACE":[x/(GN-1), y/(GN-1), 0.0 if s.at(x,y) else 1.0],
                "GROUP_LIBERTY":[libs, caps, own],
                "INFLUENCE":[min(near_own,4.0)/4.0, min(near_opp,4.0)/4.0,
                             d_own, d_opp, edge],
                "CONNECTION":[min(len(roots),3)/3.0, adj_opp/4.0,
                              1.0 if len(roots) >= 2 else 0.0],
                "PASS_ALLOWED":[0.0],
                "SIDE":[1.0 if s.turn=="b" else 0.0]}
    def value(self, s, me):
        bs, ws = s.score()
        d = (bs - ws) if me == "b" else (ws - bs)
        return max(-1.0, min(1.0, d / (GN*GN/2.0)))

    def reward(self, s, mv):
        if mv == PASS: return -0.5           # passing is rarely the best move
        return self.generalise(s, mv)["GROUP_LIBERTY"][1]


ALL = {g.name: g for g in (ChessGame(), SudokuGame(), CheckersGame(), GoGame())}
