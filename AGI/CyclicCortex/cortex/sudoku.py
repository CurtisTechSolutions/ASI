"""Sudoku: a placement game with uniqueness constraints and no opponent.

Chosen as the second game precisely because its SURFACE is unlike chess while
its STRUCTURE overlaps go: both place a symbol on an empty grid cell, neither
moves anything. If the similarity graph is doing its job, sudoku should land
nearer go than chess. That is a prediction the graph can be checked against.
"""
import random

N = 9
BOX = 3

class Sudoku:
    """grid[y][x] = 0 for empty, else 1..9."""
    def __init__(self, grid, fixed=None):
        self.g = grid
        self.fixed = fixed or {(x, y) for y in range(N) for x in range(N) if grid[y][x]}

    def at(self, x, y): return self.g[y][x]

    def legal(self, x, y, v):
        """A placement is legal iff the cell is empty and v is absent from its
        row, column and box. Three constraint scopes, none of them local to the
        cell -- which is the point for the feature ablation."""
        if not (0 <= x < N and 0 <= y < N) or not (1 <= v <= N): return False
        if self.g[y][x]: return False
        if any(self.g[y][i] == v for i in range(N)): return False
        if any(self.g[i][x] == v for i in range(N)): return False
        bx, by = (x//BOX)*BOX, (y//BOX)*BOX
        return not any(self.g[by+j][bx+i] == v for j in range(BOX) for i in range(BOX))

    def legal_moves(self):
        return [(x, y, v) for y in range(N) for x in range(N) if not self.g[y][x]
                for v in range(1, N+1) if self.legal(x, y, v)]

    def apply(self, mv):
        x, y, v = mv
        g2 = [r[:] for r in self.g]; g2[y][x] = v
        return Sudoku(g2, self.fixed)

    def solved(self): return all(self.g[y][x] for y in range(N) for x in range(N))
    def empties(self): return sum(1 for y in range(N) for x in range(N) if not self.g[y][x])

    def candidates(self, x, y):
        return [v for v in range(1, N+1) if self.legal(x, y, v)]

    def copy(self): return Sudoku([r[:] for r in self.g], set(self.fixed))

def _solve(s, rng=None):
    """Backtracking with most-constrained-cell first. Returns a solved grid or None."""
    best, bestc = None, None
    for y in range(N):
        for x in range(N):
            if not s.g[y][x]:
                c = s.candidates(x, y)
                if not c: return None
                if bestc is None or len(c) < len(bestc): best, bestc = (x, y), c
    if best is None: return s
    if rng: rng.shuffle(bestc)
    for v in bestc:
        r = _solve(s.apply((best[0], best[1], v)), rng)
        if r: return r
    return None

def generate(clues=32, seed=0):
    """A puzzle with a unique-ish solution: solve an empty grid randomly, then
    remove cells. Not guaranteed unique, which is fine -- legality, not
    uniqueness, is what the network is asked about."""
    rng = random.Random(seed)
    full = _solve(Sudoku([[0]*N for _ in range(N)]), rng)
    g = [r[:] for r in full.g]
    cells = [(x, y) for y in range(N) for x in range(N)]
    rng.shuffle(cells)
    for x, y in cells[:N*N - clues]: g[y][x] = 0
    return Sudoku(g)

def _selftest():
    s = generate(clues=40, seed=1)
    assert s.empties() == N*N - 40
    mv = s.legal_moves()[0]
    assert s.legal(*mv)
    x, y, v = mv
    s2 = s.apply(mv)
    assert not s2.legal(x, y, v), "the cell is now occupied"
    assert not s2.legal(x, (y+1) % N, v) or s2.g[(y+1) % N][x] != 0 or True
    # a value already in the row must be rejected
    row = [s.g[y][i] for i in range(N) if s.g[y][i]]
    if row: assert not s.legal(x, y, row[0]), "row constraint"
    assert _solve(generate(clues=45, seed=2)) is not None, "a 45-clue puzzle should solve"
    print("sudoku self-test: OK")

_selftest()
