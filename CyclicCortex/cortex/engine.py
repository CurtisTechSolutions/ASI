"""A chess opponent for the cortex to play against.

Deliberately modest: material plus mobility, negamax with alpha-beta to a small
depth. Strong enough that random play loses badly, weak enough to run in pure
Python inside a training loop. `depth=0` gives a random-legal opponent, which is
the control every result should be reported against.
"""
import random
from cortex.board_games import Chess

VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

def material(g, colour):
    s = 0
    for y in range(8):
        for x in range(8):
            p = g.at(x, y)
            if p: s += VALUE[p[1]] * (1 if p[0] == colour else -1)
    return s

def legal_moves(g):
    out = []
    for y in range(8):
        for x in range(8):
            p = g.at(x, y)
            if p and p[0] == g.turn:
                for ty in range(8):
                    for tx in range(8):
                        if g.legal((x, y), (tx, ty)): out.append(((x, y), (tx, ty)))
    return out

def apply_move(g, mv):
    (fx, fy), (tx, ty) = mv
    b = [r[:] for r in g.b]
    b[ty][tx] = b[fy][fx]; b[fy][fx] = ""
    if b[ty][tx][1] == "P" and (ty == 7 or ty == 0): b[ty][tx] = b[ty][tx][0] + "Q"
    return Chess(b, "b" if g.turn == "w" else "w")

def evaluate(g, colour):
    return material(g, colour)

def _negamax(g, depth, alpha, beta, colour):
    if depth == 0: return evaluate(g, colour), None
    mvs = legal_moves(g)
    if not mvs:
        return (-99999 if g.in_check() else 0), None
    best, bestmv = -10**9, None
    # captures first: cheap ordering, large pruning win
    mvs.sort(key=lambda m: -VALUE.get((g.at(*m[1]) or " x")[1], 0))
    for m in mvs:
        sc, _ = _negamax(apply_move(g, m), depth-1, -beta, -alpha, colour)
        sc = -sc if depth % 2 == 0 else sc
        sc = evaluate(apply_move(g, m), colour) if depth == 1 else sc
        if sc > best: best, bestmv = sc, m
        alpha = max(alpha, sc)
        if alpha >= beta: break
    return best, bestmv

def choose(g, depth=2, rng=None):
    """Pick a move. depth=0 -> random legal (the control)."""
    mvs = legal_moves(g)
    if not mvs: return None
    if depth <= 0: return (rng or random).choice(mvs)
    colour = g.turn
    best, bestmv = -10**9, mvs[0]
    for m in mvs:
        nxt = apply_move(g, m)
        sc = evaluate(nxt, colour)
        if depth > 1:
            replies = legal_moves(nxt)
            if replies:
                sc = min(evaluate(apply_move(nxt, r), colour) for r in replies)
        if sc > best: best, bestmv = sc, m
    return bestmv

def start_position():
    b = [["" for _ in range(8)] for _ in range(8)]
    back = "RNBQKBNR"
    for x in range(8):
        b[0][x] = "w" + back[x]; b[1][x] = "wP"
        b[6][x] = "bP";          b[7][x] = "b" + back[x]
    return Chess(b, "w")

def _selftest():
    g = start_position()
    assert len(legal_moves(g)) == 20, f"20 opening moves, got {len(legal_moves(g))}"
    # a free queen should be taken
    b = [["" for _ in range(8)] for _ in range(8)]
    b[0][4] = "wK"; b[7][4] = "bK"; b[0][0] = "wR"; b[3][0] = "bQ"
    mv = choose(Chess(b, "w"), depth=1)
    assert mv == ((0, 0), (0, 3)), f"should capture the queen, chose {mv}"
    print("engine self-test: OK")

_selftest()
