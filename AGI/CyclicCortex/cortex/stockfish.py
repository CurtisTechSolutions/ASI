"""Stockfish as opponent AND teacher.

Two distinct uses, and the second is the interesting one:

  OPPONENT  a real chess engine to lose to, instead of a toy that takes free
            pieces at one ply.
  TEACHER   Stockfish scores EVERY candidate move, not just the game's outcome.
            Outcome credit gives one number per game, spread over ~100 moves by a
            discount; MultiPV gives a centipawn target for each legal move in
            every position. That is dense supervision where self-play is sparse.

The board in cortex/board_games.py tracks no castling rights and no en-passant
square, so FEN is emitted with "-" for both. Stockfish then plays a legal subset
this representation can hold, rather than proposing a castle the board cannot
apply. Promotion is emitted as e.g. e7e8q and `engine.apply_move` already
promotes to a queen, so those agree.

Missing Stockfish is not an error: `available()` reports it and callers fall back
to the built-in engine.
"""
import os, shutil, subprocess, threading

PATHS = ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]
FILES = "abcdefgh"

def find():
    p = os.environ.get("STOCKFISH_PATH")
    if p and os.path.exists(p): return p
    w = shutil.which("stockfish")
    if w: return w
    for c in PATHS:
        if os.path.exists(c) and os.access(c, os.X_OK): return c
    return None

def available(): return find() is not None

# ------------------------------------------------------------------ notation
def playable(s):
    """Stockfish rejects positions that cannot arise in a real game. Random
    boards routinely produce them -- a missing king, or the side NOT to move
    already in check -- and sending one kills the process with a broken pipe."""
    if s.king_sq("w") is None or s.king_sq("b") is None: return False
    other = "b" if s.turn == "w" else "w"
    ks = s.king_sq(other)
    return not s.attacked(ks, s.turn)

def to_fen(s):
    """cortex Chess -> FEN. Castling and en passant are '-' by construction."""
    rows = []
    for y in range(7, -1, -1):
        row, gap = "", 0
        for x in range(8):
            p = s.at(x, y)
            if not p: gap += 1; continue
            if gap: row += str(gap); gap = 0
            row += p[1].upper() if p[0] == "w" else p[1].lower()
        if gap: row += str(gap)
        rows.append(row or "8")
    return f"{'/'.join(rows)} {s.turn} - - 0 1"

def to_uci(mv):
    (fx, fy), (tx, ty) = mv
    return f"{FILES[fx]}{fy+1}{FILES[tx]}{ty+1}"

def from_uci(u):
    if len(u) < 4: return None
    return ((FILES.index(u[0]), int(u[1])-1), (FILES.index(u[2]), int(u[3])-1))

# -------------------------------------------------------------------- engine
class Stockfish:
    """A persistent UCI process. Spawning one per call costs ~50ms of startup
    and NNUE load, which dwarfs a depth-8 search."""

    def __init__(self, path=None, depth=8, threads=1, hash_mb=32):
        self.path = path or find()
        if not self.path: raise RuntimeError("stockfish not found")
        self.depth = depth
        self._lock = threading.Lock()
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self._send("uci"); self._until("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._ready()

    def _restart(self):
        try: self.p.kill()
        except Exception: pass
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self._send("uci"); self._until("uciok"); self._send("isready"); self._until("readyok")

    def _send(self, line):
        try:
            self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._restart()
            self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def _until(self, token, collect=False):
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line: raise RuntimeError("stockfish exited")
            if collect: out.append(line.rstrip())
            if line.startswith(token): return out

    def _ready(self): self._send("isready"); self._until("readyok")

    def set_skill(self, level):
        """0-20. Lets the teacher be throttled to a beatable opponent while still
        scoring moves at full strength."""
        self._send(f"setoption name Skill Level value {max(0, min(20, level))}")
        self._ready()

    def best(self, state, depth=None, movetime=None):
        with self._lock:
            self._send(f"position fen {to_fen(state)}")
            self._send(f"go movetime {movetime}" if movetime
                       else f"go depth {depth or self.depth}")
            for line in self._until("bestmove", collect=True):
                if line.startswith("bestmove"):
                    tok = line.split()[1]
                    return None if tok in ("(none)", "0000") else from_uci(tok)
        return None

    def evaluate(self, state, depth=None):
        """Centipawns from the SIDE TO MOVE's point of view. A mate is clamped to
        +/-10000 so it stays comparable with material scores."""
        if not playable(state): return 0
        if not playable(state): return None
        with self._lock:
            self._send(f"position fen {to_fen(state)}")
            self._send(f"go depth {depth or self.depth}")
            score = 0
            for line in self._until("bestmove", collect=True):
                if " score cp " in line:
                    score = int(line.split(" score cp ")[1].split()[0])
                elif " score mate " in line:
                    m = int(line.split(" score mate ")[1].split()[0])
                    score = 10000 if m > 0 else -10000
            return score

    def score_moves(self, state, depth=None, multipv=24):
        """{move: centipawns} for the top `multipv` moves, from the MOVER's point
        of view, in ONE search. This is the teaching signal."""
        if not playable(state): return {}
        with self._lock:
            self._send(f"setoption name MultiPV value {multipv}")
            self._ready()
            self._send(f"position fen {to_fen(state)}")
            self._send(f"go depth {depth or self.depth}")
            best_by_pv = {}
            for line in self._until("bestmove", collect=True):
                if " multipv " not in line or " pv " not in line: continue
                pv = int(line.split(" multipv ")[1].split()[0])
                if " score cp " in line:
                    sc = int(line.split(" score cp ")[1].split()[0])
                elif " score mate " in line:
                    m = int(line.split(" score mate ")[1].split()[0])
                    sc = 10000 if m > 0 else -10000
                else: continue
                mv = from_uci(line.split(" pv ")[1].split()[0])
                if mv: best_by_pv[pv] = (mv, sc)     # later depths overwrite earlier
            self._send("setoption name MultiPV value 1"); self._ready()
            return {mv: sc for mv, sc in best_by_pv.values()}

    def close(self):
        try:
            self._send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

def _selftest():
    if not available(): print("stockfish self-test: SKIPPED (not installed)"); return
    from cortex import engine
    from cortex.board_games import Chess
    sf = Stockfish(depth=8)
    try:
        st = engine.start_position()
        assert to_fen(st).startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -")
        assert to_uci(((4,1),(4,3))) == "e2e4" and from_uci("e2e4") == ((4,1),(4,3))
        assert -100 < sf.evaluate(st) < 200, "the opening is roughly level"
        b = [["" for _ in range(8)] for _ in range(8)]
        b[0][4] = "wK"; b[7][4] = "bK"; b[0][3] = "wQ"
        assert sf.evaluate(Chess(b, "w")) > 500, "a free queen is worth a lot"
        sc = sf.score_moves(st, multipv=8)
        assert len(sc) >= 5, f"MultiPV should score several moves, got {len(sc)}"
        assert all(st.legal(*m) for m in sc), "every scored move must be legal here"
        print(f"stockfish self-test: OK ({len(sc)} opening moves scored, "
              f"best {max(sc.values())}cp)")
    finally:
        sf.close()

if __name__ == "__main__": _selftest()
