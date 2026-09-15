"""The thing that says no.

An oracle names its own refusal kinds, the way a compiler emits E0308 rather than
a bare rejection. That is not GREN cheating: reasons.py §15 says a structured
code IS the class, because rediscovering clusters over rustc output when rustc
hands you the code would be inventing a worse version of an existing answer.

What GREN discovers is WHICH codes a game actually has and in what proportion --
and that, per DESIGN §2, is the game's identity: two games are similar exactly to
the degree they refuse the same things.

The codes are drawn from one cross-game vocabulary so that "target occupied by
your own piece" in chess and "point already occupied" in go can be recognised as
the same refusal. A code appearing in the vocabulary does not mean a game has it;
only probing shows that.
"""
import sys, os, time

# CyclicCortex supplies the game adapters GREN probes. GREN discovers what those
# games refuse; CyclicCortex currently hard-codes the answer, and closing that
# loop is the point of this module.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CC = os.path.join(_REPO, "CyclicCortex")
if _CC not in sys.path: sys.path.insert(0, _CC)

from gren.verdict import Verdict, Outcome, LEGAL

# One vocabulary, shared across every domain. Which of these a game HAS is
# discovered by probing; the vocabulary only makes them comparable.
CODES = [
    "OFF_BOARD", "NOT_YOURS", "EMPTY_SOURCE", "OCCUPIED_TARGET",
    "WRONG_PATTERN", "BLOCKED_PATH", "WRONG_DIRECTION",
    "NO_CAPTURE_TARGET", "FORCED_ALTERNATIVE", "SELF_CHECK",
    "SUICIDE", "REPETITION", "OUT_OF_RANGE",
    "CONSTRAINT_ROW", "CONSTRAINT_COL", "CONSTRAINT_BOX",
]

class GameOracle:
    """Wraps a CyclicCortex game adapter and explains its refusals.

    `declared` is what a description would tell you before a single probe --
    players, turn structure, category. Free evidence. Everything else costs
    probes, which is DESIGN §9.1's declared/measured split made real.
    """
    fidelity = "exact"

    def __init__(self, game, declared=None):
        self.game = game
        self.name = game.name
        self._declared = declared or {}
        self.probes = 0
        self.refusals = 0
        self.errors = 0

    def declared(self): return dict(self._declared)
    def reset(self, seed=0): return self.game.new(seed=seed)
    def actions(self, state): return self.game.legal_moves(state)

    def candidates(self, state, rng, k=12):
        return self.game.candidates(state, rng, k)

    def probe(self, state, action):
        """Returns a Verdict. An ILLEGAL verdict carries a CODE, which is the
        whole point: a bare rejection is one bit, a code from R kinds is
        log2(R+1)."""
        t = time.time()
        self.probes += 1
        try:
            legal = self.game.is_legal(state, action)
        except Exception as e:
            self.errors += 1
            return Verdict(Outcome.ERROR, reason=repr(e), latency=time.time()-t)
        if legal:
            return Verdict(Outcome.LEGAL, latency=time.time()-t)
        self.refusals += 1
        code, why = self.why(state, action)
        return Verdict(Outcome.ILLEGAL, reason=why, reason_code=code,
                       latency=time.time()-t)

    def why(self, state, action):
        raise NotImplementedError


class ChessOracle(GameOracle):
    def declared(self):
        return {"players": "2", "turn_structure": "alternating",
                "category": "board", "information": "perfect",
                "adversarial": "competitive", "goal_type": "reach-target"}

    def why(self, s, mv):
        from cortex.board_games import Chess, on
        (fx, fy), (tx, ty) = mv
        if not (on(fx, fy) and on(tx, ty)): return "OFF_BOARD", "square off the board"
        p = s.at(fx, fy)
        if not p: return "EMPTY_SOURCE", "no piece on the source square"
        if p[0] != s.turn: return "NOT_YOURS", "that piece is not yours to move"
        t = s.at(tx, ty)
        if t and t[0] == s.turn:
            return "OCCUPIED_TARGET", "target occupied by your own piece"
        if not s.pseudo_legal(mv[0], mv[1]):
            dx, dy = tx-fx, ty-fy
            if p[1] in "BRQ" and (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
                return "BLOCKED_PATH", "a piece stands between source and target"
            return "WRONG_PATTERN", f"not a legal move pattern for {p[1]}"
        return "SELF_CHECK", "that move leaves your own king in check"


class CheckersOracle(GameOracle):
    def declared(self):
        return {"players": "2", "turn_structure": "alternating",
                "category": "board", "information": "perfect",
                "adversarial": "competitive", "goal_type": "outlast"}

    def why(self, s, mv):
        from cortex.board_games import on
        (fx, fy), (tx, ty) = mv
        if not (on(fx, fy) and on(tx, ty)): return "OFF_BOARD", "square off the board"
        p = s.at(fx, fy)
        if not p: return "EMPTY_SOURCE", "no piece on the source square"
        if p.lower() != s.turn: return "NOT_YOURS", "that piece is not yours to move"
        if s.at(tx, ty): return "OCCUPIED_TARGET", "target square is occupied"
        dx, dy = tx-fx, ty-fy
        if abs(dx) != abs(dy) or abs(dx) not in (1, 2):
            return "WRONG_PATTERN", "not a diagonal step or jump"
        if (dx, dy) not in [(d[0]*abs(dx), d[1]*abs(dx)) for d in s.dirs(p)]:
            return "WRONG_DIRECTION", "an uncrowned man may not move that way"
        if abs(dx) == 2:
            m = s.at((fx+tx)//2, (fy+ty)//2)
            if not m or m.lower() == s.turn:
                return "NO_CAPTURE_TARGET", "no enemy piece on the jumped square"
        if s.jumps():
            return "FORCED_ALTERNATIVE", "a capture is available and must be taken"
        return "WRONG_PATTERN", "not a legal move"


class GoOracle(GameOracle):
    def declared(self):
        return {"players": "2", "turn_structure": "alternating",
                "category": "board", "information": "perfect",
                "adversarial": "competitive", "goal_type": "maximise-score"}

    def why(self, s, mv):
        from cortex.board_games import GN, PASS
        if mv == PASS: return None, None
        x, y = mv
        if not (0 <= x < GN and 0 <= y < GN): return "OFF_BOARD", "point off the board"
        if s.at(x, y): return "OCCUPIED_TARGET", "point already occupied"
        b2 = [r[:] for r in s.b]; b2[y][x] = s.turn
        opp = "w" if s.turn == "b" else "b"
        captured = False
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < GN and 0 <= ny < GN and b2[ny][nx] == opp:
                g, l = s.group(nx, ny, b2)
                if not l:
                    captured = True
                    for gx, gy in g: b2[gy][gx] = ""
        if not captured:
            _, libs = s.group(x, y, b2)
            if not libs: return "SUICIDE", "the group would have no liberties"
        return "REPETITION", "that recreates the previous position"


class SudokuOracle(GameOracle):
    def declared(self):
        return {"players": "1", "turn_structure": "free", "category": "puzzle",
                "information": "perfect", "adversarial": "solitaire",
                "goal_type": "produce-artifact"}

    def why(self, s, mv):
        x, y, v = mv
        if not (0 <= x < 9 and 0 <= y < 9): return "OFF_BOARD", "cell off the grid"
        if not (1 <= v <= 9): return "OUT_OF_RANGE", "value outside 1..9"
        if s.g[y][x]: return "OCCUPIED_TARGET", "that cell is already filled"
        if any(s.g[y][i] == v for i in range(9)):
            return "CONSTRAINT_ROW", "that value already appears in the row"
        if any(s.g[i][x] == v for i in range(9)):
            return "CONSTRAINT_COL", "that value already appears in the column"
        bx, by = (x//3)*3, (y//3)*3
        if any(s.g[by+j][bx+i] == v for j in range(3) for i in range(3)):
            return "CONSTRAINT_BOX", "that value already appears in the box"
        return "WRONG_PATTERN", "not a legal placement"


def build_all():
    from cortex.games import ALL
    return {"chess": ChessOracle(ALL["chess"]), "checkers": CheckersOracle(ALL["checkers"]),
            "go": GoOracle(ALL["go"]), "sudoku": SudokuOracle(ALL["sudoku"])}
