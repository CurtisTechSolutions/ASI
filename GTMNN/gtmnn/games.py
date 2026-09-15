"""The universal game interface: one micro geometry for every game.

Chess has ~4000 candidate moves, go has 361, checkers ~50. A micro has FIXED
geometry, so the action-space size must never touch it. The resolution
(GREN/DESIGN.md 22.4) is to score ONE candidate at a time: the micro is never
asked "which of the 4000 moves", it is asked "this one: legal?". Games then
differ only in how many times the same network is evaluated.

    micro input   F_state hashed features of the position
                + F_cand  hashed features of the ONE candidate
                + F_mod   the game modifier -- the mechanic set, hashed

    micro action  LEGAL | ILLEGAL | ABSTAIN

The modifier is why this transfers. Hashing a SET preserves overlap, so for two
games A and B with few collisions

    <v(A), v(B)>  ~  |A n B| / sqrt(|A||B|)     = the Ochiai similarity

and two similar games land at nearby points in modifier space automatically. A
micro that learned "in games shaped like this, a piece cannot move through an
occupant" generalises to every game with a similar modifier without being told
to, because its inputs barely moved. Transfer is not a mechanism that had to be
designed; it is a consequence of hashing a set instead of assigning an id.

The mechanic sets come from GREN's handoff -- the same JSON CyclicCortex reads --
so all three projects agree on what a game IS by construction rather than by
three hand-written copies of the same frozenset.
"""
import json, math, os, random
from array import array

from gtmnn.features import fnv1a

ILLEGAL, LEGAL = 0, 1
VERDICTS = ["illegal", "legal"]

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_HERE)
HANDOFF = os.path.join(_REPO, "CyclicCortex", "data", "gren_packages.json")


def modifier(mechanics, dim=128, seed=0, out=None):
    """v[h(m) % dim] += +-1 for each mechanic m, then L2-normalise."""
    v = out if out is not None else array("d", [0.0]) * dim
    for i in range(dim): v[i] = 0.0
    for m in mechanics:
        hv = fnv1a(str(m).encode("utf-8"), seed)
        v[hv % dim] += 1.0 if (hv >> 32) & 1 else -1.0
    norm = sum(x * x for x in v) ** 0.5
    if norm > 0.0:
        for i in range(dim): v[i] /= norm
    return v


def ochiai(a, b):
    """|A n B| / sqrt(|A| |B|) -- what modifier cosine is meant to approximate."""
    a, b = set(a), set(b)
    if not a or not b: return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def cosine(u, v):
    du = sum(x * x for x in u) ** 0.5
    dv = sum(x * x for x in v) ** 0.5
    if du < 1e-12 or dv < 1e-12: return 0.0
    return sum(x * y for x, y in zip(u, v)) / (du * dv)


def load_mechanics(path=HANDOFF):
    """Mechanic sets straight out of GREN's handoff.

    GREN probed each game and recorded what it refuses; CyclicCortex reads the
    same file to decide which games share a region. Reading it here too is what
    makes "similar games have similar modifiers" mean the same thing in all three
    projects."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != 1:
        raise ValueError(f"{path}: unknown handoff format {doc.get('format')!r}")
    return {n: list(p["signature"]) for n, p in doc["packages"].items()}, doc


class GameFeatures:
    """state ++ candidate ++ modifier, in one fixed-width vector.

    The three blocks occupy disjoint index ranges, so a micro's receptive field
    can land wholly in one of them -- a micro that reads only modifier bits is a
    game-identifier, one that reads only candidate bits is a move-shape
    specialist, and nobody assigned either role."""

    def __init__(self, mechanics_by_game, f_state=128, f_cand=64, f_mod=64,
                 context=1, seed=0):
        self.f_state, self.f_cand, self.f_mod = f_state, f_cand, f_mod
        self.dim = f_state + f_cand + f_mod
        self.seed = seed
        self.mech = dict(mechanics_by_game)
        self.mods = {g: modifier(m, f_mod, seed) for g, m in self.mech.items()}

    def _hash_into(self, tokens, out, offset, width, decay=1.0):
        for o, tok in enumerate(tokens):
            hv = fnv1a(tok.encode("utf-8"), self.seed)
            out[offset + hv % width] += (1.0 if (hv >> 32) & 1 else -1.0) * (decay ** o)

    @staticmethod
    def _l2(out, lo, hi):
        n = sum(out[i] * out[i] for i in range(lo, hi)) ** 0.5
        if n > 0.0:
            for i in range(lo, hi): out[i] /= n

    def transform(self, game, state_tokens, cand_tokens, out=None):
        out = out if out is not None else array("d", [0.0]) * self.dim
        for i in range(self.dim): out[i] = 0.0
        self._hash_into(state_tokens, out, 0, self.f_state)
        self._l2(out, 0, self.f_state)
        self._hash_into(cand_tokens, out, self.f_state, self.f_cand)
        self._l2(out, self.f_state, self.f_state + self.f_cand)
        mod = self.mods.get(game)
        if mod is not None:
            base = self.f_state + self.f_cand
            for i in range(self.f_mod): out[base + i] = mod[i]
        return out


# ------------------------------------------------------------------ adapters
def state_tokens(game_name, s):
    """A few dozen strings describing the position. Deliberately coarse: the
    micro sees 8 hashed features of this, not the board."""
    toks = []
    if game_name in ("chess", "checkers"):
        for y in range(8):
            for x in range(8):
                p = s.at(x, y)
                if p: toks.append(f"p:{x}:{y}:{p}")
        toks.append(f"turn:{s.turn}")
    elif game_name == "go":
        from cortex.board_games import GN
        for y in range(GN):
            for x in range(GN):
                p = s.at(x, y)
                if p: toks.append(f"p:{x}:{y}:{p}")
        toks.append(f"turn:{s.turn}")
    elif game_name == "sudoku":
        for y in range(9):
            for x in range(9):
                v = s.grid[y][x] if hasattr(s, "grid") else 0
                if v: toks.append(f"c:{x}:{y}:{v}")
    return toks or ["empty"]


def cand_tokens(game_name, s, mv):
    if game_name in ("chess", "checkers"):
        (fx, fy), (tx, ty) = mv
        p = s.at(fx, fy) or "."
        t = s.at(tx, ty) or "."
        dx, dy = tx - fx, ty - fy
        return [f"from:{fx}:{fy}", f"to:{tx}:{ty}", f"piece:{p}", f"target:{t}",
                f"d:{dx}:{dy}", f"adx:{abs(dx)}", f"ady:{abs(dy)}",
                f"diag:{int(abs(dx) == abs(dy))}", f"straight:{int(dx == 0 or dy == 0)}",
                f"mine:{int(bool(p) and p[0] == s.turn)}"]
    if game_name == "go":
        from cortex.board_games import PASS
        if mv == PASS: return ["pass"]
        x, y = mv
        occ = 1 if s.at(x, y) else 0
        return [f"at:{x}:{y}", f"occupied:{occ}", f"edge:{min(x, y)}"]
    if game_name == "sudoku":
        x, y, v = mv
        return [f"at:{x}:{y}", f"val:{v}", f"box:{x//3}:{y//3}"]
    return ["cand"]
