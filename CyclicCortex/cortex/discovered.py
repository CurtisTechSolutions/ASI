"""The map built from what GREN measured, instead of from what I wrote down.

`cortex/games.py` carries a hand-written `mechanics` frozenset per game, and
those sets are what decide which games share a region. So the cortex's whole
geography was an artefact of my NAMING: chess and checkers land together in part
because I typed GRID_BOARD twice.

GREN never sees those names. It fires moves at a game, records what comes back
when the game refuses, and buckets a handful of declared axes (players,
information, goal type). Its signature is a measurement. This module substitutes
it, so the cortex's geography is something the system found rather than
something it was told.

The handoff is a JSON file, never an import. GREN imports CyclicCortex's games
in order to probe them; importing GREN back from here would close the cycle.
Write the file with

    cd GREN && python3 -m gren.cli package --out ../CyclicCortex/data/gren_packages.json
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(os.path.dirname(_HERE), "data", "gren_packages.json")

TAU_PLACE = 0.75          # GamePackage.placeable(): characterisation, not identity


def load(path=DEFAULT_PATH):
    """Read the handoff. Raises if it is missing -- a silent fall back to the
    hand-written sets would make the discovered run indistinguishable from the
    baseline, which is the one failure that would invalidate every comparison
    drawn from it."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != 1:
        raise ValueError(f"{path}: unknown handoff format {doc.get('format')!r}")
    return doc


class Discovered:
    """A game adapter wearing GREN's signature.

    Everything about PLAYING the game is delegated untouched -- the rules are
    the rules. What is replaced is what the cortex uses to place the game:
    `mechanics`, and, when the handoff carries a derived vocabulary, `spec` and
    the keys `generalise` emits.
    """

    def __init__(self, inner, pkg, layout=None):
        self.inner = inner
        self.name = inner.name
        self.package = pkg
        self.mechanics = frozenset(pkg["signature"])
        self.refusals = frozenset(pkg.get("mechanics", ()))
        self.confidence = pkg.get("confidence", 0.0)
        self.characterisation = pkg.get("characterisation", 0.0)
        self.layout = None
        if layout:
            # slot name -> [(block, index, sign)]. A SHARED slot holds one
            # quantity and so is one input wide, for every game that writes it;
            # a game's private block is as wide as it needs. Width therefore
            # never has to be agreed or padded -- it falls out of the alignment,
            # and two games cannot disagree about it.
            self.layout = {c: [tuple(t) for t in d] for c, d in layout.items() if d}
            self.spec = {c: len(d) for c, d in self.layout.items()}
        else:
            self.spec = dict(inner.spec)

    def generalise(self, s, mv):
        feats = self.inner.generalise(s, mv)
        if self.layout is None: return feats
        out = {}
        for code, dims in self.layout.items():
            w = self.spec[code]
            vals = [0.0] * w
            for j, (b, i, sg) in enumerate(dims):
                v = feats.get(b)
                if v is not None and i < len(v): vals[j] = sg * v[i]
            out[code] = vals
        return out

    def __getattr__(self, k):
        # Only reached for names not set above: new, legal_moves, apply,
        # candidates, random_state, terminal, value, reward, ...
        return getattr(self.inner, k)

    def __repr__(self):
        return f"<Discovered {self.name}: {len(self.mechanics)} tokens>"


def registry(base, path=DEFAULT_PATH, tau=TAU_PLACE, vocabulary=True):
    """Wrap every game in `base` that GREN characterised well enough to place.

    The gate is `characterisation`, not `confidence`. They answer different
    questions, and the distinction is what the handoff exposed: sudoku scores
    0.35 confidence because it is 0.80 away from everything GREN has seen, which
    means confidently NOVEL, not uncertain. A novel game is exactly the one that
    should found its own region -- refusing to place it would be backwards.

    Returns (games, report). Games GREN has no package for, or characterised
    below `tau`, keep their hand-written sets and are named in the report.
    """
    doc = load(path)
    packs = doc.get("packages", {})
    vocab = doc.get("vocabulary", {}) if vocabulary else {}
    games, report = {}, []
    for n, g in base.items():
        p = packs.get(n)
        if p is None:
            games[n] = g; report.append((n, "no package", None)); continue
        ch = p.get("characterisation", 0.0)
        if ch < tau:
            games[n] = g; report.append((n, "not placeable", ch)); continue
        games[n] = Discovered(g, p, layout=vocab.get(n))
        report.append((n, "discovered", ch))
    return games, report


def available(path=DEFAULT_PATH):
    return os.path.exists(path)
