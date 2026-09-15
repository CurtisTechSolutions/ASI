"""The handoff. Everything else exists to produce this object.

`GTMNet.load_package` refuses below `tau_conf`: a package that does not know
what game it is holding is refused and probing continues. That is
"once you understand the game, then and only then can you play it", enforced
mechanically rather than assumed.
"""
from dataclasses import dataclass, field
from gren.axis import distance

TAU_CONF = 0.8

@dataclass
class GamePackage:
    name: str
    signature: frozenset
    mechanics: frozenset          # the refusal codes this game actually has
    candidates: list              # [(distance, known_game_name)], nearest first
    probes: int
    legality_density: float
    reason_entropy: float
    verified_fraction: float = 1.0
    confidence: float = 0.0

    def to_dict(self):
        return {"name": self.name, "signature": sorted(self.signature),
                "mechanics": sorted(self.mechanics), "probes": self.probes,
                "candidates": [(round(d, 3), n) for d, n in self.candidates[:5]],
                "legality_density": round(self.legality_density, 3),
                "reason_entropy": round(self.reason_entropy, 2),
                "confidence": round(self.confidence, 3)}

    def accepted(self, tau=TAU_CONF): return self.confidence >= tau


def build(evidence, corpus=None, min_probes=200):
    """Confidence multiplies three independent ways of being wrong -- too few
    probes, no clear nearest neighbour, a refusal taxonomy still growing -- so
    that any one of them being bad is fatal rather than averaged away."""
    corpus = corpus or {}
    sig = evidence.signature()
    ranked = sorted(((distance(sig, s), n) for n, s in corpus.items() if n != evidence.name))
    coverage = min(1.0, evidence.probes / min_probes)
    if len(ranked) >= 2:
        margin = min(1.0, max(0.0, (ranked[1][0] - ranked[0][0]) / 0.25))
        nearest = 1.0 - ranked[0][0]
        identity = max(nearest, margin)
    elif ranked:
        identity = 1.0 - ranked[0][0]
    else:
        identity = 1.0                      # nothing to confuse it with
    settled = 1.0 if evidence.codes else 0.0
    conf = coverage * max(0.35, identity) * (0.5 + 0.5 * settled)
    return GamePackage(
        name=evidence.name, signature=sig,
        mechanics=frozenset(evidence.codes), candidates=ranked,
        probes=evidence.probes, legality_density=evidence.legality_density,
        reason_entropy=evidence.reason_entropy, confidence=conf)
