"""What the oracle says back.

The distinction that matters is ILLEGAL vs ERROR. A compiler that times out has
not taught you a rule, and folding the two together poisons the refusal boundary
with the oracle's own failures -- the single most likely way to learn confident
nonsense. `reason` is stored verbatim; normalisation produces a separate field,
because the normalisation rules are themselves a guess that will be revised.
"""
import re
from dataclasses import dataclass, field
from enum import IntEnum

class Outcome(IntEnum):
    LEGAL = 0        # accepted, play continues
    ILLEGAL = 1      # refused, with a reason -- the informative case
    TERMINAL = 2     # the game ended
    ERROR = 3        # the ORACLE failed; never a rule signal

@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reason: str | None = None          # the oracle's own words, verbatim
    reason_code: str | None = None     # a structured code when the oracle offers one
    reason_class: int | None = None    # filled by reasons.py
    payoff: float | None = None
    latency: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def legal(self): return self.outcome in (Outcome.LEGAL, Outcome.TERMINAL)
    @property
    def informative(self): return self.outcome == Outcome.ILLEGAL
    def to_dict(self):
        return {"outcome": int(self.outcome), "reason": self.reason,
                "reason_code": self.reason_code, "reason_class": self.reason_class,
                "payoff": self.payoff}

LEGAL = Verdict(Outcome.LEGAL)

_SUBS = [
    (re.compile(r"/?(?:[\w.\-]+/)+[\w.\-]+"), "<path>"),      # paths, leading slash included
    (re.compile(r"\b\d+:\d+\b"), "<pos>"),                   # line:col
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (re.compile(r"\b\d+(\.\d+)?(ms|s|us)\b"), "<dur>"),
    (re.compile(r"(?<![\"'\w])-?\d+(?![\"'\w])"), "<n>"),    # bare integers
    (re.compile(r"\s+"), " "),
]

def normalise(reason):
    """Strip what varies between runs but not between RULES. Deterministic,
    order-independent and total: the same string always maps to the same class,
    across processes."""
    if not reason: return ""
    out = reason
    for pat, rep in _SUBS: out = pat.sub(rep, out)
    return out.strip()

def signature(reason):
    return normalise(reason).lower()
