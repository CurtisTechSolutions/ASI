"""Every number the system is allowed to change about itself.

Self-improvement needs a surface to improve. Letting a model rewrite its own
control loop is the version of that idea which sounds impressive and produces an
agent that deletes its own scheduler on iteration 40. So the kernel is fixed and
this file is the surface: bounded, typed, measurable parameters, tuned by
measured outcome through `explore.self_upgrade` (DESIGN 8.3).

Capability growth happens the other way -- through tools, which are new code the
system writes, tests and keeps (DESIGN 7). The split is deliberate: **behaviour
is tuned within bounds, capability is added by construction**. Both are
reversible, because both are written down.

Every field carries an explicit range. A tuner that can set a weight to 1e9 has
not been tuned, it has been broken.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

#: name -> (low, high). The tuner cannot leave these.
BOUNDS: dict[str, tuple[float, float]] = {
    "recall_similarity_weight": (0.1, 3.0),
    "recall_credibility_weight": (0.0, 3.0),
    "recall_recency_weight": (0.0, 2.0),
    "recency_halflife_days": (0.5, 365.0),
    "credibility_prior": (0.0, 1.0),
    "credibility_strength": (0.5, 20.0),
    "dedupe_threshold": (0.80, 0.999),
    "credit_decay": (0.0, 0.95),
    "branch_factor": (2, 8),
    "max_depth": (1, 6),
    "target_success": (0.2, 0.8),
    "novelty_weight": (0.0, 2.0),
    "cost_weight": (0.0, 2.0),
    "regret_aversion": (0.0, 1.0),
}


@dataclass
class Policy:
    # --- recall: how memory ranks what it returns (memory.py) -----------------
    recall_similarity_weight: float = 1.0      # exponent on cosine
    recall_credibility_weight: float = 0.6     # exponent on graded credibility
    recall_recency_weight: float = 0.25        # exponent on recency decay
    recency_halflife_days: float = 30.0
    credibility_prior: float = 0.5             # ungraded traces sit at the midpoint
    credibility_strength: float = 3.0          # pseudo-counts; shrinkage toward prior
    dedupe_threshold: float = 0.97             # cosine above which a write merges
    credit_decay: float = 0.6                  # per hop, propagating grades over links

    # --- reasoning: how tasks become goals (reason.py, goals.py) --------------
    branch_factor: int = 4                     # goals proposed per distillation
    max_depth: int = 3                         # recursion cap on the goal tree
    regret_aversion: float = 0.5               # 0 = expected utility, 1 = minimax regret

    # --- exploration: which experiments get run (explore.py) ------------------
    target_success: float = 0.5                # the maximum-information operating point
    novelty_weight: float = 1.0
    cost_weight: float = 0.5

    def clamp(self) -> "Policy":
        for f in fields(self):
            lo, hi = BOUNDS[f.name]
            val = getattr(self, f.name)
            val = max(lo, min(hi, val))
            setattr(self, f.name, int(round(val)) if f.type == "int" else float(val))
        return self

    def mutate(self, rng: random.Random, sigma: float = 0.15) -> "Policy":
        """One neighbour, one field. Coordinate-wise because attributing an
        improvement to a change requires the change to be attributable: move six
        knobs and a gain tells you nothing about which of them earned it."""
        child = Policy(**asdict(self))
        name = rng.choice([f.name for f in fields(self)])
        lo, hi = BOUNDS[name]
        current = getattr(child, name)
        step = rng.gauss(0.0, sigma) * (hi - lo)
        setattr(child, name, current + step)
        return child.clamp()

    def distance(self, other: "Policy") -> float:
        """Normalised L1 over the bounded box, so a change to a 0-1 field and a
        change to a 0-365 field are comparable."""
        total = 0.0
        for f in fields(self):
            lo, hi = BOUNDS[f.name]
            total += abs(getattr(self, f.name) - getattr(other, f.name)) / (hi - lo)
        return total / len(fields(self))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Policy":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}).clamp()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Policy":
        if not path.exists():
            return cls()
        try:
            return cls.from_json(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            return cls()          # a corrupt policy file is not worth a crash

    def diff(self, other: "Policy") -> dict[str, tuple[float, float]]:
        return {f.name: (getattr(self, f.name), getattr(other, f.name))
                for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)}
