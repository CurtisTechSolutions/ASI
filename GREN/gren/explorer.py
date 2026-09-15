"""The loop: probe, observe, build a signature, identify by similarity."""
import random
from gren.axis import Evidence, distance
from gren.probe import POLICIES
from gren.reasons import ReasonTaxonomy

class Explorer:
    def __init__(self, oracle, policy="eig", seed=0):
        self.oracle = oracle
        self.policy = POLICIES[policy](seed=seed)
        self.evidence = Evidence(oracle.name)
        self.evidence.declare(oracle.declared())
        self.taxonomy = ReasonTaxonomy()
        self.rng = random.Random(seed + 1)

    def run(self, budget=600, k=10):
        """Probe until the budget is spent. Declared evidence is already in hand
        before the first probe, which is the point of the split."""
        g = self.oracle.game
        while self.evidence.probes < budget:
            st = (g.random_state(self.rng) if hasattr(g, "random_state")
                  else g.new(seed=self.rng.randrange(10**6)))
            for mv in self.policy.propose(self.oracle, st, k):
                if self.evidence.probes >= budget: break
                v = self.oracle.probe(st, mv)
                self.taxonomy.classify(v)
                self.evidence.observe(v)
                if hasattr(self.policy, "observe"): self.policy.observe(v)
                else: self.policy.note(v)
        return self.report()

    def report(self):
        e = self.evidence
        return {"game": e.name, "probes": e.probes,
                "failure_rate": round(self.policy.failure_rate, 3),
                "codes": len(e.codes), "taxonomy": len(self.taxonomy),
                "legality_density": round(e.legality_density, 3),
                "reason_entropy": round(e.reason_entropy, 2),
                "signature_size": len(e.signature())}

def identify(evidence, corpus):
    """Nearest known game by signature distance. `corpus` maps name -> signature."""
    sig = evidence.signature()
    ranked = sorted(((distance(sig, s), n) for n, s in corpus.items()))
    return ranked
