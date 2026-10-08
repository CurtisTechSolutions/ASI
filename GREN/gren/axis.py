"""Evidence: what we know about a game so far.

Split into DECLARED (free -- you read the box before you play) and MEASURED
(costs probes). Retrieval on declared axes alone routinely cuts a corpus to a
handful, which is the single largest efficiency in the design; probes are then
spent only on what the description did not say.

The signature is a set of `axis=value` tokens, so two games can be compared by
set overlap -- symmetric, order-free, and immune to the ordering pathology a
tree has (DESIGN §8.1).
"""
import math

def bucket(x, edges, names):
    for e, n in zip(edges, names):
        if x <= e: return n
    return names[-1]

class Evidence:
    def __init__(self, name):
        self.name = name
        self.declared = {}
        self.codes = {}            # refusal code -> count
        self.first_seen = {}       # refusal code -> the probe that discovered it
        self.legal = 0
        self.illegal = 0
        self.probes = 0

    def declare(self, d): self.declared.update(d)

    def observe(self, verdict):
        self.probes += 1
        if verdict.outcome.name == "LEGAL": self.legal += 1
        elif verdict.outcome.name == "ILLEGAL":
            self.illegal += 1
            if verdict.reason_code:
                if verdict.reason_code not in self.codes:
                    self.first_seen[verdict.reason_code] = self.probes
                self.codes[verdict.reason_code] = self.codes.get(verdict.reason_code, 0) + 1

    # ------------------------------------------------------------- measured
    @property
    def legality_density(self):
        n = self.legal + self.illegal
        return self.legal / n if n else 0.0

    @property
    def reason_entropy(self):
        tot = sum(self.codes.values())
        if tot <= 0: return 0.0
        h = 0.0
        for c in self.codes.values():
            p = c / tot
            if p > 0: h -= p * math.log2(p)
        return h

    @property
    def settled(self):
        """How long since a NEW refusal kind turned up, as a fraction of the run.

        A taxonomy still growing at the end of probing has not been explored;
        one whose last discovery was long ago has. This is the honest measure of
        `do we know what this game is like`, and it is a different question from
        `which known game is this` -- a genuinely novel game is FAR from every
        neighbour and that is a confident answer, not an uncertain one."""
        if not self.first_seen or self.probes <= 0: return 0.0
        return max(0.0, 1.0 - (max(self.first_seen.values()) / self.probes))

    def measured(self, min_share=0.005):
        """A code seen once in a thousand probes is noise, not a rule. The
        threshold is what keeps a signature stable across runs."""
        tot = max(1, sum(self.codes.values()))
        out = {f"refuses={c}" for c, n in self.codes.items() if n / tot >= min_share}
        out.add("legality=" + bucket(self.legality_density, [0.01, 0.1, 0.5, 0.9],
                                     ["<1%", "1-10%", "10-50%", "50-90%", ">90%"]))
        out.add("reason_kinds=" + bucket(len(self.codes), [1, 4, 16],
                                         ["1", "2-4", "5-16", "17+"]))
        out.add("reason_entropy=" + bucket(self.reason_entropy, [0.8, 1.8],
                                           ["low", "medium", "high"]))
        return out

    def signature(self, include_measured=True):
        sig = {f"{k}={v}" for k, v in self.declared.items()}
        if include_measured: sig |= self.measured()
        return frozenset(sig)

    def to_dict(self):
        return {"name": self.name, "declared": self.declared, "codes": self.codes,
                "first_seen": self.first_seen, "settled": round(self.settled, 3),
                "legal": self.legal, "illegal": self.illegal, "probes": self.probes,
                "signature": sorted(self.signature())}

def distance(a, b):
    """Jaccard distance -- a proper metric, so `distance is similarity` has a
    consistent geometry (CyclicCortex/DESIGN.md §5.1)."""
    u = a | b
    return 1.0 - len(a & b) / len(u) if u else 1.0
