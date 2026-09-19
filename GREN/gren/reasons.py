"""Clustering the oracle's own words into a taxonomy.

The reason text is the highest-bandwidth channel in the system. A bare rejection
carries one bit; a reason drawn from R classes carries log2(R+1), which is why
`information.py` found a rich oracle identifies far faster than the halving
bound suggests. Most learners keep the bit and throw the sentence away.
"""
from gren.verdict import signature

class ReasonTaxonomy:
    def __init__(self, merge_tau=0.6):
        self.classes = []          # [{id, exemplar, sig, code, count}]
        self.by_sig = {}
        self.by_code = {}
        self.merge_tau = merge_tau

    def __len__(self): return len(self.classes)

    def classify(self, verdict):
        """1. A structured code IS the class -- compilers already did this work,
        and rediscovering clusters over rustc output when rustc hands you E0308
        would be inventing a worse version of an existing answer.
           2. Else an exact normalised-signature match.
           3. Else agglomerate by token Jaccard against exemplars.
           4. Else open a new class."""
        if verdict.reason_code:
            c = self.by_code.get(verdict.reason_code)
            if c is None: c = self._new(verdict, verdict.reason_code)
            self.classes[c]["count"] += 1
            return c
        sig = signature(verdict.reason)
        if not sig: return None
        c = self.by_sig.get(sig)
        if c is None:
            c = self._nearest(sig)
            if c is None: c = self._new(verdict, None, sig)
            else: self.by_sig[sig] = c
        self.classes[c]["count"] += 1
        return c

    def _tokens(self, sig): return set(sig.split())

    def _nearest(self, sig):
        best, bi = 0.0, None
        ts = self._tokens(sig)
        for i, c in enumerate(self.classes):
            other = self._tokens(c["sig"])
            u = ts | other
            if not u: continue
            j = len(ts & other) / len(u)
            if j > best: best, bi = j, i
        return bi if best >= self.merge_tau else None

    def _new(self, verdict, code=None, sig=None):
        i = len(self.classes)
        sig = sig if sig is not None else signature(verdict.reason)
        self.classes.append({"id": i, "exemplar": verdict.reason, "sig": sig,
                             "code": code, "count": 0})
        if code: self.by_code[code] = i
        if sig: self.by_sig.setdefault(sig, i)
        return i

    def entropy(self):
        import math
        tot = sum(c["count"] for c in self.classes)
        if tot <= 0: return 0.0
        h = 0.0
        for c in self.classes:
            p = c["count"] / tot
            if p > 0: h -= p * math.log2(p)
        return h

    def to_dict(self):
        return {"classes": self.classes, "merge_tau": self.merge_tau}
