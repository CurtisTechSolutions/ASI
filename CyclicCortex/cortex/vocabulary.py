"""A region's input vocabulary: mechanic -> a fixed block of input slots.

The one hard rule is that a slot's index is PERMANENT. Extension only appends.
A slot index is a contract with every weight that reads it, so reordering would
silently scramble everything the region has learned -- the failure would look
like a training bug, not like a data-structure bug, which is why it is an
invariant and not a convention.
"""
from array import array

class Vocabulary:
    def __init__(self):
        self.order = []                 # mechanic names, append-only
        self.slot = {}                  # name -> (offset, width)
        self.width = 0
        self.epoch = 0                  # bumped on every extension

    def has(self, name): return name in self.slot

    def extend(self, spec):
        """spec: {mechanic_name: width}. Appends any name not already present.
        Returns (old_width, new_width)."""
        old = self.width
        for name in sorted(spec):
            if name in self.slot: continue
            w = spec[name]
            self.slot[name] = (self.width, w)
            self.order.append(name)
            self.width += w
        if self.width != old: self.epoch += 1
        return old, self.width

    def encode(self, feats):
        """feats: {mechanic_name: [values]}. Unknown mechanics are IGNORED rather
        than erroring: a game may exercise a mechanic this region has not adopted,
        and dropping it is correct -- the region genuinely does not model it."""
        v = array("d", [0.0]) * self.width if self.width else array("d")
        v = array("d", [0.0] * self.width)
        for name, vals in feats.items():
            s = self.slot.get(name)
            if s is None: continue
            off, w = s
            for i in range(min(w, len(vals))): v[off + i] = vals[i]
        return v

    def to_dict(self):
        return {"order": self.order, "slot": {k: list(v) for k, v in self.slot.items()},
                "width": self.width, "epoch": self.epoch}

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.order = list(d["order"])
        v.slot = {k: tuple(t) for k, t in d["slot"].items()}
        v.width = d["width"]; v.epoch = d["epoch"]
        return v
