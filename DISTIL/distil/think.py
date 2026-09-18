"""Reasoning performed *in* the embedding space, producing new memories.

Everything else in this package treats the embedding layer as something to read:
embed a query, retrieve neighbours, put them in front of a model. That makes the
space a lookup table with good ranking. This module treats it as something to
*compute with*, and every operation here writes its result back as a new trace --
so the space grows from its own structure rather than only from what the system
was asked.

Four operations, each a question about the geometry that produces an answer the
system did not previously hold:

    CONCEPT       a dense cluster has a centre nobody has named. Name it, and
                  embed the name. The centroid is a point the store can now
                  retrieve *through*, one hop closer to everything under it.
    ANALOGY       a - b + c. The classic vector-arithmetic completion, and the
                  only operation here that can reach a region no single memory
                  is near: the offset between two things it knows, applied to a
                  third, lands somewhere it has never looked.
    BRIDGE        two clusters far apart in the space, with nothing between
                  them. The midpoint is a question: what would sit here? An
                  empty region between two populated ones is where an invention
                  goes, which is `explore.py`'s analogy argument done with
                  coordinates instead of a similarity band.
    TENSION       two memories that are near-identical in the space and carry
                  opposite grades. Same subject, contradictory verdicts -- a
                  real problem, and one nothing else in this system detects
                  (`memory.gaps` finds *diffuse* disagreement; this finds a
                  direct contradiction between two specific traces).

**What makes this honest rather than hallucination.** Nothing written here is
graded, and every trace carries `derived: True`. These are *conjectures read off
the geometry*: "these six memories share a centre", "this pair contradicts".
They enter memory ungraded and at the credibility prior, exactly like an MCP tool
nobody has run -- earning credibility only if something later confirms them. A
derived trace that asserted itself as fact would be the system manufacturing
evidence about its own contents, which is the failure `explore.py` and
`toolsmith.py` are both arranged against.

**Why it must not feed on itself.** A derived trace is a point in the same space
the next pass reads, so without a guard the centroid of a set of centroids
becomes a concept, and the analogy between two analogies becomes an analogy. The
same self-reference that broke `auto.py`'s why-chains twice. `derived` is the
marker, and every source pool here excludes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import Kind
from .vector import centroid, cosine, normalise, spread


class Thought:
    CONCEPT = "concept"
    ANALOGY = "analogy"
    BRIDGE = "bridge"
    TENSION = "tension"
    ALL = (CONCEPT, ANALOGY, BRIDGE, TENSION)


@dataclass
class Derived:
    """Something read off the geometry, and the evidence for it."""
    kind: str
    text: str
    vector: list
    sources: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": self.kind, "text": self.text,
                "sources": [t.id for t in self.sources], "detail": self.detail}


class Thinker:
    """Operations on the space, and the writer that puts results back into it.

    Every method returns candidates without writing; `absorb` writes. The split
    exists so the frontend can show what the geometry suggests before any of it
    becomes memory, and so a caller can rank across operations rather than
    taking whatever the first one produced.
    """

    #: Below this a "cluster" is a coincidence, not a concept.
    COHESION = 0.30
    #: Above this two traces are the same memory, not two that contradict.
    SAME = 0.92
    #: Below this two things are anti-correlated rather than merely distant, and
    #: the midpoint of those is noise, not an invention.
    FAR = 0.0
    #: A third trace this close to the midpoint means the region is occupied.
    #: Deliberately generous: the midpoint of two near-orthogonal vectors sits at
    #: cos 45deg ~ 0.707 from BOTH parents, so anything genuinely between them
    #: has to beat the parents to be interesting at all.
    OCCUPIED = 0.62

    def __init__(self, memory, policy, rng=None) -> None:
        self.memory = memory
        self.policy = policy
        self.rng = rng

    # -- the pool ------------------------------------------------------------

    def pool(self, kinds: tuple[str, ...] | None = None) -> list:
        """Traces this may reason from: everything it did not itself derive.

        Excluding its own output is not an optimisation. A centroid of centroids
        is a point about the shape of its own conclusions, phrased as a claim
        about the world, and two passes of that produces confident nonsense.
        """
        return [t for t in self.memory.store.all()
                if not t.meta.get("derived") and (not kinds or t.kind in kinds)]

    def _derived_ids(self) -> set:
        """Every trace this module has written, to exclude from k-nearest search.

        Filtering derived traces out of the *results* is not enough: they still
        occupy slots in the top-k, so each pass saw a different neighbourhood
        than the last and re-derived the same conjecture under a different set of
        sources. The identity key is built from the sources, so that meant the
        store accumulated near-duplicates forever instead of merging them.
        """
        return {t.id for t in self.memory.store.all() if t.meta.get("derived")}

    # -- 1. concepts ---------------------------------------------------------

    def concepts(self, limit: int = 2, size: int = 4) -> list[Derived]:
        """Name the centre of a dense cluster.

        The centroid of a tight group is a point that is *near everything in the
        group and identical to none of it* -- which is what a concept is. Writing
        it back gives recall a single hop to the whole neighbourhood, where
        before it had to be similar to one specific member to find any of them.
        """
        out, taken = [], []
        skip = self._derived_ids()
        for seed in self.pool():
            near = self.memory.recall_vector(seed.vector, k=size, exclude=skip)
            if len(near) < size:
                continue
            traces = [h.trace for h in near if not h.trace.meta.get("derived")]
            if len(traces) < size:
                continue
            vectors = [t.vector for t in traces]
            centre = centroid(vectors)
            cohesion = sum(cosine(v, centre) for v in vectors) / len(vectors)
            if cohesion < self.COHESION:
                continue
            if any(cosine(centre, c) > 0.85 for c in taken):
                continue                  # the same neighbourhood twice
            taken.append(centre)
            shared = _shared_words(traces)
            out.append(Derived(
                Thought.CONCEPT,
                f"these {len(traces)} memories share a centre"
                + (f", about: {shared}" if shared else "")
                + f" -- {_clip(traces[0].text)}",
                centre, traces,
                {"cohesion": round(cohesion, 4), "size": len(traces), "shared": shared}))
            if len(out) >= limit:
                break
        return out

    # -- 2. analogy by arithmetic --------------------------------------------

    def analogies(self, limit: int = 2) -> list[Derived]:
        """a - b + c, and whatever is nearest the result.

        The one operation here that can reach somewhere no single memory sits.
        The offset `a - b` is a *relation* held as a direction; adding it to `c`
        asks "what stands to c as a stands to b?" -- and the answer is a point,
        which may or may not have a memory near it. Both outcomes are worth
        writing: a near hit is a relation the store confirms, and a miss is a
        region the system has reason to look at and nothing in.
        """
        pool = [t for t in self.pool() if t.graded]
        if len(pool) < 3:
            return []
        pool.sort(key=lambda t: t.credibility(self.policy), reverse=True)
        skip = self._derived_ids()
        out = []
        for i, a in enumerate(pool[:6]):
            for b in pool[i + 1:i + 4]:
                if cosine(a.vector, b.vector) > self.SAME:
                    continue              # no relation between a thing and itself
                for c in pool[:4]:
                    if c.id in (a.id, b.id):
                        continue
                    target = normalise([x - y + z for x, y, z in
                                        zip(a.vector, b.vector, c.vector)])
                    near = [h for h in self.memory.recall_vector(
                                target, k=2, exclude=skip | {a.id, b.id, c.id})]
                    landed = near[0].trace if near else None
                    out.append(Derived(
                        Thought.ANALOGY,
                        f"{_clip(a.text, 44)} is to {_clip(b.text, 44)} as "
                        f"{_clip(c.text, 44)} is to "
                        + (_clip(landed.text, 44) if landed
                           else "nothing the store holds -- an unoccupied region"),
                        target, [t for t in (a, b, c, landed) if t is not None],
                        {"landed": bool(landed),
                         "similarity": round(near[0].similarity, 4) if near else 0.0}))
                    if len(out) >= limit:
                        return out
        return out

    # -- 3. bridges ----------------------------------------------------------

    def bridges(self, limit: int = 1) -> list[Derived]:
        """The empty midpoint between two populated regions.

        A question rather than a claim, and phrased as one.

        **The parents have to be excluded from the search**, and getting this
        wrong made the operation return nothing at all. The midpoint of two
        near-orthogonal vectors sits at cos 45deg ~ 0.707 from *each of them*, so
        the nearest neighbour to any midpoint is always one of the two traces
        that defined it. Asking "is anything near the midpoint?" therefore always
        answered yes, and every candidate was discarded. The question that means
        something is whether any *third* memory sits between them.
        """
        pool = self.pool()
        if len(pool) < 4:
            return []
        pool.sort(key=lambda t: t.credibility(self.policy), reverse=True)
        skip = self._derived_ids()
        out = []
        for i, a in enumerate(pool[:8]):
            for b in pool[i + 1:8]:
                sim = cosine(a.vector, b.vector)
                if not (self.FAR <= sim <= 0.45):
                    continue
                mid = normalise([(x + y) / 2.0 for x, y in zip(a.vector, b.vector)])
                near = [h for h in self.memory.recall_vector(
                            mid, k=3, exclude=skip | {a.id, b.id})]
                # Only interesting if no THIRD memory sits between them.
                if near and near[0].similarity > self.OCCUPIED:
                    continue
                out.append(Derived(
                    Thought.BRIDGE,
                    f"what connects {_clip(a.text, 50)} and {_clip(b.text, 50)}? "
                    f"nothing sits between them",
                    mid, [a, b],
                    {"separation": round(sim, 4),
                     "nearest": round(near[0].similarity, 4) if near else 0.0}))
                if len(out) >= limit:
                    return out
        return out

    # -- 4. tensions ---------------------------------------------------------

    def tensions(self, limit: int = 2) -> list[Derived]:
        """Two memories about the same thing that disagree about it.

        `memory.gaps` finds a *neighbourhood* of low mean credibility -- diffuse
        uncertainty. This finds a specific pair: near-identical in the space, one
        graded up and one graded down. That is not uncertainty, it is a
        contradiction with two named sides, and it is directly actionable.
        """
        pool = [t for t in self.pool() if t.graded and t.mean_grade is not None]
        out = []
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if a.mean_grade * b.mean_grade >= 0:
                    continue              # they agree, or one is neutral
                sim = cosine(a.vector, b.vector)
                if sim < 0.55 or sim > self.SAME:
                    continue
                good, bad = (a, b) if a.mean_grade > 0 else (b, a)
                out.append(Derived(
                    Thought.TENSION,
                    f"contradiction: {_clip(good.text, 46)} graded "
                    f"{good.mean_grade:+.2f} while {_clip(bad.text, 46)} graded "
                    f"{bad.mean_grade:+.2f}, and they are {sim:.2f} alike",
                    centroid([a.vector, b.vector]), [good, bad],
                    {"similarity": round(sim, 4),
                     "spread": round(abs(good.mean_grade - bad.mean_grade), 4)}))
                if len(out) >= limit:
                    return out
        return out

    # -- writing it back -----------------------------------------------------

    def think(self, limit: int = 4) -> list[Derived]:
        """One pass over all four operations, best-first."""
        found = (self.concepts() + self.analogies() + self.tensions() + self.bridges())
        found.sort(key=lambda d: -len(d.sources))
        return found[:limit]

    def absorb(self, derived: list[Derived]) -> list:
        """Write conjectures into memory, ungraded.

        `Kind.FACT` for a concept or an analogy that landed, `Kind.IDEA` for a
        bridge or an analogy that did not, `Kind.FAILURE` for a tension -- which
        is what a contradiction is: a record that something here does not work.

        Ungraded on purpose, and marked `derived`. These are read off the shape
        of the store, not observed, and entering them as established would be the
        system manufacturing evidence about its own contents.
        """
        written = []
        for d in derived:
            kind = {Thought.CONCEPT: Kind.FACT,
                    Thought.BRIDGE: Kind.IDEA,
                    Thought.TENSION: Kind.FAILURE}.get(d.kind)
            if kind is None:              # analogy: fact if it landed, idea if not
                kind = Kind.FACT if d.detail.get("landed") else Kind.IDEA
            written.append(self.memory.remember(
                kind, d.text,
                meta={"derived": True, "thought": d.kind, **d.detail},
                links=[t.id for t in d.sources],
                # Identity keyed on the operation and its sources, so the same
                # conjecture re-derived next pass merges instead of accumulating.
                identity=f"think:{d.kind}:" + ",".join(sorted(t.id for t in d.sources))))
        return written


def _shared_words(traces, floor: int = 2) -> str:
    """Words common to most of a cluster: a cheap name for what it is about."""
    import collections
    import re
    counts = collections.Counter()
    for t in traces:
        counts.update(set(re.findall(r"[a-z]{4,}", t.text.lower())))
    common = [w for w, n in counts.most_common(6) if n >= max(floor, len(traces) // 2)]
    return ", ".join(common[:4])


def _clip(text: str, n: int = 60) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")
