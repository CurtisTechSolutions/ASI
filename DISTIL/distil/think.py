"""Thinking by clustering the embedding space, and writing what it finds back.

Everything else in this package treats the embedding layer as something to read:
embed a query, retrieve neighbours, rank them. That makes the space a lookup
table with good ordering. This module treats its *shape* as information — it
clusters what the system knows and names each cluster, and the name goes back in
as a new memory.

**Why a name is worth storing.** The centroid of a cluster is a point near every
member and identical to none of them, which is what a concept is. Writing it back
gives recall a single hop to a whole neighbourhood where before it had to be
similar to one specific member to reach any of them. The store gets denser rather
than only longer.

**And the name carries the cluster's compressed context.** A label alone --
"these six memories share a centre, about: median, sequence" -- tells you a group
exists and nothing about what is in it, so recalling it teaches you nothing you
could act on. Each concept is instead summarised by `compress.Compressor`, the
same detail-preserving compression used on cold memory (13): the terms that
distinguish this cluster from the rest of the store, the best-graded member
verbatim, the spread of grades across it. One recalled concept then answers
"what does this system know about X?" without fetching the cluster.

That compression *adds* a trace; it never replaces one. `compress.py` groups by
access and folds what has gone cold; this groups by structure and leaves every
member exactly where it was. The two are deliberately separate -- a cluster being
coherent is not a reason to forget its members.

**Real clustering, not nearest-neighbours-of-a-seed.** An earlier version picked a
trace, took its k nearest and called that a cluster. That is seed-dependent and
produces overlapping near-duplicate groups: two adjacent seeds give two clusters
that are mostly the same traces, and which ones you get depends on which trace
happened to be iterated first. Agglomerative average-linkage is used instead —
deterministic, disjoint, and it needs no k, because how many concepts a store
contains is not something the caller knows in advance.

**What is written is a conjecture, not an observation.** Every derived trace is
ungraded and carries `derived: True`. "These six memories share a centre" is read
off the geometry; it earns credibility only if something later confirms it. A
derived trace entering as established would be the system manufacturing evidence
about its own contents — the failure `explore.py` and `toolsmith.py` are both
arranged against, turned inward.

**It must not feed on itself.** A derived trace is a point in the same space the
next pass reads, so a centroid of centroids would become a concept, and two passes
of that is confident nonsense. Derived traces are excluded from the pool *and*
from the similarity search — filtering them out of the results is not enough,
because they still occupy slots and shift which cluster forms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import Kind
from .vector import centroid, cosine


@dataclass
class Cluster:
    """A group of memories that sit together, and the point at the middle."""
    members: list
    centre: list
    cohesion: float                       # mean similarity of members to the centre
    shared: str = ""                      # words most of them have in common

    def to_json(self) -> dict:
        return {"size": len(self.members), "cohesion": round(self.cohesion, 4),
                "shared": self.shared, "members": [t.id for t in self.members]}


@dataclass
class Derived:
    """A named cluster, ready to be written back."""
    text: str
    vector: list
    sources: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": "concept", "text": self.text,
                "sources": [t.id for t in self.sources], "detail": self.detail}


class Thinker:
    """Cluster the space, name what forms, put the names back.

    `clusters` and `concepts` compute without writing; `absorb` writes. The split
    lets the frontend show what the geometry suggests before any of it becomes
    memory.
    """

    #: How far above the store's OWN mean pairwise similarity two groups must be
    #: to merge, in standard deviations. A fixed cut cannot work: what counts as
    #: "similar" depends on the embedder and the corpus, and a constant tuned for
    #: one is wrong for the other. Measured here, the lexical embedder puts
    #: unrelated short texts at ~0.01 and related ones at 0.13-0.40, while a
    #: provider embedder is compressed into a much narrower, much higher band --
    #: no single number separates both, but "well above this store's average"
    #: does.
    SPREAD = 0.75
    #: The floor, for a store with no structure at all. Without it a uniformly
    #: unrelated set has a tiny standard deviation, the cut collapses towards the
    #: mean, and everything merges into one meaningless cluster.
    FLOOR = 0.10
    #: A pair is not a concept. Three is the smallest group whose centre says
    #: something none of its members does.
    MIN_SIZE = 3
    #: Below this a group is a coincidence rather than a concept, even if
    #: linkage put it together.
    COHESION = 0.30
    #: Agglomerative linkage is O(n^2) in time and memory. At a few hundred
    #: traces that is instant; at fifty thousand it is not, so the most credible
    #: are clustered and the rest wait for a later pass. `skipped` reports it
    #: rather than silently clustering a sample and calling it the store.
    POOL = 400

    def __init__(self, memory, policy, rng=None, compressor=None) -> None:
        self.memory = memory
        self.policy = policy
        self.rng = rng                    # unused: clustering here is deterministic
        # Optional: without one, concepts fall back to a bare label. The caller
        # usually has a Compressor already (Distil builds one), and sharing it
        # means the concept is summarised by the same code that summarises cold
        # memory rather than a second, quietly different implementation.
        self.compressor = compressor

    # -- what may be clustered -----------------------------------------------

    def pool(self, kinds: tuple[str, ...] | None = None) -> list:
        """Traces this may reason from: everything it did not itself derive.

        Excluding its own output is not an optimisation. The centroid of a set of
        centroids is a claim about the shape of its own conclusions, phrased as a
        claim about the world.
        """
        found = [t for t in self.memory.store.all()
                 if not t.meta.get("derived") and (not kinds or t.kind in kinds)]
        found.sort(key=lambda t: t.credibility(self.policy), reverse=True)
        return found[:self.POOL]

    def skipped(self, kinds: tuple[str, ...] | None = None) -> int:
        """How many eligible traces the pool cap left out of this pass."""
        total = sum(1 for t in self.memory.store.all()
                    if not t.meta.get("derived") and (not kinds or t.kind in kinds))
        return max(0, total - self.POOL)

    # -- clustering ----------------------------------------------------------

    def clusters(self, kinds: tuple[str, ...] | None = None) -> list[Cluster]:
        """Agglomerative average-linkage over the pool, cut at `LINKAGE`.

        Average linkage (UPGMA) rather than single: single linkage chains, so one
        trace sitting between two unrelated groups merges them into a cluster
        whose centre means nothing. Average asks whether the groups are alike *on
        the whole*, which is the question a concept needs answered.

        Deterministic given the store, so the same memories always produce the
        same concepts and a new one means the store changed.

        O(n^3) worst case, which is why `POOL` is small. The straightforward
        implementation is worth more here than a fast one that is hard to check:
        this decides what the system believes it knows about itself.
        """
        traces = self.pool(kinds)
        if len(traces) < self.MIN_SIZE:
            return []

        # The FULL pairwise matrix. Storing only pairs above the cut was wrong in
        # a way that silently lost clusters: average linkage averages over every
        # cross-pair, so a missing entry has to mean "low", not "absent". With
        # them dropped, two groups whose members were mostly-but-not-all above
        # the cut never merged, and three obvious groups came back as one.
        n = len(traces)
        sim = [[0.0] * n for _ in range(n)]
        flat = []
        for i in range(n):
            for j in range(i + 1, n):
                sim[i][j] = sim[j][i] = cosine(traces[i].vector, traces[j].vector)
                flat.append(sim[i][j])

        cut = self.linkage(flat)
        groups = {i: [i] for i in range(n)}
        while len(groups) > 1:
            # UPGMA: merge the closest pair by average linkage, which is the mean
            # similarity over all cross-pairs -- held here as group sizes and a
            # running average rather than by revisiting the members.
            best, pair = cut, None
            keys = sorted(groups)
            for x_i, x in enumerate(keys):
                for y in keys[x_i + 1:]:
                    total = sum(sim[i][j] for i in groups[x] for j in groups[y])
                    avg = total / (len(groups[x]) * len(groups[y]))
                    if avg > best:
                        best, pair = avg, (x, y)
            if pair is None:
                break                     # nothing left above the cut
            x, y = pair
            groups[x] = groups[x] + groups[y]
            del groups[y]

        groups = {k: [traces[i] for i in v] for k, v in groups.items()}
        out = []
        for members in groups.values():
            if len(members) < self.MIN_SIZE:
                continue                  # a pair is not a concept
            vectors = [t.vector for t in members]
            centre = centroid(vectors)
            cohesion = sum(cosine(v, centre) for v in vectors) / len(vectors)
            if cohesion < self.COHESION:
                continue
            out.append(Cluster(members, centre, cohesion, _shared_words(members)))
        out.sort(key=lambda c: (c.cohesion * len(c.members)), reverse=True)
        return out

    def linkage(self, similarities: list[float]) -> float:
        """The cut, taken from this store's own distribution.

        `mean + SPREAD * stdev` over the observed pairs. A pair that is merely
        average for this corpus is not evidence of anything -- everything is
        average somewhere -- so what matters is being unusually close *here*.
        """
        import statistics
        if len(similarities) < 2:
            return self.FLOOR
        return max(self.FLOOR,
                   statistics.mean(similarities)
                   + self.SPREAD * statistics.pstdev(similarities))

    # -- naming --------------------------------------------------------------

    def concepts(self, limit: int = 3, kinds: tuple[str, ...] | None = None) -> list[Derived]:
        """Every cluster worth naming, best first."""
        return [self._name(c) for c in self.clusters(kinds)[:limit]]

    def _name(self, cluster: Cluster) -> Derived:
        """Name a cluster, and compress its context into the name.

        The heading is the label; the body is the cluster compressed -- its
        distinguishing terms, its best-graded member verbatim, the spread of its
        grades. A concept recalled later then carries what the cluster actually
        *contains*, which is the difference between an index entry and an answer.
        """
        members = list(cluster.members)
        heading = (f"concept over {len(members)} memories"
                   + (f", about: {cluster.shared}" if cluster.shared else ""))
        detail = {"cohesion": round(cluster.cohesion, 4), "size": len(members),
                  "shared": cluster.shared,
                  "kinds": sorted({t.kind for t in members})}

        if self.compressor is None:
            # No compressor: a label and one example, which is what this used to
            # be. Honest, and much less useful.
            return Derived(f"{heading} -- e.g. {_clip(members[0].text)}",
                           cluster.centre, members, {**detail, "compressed": False})

        digest = self.compressor.summarise_members(members, label=heading)
        detail.update({"compressed": True, "terms": digest.kept_terms,
                       "exemplar": digest.exemplar[:400],
                       "mean_grade": digest.mean_grade})
        return Derived(digest.text, cluster.centre, members, detail)

    def think(self, limit: int = 3) -> list[Derived]:
        return self.concepts(limit=limit)

    # -- writing it back -----------------------------------------------------

    def absorb(self, derived: list[Derived]) -> list:
        """Write the names into memory, ungraded.

        `Kind.FACT`, because a cluster's centre is an observation about the store
        — but ungraded and marked `derived`, because it is read off the geometry
        rather than checked against anything.
        """
        written = []
        for d in derived:
            written.append(self.memory.remember(
                Kind.FACT, d.text,
                meta={"derived": True, "thought": "concept", **d.detail},
                links=[t.id for t in d.sources],
                # Keyed on the members, so the same cluster re-derived next pass
                # merges instead of accumulating a near-duplicate.
                identity="think:concept:" + ",".join(sorted(t.id for t in d.sources))))
        return written


def _shared_words(traces, floor: int = 2) -> str:
    """Words most of a cluster has in common: a cheap name for what it is about."""
    import collections
    import re
    counts = collections.Counter()
    for t in traces:
        counts.update(set(re.findall(r"[a-z]{4,}", t.text.lower())))
    common = [w for w, n in counts.most_common(8) if n >= max(floor, len(traces) // 2)]
    return ", ".join(common[:4])


def _clip(text: str, n: int = 60) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")
