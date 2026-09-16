"""The embedding layer, which is the memory. Not a cache in front of one.

Everything the system does becomes a `Trace`: the query it was asked, the goals
it distilled, the reasoning chain it followed, the tool it wrote, the error the
interpreter returned, the idea it abandoned. One store, one vector space, one
recall path. There is no separate "tool index" or "fact table" -- a tool is
retrieved by the same query that retrieves the bug it once fixed, because at the
moment of recall they are the same question: *what do I know that bears on this?*

The design decision that makes this more than a vector database:

> **Recall ranks by similarity times credibility times recency, not similarity.**

A vector store answers "what is closest?". That is the wrong question for a
system that learns. Closest includes the answer that looked right and did not
compile. So every trace carries a grade -- from a verifier where one exists,
from the user where one does not -- and recall multiplies similarity by a
shrunk posterior over those grades (4.2). A slightly-less-similar memory that
has been right four times outranks a closer one that has never been checked.
Being wrong pushes a memory *down*, which means the store gets better at
answering without getting bigger.

Grades propagate. A tool that produced a passing answer credits the goal that
called for it and the reasoning step that chose it, decayed per hop. Credit
assignment over the link graph is what turns a single verified outcome into
evidence about the whole path that produced it (4.6).
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .policy import Policy
from .store import OVERFETCH, InProcessStore, TraceStore
from .vector import Vector, centroid, cosine, spread

DAY = 86400.0


class Kind:
    """What a trace is. Recall can filter on it; scoring never uses it, so a
    kind is a label rather than a privilege."""
    QUERY = "query"          # something asked of the system
    ANSWER = "answer"        # something it produced
    GOAL = "goal"            # a distilled objective
    CHAIN = "chain"          # a reasoning trace, stored so reasoning is retrievable
    TOOL = "tool"            # a tool it wrote and verified
    FACT = "fact"            # an observation about the world
    FAILURE = "failure"      # something that did not work, and why
    IDEA = "idea"            # a proposal, usually a bad one, usually on purpose
    GAME = "game"            # a framed game: players, actions, payoffs, referee
    SOLUTION = "solution"    # a problem paired with what actually solved it
    DIGEST = "digest"        # several cold traces compressed into one

    ALL = (QUERY, ANSWER, GOAL, CHAIN, TOOL, FACT, FAILURE, IDEA, GAME, SOLUTION, DIGEST)


class Source:
    SELF = "self"            # a verifier said so: compiled, ran, passed
    USER = "user"            # a human said so
    NONE = "ungraded"        # nothing checkable was available -- the honest default


@dataclass
class Trace:
    id: str
    kind: str
    text: str
    vector: Vector
    embedder: str                                   # vectors from different backends
    created: float                                  # are not comparable; recall enforces it
    meta: dict = field(default_factory=dict)
    links: list[str] = field(default_factory=list)  # ids this trace depends on
    grades: list[tuple[float, str, float]] = field(default_factory=list)   # (score, source, when)
    hits: int = 0                                   # times recalled
    seen: int = 1                                   # times written (dedupe increments)
    last_used: float = 0.0

    @property
    def graded(self) -> bool:
        return bool(self.grades)

    @property
    def mean_grade(self) -> float | None:
        if not self.grades:
            return None
        return sum(g for g, _, _ in self.grades) / len(self.grades)

    @property
    def verified(self) -> bool:
        """Graded by a verifier rather than an opinion. The distinction is the
        whole point of the grading layer, so it gets a name."""
        return any(src == Source.SELF for _, src, _ in self.grades)

    def credibility(self, policy: Policy) -> float:
        """Shrunk posterior in [0,1] over this trace's grades.

        Grades live in [-1,1] and are mapped to [0,1]; the estimate is pulled
        toward `credibility_prior` with `credibility_strength` pseudo-counts.
        Shrinkage is not decoration -- without it a single lucky +1 outranks a
        memory verified twenty times, and recall becomes a recency lottery. A
        user grade counts double: a human bothering to grade is a stronger
        signal than an automatic check firing, and rarer.
        """
        m = policy.credibility_strength
        total = policy.credibility_prior * m
        n = m
        for score, src, _ in self.grades:
            w = 2.0 if src == Source.USER else 1.0
            total += w * (score + 1.0) / 2.0
            n += w
        return total / n

    def recency(self, policy: Policy, now: float | None = None) -> float:
        now = time.time() if now is None else now
        age_days = max(0.0, (now - max(self.created, self.last_used))) / DAY
        return 0.5 ** (age_days / policy.recency_halflife_days)

    def heat(self, policy: Policy, now: float | None = None) -> float:
        """How live this memory is, in [0, 1).

        Access recency and access *frequency*, combined. Recency alone buries
        something consulted constantly for a year and then left alone for a
        month; frequency alone keeps a trace hot forever because it was popular
        once. The product of a saturating hit count and the recency decay is
        what the compression phase reads to decide what may be summarised: cold
        means neither recently wanted nor often wanted.
        """
        used = math.log1p(self.hits + self.seen - 1) / math.log(20.0)
        return min(0.999, min(1.0, used) * self.recency(policy, now))

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text, "vector": self.vector,
                "embedder": self.embedder, "created": self.created, "meta": self.meta,
                "links": self.links, "grades": [list(g) for g in self.grades],
                "hits": self.hits, "seen": self.seen, "last_used": self.last_used}

    @classmethod
    def from_json(cls, d: dict) -> "Trace":
        return cls(id=d["id"], kind=d["kind"], text=d["text"], vector=d["vector"],
                   embedder=d.get("embedder", "hash-v1"), created=d.get("created", 0.0),
                   meta=d.get("meta", {}), links=d.get("links", []),
                   grades=[tuple(g) for g in d.get("grades", [])],
                   hits=d.get("hits", 0), seen=d.get("seen", 1),
                   last_used=d.get("last_used", 0.0))


@dataclass
class Recollection:
    """A recall hit with its arithmetic exposed. The breakdown is kept because
    "why did it surface that?" is the first question asked of any retrieval
    system, and a score with no parts cannot answer it."""
    trace: Trace
    similarity: float
    credibility: float
    recency: float
    score: float

    def explain(self) -> str:
        # The parts are reported without an "=", because each is raised to its
        # own policy weight before multiplication -- printing "score = sim x cred
        # x rec" stated an equation the numbers do not satisfy, and anyone
        # checking it by hand would conclude the ranking was broken.
        return (f"{self.score:.3f}  from sim {self.similarity:.3f}, "
                f"cred {self.credibility:.3f}, rec {self.recency:.3f} "
                f"(each weighted)  [{self.trace.kind}] {self.trace.text[:60]!r}")


class Memory:
    #: The policy recorded in the last loaded store's header, if any. Kept for
    #: inspection rather than applied: `policy.json` is the live setting, and a
    #: store silently overriding it would give one setting two sources of truth.
    saved_policy: Policy | None = None

    def __init__(self, embedder, policy: Policy | None = None, clock=time.time,
                 store: TraceStore | None = None) -> None:
        self.embedder = embedder
        self.policy = policy or Policy()
        self.clock = clock                      # injectable so tests own the timeline
        self.store = store or InProcessStore()

    @property
    def traces(self) -> dict[str, Trace]:
        """Every trace, by id. Convenient on the in-process store and a full scan
        on a remote one -- which is why nothing on the hot path uses it."""
        return {t.id: t for t in self.store.all()}

    # -- writing ----------------------------------------------------------------

    def remember(self, kind: str, text: str, meta: dict | None = None,
                 links: list[str] | None = None, grade: float | None = None,
                 source: str = Source.NONE, identity: str | None = None) -> Trace:
        """Write a trace, or merge into the one it duplicates.

        Near-duplicates are merged rather than appended. An agent that re-reads
        the same file every loop would otherwise fill its own memory with one
        fact stated ten thousand times, and recall would return ten copies of it
        instead of ten things. Merging increments `seen`, which is itself a
        signal: a memory the system keeps re-deriving is one it relies on.

        `identity` overrides that with an exact key, and some records need it.
        Two solutions to the *same* problem embed almost identically -- the
        problem text dominates both vectors -- so cosine dedupe merges them and
        the second silently overwrites the first. That is catastrophic exactly
        where it matters most: the working fix and the refuted one for one
        problem are the most valuable pair in the store, and merging them
        averages a +1 and a -1 into an opinionless 0. Any caller whose records
        have a natural key should pass it.
        """
        vector = self.embedder.embed(text)
        # The backend that actually produced THIS vector, which is not always the
        # one the embedder is named after: a provider embedder falls back to the
        # lexical one on a failed call, and tagging that vector as a provider
        # vector silently mixes two geometries in one space.
        backend = getattr(self.embedder, "last_backend", None) or \
            getattr(self.embedder, "name", "unknown")
        twin = (self._by_identity(kind, identity) if identity
                else self._duplicate(kind, vector, backend))
        if identity:
            meta = {**(meta or {}), "_identity": identity}
        now = self.clock()
        if twin is not None:
            twin.seen += 1
            twin.last_used = now
            twin.meta.update(meta or {})
            if identity and twin.text != text:
                # An identity-keyed record is THE record for that key, so a
                # changed description replaces the old one. Keeping the first
                # text meant a re-forged tool stayed indexed under the
                # description of the version it replaced.
                twin.text = text
                twin.vector = vector
                # And the backend tag with it. Swapping the vector while leaving
                # the old tag put a lexical vector under a provider label (or the
                # reverse), which is the one thing recall cannot detect: it would
                # then compare two geometries as though they were one.
                twin.embedder = backend
            for lid in (links or []):
                if lid not in twin.links:
                    twin.links.append(lid)
            self.store.touch(twin)
            if grade is not None:
                self.grade(twin.id, grade, source)
            return twin
        trace = Trace(id=uuid.uuid4().hex[:12], kind=kind, text=text, vector=vector,
                      embedder=backend, created=now,
                      meta=dict(meta or {}), links=list(links or []))
        self.store.put(trace)
        if grade is not None:
            self.grade(trace.id, grade, source)
        return trace

    def _by_identity(self, kind: str, identity: str) -> Trace | None:
        """Exact-key lookup. Linear over one kind, which is the right cost here:
        keyed writes are rare next to reads, and an index that can disagree with
        the store is a worse problem than a scan."""
        for t in self.store.all():
            if t.kind == kind and t.meta.get("_identity") == identity:
                return t
        return None

    def _duplicate(self, kind: str, vector: Vector, backend: str) -> Trace | None:
        """The nearest same-kind trace, if it is near enough to be the same thing.

        `backend` is the tag the incoming write will carry, not the embedder's
        name. Those differ whenever a provider embedder falls back to lexical,
        and filtering on the name meant dedupe searched a space nothing had been
        written to -- so it silently stopped finding duplicates for the whole
        duration of an outage.
        """
        hits = self.store.search(vector, k=1, kinds=(kind,),
                                 min_similarity=self.policy.dedupe_threshold,
                                 embedder=backend)
        return hits[0][0] if hits else None

    # -- grading ----------------------------------------------------------------

    def grade(self, trace_id: str, score: float, source: str = Source.USER,
              propagate: bool = True) -> Trace:
        """Attach a grade and push a decayed share of it back along `links`.

        Credit assignment is breadth-first with a decay of `credit_decay` per hop
        and a visited set, so a cycle in the link graph terminates instead of
        amplifying. Propagated grades keep the *originating* source, so a chain
        credited by a passing test still reads as self-verified -- losing that
        would let an opinion launder itself into evidence two hops away.
        """
        trace = self.store.get(trace_id)
        if trace is None:
            raise KeyError(f"no such trace: {trace_id}")
        score = max(-1.0, min(1.0, float(score)))
        trace.grades.append((score, source, self.clock()))
        self.store.touch(trace)
        if propagate and self.policy.credit_decay > 0:
            self._propagate(trace, score, source)
        return trace

    def _propagate(self, origin: Trace, score: float, source: str) -> None:
        frontier = [(lid, score * self.policy.credit_decay) for lid in origin.links]
        seen = {origin.id}
        while frontier:
            tid, share = frontier.pop(0)
            if tid in seen or abs(share) < 0.02:
                continue          # below 0.02 the hop is noise; stop walking
            seen.add(tid)
            t = self.store.get(tid)
            if t is None:
                continue
            t.grades.append((max(-1.0, min(1.0, share)), source, self.clock()))
            self.store.touch(t)
            frontier.extend((lid, share * self.policy.credit_decay) for lid in t.links)

    # -- reading ----------------------------------------------------------------

    def recall(self, query: str, k: int = 5, kinds: tuple[str, ...] | None = None,
               min_similarity: float = 0.02, exclude: set[str] | None = None,
               verified_only: bool = False) -> list[Recollection]:
        if not self.traces:
            return []
        qv = self.embedder.embed(query, learn=False)
        return self.recall_vector(qv, k, kinds, min_similarity, exclude, verified_only)

    def recall_vector(self, qv: Vector, k: int = 5, kinds: tuple[str, ...] | None = None,
                      min_similarity: float = 0.02, exclude: set[str] | None = None,
                      verified_only: bool = False) -> list[Recollection]:
        """Vector search, then re-rank by credibility and recency.

        The over-fetch is the important line. An approximate index returns the
        top-k by distance alone, so a verified memory that ranks 12th by cosine
        is invisible to a re-ranker that only ever sees the top 5 -- and the
        symptom is a store that quietly gets worse at surfacing its best material
        as it grows. Exact backends fetch exactly what is asked for.
        """
        p, now = self.policy, self.clock()
        # The backend that produced the QUERY vector, for the same reason writes
        # are tagged that way: a degraded provider embedder returns a lexical
        # vector, and a query embedded lexically must be compared against
        # lexically embedded traces. Reading `name` here made every trace written
        # during an outage invisible to every query made during the same outage.
        mine = getattr(self.embedder, "last_backend", None) or \
            getattr(self.embedder, "name", "unknown")
        fetch = k * OVERFETCH if self.store.approximate else 0
        candidates = self.store.search(qv, k=fetch, kinds=kinds,
                                       min_similarity=min_similarity,
                                       exclude=exclude, embedder=mine)
        hits: list[Recollection] = []
        for t, sim in candidates:
            if verified_only and not t.verified:
                continue
            cred = t.credibility(p)
            rec = t.recency(p, now)
            score = (max(sim, 1e-9) ** p.recall_similarity_weight
                     * max(cred, 1e-9) ** p.recall_credibility_weight
                     * max(rec, 1e-9) ** p.recall_recency_weight)
            hits.append(Recollection(t, sim, cred, rec, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:k]
        for h in top:
            h.trace.hits += 1
            h.trace.last_used = now
            self.store.touch(h.trace)
        return top

    def get(self, trace_id: str) -> Trace | None:
        return self.store.get(trace_id)

    def of_kind(self, kind: str) -> list[Trace]:
        return [t for t in self.store.all() if t.kind == kind]

    # -- structure ---------------------------------------------------------------

    def gaps(self, kinds: tuple[str, ...] | None = None, k: int = 3) -> list[dict]:
        """Regions the system has touched but not mastered.

        A gap is a neighbourhood with low mean credibility and high spread: many
        memories, disagreeing, none verified. That is not ignorance -- ignorance
        is empty space, and empty space is unreachable by definition. It is the
        far more actionable case of *confident inconsistency*, and it is where an
        experiment buys the most (8.1).
        """
        pool = [t for t in self.store.all() if not kinds or t.kind in kinds]
        if len(pool) < 3:
            return []
        out = []
        for seed in pool:
            near = self.recall_vector(seed.vector, k=6, kinds=kinds)
            if len(near) < 3:
                continue
            traces = [h.trace for h in near]
            cred = sum(t.credibility(self.policy) for t in traces) / len(traces)
            sprd = spread([t.vector for t in traces])
            out.append({"centre": seed, "members": traces, "credibility": cred,
                        "spread": sprd, "gap": (1.0 - cred) * (0.5 + sprd),
                        "vector": centroid([t.vector for t in traces])})
        out.sort(key=lambda g: g["gap"], reverse=True)
        deduped, taken = [], []
        for g in out:                       # one representative per neighbourhood
            if all(cosine(g["vector"], v) < 0.6 for v in taken):
                deduped.append(g)
                taken.append(g["vector"])
            if len(deduped) >= k:
                break
        return deduped

    def stats(self) -> dict:
        every = self.store.all()
        by_kind: dict[str, int] = {}
        for t in every:
            by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
        graded = [t for t in every if t.graded]
        verified = [t for t in graded if t.verified]
        return {
            "traces": len(every), "by_kind": by_kind,
            "graded": len(graded), "verified": len(verified),
            "mean_grade": round(sum(t.mean_grade for t in graded) / len(graded), 4) if graded else None,
            "vocabulary": getattr(getattr(self.embedder, "vocab", None), "docs", 0),
        }

    # -- persistence ---------------------------------------------------------------

    def save(self, path: Path) -> None:
        """JSONL, one trace per line, written through a temp file and renamed.

        Append-only would be cheaper, but grades mutate traces in place, so the
        file is rewritten. `os.replace` makes it atomic: a crash mid-write leaves
        the previous store intact rather than a half-truncated memory.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            vocab = getattr(self.embedder, "vocab", None)
            fh.write(json.dumps({"__header__": 1, "policy": self.policy.to_json(),
                                 "vocab": vocab.to_json() if vocab else None}) + "\n")
            for t in self.store.all():
                fh.write(json.dumps(t.to_json()) + "\n")
        tmp.replace(path)

    def load(self, path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        from .embed import Vocabulary
        count = 0
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("__header__"):
                    if d.get("vocab") and hasattr(self.embedder, "vocab"):
                        self.embedder.vocab = Vocabulary.from_json(d["vocab"])
                    if d.get("policy"):
                        self.saved_policy = Policy.from_json(d["policy"])
                    continue
                self.store.put(Trace.from_json(d))
                count += 1
        return count
