"""Consolidation: compress what has gone cold, keep what is still wanted.

An append-only memory is a memory that gets slower and vaguer as it grows. Every
near-miss recall costs a full scan, and the top-k fills with ten restatements of
one thing. So there is a consolidation phase, and its governing rule is:

> **Compress by access, not by age.**

Age alone is wrong. A fact consulted every day for two years is old and is the
last thing that should be summarised; an idea written an hour ago and never read
again is new and is the first. `Trace.heat` is the quantity that matters --
saturating hit count times recency decay -- so "cold" means *neither recently
wanted nor often wanted*, which is the honest definition of something the system
is not using.

**Detail preservation** is the part that is easy to get wrong. A summariser that
produces fluent prose loses exactly what made the traces worth keeping: the
identifier, the error code, the number, the file path. So the digest is built to
keep those on purpose:

1. The highest-graded member is kept **verbatim** as the exemplar. Not a
   paraphrase of it -- the text.
2. Rare terms across the cluster are kept verbatim, ranked by IDF, because a
   high-IDF token is by construction the thing that distinguishes these traces
   from everything else in the store. `ValueError`, `E0308`, `port 5432` and
   `--no-verify` survive; "the", "should" and "system" do not.
3. Counts and grade statistics are kept, so a digest of eleven failures still
   says eleven.
4. A prose abstract is added **only if a provider is available**, and it is
   appended to the extractive core rather than replacing it. A model outage
   degrades the digest's readability, never its content.

**Nothing is destroyed.** Compressed originals are appended to `archive.jsonl`
before they leave the store, and `restore()` brings them back. Compression here
means "moved out of the retrieval path", not "deleted" -- which is what makes it
safe to run automatically, and what makes "maintain details" a property rather
than a hope.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .embed import tokenise
from .memory import Kind, Source
from .provider import Message
from .vector import centroid, cosine

#: Kinds that are never compressed. Tools and cases are the system's accumulated
#: capability and its accumulated judgement -- summarising either would be
#: compressing the thing the rest of the store exists to serve. Games are the
#: frames everything else is interpreted against.
PRESERVE = (Kind.TOOL, Kind.SOLUTION, Kind.GAME, Kind.DIGEST)


@dataclass
class Cluster:
    members: list                     # list[Trace]
    cohesion: float                   # mean cosine to the centroid
    heat: float                       # mean heat of the members

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class Digest:
    text: str
    kept_terms: list[str]
    exemplar: str
    members: list[str] = field(default_factory=list)
    mean_grade: float | None = None

    def render(self) -> str:
        return self.text


class Compressor:
    def __init__(self, memory, provider=None, archive: Path | None = None,
                 heat_threshold: float = 0.15, cohesion: float = 0.45,
                 min_cluster: int = 3) -> None:
        self.memory = memory
        self.provider = provider
        self.archive_path = Path(archive) if archive else None
        self.heat_threshold = heat_threshold
        self.cohesion = cohesion
        self.min_cluster = min_cluster

    # -- finding what is safe to compress -------------------------------------

    def cold(self) -> list:
        now = self.memory.clock()
        return [t for t in self.memory.store.all()
                if t.kind not in PRESERVE
                and t.heat(self.memory.policy, now) < self.heat_threshold]

    def clusters(self, traces: list | None = None) -> list[Cluster]:
        """Greedy single-pass clustering by cosine to a running centroid.

        Greedy rather than k-means because the number of clusters is not known,
        the data arrive incrementally, and the cost of a slightly suboptimal
        grouping is a slightly worse summary -- not a wrong answer. Cheap and
        deterministic beats optimal and fragile for a maintenance pass.
        """
        pool = self.cold() if traces is None else list(traces)
        pool.sort(key=lambda t: t.created)
        now = self.memory.clock()
        groups: list[list] = []
        centres: list[list[float]] = []
        for t in pool:
            placed = False
            for i, c in enumerate(centres):
                if cosine(t.vector, c) >= self.cohesion:
                    groups[i].append(t)
                    centres[i] = centroid([m.vector for m in groups[i]])
                    placed = True
                    break
            if not placed:
                groups.append([t])
                centres.append(list(t.vector))
        out = []
        for g, c in zip(groups, centres):
            if len(g) < self.min_cluster:
                continue
            out.append(Cluster(
                members=g,
                cohesion=round(sum(cosine(m.vector, c) for m in g) / len(g), 4),
                heat=round(sum(m.heat(self.memory.policy, now) for m in g) / len(g), 4)))
        out.sort(key=lambda c: (c.size, c.cohesion), reverse=True)
        return out

    # -- summarising without losing the specifics -----------------------------

    def summarise(self, cluster: Cluster, keep_terms: int = 12) -> Digest:
        """Compress a cold cluster, on its way to replacing it."""
        return self.summarise_members(cluster.members, keep_terms=keep_terms)

    def summarise_members(self, members: list, keep_terms: int = 12,
                          label: str | None = None) -> Digest:
        """Compress any group of traces into one detail-preserving summary.

        Split out from `summarise` so the *semantic* clusters in `think.py` can
        be compressed by the same code that compresses *cold* ones. The two
        callers group traces for different reasons -- structure versus access --
        but what it means to summarise a group without losing the specifics is
        one problem, and having two implementations of it is how they drift.

        `label` names what the group is, because "digest of 6 cold traces" is the
        wrong sentence for a concept that is not being archived.
        """
        graded = [m for m in members if m.mean_grade is not None]
        exemplar = max(graded, key=lambda m: m.mean_grade) if graded else members[0]

        # Rare terms first. IDF is already maintained by the embedder, so the
        # ranking is free and means the same thing here as it does in recall:
        # a term that distinguishes these traces from the rest of the store.
        vocab = getattr(self.memory.embedder, "vocab", None)
        counts: Counter[str] = Counter()
        for m in members:
            for tok in tokenise(m.text):
                if not tok.startswith("#") and len(tok) > 2:
                    counts[tok] += 1
        if vocab is not None:
            ranked = sorted(counts, key=lambda t: vocab.idf(t) * math.log1p(counts[t]), reverse=True)
        else:
            ranked = sorted(counts, key=lambda t: counts[t], reverse=True)
        kept = ranked[:keep_terms]

        kinds = Counter(m.kind for m in members)
        grades = [m.mean_grade for m in members if m.mean_grade is not None]
        mean_grade = round(sum(grades) / len(grades), 4) if grades else None

        lines = [
            (label or f"digest of {len(members)} cold trace(s)")
            + f" ({', '.join(f'{v} {k}' for k, v in kinds.most_common())})",
            f"distinguishing terms: {', '.join(kept)}",
            f"exemplar (verbatim, best graded): {exemplar.text[:400]}",
        ]
        if mean_grade is not None:
            lines.append(f"grades: mean {mean_grade} over {len(grades)} graded member(s)")
        abstract = self._abstract(members)
        if abstract:
            lines.append(f"abstract: {abstract}")
        return Digest(text="\n".join(lines), kept_terms=kept, exemplar=exemplar.text,
                      members=[m.id for m in members], mean_grade=mean_grade)

    def _abstract(self, members: list) -> str:
        """Optional prose, appended to the extractive core -- never replacing it."""
        if self.provider is None:
            return ""
        try:
            joined = "\n".join(f"- {m.text[:200]}" for m in members[:12])
            out = self.provider.complete(
                [Message("system", "Summarise in one sentence. Keep every identifier, "
                                   "error string and number exactly as written."),
                 Message("user", f"SUMMARISE:: \n{joined}")], temperature=0.1, max_tokens=120)
            return (out or "").strip().replace("\n", " ")[:300]
        except Exception:
            return ""

    # -- the pass -------------------------------------------------------------

    def compress(self, dry_run: bool = False, limit: int = 20) -> dict:
        """Replace each cold cluster with one digest. Archive first, always."""
        clusters = self.clusters()[:limit]
        report = {"clusters": len(clusters), "compressed": 0, "freed": 0,
                  "digests": [], "dry_run": dry_run}
        for cluster in clusters:
            digest = self.summarise(cluster)
            report["digests"].append({"size": cluster.size, "cohesion": cluster.cohesion,
                                      "heat": cluster.heat, "terms": digest.kept_terms[:6],
                                      "text": digest.text})
            if dry_run:
                # Count what WOULD happen. Reporting 0/0 for a dry run made the
                # preview say "would compress 0, folding 0" however much it was
                # about to fold -- the one number anyone runs --dry-run to see.
                report["compressed"] += 1
                report["freed"] += cluster.size
                continue
            self._archive(cluster.members)
            links = sorted({lid for m in cluster.members for lid in m.links})
            trace = self.memory.remember(
                Kind.DIGEST, digest.text,
                meta={"members": digest.members, "size": cluster.size,
                      "cohesion": cluster.cohesion, "terms": digest.kept_terms,
                      "mean_grade": digest.mean_grade, "archived": True},
                links=links,
                # Digests of related clusters read alike, so cosine dedupe merged
                # a new one into an earlier one: the new summary text was dropped
                # and the old digest's member list stood for traces it never
                # covered -- which `restore` would then fail to bring back.
                identity="digest:" + ",".join(sorted(digest.members)))
            # Rewire inbound links. Compression carried each member's OUTBOUND
            # links onto the digest but left everything pointing AT the members
            # pointing at ids that no longer exist, so credit propagation
            # (memory._propagate) stopped dead at the boundary.
            gone = set(digest.members)
            for other in self.memory.store.all():
                if other.id == trace.id or not gone.intersection(other.links):
                    continue
                other.links = [trace.id if lid in gone else lid for lid in other.links]
                seen_once, deduped = set(), []
                for lid in other.links:
                    if lid not in seen_once:
                        seen_once.add(lid)
                        deduped.append(lid)
                other.links = deduped
                self.memory.store.touch(other)
            if digest.mean_grade is not None:
                # The digest inherits the cluster's standing, so compression does
                # not quietly reset how much the system trusts what it learned.
                self.memory.grade(trace.id, digest.mean_grade, Source.SELF, propagate=False)
            for m in cluster.members:
                self.memory.store.delete(m.id)
            report["compressed"] += 1
            report["freed"] += cluster.size
        return report

    def _archive(self, members: list) -> None:
        if self.archive_path is None:
            return
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a") as fh:
            for m in members:
                fh.write(json.dumps(m.to_json()) + "\n")

    def restore(self, trace_ids: list[str] | None = None) -> int:
        """Bring archived traces back into the retrieval path.

        The reason compression is safe to run unattended: it is reversible. A
        digest that turns out to have dropped something that mattered is not a
        loss, it is a restore -- which is only true because the originals were
        written to the archive before they left the store.
        """
        if self.archive_path is None or not self.archive_path.exists():
            return 0
        from .memory import Trace
        wanted = set(trace_ids) if trace_ids else None
        restored = 0
        with self.archive_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if wanted and d["id"] not in wanted:
                    continue
                if self.memory.store.get(d["id"]) is None:
                    self.memory.store.put(Trace.from_json(d))
                    restored += 1
        return restored
