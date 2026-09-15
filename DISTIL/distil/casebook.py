"""Problems, and what actually solved them.

The tool registry answers "what can I do?". This answers the question that comes
up far more often: **"what did I do last time something looked like this?"** --
which is case-based reasoning, and it is the oldest idea in this package and the
one that earns its keep fastest.

A `Case` pairs a problem with the solution that worked, the evidence that it
worked, and what it cost. Both halves are embedded into one vector, so a case is
retrievable from either end: by a problem that resembles the problem, or by a
solution that resembles the solution. The second direction is not decoration --
"where else have I used this approach?" is how a fix gets applied to the four
other places it applies to.

Two properties are enforced because without them a casebook rots:

**A case records the grade it was given, and recall is ranked by it.** A solution
that was tried and refuted is stored as a case with a negative grade. It stays
retrievable, ranked below the working ones, and it is the difference between
"nobody has tried that" and "that has been tried and it does not work" -- the
second is worth far more and is what an ungraded store throws away.

**A retrieved case reports what differs.** `adapt` returns the precedent
alongside the tokens present in the new problem and absent from the old one. A
precedent applied without noticing what changed is how the right answer to last
month's question becomes this month's outage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import hashlib

from .embed import tokenise
from .memory import Kind, Source
from .vector import cosine


@dataclass
class Case:
    problem: str
    solution: str
    grade: float = 0.0
    via: str = ""                     # the tool or method that carried it
    evidence: str = ""                # why we believe it worked
    cost: float = 1.0
    trace_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def embed_text(self) -> str:
        """Both halves in one vector, each labelled. The labels matter: without
        them "parse" in the problem and "parse" in the solution collapse into one
        another and the case stops being retrievable by approach."""
        parts = [f"problem: {self.problem}", f"solution: {self.solution}"]
        if self.via:
            parts.append(f"via: {self.via}")
        if self.evidence:
            parts.append(f"evidence: {self.evidence}")
        if self.tags:
            parts.append("tags: " + ", ".join(self.tags))
        return "\n".join(parts)

    def to_meta(self) -> dict:
        # `trace_id` is what makes a case gradeable after the fact. Dropping it
        # meant every case returned by `Casebook.all()` had trace_id None and
        # could never be re-graded by anyone reading them back.
        return {"problem": self.problem, "solution": self.solution, "grade": self.grade,
                "via": self.via, "evidence": self.evidence, "cost": self.cost,
                "tags": self.tags, "trace_id": self.trace_id}

    @classmethod
    def from_meta(cls, meta: dict) -> "Case":
        return cls(problem=meta.get("problem", ""), solution=meta.get("solution", ""),
                   grade=meta.get("grade", 0.0), via=meta.get("via", ""),
                   evidence=meta.get("evidence", ""), cost=meta.get("cost", 1.0),
                   tags=meta.get("tags", []), trace_id=meta.get("trace_id"))


@dataclass
class Precedent:
    case: Case
    similarity: float
    score: float
    novel_terms: list[str]           # in the new problem, absent from the old one

    @property
    def worked(self) -> bool:
        return self.case.grade > 0

    def render(self) -> str:
        verdict = "worked" if self.worked else "did NOT work"
        line = (f"  {self.score:.3f} ({verdict}, sim {self.similarity:.2f}) "
                f"{self.case.problem[:60]!r}\n    -> {self.case.solution[:90]}")
        if self.novel_terms:
            line += f"\n    but this time also involves: {', '.join(self.novel_terms[:6])}"
        return line


class Casebook:
    def __init__(self, memory) -> None:
        self.memory = memory

    def record(self, problem: str, solution: str, grade: float, via: str = "",
               evidence: str = "", cost: float = 1.0, links: list[str] | None = None,
               source: str = Source.SELF, tags: list[str] | None = None) -> Case:
        """File a case. Failures are filed too, and that is the point."""
        case = Case(problem=problem, solution=solution, grade=grade, via=via,
                    evidence=evidence, cost=cost, tags=tags or [])
        # A case is identified by the pair, never by similarity: "pad the short
        # rows" and "drop the short rows" answer the same problem and must stay
        # two records, or the store forgets which one worked.
        key = hashlib.blake2b(f"{problem}\x00{solution}".encode(), digest_size=8).hexdigest()
        trace = self.memory.remember(Kind.SOLUTION, case.embed_text(), meta=case.to_meta(),
                                     links=links, grade=grade, source=source, identity=key)
        case.trace_id = trace.id
        trace.meta["trace_id"] = trace.id     # the id is only known after the write
        self.memory.store.touch(trace)
        return case

    def precedents(self, problem: str, k: int = 3, worked_only: bool = False) -> list[Precedent]:
        """Two stages: recall on the whole case, re-rank on the problem half.

        A case embeds problem *and* solution in one vector so it is reachable
        from either end. That blend is wrong for this query, though: asked "what
        looks like this problem?", a case whose solution happens to share wording
        with the question outranks a case whose *problem* is the same one. The
        first version of this method did exactly that and put an unrelated JSON
        case above the CSV case it was asked about.

        So the blended vector fans out (it decides what is even a candidate), and
        the ranking is then similarity to the recorded problem alone, multiplied
        by the credibility the grading layer already computed. Standard two-stage
        retrieval, and the reason it is here is a wrong answer rather than a
        preference.
        """
        candidates = self.memory.recall(problem, k=max(k * 4, 12), kinds=(Kind.SOLUTION,))
        if not candidates:
            return []
        qv = self.memory.embedder.embed(problem, learn=False)
        new_terms = set(tokenise(problem))
        scored: list[Precedent] = []
        for hit in candidates:
            case = Case.from_meta(hit.trace.meta)
            case.trace_id = hit.trace.id
            if worked_only and case.grade <= 0:
                continue
            problem_sim = cosine(qv, self.memory.embedder.embed(case.problem, learn=False))
            credibility = hit.trace.credibility(self.memory.policy)
            old_terms = set(tokenise(case.problem))
            novel = sorted(t for t in (new_terms - old_terms) if not t.startswith("#"))
            scored.append(Precedent(case, problem_sim, problem_sim * credibility, novel))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:k]

    def by_solution(self, approach: str, k: int = 3) -> list[Precedent]:
        """The other direction: where else has this approach been used?

        Same two stages, re-ranked on the solution half. This is the query that
        turns one fix into every place it applies.
        """
        candidates = self.memory.recall(approach, k=max(k * 4, 12), kinds=(Kind.SOLUTION,))
        qv = self.memory.embedder.embed(approach, learn=False)
        out = []
        for hit in candidates:
            case = Case.from_meta(hit.trace.meta)
            case.trace_id = hit.trace.id
            sim = cosine(qv, self.memory.embedder.embed(case.solution, learn=False))
            out.append(Precedent(case, sim, sim * hit.trace.credibility(self.memory.policy), []))
        out.sort(key=lambda p: p.score, reverse=True)
        return out[:k]

    def adapt(self, problem: str) -> dict:
        """The best precedent, what it cost, and what is different this time.

        Returns the failures as well as the winner. A caller that only sees the
        best match will cheerfully re-run an approach that has already been
        refuted twice, because the refutations ranked fourth.
        """
        found = self.precedents(problem, k=5)
        if not found:
            return {"precedent": None, "avoid": [], "note": "no comparable case on record"}
        worked = [p for p in found if p.worked]
        failed = [p for p in found if not p.worked]
        best = worked[0] if worked else None
        return {
            "precedent": best,
            "avoid": failed,
            "note": ("no case on record worked; "
                     f"{len(failed)} known-bad approach(es) to avoid" if best is None
                     else f"closest working case differs on: "
                          f"{', '.join(best.novel_terms[:5]) or 'nothing lexical'}"),
        }

    def all(self) -> list[Case]:
        return [Case.from_meta(t.meta) for t in self.memory.of_kind(Kind.SOLUTION)]
