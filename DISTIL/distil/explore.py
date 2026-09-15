"""Curiosity, on purpose, with a ranking function.

Three claims, in order of how much they change the behaviour.

**1. Curiosity is a why-chain that did not bottom out.** The system interrogates
its own tasks (`challenge.interrogate`), and every chain terminating in ASSUMED
or CIRCULAR is an unexamined premise sitting in memory. Those are the first
place experiments come from, ahead of gaps and ahead of anything a model
suggests, because an unexamined premise is a *known* hole with a *known*
location. Asking why is not commentary here -- it is the experiment generator.

**2. Bad ideas are the high-information ones, and this is not a metaphor.** The
value of an experiment is the entropy of its outcome:

    H(p) = -p log2 p - (1-p) log2 (1-p),  maximised at p = 0.5

An idea you are sure will work teaches nothing. An idea you are sure will fail
teaches nothing *either*, which is the half that gets skipped. So candidates are
ranked by `H(p)/cost`, and the system deliberately manufactures ideas it expects
to fail about half the time -- inverting things it believes, over-generalising
what worked once, applying a tool outside its range. `GREN/DESIGN.md` 4 derives
the same 0.5 operating point for probe policies against a game oracle; this is
that result pointed at self-improvement instead.

**3. A failed experiment is a stored result, not a wasted cycle.** Every refuted
idea becomes a `Kind.FAILURE` trace with its diagnostic, which does three jobs:
it stops the idea being re-derived, it gives `payoff_matrix` evidence that pulls
similar plans down, and it is the raw material the mutation source recombines.
The store gets better by being wrong, on the record.

Self-upgrade is separate and duller by design (`upgrade`): propose a bounded
policy mutation, measure it against a held task set, keep it only if the
measured score improves, and let regret matching decide which parameter to try
next. Bounded, measured, reversible. The exciting kind of self-improvement is
`selfedit.py`, and it is guarded there.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .challenge import Ground, Persistence, interrogate
from .game import RegretMatching, information_gain
from .goals import checkability
from .grade import grade_python
from .memory import Kind, Source
from .policy import Policy
from .provider import Message
from .toolsmith import Toolsmith
from .vector import cosine


class Origin:
    """Where an idea came from. Tracked because the mix matters: a loop running
    only on `MUTATION` has stopped exploring and started grinding."""
    CURIOSITY = "curiosity"      # an unexamined premise from a why-chain
    GAP = "gap"                  # a low-credibility, high-spread neighbourhood
    MUTATION = "mutation"        # a recombination of past failures
    INVERSION = "inversion"      # the opposite of something believed
    ANALOGY = "analogy"          # what worked over there, tried over here
    PROPOSAL = "proposal"        # the model suggested it


@dataclass
class Idea:
    text: str
    origin: str
    p_success: float = 0.5
    cost: float = 1.0
    novelty: float = 1.0
    source_ids: list[str] = field(default_factory=list)

    @property
    def claim(self) -> bool:
        """Does this idea *assert* something, or propose *building* something?

        The distinction decides what counts as testing it. A proposal is tested
        by building the thing and checking it works. A claim is tested by a check
        that could come back false -- which is a different artefact, and one the
        rule-based synthesiser cannot write at all.
        """
        return self.origin in (Origin.CURIOSITY, Origin.INVERSION)

    def value(self, policy: Policy) -> float:
        """Bits per unit cost, scaled by novelty and by how close the prior sits
        to the policy's target operating point."""
        bits = information_gain(self.p_success, self.cost ** policy.cost_weight)
        focus = 1.0 - abs(self.p_success - policy.target_success)
        return bits * (self.novelty ** policy.novelty_weight) * max(focus, 0.05)

    def __str__(self) -> str:
        return f"[{self.origin}] {self.text}"


@dataclass
class Experiment:
    idea: Idea
    ran: bool
    passed: bool
    detail: str
    grade: float
    seconds: float
    trace_id: str | None = None

    def summary(self) -> str:
        verdict = "confirmed" if self.passed else ("refuted" if self.ran else "not run")
        return f"{verdict:<10} {self.idea.origin:<10} {self.idea.text[:64]!r} -- {self.detail[:70]}"


class Explorer:
    def __init__(self, memory, provider, policy: Policy, workspace, toolsmith: Toolsmith,
                 toolbox, rng: random.Random | None = None) -> None:
        self.memory = memory
        self.provider = provider
        self.policy = policy
        self.workspace = workspace
        self.toolsmith = toolsmith
        self.toolbox = toolbox
        self.rng = rng or random.Random()
        self.regret = self._load_regret()
        self._observed: list[list[float]] = [[] for _ in _ORIGINS]

    # -- idea generation -------------------------------------------------------

    def curiosity(self, limit: int = 3) -> list[Idea]:
        """Unexamined premises, turned into experiments.

        Runs a why-chain over recent queries and goals. Where it bottoms out in
        ASSUMED or CIRCULAR, the terminal claim becomes an experiment: *establish
        whether this holds*. The prior is deliberately near 0.5 -- that is what
        "unexamined" means, and it puts these at the top of the ranking by
        construction rather than by a hand-set priority.
        """
        out: list[Idea] = []
        pool = (self.memory.of_kind(Kind.QUERY) + self.memory.of_kind(Kind.GOAL))[-12:]
        for trace in reversed(pool):
            if len(out) >= limit:
                break
            chain = interrogate(trace.text, self._ask, self.memory.embedder, max_depth=3)
            if chain.terminal in (Ground.ASSUMED, Ground.CIRCULAR) and chain.links:
                claim = chain.links[-1].answer
                out.append(Idea(
                    text=f"establish whether it holds that {claim[:120]}",
                    origin=Origin.CURIOSITY,
                    p_success=0.5 if chain.terminal == Ground.ASSUMED else 0.4,
                    cost=1.0, novelty=self._novelty(claim), source_ids=[trace.id]))
        return out

    def _ask(self, question: str, context: list[str]) -> str:
        hits = self.memory.recall(question, k=2, kinds=(Kind.ANSWER, Kind.FACT, Kind.FAILURE))
        if hits and hits[0].similarity > 0.55:
            return hits[0].trace.text
        try:
            return self.provider.complete([Message("user", "WHY:: " + question)],
                                          temperature=0.3, max_tokens=120)
        except Exception:
            return ""

    def gaps(self, limit: int = 2) -> list[Idea]:
        out = []
        for gap in self.memory.gaps(k=limit):
            out.append(Idea(
                text=f"resolve the disagreement around: {gap['centre'].text[:110]}",
                origin=Origin.GAP,
                p_success=max(0.25, min(0.75, gap["credibility"])),
                cost=1.2, novelty=0.5 + 0.5 * gap["spread"],
                source_ids=[t.id for t in gap["members"][:3]]))
        return out

    def mutations(self, limit: int = 2) -> list[Idea]:
        """Recombine past failures.

        Two things that did not work, crossed, are not obviously a third thing
        that will not work -- failures usually have different causes, and the
        combination tests whether either cause was the real one. The prior is set
        low (0.3) and honestly: most of these are bad. Bad at p=0.3 still carries
        0.88 bits.
        """
        failures = self.memory.of_kind(Kind.FAILURE)
        if len(failures) < 2:
            return []
        out = []
        for _ in range(limit):
            a, b = self.rng.sample(failures, 2)
            out.append(Idea(
                text=f"combine the approaches behind {_clip(a.text)} and {_clip(b.text)}",
                origin=Origin.MUTATION, p_success=0.3, cost=1.0,
                novelty=1.0 - cosine(a.vector, b.vector), source_ids=[a.id, b.id]))
        return out

    def inversions(self, limit: int = 2) -> list[Idea]:
        """Deliberately bad ideas: the negation of something well graded.

        This is the part that sounds like vandalism and is not. A belief the
        system holds with high credibility and has never tested the converse of
        is a belief it cannot distinguish from a convention. Inverting it is
        cheap and occasionally the inversion holds -- which is the only way that
        error ever gets found.
        """
        believed = [t for t in self.memory.store.all()
                    if t.graded and (t.mean_grade or 0) > 0.5 and t.kind in (Kind.FACT, Kind.TOOL, Kind.ANSWER)]
        out = []
        for trace in believed[-limit:]:
            out.append(Idea(
                text=f"test the opposite: what if it is false that {_clip(trace.text, 110)}",
                origin=Origin.INVERSION, p_success=0.25, cost=0.8,
                novelty=0.9, source_ids=[trace.id]))
        return out

    def analogies(self, limit: int = 3, band: tuple[float, float] = (0.22, 0.60)) -> list[Idea]:
        """Invention by transfer: what worked over there, tried over here.

        The mechanism is a similarity *band*, and the band is the whole idea.
        Two memories above ~0.6 are the same thing said twice -- transferring
        between them invents nothing. Two below ~0.22 have no shared structure to
        carry, so the transfer is noise dressed as creativity. The productive
        distance is the middle: near enough that the mapping is meaningful, far
        enough that the result is new. Retrieval-by-nearest-neighbour is
        structurally incapable of finding these, because it is built to return
        the top of the range and analogy lives in the middle of it.

        The source is always something that *worked* (positively graded) and the
        target is something unresolved. That direction matters: transferring a
        known-good approach onto an open problem is invention; transferring an
        open problem onto a solved one is just noise.

        The prior sits at 0.45 -- deliberately near the maximum-information
        point, because whether a transfer holds is genuinely unknown, which is
        exactly what makes these worth running.
        """
        every = self.memory.store.all()
        sources = [t for t in every
                   if (t.mean_grade or 0) > 0.4 and t.kind in (Kind.TOOL, Kind.FACT, Kind.SOLUTION, Kind.ANSWER)]
        targets = [t for t in every
                   if t.kind in (Kind.GOAL, Kind.QUERY, Kind.FAILURE, Kind.IDEA)
                   and (t.mean_grade or 0) <= 0.4]
        if not sources or not targets:
            return []
        lo, hi = band
        pairs = []
        for src in sources:
            for tgt in targets:
                sim = cosine(src.vector, tgt.vector)
                if lo <= sim <= hi:
                    pairs.append((sim, src, tgt))
        if not pairs:
            return []
        # Mid-band first: the most transferable distance, not the closest pair.
        mid = (lo + hi) / 2.0
        pairs.sort(key=lambda p: abs(p[0] - mid))
        out = []
        for sim, src, tgt in pairs[:limit]:
            out.append(Idea(
                text=f"apply what worked for {_clip(src.text, 70)} to {_clip(tgt.text, 70)}",
                origin=Origin.ANALOGY, p_success=0.45, cost=1.0,
                novelty=1.0 - sim, source_ids=[src.id, tgt.id]))
        return out

    def proposals(self, seed: str, limit: int = 3) -> list[Idea]:
        try:
            reply = self.provider.complete(
                [Message("system", "Propose experiments, including ones likely to fail. "
                                   "One per line."),
                 Message("user", f"EXPERIMENT:: {seed}")], temperature=0.9, max_tokens=250)
        except Exception:
            return []
        out = []
        for line in (reply or "").splitlines()[:limit]:
            text = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip()
            if len(text) > 8:
                out.append(Idea(text=text, origin=Origin.PROPOSAL, p_success=0.45,
                                cost=1.0, novelty=self._novelty(text)))
        return out

    def _novelty(self, text: str) -> float:
        """1 - similarity to the nearest thing already known. An idea the store
        has already seen is not an idea, whatever it costs to generate."""
        hits = self.memory.recall(text, k=1)
        return 1.0 - (hits[0].similarity if hits else 0.0)

    def brainstorm(self, seed: str | None = None, n: int = 6) -> list[Idea]:
        """All sources, ranked by bits per unit cost. Curiosity is not given a
        bonus -- it wins on the ranking when it deserves to, and when the store
        holds no unexamined premises it correctly yields to the other sources."""
        ideas = (self.curiosity() + self.gaps() + self.mutations()
                 + self.inversions() + self.analogies())
        if seed:
            ideas += self.proposals(seed)
        seen: list = []
        unique: list[Idea] = []
        for idea in ideas:
            v = self.memory.embedder.embed(idea.text, learn=False)
            if any(cosine(v, u) > 0.9 for u in seen):
                continue
            seen.append(v)
            unique.append(idea)
        unique.sort(key=lambda i: i.value(self.policy), reverse=True)
        return unique[:n]

    # -- running an experiment -------------------------------------------------

    def test(self, idea: Idea) -> Experiment:
        """Turn an idea into something that runs, then run it.

        An experiment is only an experiment if it can come back false, so the
        idea is first turned into a tool with contract tests. Where no tool can
        be written the result is recorded as `ran=False` -- untested, not
        confirmed, and the distinction is kept in the store rather than rounded
        off to a small positive.
        """
        started = time.monotonic()
        spec = self.toolsmith.forge(idea.text)
        if spec is not None and idea.claim and spec.built_by == "template":
            # A function template can only ever demonstrate that a function works.
            # It cannot refute "X is false" -- so pairing the two would record a
            # confirmation that nothing established. Offline, claims go untested,
            # and that is the truthful outcome rather than a convenient one.
            self.memory.remember(
                Kind.IDEA, idea.text,
                meta={"origin": idea.origin, "status": "claim needs a falsifiable check",
                      "p": idea.p_success}, links=idea.source_ids)
            return Experiment(idea, False, False,
                              "a claim needs a check that can come back false; "
                              "the rule synthesiser only writes functions",
                              0.0, time.monotonic() - started)
        if spec is not None and not self._tests_the_claim(idea, spec):
            # The synthesiser matched on words the idea happens to contain rather
            # than on what it asserts, so running it would "confirm" a claim it
            # has nothing to do with. An experiment tested by an unrelated check
            # is worse than an untested one: it manufactures evidence.
            self.memory.remember(
                Kind.IDEA, idea.text,
                meta={"origin": idea.origin, "status": "no relevant check", "p": idea.p_success},
                links=idea.source_ids)
            return Experiment(idea, False, False,
                              f"nearest runnable form ({spec.name}) does not test the claim",
                              0.0, time.monotonic() - started)
        if spec is None:
            trace = self.memory.remember(
                Kind.IDEA, idea.text,
                meta={"origin": idea.origin, "status": "untestable", "p": idea.p_success},
                links=idea.source_ids)
            return Experiment(idea, False, False, "no runnable form could be written", 0.0,
                              time.monotonic() - started, trace.id)
        grade = self.toolsmith.validate(spec)
        kept = self.toolsmith.register(spec)
        passed = grade.passed and kept
        kind = Kind.FACT if passed else Kind.FAILURE
        text = (f"experiment ({idea.origin}) {'held' if passed else 'refuted'}: {idea.text} "
                f"-- {grade.diagnostic}")
        trace = self.memory.remember(
            kind, text,
            meta={"origin": idea.origin, "p_prior": idea.p_success, "grade": grade.score,
                  "tool": spec.name, "bits": round(information_gain(idea.p_success), 3)},
            links=idea.source_ids, grade=grade.score, source=Source.SELF)
        return Experiment(idea, True, passed, grade.diagnostic, grade.score,
                          time.monotonic() - started, trace.id)

    def _tests_the_claim(self, idea: Idea, spec, floor: float = 0.25) -> bool:
        """Is the forged check actually about the idea?

        Cosine between the claim and what the tool advertises. The floor is low
        on purpose -- the question is whether the two are about the same subject
        at all, not whether they are paraphrases.
        """
        a = self.memory.embedder.embed(idea.text, learn=False)
        b = self.memory.embedder.embed(spec.embed_text(), learn=False)
        return cosine(a, b) >= floor

    def step(self, seed: str | None = None) -> Experiment | None:
        ideas = self.brainstorm(seed, n=self.policy.branch_factor)
        if not ideas:
            return None
        chosen = ideas[0]
        result = self.test(chosen)
        self._journal(result)
        self.regret.observe(self._utilities(ideas, chosen, result),
                            played=_origin_index(chosen.origin))
        self._save_regret()
        return result

    def loop(self, steps: int = 5, seed: str | None = None) -> list[Experiment]:
        return [e for e in (self.step(seed) for _ in range(steps)) if e is not None]

    def _utilities(self, ideas: list[Idea], chosen: Idea, result: Experiment) -> list[float]:
        """Payoff per idea *origin*, so regret matching learns which generator is
        earning its keep.

        Information, not success: an experiment that refutes something scores as
        well as one that confirms it. That is the point of the whole file -- a
        generator rewarded for being right would stop proposing the ideas worth
        running.

        Only the played arm's payoff is observed, so the others are credited with
        their own running mean. Scoring them zero instead would make every arm
        look bad the moment any arm did well, and the sampler would collapse onto
        whichever generator happened to fire first.
        """
        played = _origin_index(chosen.origin)
        gained = (information_gain(chosen.p_success) * chosen.novelty) if result.ran else -0.25
        self._observed[played].append(gained)
        util = []
        for i in range(len(_ORIGINS)):
            if i == played:
                util.append(gained)
            else:
                seen = self._observed[i]
                util.append(sum(seen) / len(seen) if seen else 0.0)
        return util

    # -- self-upgrade -----------------------------------------------------------

    def upgrade(self, tasks: list[str], trials: int = 6) -> dict:
        """Tune the policy against measured outcomes, one parameter at a time.

        The objective is `score()` below: how well graded recall ranks what the
        store already knows to be good. A mutation is kept only if it beats the
        incumbent on that measure. Coordinate-wise, bounded, and reversible --
        the boring kind of self-improvement, which is the kind that compounds
        instead of diverging.
        """
        best, best_score = self.policy, self.score(tasks)
        history = [{"policy": "incumbent", "score": round(best_score, 5)}]
        for _ in range(trials):
            candidate = best.mutate(self.rng)
            saved, self.memory.policy = self.memory.policy, candidate
            try:
                s = self.score(tasks)
            finally:
                self.memory.policy = saved
            changed = best.diff(candidate)
            history.append({"policy": {k: round(v[1], 4) for k, v in changed.items()},
                            "score": round(s, 5), "kept": s > best_score})
            if s > best_score:
                best, best_score = candidate, s
        moved = self.policy.diff(best)
        self.policy = best
        self.memory.policy = best
        best.save(self.workspace.policy)
        return {"score_before": round(history[0]["score"], 5), "score_after": round(best_score, 5),
                "changed": {k: (round(a, 4), round(b, 4)) for k, (a, b) in moved.items()},
                "history": history}

    def score(self, tasks: list[str], k: int = 3) -> float:
        """Rank-discounted **similarity-weighted** grade over a held task set.

        The weighting is not decoration, and the reason is worth recording: the
        first version of this function averaged the grades of the top hits and
        nothing else. Self-upgrade promptly drove `recall_similarity_weight` to
        its floor and scored *better*, because a recall that ignores the query
        returns the best-graded traces in the store for every task -- which is a
        perfect score and a useless memory.

        That is the metric-gaming failure appearing inside this package rather
        than in a paper about it, and the fix is to make the objective require
        both properties at once: each hit contributes `grade x similarity`, so a
        policy that surfaces well-graded irrelevance earns nothing, and an
        ungraded hit at rank 1 costs the slot. Anything found by recall but not
        graded counts as zero rather than being skipped, so padding the top-k
        with unverified material cannot raise the mean.
        """
        total, n = 0.0, 0
        for task in tasks:
            hits = self.memory.recall(task, k=k)
            for rank, h in enumerate(hits):
                g = h.trace.mean_grade or 0.0
                total += g * h.similarity / (1.0 + rank)
                n += 1
            n += max(0, k - len(hits))          # a slot recall could not fill is a zero
        return total / n if n else 0.0

    # -- bookkeeping -------------------------------------------------------------

    def _journal(self, result: Experiment) -> None:
        with self.workspace.journal.open("a") as fh:
            fh.write(json.dumps({
                "at": time.time(), "origin": result.idea.origin, "idea": result.idea.text,
                "ran": result.ran, "passed": result.passed, "grade": result.grade,
                "detail": result.detail[:300], "seconds": round(result.seconds, 3)}) + "\n")

    def _load_regret(self) -> RegretMatching:
        path = self.workspace.regret
        if path.exists():
            try:
                return RegretMatching.from_json(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError):
                pass
        return RegretMatching(list(_ORIGINS))

    def _save_regret(self) -> None:
        self.workspace.regret.write_text(json.dumps(self.regret.to_json(), indent=2))


_ORIGINS = (Origin.CURIOSITY, Origin.GAP, Origin.MUTATION, Origin.INVERSION,
            Origin.ANALOGY, Origin.PROPOSAL)


def _origin_index(origin: str) -> int:
    return _ORIGINS.index(origin) if origin in _ORIGINS else 0


def _clip(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")
