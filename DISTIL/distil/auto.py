"""The system running itself.

Every other entry point waits to be asked something. This one does not: it picks
its own next move, does it, grades the result and folds it back into the same
memory everything else reads from. Nothing here is a new capability -- it is the
existing ones, sequenced by something that has an opinion about what is worth
doing next.

**What makes it curious rather than merely busy.** A loop that does the same
thing every cycle is a cron job. Five moves compete for each cycle and the choice
is regret-matched (`game.RegretMatching`), so the mix is learned from what each
move actually returned rather than fixed in advance -- and because the system is
changing underneath it, no-regret learning is the right tool where a stationary
bandit would be wrong:

    QUESTION    take something it believes and ask why until the chain
                terminates, then attack the premises. The highest-value
                outcome is finding out that something it recorded is circular.
    EXPERIMENT  brainstorm and run one, ranked by information gain: the ideas
                worth running are the ones it cannot call, which is why bad
                ideas are ranked rather than avoided.
    BUILD       find a capability gap and forge a tool through the grader.
    CONSOLIDATE compress cold memory into digests that keep the detail.
    TUNE        re-fit its own policy against measured outcomes.
    PURSUE      set itself a new problem and actually solve it.
    THINK       cluster the embedding space and name what forms, writing the
                names back as new memories. The only move whose input is the
                *shape* of the store rather than its contents.

**Running dry.** The first five moves all draw from pools that empty: there is a
last unquestioned belief, a last known gap, a last cold cluster. When they are
gone every move returns "nothing to do" and the loop spins without stopping,
which is the worst of both -- it is neither working nor finished. PURSUE is the
answer: it invents a direction and runs the full solve loop on it, which produces
new frames, goals, tools and cases, and so refills the pools the other five draw
from. It is a normal competing move *and* it is forced after two barren cycles,
because a loop with nothing to do should change what it is doing rather than wait
to be told.

**What it is rewarded for.** Information, not success. A move that confirms what
was already believed scores near zero however cleanly it ran; a move that refutes
something scores highly. Rewarding correctness would teach it to stop proposing
the experiments worth running, which is the failure mode this whole design is
arranged against.

**What it may not do.** It cannot edit its own source -- `selfedit` is reachable
only from the command line, deliberately, and an unattended loop with write
access to its own grader is the one thing that makes every grade in the store
meaningless. It cannot spend unbounded time: every cycle is counted and the
caller holds the stop flag.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .challenge import Ground, challenge, interrogate
from .game import RegretMatching, entropy
from .memory import Kind, Source


class Move:
    QUESTION = "question"
    EXPERIMENT = "experiment"
    BUILD = "build"
    CONSOLIDATE = "consolidate"
    TUNE = "tune"
    PURSUE = "pursue"
    THINK = "think"
    ALL = (QUESTION, EXPERIMENT, BUILD, CONSOLIDATE, TUNE, PURSUE, THINK)


@dataclass
class Cycle:
    """One turn of the loop, and what it bought.

    `learned` is the honest measure: bits of information, not whether the move
    worked. `note` is what a person reads.
    """
    n: int
    move: str
    note: str
    learned: float = 0.0
    detail: dict = field(default_factory=dict)
    seconds: float = 0.0
    #: Nothing was found to do. Distinct from a move that ran and learned
    #: nothing: that is a result, this is an empty pool.
    barren: bool = False

    def to_json(self) -> dict:
        return {"n": self.n, "move": self.move, "note": self.note,
                "learned": round(self.learned, 4), "detail": self.detail,
                "barren": self.barren, "seconds": round(self.seconds, 3)}


class Auto:
    """The loop. One `Distil`, driven by its own choices.

    Not thread-safe and not meant to be: the caller holds the agent lock for the
    duration, exactly as every other endpoint does.
    """

    def __init__(self, agent, stop=None) -> None:
        self.agent = agent
        self.stop = stop or (lambda: False)
        self.regret = RegretMatching(list(Move.ALL))
        # What each move has actually paid, and how often it has been tried.
        # `RegretMatching.cumulative` is the running sum of STRATEGY weights --
        # it is what `average_strategy` divides, not a payoff history -- so the
        # counterfactual below keeps its own.
        self.paid = [0.0] * len(Move.ALL)
        self.tried = [0] * len(Move.ALL)
        self.n = 0
        self.asked: set[str] = set()
        self.attempted: set[str] = set()
        self.directions: list[str] = []
        self.barren = 0
        #: How many chains have already bottomed out the same way. The first
        #: circular belief is a discovery; the tenth is a pattern you already
        #: know, and paying full price for it would pin the loop on one move.
        self.terminals: dict[str, int] = {}

    # -- the loop ------------------------------------------------------------

    def run(self, cycles: int = 0, on_cycle=None):
        """Run until `cycles` is reached, or forever while `cycles` is 0.

        `on_cycle` is called with each `Cycle` as it completes, which is how the
        frontend watches. An unbounded run is deliberate: the interesting
        property of a self-directed loop is what it does on the thousandth turn,
        and a hard limit baked in here would just be a smaller number than the
        caller's.
        """
        out = []
        while not self.stop() and (cycles <= 0 or self.n < cycles):
            cycle = self.once()
            out.append(cycle)
            if on_cycle is not None:
                try:
                    on_cycle(cycle)
                except Exception:
                    on_cycle = None          # a dead watcher must not stop the loop
        return out

    #: Barren cycles tolerated before a new direction is forced. One is noise --
    #: a move can legitimately find its pool empty this turn and full the next.
    #: Two in a row means the loop has genuinely run out of things to do.
    PATIENCE = 2

    def once(self) -> Cycle:
        started = time.time()
        self.n += 1
        if self.barren >= self.PATIENCE:
            index, move = Move.ALL.index(Move.PURSUE), Move.PURSUE
        else:
            index = self.regret.sample(self.agent.rng)
            move = Move.ALL[index]
        try:
            cycle = getattr(self, f"_{move}")()
        except Exception as exc:
            # A move that throws is a result too: it is recorded, scored zero,
            # and the regret matcher stops choosing it. Crashing the loop over
            # one bad move would lose everything learned before it.
            cycle = Cycle(self.n, move, f"{move} failed: {type(exc).__name__}: {exc}")
        cycle.n, cycle.move = self.n, move
        cycle.seconds = time.time() - started
        self.barren = self.barren + 1 if cycle.barren else 0
        self._learn(index, cycle)
        self.agent.save()
        return cycle

    def _learn(self, index: int, cycle: Cycle) -> None:
        """Regret is measured against what the other moves would plausibly have
        returned. Their counterfactual is the running average of what they *have*
        returned, which is the standard substitute when you cannot replay the
        world."""
        self.paid[index] += cycle.learned
        self.tried[index] += 1
        utilities = [cycle.learned if i == index else self._average(i)
                     for i in range(len(Move.ALL))]
        self.regret.observe(utilities, played=index)

    def _average(self, index: int) -> float:
        """What this move has paid on average, or an optimistic prior.

        Untried moves are given the best average seen so far rather than zero, so
        the loop explores all five before settling. Zero would let one lucky first
        cycle starve the other four permanently.
        """
        if self.tried[index]:
            return self.paid[index] / self.tried[index]
        seen = [self.paid[i] / self.tried[i] for i in range(len(Move.ALL)) if self.tried[i]]
        return max(seen) if seen else 0.5

    # -- the moves -----------------------------------------------------------

    def _question(self) -> Cycle:
        """Ask why about something it believes, then attack the premise.

        Subjects are drawn from what it has recorded and not yet questioned, so
        the loop works through its own beliefs rather than re-interrogating the
        same convenient one.
        """
        subject = self._unquestioned()
        if subject is None:
            return Cycle(self.n, Move.QUESTION, "nothing left unquestioned right now", barren=True)
        self.asked.add(subject)
        chain = interrogate(subject, self._answer, self.agent.memory.embedder, max_depth=4)
        attacks = challenge(subject, self.agent.memory, limit=3)

        # A chain that bottoms out in an assumption or a circle found something:
        # a belief resting on nothing. One that reaches ground confirmed what was
        # already believed, which is worth much less.
        shaky = chain.terminal in (Ground.ASSUMED, Ground.CIRCULAR)
        seen = self.terminals.get(chain.terminal, 0)
        self.terminals[chain.terminal] = seen + 1
        # Diminishing returns on a finding already made. Without this the offline
        # provider -- which cannot really answer "why", so every chain comes back
        # circular -- made QUESTION score identically forever and starve the rest.
        novelty = 1.0 / (1.0 + seen)
        learned = ((0.8 if shaky else 0.15) + sum(a.value for a in attacks[:1])) * novelty
        self.agent.memory.remember(
            Kind.FACT if not shaky else Kind.FAILURE,
            f"why {subject[:70]!r} -> {chain.terminal} after {chain.depth} question(s)"
            + (f"; sharpest attack: {attacks[0]}" if attacks else ""),
            meta={"auto": Move.QUESTION, "terminal": chain.terminal, "depth": chain.depth})
        return Cycle(self.n, Move.QUESTION,
                     f"asked why about {subject[:60]!r} — {chain.terminal} after "
                     f"{chain.depth} question(s)"
                     + (", so it rests on nothing checked" if shaky else ""),
                     learned=learned,
                     detail={"subject": subject, "terminal": chain.terminal,
                             "depth": chain.depth,
                             "chain": [{"q": w.question, "a": w.answer, "ground": w.ground}
                                       for w in chain.links],
                             "attacks": [{"text": str(a), "bits": round(a.value, 4)}
                                         for a in attacks]})

    def _experiment(self) -> Cycle:
        """Run one experiment, chosen for what it would teach."""
        result = self.agent.explorer.step()
        if result is None:
            return Cycle(self.n, Move.EXPERIMENT, "no idea worth running this cycle", barren=True)
        # Bits are what the idea was worth before it ran: an idea you could call
        # in advance teaches nothing whichever way it lands.
        bits = entropy(result.idea.p_success)
        surprise = 1.0 if result.ran and not result.passed else 0.35
        return Cycle(self.n, Move.EXPERIMENT,
                     f"{result.idea.origin}: {result.idea.text[:70]} — "
                     + ("not runnable" if not result.ran
                        else "held" if result.passed else "refuted, which is the useful outcome"),
                     learned=bits * surprise if result.ran else 0.0,
                     detail={"idea": result.idea.text, "origin": result.idea.origin,
                             "ran": result.ran, "passed": result.passed,
                             "bits": round(bits, 4), "detail": result.detail})

    def _build(self) -> Cycle:
        """Forge a tool for something it cannot currently do."""
        gap = self._gap()
        if gap is None:
            return Cycle(self.n, Move.BUILD, "no capability gap it knows how to close", barren=True)
        spec = self.agent.toolsmith.forge(gap)
        if spec is None:
            self.agent.memory.remember(
                Kind.FAILURE, f"no tool could be written for {gap[:70]!r}",
                meta={"auto": Move.BUILD}, grade=-0.3, source=Source.SELF)
            return Cycle(self.n, Move.BUILD, f"could not write anything for {gap[:60]!r}",
                         learned=0.25, detail={"goal": gap})
        grade = self.agent.toolsmith.validate(spec, self.agent.toolbox)
        kept = self.agent.toolsmith.register(spec)
        return Cycle(self.n, Move.BUILD,
                     f"{'registered' if kept else 'rejected'} {spec.name} for {gap[:50]!r}",
                     learned=0.9 if kept else 0.4,
                     detail={"goal": gap, "name": spec.name, "registered": kept,
                             "grade": grade.score,
                             "stages": [{"name": s.name, "passed": s.passed} for s in grade.stages]})

    def _think(self) -> Cycle:
        """Cluster what it knows, name what forms, and remember the names.

        Every other move consumes memory or produces it from outside; this one
        produces memory *from memory*, by computing on the shape of the store. It
        is what makes memory grow denser rather than only longer: a cluster's
        centre is a point recall can reach a whole neighbourhood through, where
        before it had to be similar to one specific member.

        Rewarded by how much structure the concept actually carries -- a tight
        cluster of many memories is worth more than a loose cluster of three,
        because it is a stronger claim that those memories belong together.
        """
        from .think import Thinker
        thinker = Thinker(self.agent.memory, self.agent.policy, self.agent.rng,
                          compressor=self.agent.compressor)
        found = thinker.think(limit=3)
        if not found:
            return Cycle(self.n, Move.THINK,
                         "no structure in what it knows worth naming yet", barren=True)
        written = thinker.absorb(found)
        # Cohesion says how tightly the cluster holds together; size says how
        # much it covers. Their product, normalised, is how much was learned.
        worth = sum(d.detail["cohesion"] * min(1.0, d.detail["size"] / 8.0)
                    for d in found) / len(found)
        return Cycle(self.n, Move.THINK,
                     f"named and compressed {len(found)} cluster(s) -- "
                     + _clip(found[0].text.splitlines()[0], 78),
                     learned=round(worth, 4),
                     detail={"found": [d.to_json() for d in found],
                             "skipped": thinker.skipped(),
                             "written": [t.id for t in written]})

    def _consolidate(self) -> Cycle:
        report = self.agent.compress(dry_run=False)
        return Cycle(self.n, Move.CONSOLIDATE,
                     f"{report['clusters']} cold cluster(s); compressed {report['compressed']}, "
                     f"folding {report['freed']} trace(s)",
                     learned=0.3 if report["compressed"] else 0.0,
                     barren=not report["compressed"],
                     detail={k: v for k, v in report.items() if k != "digests"})

    def _tune(self) -> Cycle:
        out = self.agent.upgrade(trials=4)
        changed = out.get("changed", {})
        return Cycle(self.n, Move.TUNE,
                     (f"re-fitted {len(changed)} policy value(s): "
                      + ", ".join(sorted(changed))) if changed
                     else "policy already the best of the trials",
                     learned=0.5 if changed else 0.05,
                     barren=not changed,
                     detail=out)

    def _pursue(self) -> Cycle:
        """Set itself a new problem, and do the work.

        This is the move that keeps the loop alive. Everything else consumes
        something already in the store; this produces -- a full `solve` writes a
        frame, an agenda, goals, a chain, usually a tool and a case, which is
        exactly what the other five had run out of.

        It runs with `ask=None`, so a direction it cannot state the objective of
        comes back gated rather than guessed at. That is not a failure: the gate
        firing on a question it set *itself* is the system telling us the
        direction was vague, and the next cycle picks a different one.
        """
        direction = self._direction()
        if direction is None:
            return self._bootstrap()
        self.directions.append(direction)
        result = self.agent.solve(direction, ask=None)

        if result.get("needs_clarification"):
            # It could not say what winning looks like. Recording that is worth
            # something -- it is a direction found to be unstated, not a dead end.
            return Cycle(self.n, Move.PURSUE,
                         f"took up {direction[:58]!r} and found it too vague to state: "
                         f"{result['reason']}",
                         learned=0.35,
                         detail={"direction": direction, "gated": True,
                                 "questions": [q.text for q in result.get("questions", [])]})
        solved = bool(result.get("solved"))
        self.agent.memory.remember(
            Kind.GOAL, f"took up on its own: {direction}",
            meta={"auto": Move.PURSUE, "solved": solved},
            grade=1.0 if solved else -0.25, source=Source.SELF)
        return Cycle(self.n, Move.PURSUE,
                     f"took up {direction[:58]!r} on its own and "
                     + ("solved it" if solved else f"did not: {result.get('reason')}"),
                     learned=0.9 if solved else 0.5,
                     detail={"direction": direction, "solved": solved,
                             "reason": result.get("reason"),
                             "attempts": result.get("attempts", [])})

    def _bootstrap(self) -> Cycle:
        """Nothing to pursue, because there is nothing to pursue it *from*.

        Every direction source reads memory, so an empty store suggests nothing
        and the loop would sit still on a fresh instance -- waiting to be told to
        do the one thing it can always do. Planting the starter kit is that
        thing: twelve tools, each through the grader, which is enough for the
        disagreement, unused-tool and composition sources to start producing.

        Only ever useful once. After it has run, the toolbox is non-empty and
        this returns barren, which is correct -- at that point the loop really
        has run out, and the caller decides what to do about it.
        """
        from .seed import plant
        if self.agent.toolbox.names():
            return Cycle(self.n, Move.PURSUE,
                         "nothing left to pursue -- every direction it can derive "
                         "has been taken", barren=True)
        report = plant(self.agent.toolsmith, self.agent.toolbox)
        planted = report["planted"]
        return Cycle(self.n, Move.PURSUE,
                     f"had nothing at all, so it planted {len(planted)} verified tool(s) "
                     f"to have something to reason from",
                     learned=0.8 if planted else 0.0,
                     barren=not planted,
                     detail={"bootstrap": True,
                             "planted": [name for name, _ in planted],
                             "rejected": [name for name, _ in report["rejected"]]})

    def _direction(self) -> str | None:
        """Invent something worth working on.

        Four sources, tried in order of how much they are grounded in what the
        system has actually run into. A disagreement between memories is the
        strongest: two things it believes that do not fit are a real problem
        whether or not anyone asked. An invented composition is the weakest, but
        it is still a task with a checkable outcome, which is the bar.

        Every candidate is checked against the directions already taken, because
        a loop that keeps re-proposing the same direction has not picked a new
        one.
        """
        for candidate in self._candidates():
            text = " ".join((candidate or "").split())
            if text and not self._already(text):
                return text
        return None

    def _already(self, candidate: str) -> bool:
        """Is this a direction it has effectively taken before?

        Not just string equality. Pursuing a direction runs the full solve loop,
        which writes goals, chains and solutions whose text *contains* that
        direction -- and those traces carry no `auto` marker, because the solver
        wrote them, not this loop. The next cycle then found one of them and
        proposed "work out what is true about work out what is true about ...".
        Checking the taken directions themselves catches every such descendant
        whoever wrote it.
        """
        if candidate in self.directions:
            return True
        # A candidate whose SUBJECT is itself phrased as a direction is a
        # descendant of one, whichever direction it came from. Solving "work out
        # what is true about X" writes traces reading "problem: ... work out what
        # is true about X ...", and proposing one back gave "work out what is
        # true about work out what is true about X". Comparing against each taken
        # direction misses this, because the descendant may belong to a different
        # one; the shape is the reliable signal.
        subject = _gist(candidate, words=200)
        if any(core in subject for core in LEAD_CORES):
            return True
        return any(_gist(taken) in candidate for taken in self.directions if _gist(taken))

    def _candidates(self):
        """Four sources, interleaved rather than exhausted in turn.

        Taking them in order meant the first source supplied every direction
        until it ran out, and since it draws on memory -- which this loop is
        busy filling -- it never did. Round-robin keeps the directions varied,
        which is the whole point of having four sources.
        """
        sources = [self._disagreements, self._unused_tools,
                   self._compositions, self._past_failures, self._far_pairs]
        # Rotate which source leads. Without this the first one supplied every
        # direction until it ran out -- and since it draws on memory, which this
        # loop is busy filling, it never did.
        lead = self.n % len(sources)
        rounds = [make() for make in sources[lead:] + sources[:lead]]
        while rounds:
            for source in list(rounds):
                found = next(source, None)
                if found is None:
                    rounds.remove(source)
                else:
                    yield found

    def _mine(self, trace) -> bool:
        """Did this loop write it?

        Directions built from its own output degenerate the same way the why-
        chains did: "work out what is true about work out what is true about".
        Anything the loop wrote carries an `auto` key.
        """
        return bool(trace is not None and trace.meta.get("auto"))

    def _disagreements(self):
        """Two memories that should agree and do not. The strongest source: a
        real problem whether or not anyone asked about it."""
        for idea in self.agent.explorer.gaps(limit=4):
            centre = idea.source_ids and self.agent.memory.store.get(idea.source_ids[0])
            if centre is not None and not self._mine(centre):
                yield f"work out what is actually true about {_subject(centre.text)}"

    def _unused_tools(self):
        """A capability that has never solved anything: find it a use, or
        establish that it does not have one."""
        for spec in self.agent.toolbox.all():
            if spec.transport == "python" and not spec.solved:
                yield f"find a real problem that {spec.name} solves, and solve it"

    def _compositions(self):
        """Two capabilities it has, joined into one it does not."""
        built = [s for s in self.agent.toolbox.all() if s.transport == "python"]
        for a, b in zip(built, built[1:]):
            yield (f"build something that uses both {a.name} and {b.name}: "
                   f"{a.purpose}, then {b.purpose}")

    def _past_failures(self):
        """Something it failed at before, revisited with more tools than it had.
        The cheapest new direction available."""
        for trace in self.agent.memory.of_kind(Kind.FAILURE):
            if not self._mine(trace):
                yield f"try again at what failed before: {_subject(trace.text)}"

    def _far_pairs(self):
        """The floor: two memories that have never been considered together.

        The other four sources all exhaust -- there is a last disagreement, a
        last unused tool -- and when they did, `_direction` returned None, PURSUE
        came back barren and the loop sat doing nothing, which is precisely the
        state this move exists to prevent. Pairs do not exhaust: there are
        O(n^2) of them and every cycle adds to n.

        Distant pairs on purpose. Two memories that are already similar suggest
        nothing that is not already implied; the inventive combination is the one
        between things filed far apart, which is what `explore.py` means by
        analogy.
        """
        from .vector import cosine
        traces = [t for t in self.agent.memory.store.all()
                  if t.kind in (Kind.TOOL, Kind.SOLUTION, Kind.FACT) and not self._mine(t)]
        if len(traces) < 2:
            return
        # Most credible first, so the pairing is between things it actually
        # relies on rather than between two half-refuted guesses.
        traces.sort(key=lambda t: t.credibility(self.agent.policy), reverse=True)
        for i, a in enumerate(traces):
            for b in traces[i + 1:]:
                if cosine(a.vector, b.vector) < 0.25:
                    yield (f"find what {_subject(a.text, 44)} and {_subject(b.text, 44)} "
                           f"have in common, and build whatever that suggests")

    # -- choosing what to work on --------------------------------------------

    def _unquestioned(self) -> str | None:
        """The most credible thing it has not yet asked why about.

        Most credible on purpose: questioning what it is least sure of is easy
        and teaches little. The expensive discovery is that something it relies
        on rests on nothing.
        """
        # Never question its own output. Each QUESTION writes a trace whose text
        # begins "why <subject>", and that trace was then eligible as the next
        # subject -- so the loop asked why about why about why, escaping
        # accumulating at each turn, and curiosity degenerated into self-
        # reference. Anything this loop wrote carries an `auto` key; that is what
        # excludes it, rather than a string test on "why", which would also
        # exclude a legitimate memory that happens to start with the word.
        pool = [t for t in self.agent.memory.store.all()
                if t.kind in (Kind.SOLUTION, Kind.FACT, Kind.TOOL, Kind.ANSWER)
                and t.text not in self.asked
                and not t.meta.get("auto")]
        if not pool:
            self.asked.clear()          # a second pass asks harder questions
            return None
        pool.sort(key=lambda t: t.credibility(self.agent.policy), reverse=True)
        return pool[0].text

    def _gap(self) -> str | None:
        """A capability some real task needed and nothing covers.

        Recomputed from the frames already in memory rather than read off them:
        a `Kind.GAME` trace stores the frame, not the agenda, and the agenda is
        what knows which actions are uncovered. Recomputing also means the answer
        tracks the toolbox as it grows, so a gap closed last cycle is not
        proposed again this one.

        Deliberately not `Explorer.gaps`, which is a different thing wearing a
        similar name: those are disagreements between memories, phrased as prose
        to investigate. Handing "resolve the disagreement around: tool X" to the
        toolsmith asked it to write a function whose name is an essay, and it
        correctly refused every time.
        """
        from .frame import GameFrame, agenda, capabilities
        for trace in self.agent.memory.of_kind(Kind.GAME):
            meta = trace.meta.get("frame")
            if not meta:
                continue
            try:
                frame = GameFrame(**{k: v for k, v in meta.items()
                                     if k in GameFrame.__annotations__})
                plan = agenda(frame, capabilities(frame, self.agent.memory, self.agent.toolbox))
            except Exception:
                continue
            for cap in plan.gaps:
                subject = (frame.objective or frame.task or "").strip()
                # The action is usually already a word in the objective, and
                # gluing them together gave the toolsmith goals like "compute
                # build a csv cleaner and compute the median".
                goal = (subject if cap.action.lower() in subject.lower()
                        else f"{cap.action} {subject}").strip()
                # `covered` holds TOOL names and `cap.action` is a verb, so they
                # are different namespaces and comparing them never matched --
                # the same gap came back every cycle and was re-forged and
                # re-rejected forever. What actually has to be remembered is
                # which goals this loop has already tried.
                if cap.action and goal not in self.attempted:
                    self.attempted.add(goal)
                    return goal
        return None

    def _answer(self, question: str, context: list[str]) -> str:
        """Answer a why-question from memory first, then the provider.

        Memory first because an answer it already holds is the one whose
        credibility is known; the provider is the fallback, and offline it is a
        rule-based decomposer that will often decline -- which is itself an
        honest terminal for the chain.
        """
        hits = self.agent.memory.recall(question, k=2,
                                        kinds=(Kind.ANSWER, Kind.FACT, Kind.FAILURE))
        if hits and hits[0].similarity > 0.55:
            return hits[0].trace.text
        from .provider import Message, ProviderError
        try:
            return self.agent.provider.complete([Message("user", "WHY:: " + question)],
                                                temperature=0.3, max_tokens=150)
        except ProviderError:
            return ""

    def strategy(self) -> dict:
        """The learned mix over moves, for the frontend to show."""
        return dict(zip(Move.ALL, [round(x, 4) for x in self.regret.average_strategy()]))


#: How each source phrases a direction. Stripped before comparing, because every
#: direction from one source shares its lead-in and a naive match would treat
#: them all as the same direction.
LEAD_INS = ("work out what is actually true about ",
            "find a real problem that ",
            "build something that uses both ",
            "find what ",
            "try again at what failed before: ")

#: The same phrases without their trailing connector. A trace written while
#: pursuing a direction quotes the phrasing but not always the preposition --
#: "…work out what is actually true solution: none" rather than "…true about X"
#: -- so matching the full lead-in missed exactly the descendants this is for.
LEAD_CORES = ("work out what is actually true",
              "find a real problem that",
              "build something that uses both",
              "try again at what failed before")


def _gist(direction: str, words: int = 7) -> str:
    """What a direction is actually *about*, for spotting its own descendants.

    Pursuing a direction writes traces containing it, under any lead-in: a
    failure recorded while pursuing "work out what is true about X" reads "refused
    while pursuing X", and proposing that as "try again at X" is the same
    direction wearing a different hat. Comparing the subject rather than the
    whole string catches it whichever source re-proposes it.
    """
    text = " ".join(direction.split())
    for lead in LEAD_INS:
        if text.startswith(lead):
            text = text[len(lead):]
            break
    return " ".join(text.split()[:words])


def _subject(text: str, limit: int = 70) -> str:
    """The readable head of a trace, for phrasing a direction around it.

    Traces carry signatures and source after their description; a direction built
    from the whole thing is a paragraph, and the frame confidence on a paragraph
    is meaningless.
    """
    head = " ".join((text or "").split())
    for cut in (": ", " -- ", "\n"):
        if cut in head:
            head = head.split(cut, 1)[-1] if head.startswith("tool ") else head
            break
    return head[:limit].strip()


def _clip(text: str, n: int = 60) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")
