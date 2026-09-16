"""The chain of thought, as a sequence of typed steps rather than a paragraph.

Free-form chain-of-thought has one property that disqualifies it here: it cannot
be checked. It is prose that narrates a decision already made, and when it is
wrong it is wrong persuasively. Every step in this file is a typed object with
its inputs attached, which buys three things prose does not:

- **Replay.** A chain can be re-executed against a changed memory, and the step
  where the answer diverges is the step that mattered.
- **Retrieval.** Chains are embedded into memory as `Kind.CHAIN`, so the system
  recalls *how it reasoned* about a similar task, not only what it concluded.
  This is the part that makes the memory layer more than a passage store.
- **Attribution.** SELECT carries the payoff matrix it chose from. "Why that
  goal?" has a numeric answer.

The spine is `FRAME -> AGENDA -> PRECEDENT -> WHY -> CHALLENGE -> RETRIEVE ->
DISTILL -> PAYOFF -> SELECT -> ACT -> VERIFY -> CREDIT`, and the order is the
argument. FRAME comes first because which solution concept is valid depends on
what game this is, and `PAYOFF`/`SELECT` below are the right machinery for
exactly one class of them (see `frame.py`). Questioning comes before
decomposition, because distilling a task nobody questioned produces a
beautifully organised tree of the wrong goals.

Goal selection (SELECT) is a normal-form game against Nature: rows are the goals
on the frontier, columns are the states the world might be in, entries are
payoffs from the role/state table below modulated by what memory says about
similar goals. The rule is `game.choose` -- a blend of expected utility and
minimax regret, set by `policy.regret_aversion`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import game
from .challenge import Ground, challenge, interrogate
from .goals import GoalTree, Status, Verifier, checkability
from .memory import Kind, Source
from .policy import Policy
from .provider import Message
from .vector import cosine


class Step:
    FRAME = "FRAME"             # what game is this? -- runs before everything
    AGENDA = "AGENDA"           # what the game needs vs what this system can do
    PRECEDENT = "PRECEDENT"     # what solved something like this before
    CHALLENGE = "CHALLENGE"     # question the premises before accepting the task
    WHY = "WHY"                 # chase the justification to its ground
    DISTILL = "DISTILL"         # task -> goals
    RETRIEVE = "RETRIEVE"       # what does memory already know?
    PAYOFF = "PAYOFF"           # build the matrix
    SELECT = "SELECT"           # choose, by a named solution concept
    ACT = "ACT"                 # do the thing
    VERIFY = "VERIFY"           # did it work? (the only opinion that counts)
    CREDIT = "CREDIT"           # who earned it -- Shapley over subgoals
    NOTE = "NOTE"


@dataclass
class Thought:
    kind: str
    text: str
    payload: dict = field(default_factory=dict)

    def render(self) -> str:
        return f"{self.kind:<9} {self.text}"


@dataclass
class Chain:
    task: str
    steps: list[Thought] = field(default_factory=list)

    def add(self, kind: str, text: str, **payload) -> Thought:
        t = Thought(kind, text, payload)
        self.steps.append(t)
        return t

    def of(self, kind: str) -> list[Thought]:
        return [s for s in self.steps if s.kind == kind]

    def render(self) -> str:
        return "\n".join([f"task: {self.task}"] + [f"  {s.render()}" for s in self.steps])

    def to_text(self) -> str:
        """The form written into memory. Keeps step kinds so a recalled chain is
        still structured, and stays short enough that the embedding is about the
        reasoning rather than about the volume of text."""
        return f"task: {self.task}\n" + "\n".join(f"{s.kind}: {s.text}" for s in self.steps)


# --------------------------------------------------------------------------- #
# the game against Nature
# --------------------------------------------------------------------------- #

class State:
    """Nature's moves: the ways a task turns out not to be what it looked like.
    Drawn from what actually goes wrong, not from a taxonomy."""
    AS_STATED = "as-stated"              # it means what it says
    UNDERSPECIFIED = "underspecified"    # a load-bearing term is undefined
    HARDER = "harder"                    # more work than it looks
    WRONG_FRAME = "wrong-frame"          # the stated goal is not the real goal
    TOOL_MISSING = "tool-missing"        # needs a capability that does not exist

    ALL = (AS_STATED, UNDERSPECIFIED, HARDER, WRONG_FRAME, TOOL_MISSING)


class Role:
    """What pursuing a goal actually buys. A goal's role decides how it pays off
    in each state, which is the entire content of the payoff table."""
    BUILD = "build"              # produce the artefact
    CLARIFY = "clarify"          # pin down a definition
    INTERROGATE = "interrogate"  # test whether the frame is right
    TOOL = "tool"                # build a capability
    CHECK = "check"              # establish whether something holds

    ALL = (BUILD, CLARIFY, INTERROGATE, TOOL, CHECK)


#: role x state -> payoff in [-1, 1]. Read a row as: "if the world is like this,
#: how much does doing this kind of work help?"
#:
#: The shape of the table is the claim. BUILD pays best when the task means what
#: it says and is actively harmful when the frame is wrong -- that is the cost of
#: building the right thing for the wrong problem, and it is the most expensive
#: mistake in the table. CLARIFY and INTERROGATE are cheap insurance: mildly
#: wasteful when everything is fine, decisive when it is not. CHECK never hurts,
#: which is why it is never the maximin loser and why a nervous system (high
#: regret_aversion) drifts toward verifying things.
PAYOFF = {
    #                AS_STATED  UNDERSPEC  HARDER  WRONG_FRAME  TOOL_MISSING
    Role.BUILD:        [ 0.90,   -0.30,    0.10,     -0.80,       -0.50],
    Role.CLARIFY:      [-0.20,    0.85,    0.15,      0.55,        0.00],
    Role.INTERROGATE:  [-0.30,    0.45,    0.20,      0.90,        0.20],
    Role.TOOL:         [ 0.20,   -0.10,    0.60,     -0.40,        0.95],
    Role.CHECK:        [ 0.35,    0.30,    0.40,      0.25,        0.10],
}

_ROLE_WORDS = {
    Role.CLARIFY: re.compile(r"\b(define|definition|clarify|specify|decide|choose|what counts|scope)\b", re.I),
    Role.INTERROGATE: re.compile(r"\b(why|question|challenge|establish whether|assumption|premise|is it)\b", re.I),
    Role.TOOL: re.compile(r"\b(tool|helper|library|utility|reusable|harness|wrapper)\b", re.I),
    Role.CHECK: re.compile(r"\b(verify|check|test|assert|validate|confirm|measure|benchmark)\b", re.I),
}


def role_of(text: str) -> str:
    for role, pattern in _ROLE_WORDS.items():
        if pattern.search(text):
            return role
    return Role.BUILD


# --------------------------------------------------------------------------- #

@dataclass
class Session:
    task: str
    chain: Chain
    tree: GoalTree
    chosen: object = None
    questions: list = field(default_factory=list)
    why: object = None
    trace_id: str | None = None
    frame: object = None          # the game, as understood before any reasoning
    agenda: object = None         # what it needs vs what this system can do
    precedent: dict = field(default_factory=dict)


class Reasoner:
    def __init__(self, memory, provider, policy: Policy | None = None, toolbox=None) -> None:
        self.memory = memory
        self.provider = provider
        self.policy = policy or Policy()
        self.toolbox = toolbox

    # -- 1. question it ------------------------------------------------------

    def interrogate_task(self, task: str, chain: Chain, depth: int = 4):
        """Ask why, and record where the chain bottomed out.

        An ASSUMED or CIRCULAR terminal is not a failure of the interrogation --
        it is its product. It says the task rests on something nobody checked,
        which raises the odds of WRONG_FRAME in the belief below and feeds
        `explore.py` an experiment.
        """
        def ask(question: str, context: list[str]) -> str:
            # Only kinds that can *contain* a justification. Recalling the stored
            # QUERY would hand the question back as its own answer, which the
            # circularity check then dutifully flags -- a true statement about a
            # fake chain, and the most confusing possible output.
            hits = self.memory.recall(question, k=2,
                                      kinds=(Kind.ANSWER, Kind.FACT, Kind.FAILURE))
            if hits and hits[0].similarity > 0.55:
                return hits[0].trace.text            # memory answers before the model does
            try:
                prompt = "WHY:: " + question + "\ncontext: " + " -> ".join(context[-2:])
                return self.provider.complete([Message("user", prompt)], temperature=0.3, max_tokens=120)
            except Exception:
                return ""

        why = interrogate(task, ask, self.memory.embedder, max_depth=depth)
        chain.add(Step.WHY, f"why-chain depth {why.depth}, terminal {why.terminal}",
                  ground=why.terminal, links=[w.answer for w in why.links])
        questions = challenge(task, self.memory, limit=self.policy.branch_factor)
        if questions:
            chain.add(Step.CHALLENGE,
                      f"{len(questions)} premise attack(s); sharpest: {questions[0].text[:80]}",
                      questions=[q.text for q in questions],
                      bits=[round(q.value, 3) for q in questions])
        return why, questions

    # -- 2. distil it --------------------------------------------------------

    def distill(self, task: str, chain: Chain | None = None, depth: int | None = None) -> GoalTree:
        """Recursive decomposition, stopping where a goal becomes verifiable."""
        depth = self.policy.max_depth if depth is None else depth
        tree = GoalTree(task)
        self._expand(tree, "root", task, depth, chain)
        if len(tree.goals) == 1:
            # Every proposal was filtered as a restatement of the parent, which is
            # the common case for a single-clause task -- and the threshold that
            # decides it moves with the vocabulary, so the same task can decompose
            # one day and not the next. Returning an empty tree made `solve`
            # report "nothing on the frontier to act on", which is false: a task
            # that resists decomposition is not unworkable, it is already atomic.
            prior = checkability(task)
            tree.add(task, verifier=self._verifier_for(task) if prior >= 0.6 else None,
                     cost=1.0, p_success=prior)
        if chain:
            chain.add(Step.DISTILL,
                      f"{len(tree.goals) - 1} goals, {len(tree.atoms())} verifiable, "
                      f"{len(tree.unverifiable())} still need a check",
                      goals=[g.text for g in tree.goals.values() if g.id != "root"])
        return tree

    def _expand(self, tree: GoalTree, parent_id: str, text: str, depth: int, chain) -> None:
        if depth <= 0:
            return
        for sub in self._propose(text)[:self.policy.branch_factor]:
            prior = checkability(sub)
            verifier = self._verifier_for(sub) if prior >= 0.6 else None
            goal = tree.add(sub, parent=parent_id, verifier=verifier,
                            cost=1.0 + 0.5 * (1.0 - prior), p_success=prior)
            # Stop at a verifiable goal. Keep going where nothing can be checked --
            # that is precisely the branch that needs breaking down further.
            if verifier is None and depth > 1:
                self._expand(tree, goal.id, sub, depth - 1, chain)

    def _propose(self, text: str) -> list[str]:
        """Ask for subgoals, then throw away the ones that are not subgoals.

        Any decomposer -- a stub or a frontier model -- will sometimes return the
        parent restated. Left in, that is an infinite recursion dressed as a plan
        ("define define define X"), so restatements are filtered by cosine
        against the parent rather than by string equality, which catches the
        rephrased ones too. The embedding layer is already here; this is one of
        the places it pays for itself outside of recall.
        """
        try:
            out = self.provider.complete(
                [Message("system", "Break the task into independent, checkable goals. "
                                   "One per line, imperative, no numbering."),
                 Message("user", f"DECOMPOSE:: {text}")], temperature=0.4, max_tokens=300)
        except Exception:
            out = ""
        raw = [re.sub(r"^\s*[-*\d.)\s]+", "", l).strip() for l in (out or "").splitlines()]
        parent = self.memory.embedder.embed(text, learn=False)
        goals, taken = [], []
        for g in raw:
            if len(g) <= 3 or g.lower() == text.lower().strip():
                continue
            v = self.memory.embedder.embed(g, learn=False)
            if cosine(v, parent) > 0.85:                 # a restatement, not a subgoal
                continue
            if any(cosine(v, seen) > 0.85 for seen in taken):   # two ways of saying one goal
                continue
            goals.append(g)
            taken.append(v)
        return goals

    def _verifier_for(self, text: str) -> Verifier | None:
        """A verifier is only claimed where one could plausibly be written. The
        body is a placeholder the toolsmith fills; what matters at this stage is
        the *commitment* that a check exists, because that commitment is what
        stops the recursion."""
        return Verifier("python", f"# check: {text}\n")

    # -- 3. what do we already know? -----------------------------------------

    def beliefs(self, task: str, why=None, questions=None) -> list[float]:
        """A distribution over Nature's states, from memory and the interrogation.

        Start from frequencies of recorded outcomes on similar tasks (Laplace
        smoothed, so early on this is near-uniform -- which is the honest prior).
        Then shift it: a why-chain that bottomed out ASSUMED or CIRCULAR is direct
        evidence for WRONG_FRAME, and high-value open questions are evidence for
        UNDERSPECIFIED. This is the one place the interrogation layer changes a
        number rather than producing commentary.
        """
        counts = {s: 1.0 for s in State.ALL}                # Laplace
        for hit in self.memory.recall(task, k=12, kinds=(Kind.CHAIN, Kind.ANSWER, Kind.FAILURE)):
            s = hit.trace.meta.get("state")
            if s in counts:
                counts[s] += hit.similarity * 2.0
        if why is not None:
            if why.terminal == Ground.CIRCULAR:
                counts[State.WRONG_FRAME] += 3.0
            elif why.terminal == Ground.ASSUMED:
                counts[State.WRONG_FRAME] += 1.5
            elif why.terminal == Ground.GROUNDED:
                counts[State.AS_STATED] += 2.0
        if questions:
            sharp = sum(1 for q in questions if q.value > 0.9)
            counts[State.UNDERSPECIFIED] += 0.75 * sharp
        if self.toolbox is not None and not self.toolbox.names():
            counts[State.TOOL_MISSING] += 1.0
        total = sum(counts.values())
        return [counts[s] / total for s in State.ALL]

    # -- 4. choose --------------------------------------------------------

    def payoff_matrix(self, goals: list, chain: Chain | None = None) -> list[list[float]]:
        """Rows = goals, columns = states. Table prior, modulated by evidence.

        Two modulations, both multiplicative on the positive part of the payoff
        so that evidence can dampen an upside but never flip a downside into a
        benefit: memory's verdict on similar goals, and the goal's own cost. A
        goal memory has seen fail keeps its shape across states and loses its
        height, which is what "we tried that" should do to a plan.
        """
        rows: list[list[float]] = []
        for g in goals:
            role = role_of(g.text)
            base = PAYOFF[role]
            hits = self.memory.recall(g.text, k=5, kinds=(Kind.GOAL, Kind.ANSWER, Kind.FAILURE))
            evidence = 1.0
            if hits:
                weighted = sum(h.similarity * ((h.trace.mean_grade or 0.0)) for h in hits)
                mass = sum(h.similarity for h in hits) or 1.0
                evidence = 1.0 + 0.5 * (weighted / mass)     # in [0.5, 1.5]
            cost_penalty = 1.0 / (1.0 + 0.25 * max(g.cost - 1.0, 0.0))
            rows.append([round((u * evidence * cost_penalty) if u > 0 else u, 4) for u in base])
        if chain:
            chain.add(Step.PAYOFF, f"{len(rows)}x{len(State.ALL)} matrix over "
                                   f"{[role_of(g.text) for g in goals]}",
                      matrix=rows, states=list(State.ALL))
        return rows

    def select(self, tree: GoalTree, belief: list[float], chain: Chain | None = None):
        frontier = tree.frontier()
        if not frontier:
            return None, None
        matrix = self.payoff_matrix(frontier, chain)
        decision = game.choose(matrix, belief, aversion=self.policy.regret_aversion)
        goal = frontier[decision.index]
        if chain:
            likely = State.ALL[max(range(len(belief)), key=lambda i: belief[i])]
            chain.add(Step.SELECT,
                      f"{goal.text}  <- {decision.rule}, world most likely {likely} "
                      f"(p={max(belief):.2f})",
                      goal=goal.text, rule=decision.rule,
                      belief=dict(zip(State.ALL, [round(b, 3) for b in belief])),
                      scores=[round(s, 3) for s in decision.scores])
        return goal, decision

    # -- 5. credit ----------------------------------------------------------

    def credit(self, tree: GoalTree, outcome: dict, chain: Chain | None = None) -> dict[str, float]:
        """Shapley values over the subgoals actually attempted.

        The coalition function is "would the parent have been met with only these
        subgoals?", read off the recorded per-goal outcomes. Shapley rather than
        "the last step before it worked" because the last step is the one that
        gets credited by every naive scheme and is almost never the one that
        mattered -- and because efficiency guarantees the shares sum to the
        outcome, so credit cannot be conjured.
        """
        attempted = [g for g in tree.goals.values() if g.id != "root" and g.status != Status.OPEN]
        if not attempted:
            return {}
        met = {g.id for g in attempted if g.status == Status.MET}

        def value(subset: list[str]) -> float:
            if not subset:
                return 0.0
            got = len([s for s in subset if s in met])
            return got / len(attempted)          # fraction of the task carried

        shares = game.shapley([g.id for g in attempted], value, samples=200)
        if chain:
            named = {tree.goals[k].text[:40]: round(v, 3) for k, v in shares.items() if abs(v) > 1e-9}
            chain.add(Step.CREDIT, f"shapley over {len(attempted)} subgoal(s): {named}", shares=shares)
        for gid, share in shares.items():
            g = tree.goals[gid]
            if g.trace_id and abs(share) > 0.01:
                self.memory.grade(g.trace_id, max(-1.0, min(1.0, share * 2.0)),
                                  outcome.get("source", Source.SELF), propagate=False)
        return shares

    # -- the whole thing ----------------------------------------------------

    def run(self, task: str, interrogate_first: bool = True,
            objective: str | None = None) -> Session:
        """`objective`, when the clarification loop established one, is what the
        work is actually distilled and recalled against.

        Without it the answers went into the frame and no further: `run` was
        called on the raw task string, so a user who patiently explained what
        done means got the same goal tree as one who said nothing. The whole
        point of asking is that the answer changes what happens next.
        """
        # The objective IS what done means, so it is what gets distilled.
        # Concatenating it onto the task produced goals carrying both, which read
        # as neither.
        subject = (objective or "").strip() or task
        chain = Chain(task)
        query = self.memory.remember(Kind.QUERY, subject)
        why, questions = (None, [])
        if interrogate_first:
            why, questions = self.interrogate_task(task, chain)
        hits = self.memory.recall(subject, k=5)
        chain.add(Step.RETRIEVE, f"{len(hits)} recollection(s)"
                  + (f"; best: {hits[0].trace.text[:60]!r} ({hits[0].score:.3f})" if hits else ""),
                  hits=[h.trace.id for h in hits])
        tree = self.distill(subject, chain)
        for g in tree.goals.values():
            if g.id != "root":
                t = self.memory.remember(Kind.GOAL, g.text, links=[query.id],
                                         meta={"task": task, "role": role_of(g.text),
                                               "verifiable": g.atomic})
                g.trace_id = t.id
        belief = self.beliefs(subject, why, questions)
        goal, decision = self.select(tree, belief, chain)
        trace = self.memory.remember(Kind.CHAIN, chain.to_text(), links=[query.id],
                                     meta={"task": task, "selected": goal.text if goal else None,
                                           "belief": dict(zip(State.ALL, belief))})
        return Session(task=task, chain=chain, tree=tree, chosen=goal,
                       questions=questions, why=why, trace_id=trace.id)
