"""Question everything. Ask why. Do not accept a refusal as an answer.

Three mechanisms, none of them a system prompt telling a model to be sceptical.
A prompt that says "be rigorous" produces prose that sounds rigorous. These
produce objects that can be checked.

**1. The why-chain (`interrogate`).** Ask why, then ask why of the answer, until
the chain terminates. What matters is not the asking, it is the *classification
of where it stops*: GROUNDED (the premise is checkable, and here is the check),
ASSUMED (nothing below it -- an axiom, a preference, a value judgement), or
CIRCULAR (the justification restates something already in the chain). Circularity
is detected by embedding the answer against its own ancestors, so "because it is
faster" -> "because it does less work" -> "because it is faster" is caught by
cosine rather than by hoping someone notices. A task whose why-chain bottoms out
in ASSUMED at depth one is a task built on an unexamined premise, and that is
worth knowing before any work is done.

**2. Premise attack (`challenge`).** Extract what a task takes for granted and
attack each premise along six axes -- definition, necessity, evidence,
alternative, scope, cost. Then rank the questions by *expected information gain*
and ask only the top ones. This is the part that keeps scepticism from becoming
noise: a question whose answer you can already predict is rhetoric, not enquiry.
The ranking is `entropy(p(answer))`, the same 0.5 operating point the rest of the
system runs on.

**3. Persistence (`Persistence`).** "Never take no for an answer" is a real
policy and it needs a real stopping rule, or it is an infinite loop with
ambition. The rule:

> Keep re-attacking while the refusals are still *new*. Stop when they repeat.

Every refusal is embedded. A refusal that is near-identical to one already
collected has taught nothing, and a second identical no means the boundary has
been found, not that persistence has failed. Until then the goal is reframed --
relax a constraint, substitute the means, invert the problem, generalise it,
or challenge the refuser itself -- and tried again. This is `GREN/DESIGN.md` §2
turned from an observation into a control loop: a game's identity is its refusal
boundary, so a system that collects refusals is mapping the problem, and one
that stops at the first no has collected one bit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .game import entropy
from .vector import cosine


# --------------------------------------------------------------------------- #
# 1. the why-chain
# --------------------------------------------------------------------------- #

class Ground:
    GROUNDED = "grounded"    # bottoms out in something checkable
    ASSUMED = "assumed"      # bottoms out in a premise, stated as such
    CIRCULAR = "circular"    # bottoms out in itself
    OPEN = "open"            # ran out of depth, not out of question


@dataclass
class Why:
    question: str
    answer: str
    depth: int
    ground: str = Ground.OPEN
    evidence: str | None = None


@dataclass
class WhyChain:
    subject: str
    links: list[Why] = field(default_factory=list)

    @property
    def terminal(self) -> str:
        return self.links[-1].ground if self.links else Ground.OPEN

    @property
    def depth(self) -> int:
        return len(self.links)

    def render(self) -> str:
        out = [f"? {self.subject}"]
        for w in self.links:
            out.append("  " * w.depth + f"why -> {w.answer}   [{w.ground}]")
        return "\n".join(out)


_GROUNDING = re.compile(
    r"\b(test|tests|measure|measured|benchmark|profil\w+|compil\w+|run|ran|observ\w+|"
    r"log|logs|data|reproduc\w+|assert\w*|exit code|error|traceback)\b", re.I)
_ASSUMING = re.compile(
    r"\b(should|ought|obviously|clearly|everyone|always|never|best practice|"
    r"standard|convention|prefer\w*|want|believe|assume\w*)\b", re.I)


def classify(answer: str, ancestors: list[str], embedder, threshold: float = 0.80) -> str:
    """Where did the chain land?

    Circularity is checked first and by vector, not by string equality --
    restating a claim in new words is the usual way a justification loops, and
    exact matching never catches it. Grounding is checked by whether the answer
    names something that could be *run*: a test, a measurement, an exit code. An
    answer appealing to convention or preference is ASSUMED, which is not an
    insult -- most chains should bottom out there -- it is just worth labelling.
    """
    if ancestors:
        v = embedder.embed(answer, learn=False)
        for prior in ancestors:
            if cosine(v, embedder.embed(prior, learn=False)) >= threshold:
                return Ground.CIRCULAR
    if _GROUNDING.search(answer):
        return Ground.GROUNDED
    if _ASSUMING.search(answer):
        return Ground.ASSUMED
    return Ground.OPEN


def interrogate(subject: str, asker, embedder, max_depth: int = 5) -> WhyChain:
    """Ask why until the chain terminates or the depth budget runs out.

    `asker(question, context) -> str` is anything that answers: a provider, a
    human at a prompt, or a stub. The chain stops on GROUNDED or CIRCULAR --
    ASSUMED does not stop it, because an assumption can itself be questioned and
    the interesting chains are the ones that keep going past the first "well,
    that's just how it's done".
    """
    chain = WhyChain(subject=subject)
    current, ancestors = subject, [subject]
    for depth in range(max_depth):
        question = f"why {current}?" if depth else f"why {subject}?"
        answer = (asker(question, ancestors) or "").strip()
        if not answer:
            chain.links.append(Why(question, "(no answer)", depth, Ground.OPEN))
            break
        ground = classify(answer, ancestors[:-1] if len(ancestors) > 1 else [], embedder)
        chain.links.append(Why(question, answer, depth, ground))
        ancestors.append(answer)
        current = answer
        if ground in (Ground.GROUNDED, Ground.CIRCULAR):
            break
    return chain


# --------------------------------------------------------------------------- #
# 2. premise attack
# --------------------------------------------------------------------------- #

class Attack:
    DEFINITION = "definition"      # what exactly counts as this?
    NECESSITY = "necessity"        # is it required, or inherited from an old decision?
    EVIDENCE = "evidence"          # what observation would falsify it?
    ALTERNATIVE = "alternative"    # what if the opposite held?
    SCOPE = "scope"                # for whom, when, at what scale?
    COST = "cost"                  # what does being wrong about it cost?

    ALL = (DEFINITION, NECESSITY, EVIDENCE, ALTERNATIVE, SCOPE, COST)


_TEMPLATES = {
    Attack.DEFINITION: "what exactly counts as {p} here, and what is the nearest thing that does not?",
    Attack.NECESSITY: "is {p} actually required, or is it inherited from a decision nobody revisited?",
    Attack.EVIDENCE: "what observation would show {p} is false? has it been made?",
    Attack.ALTERNATIVE: "suppose the opposite of {p} were true -- what would be built instead?",
    Attack.SCOPE: "for whom, when, and at what scale does {p} hold? where does it stop holding?",
    Attack.COST: "if {p} is wrong, what does it cost, and when would that be discovered?",
}

#: Words that mark a clause as load-bearing rather than descriptive. A premise
#: hides behind a modal or a quantifier far more often than behind a noun.
_PREMISE_MARKERS = re.compile(
    r"\b(must|should|need|needs|requires?|always|never|all|every|because|since|"
    r"so that|in order to|obviously|clearly|the best|the only|cannot|can't)\b", re.I)


@dataclass
class Question:
    text: str
    attack: str
    premise: str
    p_answer_known: float          # how confidently the answer can be predicted
    value: float                   # bits: entropy of that prediction

    def __str__(self) -> str:
        return f"[{self.attack}] {self.text}"


def premises(task: str) -> list[str]:
    """The claims a task assumes. Clauses carrying a modal, a quantifier or a
    causal connective, plus the task itself -- which is always a premise, and
    the one least often questioned."""
    clauses = [c.strip() for c in re.split(r"[;,.]|\bbut\b|\bbecause\b|\bso that\b", task) if c.strip()]
    marked = [c for c in clauses if _PREMISE_MARKERS.search(c)]
    out = marked or clauses[:2]
    if task.strip() not in out:
        out = [task.strip()] + out
    return out[:4]


def challenge(task: str, memory=None, limit: int = 6) -> list[Question]:
    """Every premise crossed with every attack, ranked by information gain.

    `p_answer_known` is estimated from memory: if the store already holds
    well-graded traces close to the question, the answer is probably already
    known and asking it is ceremony. With no memory the prior is the attack's
    own base rate -- EVIDENCE and ALTERNATIVE questions are the ones whose
    answers are hardest to predict, so they outrank DEFINITION by default, which
    matches the experience of anyone who has sat through a requirements meeting.
    """
    base = {Attack.DEFINITION: 0.75, Attack.NECESSITY: 0.55, Attack.EVIDENCE: 0.35,
            Attack.ALTERNATIVE: 0.40, Attack.SCOPE: 0.50, Attack.COST: 0.45}
    questions: list[Question] = []
    for p in premises(task):
        for attack in Attack.ALL:
            text = _TEMPLATES[attack].format(p=p.strip().rstrip("?."))
            known = base[attack]
            if memory is not None:
                hits = memory.recall(text, k=3)
                if hits:
                    # a close, credible memory means the answer is largely known
                    known = min(0.95, known + 0.5 * hits[0].similarity * hits[0].credibility)
            questions.append(Question(text, attack, p, known, entropy(known)))
    questions.sort(key=lambda q: q.value, reverse=True)
    # Diversify on both axes. Six variations of one attack, or six attacks on one
    # premise, is a single question asked loudly -- and because the base rates are
    # shared, a pure entropy sort produces exactly that. Cap each.
    picked: list[Question] = []
    per_premise: dict[str, int] = {}
    per_attack: dict[str, int] = {}
    for q in questions:
        if per_premise.get(q.premise, 0) >= 2 or per_attack.get(q.attack, 0) >= 2:
            continue
        picked.append(q)
        per_premise[q.premise] = per_premise.get(q.premise, 0) + 1
        per_attack[q.attack] = per_attack.get(q.attack, 0) + 1
        if len(picked) >= limit:
            break
    return picked


# --------------------------------------------------------------------------- #
# 3. persistence -- never take no for an answer
# --------------------------------------------------------------------------- #

class Reframe:
    RELAX = "relax"              # drop the constraint that was violated
    SUBSTITUTE = "substitute"    # same end, different means
    DECOMPOSE = "decompose"      # attack a part instead of the whole
    INVERT = "invert"            # solve the complement, or work backwards
    GENERALISE = "generalise"    # solve a larger problem that is easier
    QUESTION_ORACLE = "question-oracle"   # is the refuser even right?

    ALL = (RELAX, SUBSTITUTE, DECOMPOSE, INVERT, GENERALISE, QUESTION_ORACLE)


_REFRAME_TEMPLATES = {
    Reframe.RELAX: "{goal} -- without the constraint that caused: {refusal}",
    Reframe.SUBSTITUTE: "{goal} -- by an entirely different means, given that: {refusal}",
    Reframe.DECOMPOSE: "the smallest part of ({goal}) that does not hit: {refusal}",
    Reframe.INVERT: "instead of ({goal}), produce what it would consume, or rule out what it is not",
    Reframe.GENERALISE: "the general problem ({goal}) is a special case of, which may avoid: {refusal}",
    Reframe.QUESTION_ORACLE: "establish whether the refusal ({refusal}) is correct, or an artefact of the checker",
}


@dataclass
class Attempt:
    goal: str
    reframe: str | None
    refusal: str | None
    novel: bool          # did this refusal say anything the others had not?


class Persistence:
    """A refusal is a constraint discovered, not a stopping condition.

    The budget is spent on *information*, not on attempts: `novelty_floor` is the
    cosine above which a new refusal counts as a repeat of one already collected,
    and the loop ends when the last `patience` attempts all came back with
    repeats. That is the honest reading of "never take no for an answer" -- keep
    going while the noes are still teaching you something, and when they stop,
    you have not given up, you have finished mapping the boundary.

    The collected refusals are the output that matters. Even a run that never
    succeeds returns a description of why the goal is impossible, which is a
    result, and is the one thing a system that stops at the first no can never
    produce.
    """

    def __init__(self, embedder, max_attempts: int = 6, novelty_floor: float = 0.85,
                 patience: int = 2) -> None:
        self.embedder = embedder
        self.max_attempts = max_attempts
        self.novelty_floor = novelty_floor
        self.patience = patience
        self.refusals: list[tuple[str, list[float]]] = []
        self.attempts: list[Attempt] = []

    def novel(self, refusal: str) -> bool:
        v = self.embedder.embed(refusal, learn=False)
        for _, prior in self.refusals:
            if cosine(v, prior) >= self.novelty_floor:
                return False
        self.refusals.append((refusal, v))
        return True

    def reframings(self, goal: str, refusal: str) -> list[tuple[str, str]]:
        return [(r, _REFRAME_TEMPLATES[r].format(goal=goal, refusal=refusal[:120]))
                for r in Reframe.ALL]

    def pursue(self, goal: str, attempt_fn, rng=None) -> dict:
        """`attempt_fn(goal_text, reframe) -> (ok: bool, detail: str)`.

        Reframes are tried in order rather than sampled, because the order is
        itself a hypothesis worth keeping fixed while everything else moves: RELAX
        first (the constraint may not be real), QUESTION_ORACLE last (the checker
        is usually right, and assuming otherwise first is how a system talks
        itself past a correct no).
        """
        ok, detail = attempt_fn(goal, None)
        self.attempts.append(Attempt(goal, None, None if ok else detail, self.novel(detail) if not ok else False))
        if ok:
            return {"solved": True, "goal": goal, "attempts": self.attempts,
                    "refusals": [r for r, _ in self.refusals], "reason": "first attempt"}
        stale = 0
        queue = self.reframings(goal, detail)
        for reframe, text in queue:
            if len(self.attempts) >= self.max_attempts:
                return self._exhausted("attempt budget")
            ok, detail = attempt_fn(text, reframe)
            novel = False if ok else self.novel(detail)
            self.attempts.append(Attempt(text, reframe, None if ok else detail, novel))
            if ok:
                return {"solved": True, "goal": text, "reframe": reframe,
                        "attempts": self.attempts, "refusals": [r for r, _ in self.refusals],
                        "reason": f"succeeded after reframing ({reframe})"}
            stale = 0 if novel else stale + 1
            if stale >= self.patience:
                return self._exhausted("refusals stopped being informative -- boundary found")
        return self._exhausted("all reframings exhausted")

    def _exhausted(self, reason: str) -> dict:
        return {"solved": False, "attempts": self.attempts,
                "refusals": [r for r, _ in self.refusals], "reason": reason,
                "boundary": self.boundary()}

    def boundary(self) -> str:
        """What was learned by failing. The deliverable of an unsuccessful run."""
        if not self.refusals:
            return "no refusals recorded"
        lines = [f"{len(self.refusals)} distinct refusal(s) over {len(self.attempts)} attempt(s):"]
        lines += [f"  - {r[:140]}" for r, _ in self.refusals]
        return "\n".join(lines)
