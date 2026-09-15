"""Understand the game before playing it. This runs first, ahead of everything.

The premise is that everything is a game -- chess, poker, a negotiation with a
landlord, a conversation with your boss, shipping a parser. What differs between
them is not whether they are games but *which* game: who moves, what may be
played, who observes what, how it is scored, and whether it is played once or
forever. Those answers decide which solution concept is even valid, so guessing
them is not a detail. A minimax play in a non-zero-sum game is a mistake; a
cooperative play in a one-shot zero-sum game is a donation.

So the first object built is a `GameFrame`, and the order is enforced:

    FRAME -> CAPABILITIES -> AGENDA -> question -> distil -> select -> act

`GREN/DESIGN.md` states the same ordering as a hard boundary -- GTMNN refuses a
game package below a confidence threshold, because a system that plays before it
understands is guessing with extra steps. This module is that boundary for
tasks rather than for board games, and it reports its confidence so the caller
can enforce it.

The frame answers seven questions:

| axis | why it changes the play |
|---|---|
| players | solitaire, versus Nature, versus an adversary, or n-party |
| actions | what may legally be done -- the move set |
| information | perfect or imperfect; complete or incomplete |
| payoff | zero-sum, common-interest, or mixed-motive |
| horizon | one-shot or repeated -- the single biggest lever there is |
| chance | deterministic or stochastic |
| referee | who says no, and how fast -- the refusal oracle |

`classify()` maps a frame to a solution concept. That mapping is the point of the
module: **the payoff-matrix-against-Nature that `reason.py` uses is not the
universal method, it is the correct method for one class**, and the frame is what
establishes which class this is.

Then `capabilities()` intersects what the game requires with what this system can
actually do -- recall, interrogate, distil, write a tool, run a check, run an
experiment, and every tool already in the toolbox. What the game needs and the
system lacks is a **capability gap**, and gaps become the first items on the
agenda, because they are the moves that are not yet playable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .memory import Kind
from .provider import Message
from .vector import cosine


class Players:
    SOLITAIRE = "solitaire"          # only you and the rules
    VS_NATURE = "vs-nature"          # an indifferent world with uncertainty
    VS_ADVERSARY = "vs-adversary"    # someone whose loss is your gain
    N_PARTY = "n-party"              # several interests, not strictly opposed


class Information:
    PERFECT = "perfect"              # everything relevant is observable
    IMPERFECT = "imperfect"          # hidden state, but known rules
    INCOMPLETE = "incomplete"        # the payoffs themselves are unknown


class Payoff:
    ZERO_SUM = "zero-sum"
    COMMON = "common-interest"
    MIXED = "mixed-motive"


class Horizon:
    ONE_SHOT = "one-shot"
    REPEATED = "repeated"


class Solution:
    """Which concept is valid given the frame. Named, so a choice can be argued
    with rather than merely disagreed with."""
    DECISION_THEORY = "decision-theory (minimax regret / expected utility)"
    MINIMAX = "minimax (fictitious play to the value of the game)"
    NASH = "nash / correlated equilibrium"
    FOLK = "repeated-game reciprocity (cooperation is enforceable)"
    BAYESIAN = "bayesian game (act on beliefs, update on observation)"
    COOPERATIVE = "cooperative (shapley allocation)"


@dataclass
class GameFrame:
    task: str
    players: str = Players.VS_NATURE
    actions: list[str] = field(default_factory=list)     # the move set
    inputs: list[str] = field(default_factory=list)      # what is observable
    outputs: list[str] = field(default_factory=list)     # what is produced
    objective: str = ""                                  # what winning means
    information: str = Information.IMPERFECT
    payoff: str = Payoff.MIXED
    horizon: str = Horizon.ONE_SHOT
    stochastic: bool = False
    referee: str = ""                                    # who says no
    confidence: float = 0.0                              # how sure of all this
    evidence: list[str] = field(default_factory=list)    # why each axis was set

    @property
    def solution(self) -> str:
        return classify(self)

    @property
    def understood(self) -> bool:
        """The boundary GREN enforces: do not play below this line."""
        return self.confidence >= 0.5 and bool(self.objective) and bool(self.actions)

    def render(self) -> str:
        return "\n".join([
            f"  game      {self.task}",
            f"  players   {self.players}",
            f"  objective {self.objective or '(not established)'}",
            f"  inputs    {', '.join(self.inputs) or '(none identified)'}",
            f"  actions   {', '.join(self.actions) or '(none identified)'}",
            f"  outputs   {', '.join(self.outputs) or '(none identified)'}",
            f"  structure {self.information}, {self.payoff}, {self.horizon}"
            + (", stochastic" if self.stochastic else ", deterministic"),
            f"  referee   {self.referee or '(none -- nothing says no)'}",
            f"  solution  {self.solution}",
            f"  confidence {self.confidence:.2f}"
            + ("" if self.understood else "   << below the play threshold"),
        ])

    def to_json(self) -> dict:
        return {"task": self.task, "players": self.players, "actions": self.actions,
                "inputs": self.inputs, "outputs": self.outputs, "objective": self.objective,
                "information": self.information, "payoff": self.payoff,
                "horizon": self.horizon, "stochastic": self.stochastic,
                "referee": self.referee, "confidence": self.confidence}

    def embed_text(self) -> str:
        """What memory indexes, so "which game is this like?" is a recall query.

        Deliberately the *structure* rather than the task wording: two tasks with
        the same players, information and horizon should cluster even when their
        subject matter is unrelated, because the same solution concept applies to
        both. That is the whole reason for framing games by structure.
        """
        return (f"game {self.task}\nplayers {self.players}\nobjective {self.objective}\n"
                f"actions {', '.join(self.actions)}\ninputs {', '.join(self.inputs)}\n"
                f"structure {self.information} {self.payoff} {self.horizon}\n"
                f"referee {self.referee}\nsolution {self.solution}")


def classify(frame: GameFrame) -> str:
    """Frame -> solution concept. Order matters; the first match wins.

    Repeated play is checked before payoff structure because it dominates: the
    folk theorem says cooperation is enforceable in a repeated mixed-motive game
    that is simply not available in the one-shot version, and that changes the
    move you make today. Incomplete information is checked before everything
    because when the payoffs themselves are unknown, no equilibrium concept
    applies yet -- you are in a Bayesian game and the first job is belief, not
    optimisation.
    """
    if frame.information == Information.INCOMPLETE:
        return Solution.BAYESIAN
    if frame.players == Players.SOLITAIRE and not frame.stochastic:
        return Solution.DECISION_THEORY
    if frame.players == Players.VS_NATURE:
        return Solution.DECISION_THEORY
    if frame.horizon == Horizon.REPEATED and frame.payoff != Payoff.ZERO_SUM:
        return Solution.FOLK
    if frame.players == Players.VS_ADVERSARY and frame.payoff == Payoff.ZERO_SUM:
        return Solution.MINIMAX
    if frame.players == Players.N_PARTY and frame.payoff == Payoff.COMMON:
        return Solution.COOPERATIVE
    return Solution.NASH


# --------------------------------------------------------------------------- #
# reading the frame off the task
# --------------------------------------------------------------------------- #

_SIGNALS = {
    "players": [
        (Players.VS_ADVERSARY, re.compile(
            r"\b(opponent|adversar\w+|beat|defeat|against|competitor|rival|attack\w*|"
            r"defend|win against|outbid|chess|poker|checkers|go\b)\b", re.I)),
        (Players.N_PARTY, re.compile(
            r"\b(negotiat\w+|team|stakeholder|partner|client|boss|manager|colleague|"
            r"customer|vendor|landlord|committee|agree\w*)\b", re.I)),
        (Players.SOLITAIRE, re.compile(
            r"\b(sudoku|puzzle|refactor|compile|parse|sort|comput\w+|implement|script)\b", re.I)),
    ],
    "repeated": re.compile(
        r"\b(every|each (day|week|sprint|time)|recurring|ongoing|continuous\w*|"
        r"repeated\w*|relationship|long.term|again|habit|maintain)\b", re.I),
    "stochastic": re.compile(
        r"\b(random|chance|probab\w+|uncertain\w*|noisy|flaky|estimate|forecast|"
        r"poker|dice|sampl\w+|luck)\b", re.I),
    "hidden": re.compile(
        r"\b(hidden|unknown|secret|private|bluff|conceal\w*|opaque|black.box|"
        r"guess|infer|do not know|don't know)\b", re.I),
    "zero_sum": re.compile(
        r"\b(zero.sum|win or lose|beat|defeat|only one|at the expense of|"
        r"chess|checkers|outbid)\b", re.I),
    "common": re.compile(
        r"\b(together|collaborat\w+|mutual|shared goal|align\w*|both benefit|team)\b", re.I),
    "referee": re.compile(
        r"\b(compiler|test suite|tests|linter|type ?check\w*|ci\b|reviewer|judge|"
        r"referee|validator|schema|assertion|benchmark|interpreter)\b", re.I),
}

_OBJECTIVE = re.compile(r"\b(?:so that|in order to|to|goal is to|aim(?:ing)? to)\s+(.{6,80})", re.I)
_VERB = re.compile(r"\b(build|write|parse|compute|sort|find|count|check|convert|read|"
                   r"fetch|clean|merge|validate|render|negotiate|persuade|decide|"
                   r"choose|plan|design|ship|fix|test|measure)\b", re.I)


class Framer:
    """Builds a `GameFrame`, preferring evidence over guesses.

    Three sources, in order. **Memory first**: if a structurally similar game has
    been framed before, its frame is inherited and only the axes that disagree
    are re-derived -- which is `GREN/DESIGN.md` 3's argument that identification
    against a corpus costs log2|G| bits where induction from nothing costs
    log2|Theta|, three to four orders of magnitude worse. **Then the provider**,
    which is asked one direct question per axis rather than for a free-form
    analysis. **Then lexical signals**, which are crude but never unavailable.

    Confidence is the fraction of axes settled by evidence rather than by
    default, and it is reported rather than rounded up, because the caller is
    entitled to refuse to play.
    """

    def __init__(self, memory, provider) -> None:
        self.memory = memory
        self.provider = provider

    def frame(self, task: str) -> GameFrame:
        f = GameFrame(task=task)
        evidence: list[str] = []
        settled = 0
        total = 7

        prior = self.similar(task)
        if prior is not None:
            f.players, f.information, f.payoff = prior.players, prior.information, prior.payoff
            f.horizon, f.stochastic = prior.horizon, prior.stochastic
            evidence.append(f"inherited structure from a similar game: {prior.task[:50]!r}")
            settled += 3

        # players
        for kind, pattern in _SIGNALS["players"]:
            if pattern.search(task):
                f.players = kind
                evidence.append(f"players={kind} from wording")
                settled += 1
                break
        else:
            if prior is None:
                evidence.append("players defaulted to vs-nature")

        # horizon, chance, information, payoff
        if _SIGNALS["repeated"].search(task):
            f.horizon = Horizon.REPEATED
            evidence.append("repeated play signalled")
            settled += 1
        if _SIGNALS["stochastic"].search(task):
            f.stochastic = True
            evidence.append("chance signalled")
            settled += 1
        if _SIGNALS["hidden"].search(task):
            f.information = Information.IMPERFECT
            evidence.append("hidden state signalled")
            settled += 1
        elif f.players == Players.SOLITAIRE:
            f.information = Information.PERFECT
            settled += 1
        if _SIGNALS["zero_sum"].search(task):
            f.payoff = Payoff.ZERO_SUM
            settled += 1
        elif _SIGNALS["common"].search(task):
            f.payoff = Payoff.COMMON
            settled += 1

        # the referee: who says no. The most valuable axis, because it decides
        # whether anything here can be self-graded at all.
        ref = _SIGNALS["referee"].search(task)
        if ref:
            f.referee = ref.group(0)
            evidence.append(f"referee={f.referee}")
            settled += 1
        elif f.players == Players.SOLITAIRE:
            f.referee = "the interpreter"
            settled += 1

        # objective, actions, inputs, outputs
        obj = _OBJECTIVE.search(task)
        f.objective = (obj.group(1).strip() if obj else task.strip())[:120]
        f.actions = self._actions(task)
        f.inputs, f.outputs = self._io(task)
        if f.actions and _VERB.search(task):
            settled += 1          # a recognised verb is evidence; a fallback is not

        asked = self._ask_provider(task, f)
        if asked:
            evidence.append("provider refined the frame")
            settled += 1

        f.confidence = round(min(1.0, settled / total), 3)
        f.evidence = evidence
        return f

    _LEADING_STOP = {"the", "a", "an", "my", "our", "this", "that", "please", "we", "i"}

    def _actions(self, task: str) -> list[str]:
        """The move set, from the verbs in the task.

        `_VERB` is a fixed list and will never be complete, so a task whose verb
        is missing from it -- *dedupe the records*, *tune the importer* -- used to
        come back with no actions at all, and an empty move set blocks the
        clarification loop with "no moves available". That gated a large share of
        perfectly clear instructions on a vocabulary gap.

        So when no known verb matches, the task's own leading word is taken as
        the action. A task is an imperative; its first content word is the verb
        by construction. This is a guess and it is labelled as one -- the frame's
        confidence does not count a fallback action as a settled axis.
        """
        found = sorted({m.group(0).lower() for m in _VERB.finditer(task)})
        if found:
            return found
        for word in re.findall(r"[A-Za-z][\w'-]*", task.lower()):
            if word not in self._LEADING_STOP:
                return [word]
        return []

    def _io(self, task: str) -> tuple[list[str], list[str]]:
        """Inputs and outputs, read off prepositions.

        "from X" and "given X" mark inputs; "into Y", "as Y" and "produce Y" mark
        outputs. Shallow, and it does not pretend otherwise -- the value is that
        an empty list is *visible*, and a game whose inputs nobody can name is a
        game nobody has understood.
        """
        inputs = re.findall(r"\b(?:from|given|using|with|out of)\s+(?:the\s+|a\s+|an\s+)?([\w .-]{3,30})", task, re.I)
        outputs = re.findall(r"\b(?:into|as|produc\w+|return\w*|output\w*|yield\w*)\s+(?:the\s+|a\s+|an\s+)?([\w .-]{3,30})", task, re.I)
        return ([i.strip() for i in inputs][:4], [o.strip() for o in outputs][:4])

    def _ask_provider(self, task: str, f: GameFrame) -> bool:
        """One structured question, parsed strictly. A reply that does not answer
        in the expected form changes nothing -- a frame half-overwritten by a
        misparse is worse than a frame built from lexical signals alone."""
        try:
            reply = self.provider.complete([
                Message("system", "Answer in exactly these lines, no prose:\n"
                                  "OBJECTIVE: <what winning means>\n"
                                  "ACTIONS: <comma separated moves available>\n"
                                  "INPUTS: <comma separated observables>\n"
                                  "OUTPUTS: <comma separated products>\n"
                                  "REFEREE: <what says no, or none>"),
                Message("user", f"FRAME:: {task}")], temperature=0.1, max_tokens=250)
        except Exception:
            return False
        if not reply or ":" not in reply:
            return False
        changed = False
        for line in reply.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().upper(), value.strip()
            if not value or value.lower() == "none":
                continue
            items = [v.strip() for v in value.split(",") if v.strip()][:6]
            if key == "OBJECTIVE":
                f.objective, changed = value[:120], True
            elif key == "ACTIONS" and items:
                f.actions, changed = items, True
            elif key == "INPUTS" and items:
                f.inputs, changed = items, True
            elif key == "OUTPUTS" and items:
                f.outputs, changed = items, True
            elif key == "REFEREE":
                f.referee, changed = value[:60], True
        return changed

    def similar(self, task: str) -> GameFrame | None:
        """The nearest game already framed, if it is near enough to inherit from."""
        hits = self.memory.recall(task, k=1, kinds=(Kind.GAME,))
        if not hits or hits[0].similarity < 0.45:
            return None
        meta = hits[0].trace.meta.get("frame")
        if not meta:
            return None
        return GameFrame(**{k: v for k, v in meta.items() if k in GameFrame.__annotations__})

    def remember(self, frame: GameFrame):
        return self.memory.remember(Kind.GAME, frame.embed_text(),
                                    meta={"frame": frame.to_json(), "task": frame.task})


# --------------------------------------------------------------------------- #
# what the system can actually do about it
# --------------------------------------------------------------------------- #

#: The primitives, independent of any tool. These are the moves the system always
#: has, and naming them is what makes "break it down into things this system can
#: do" a concrete operation rather than an aspiration.
#: The descriptions carry their own synonyms on purpose. Matching is by
#: embedding, and the default embedder is lexical, so a primitive described only
#: as "forge" is unreachable from the word "build" -- which is the word people
#: actually use. Listing the surface forms is not padding; it is what makes the
#: capability match work without a semantic model attached.
PRIMITIVES = {
    "recall": "recall remember retrieve look up find prior graded experience from memory",
    "interrogate": "interrogate ask why question justify understand reason establish why",
    "challenge": "challenge attack question dispute the premises assumptions of a claim",
    "distil": "distil decompose break down split plan a goal into checkable subgoals",
    "forge": "forge build write implement create make a python tool function script "
             "and verify it against contract tests",
    "check": "check test verify validate confirm assert measure benchmark run the tests "
             "and record the verdict",
    "experiment": "experiment try explore probe trial a deliberately uncertain idea "
                  "for the information it yields",
    "select": "select choose decide pick prioritise among goals by a named solution concept",
}


@dataclass
class Capability:
    action: str
    covered_by: str | None          # a primitive name, a tool name, or None
    confidence: float

    @property
    def gap(self) -> bool:
        return self.covered_by is None


@dataclass
class Agenda:
    frame: GameFrame
    capabilities: list[Capability]
    items: list[str]
    #: Items that no primitive and no tool can discharge -- they need a person to
    #: answer something. Tracked explicitly rather than recovered by matching on
    #: the item text, because "is this step actionable?" is the termination
    #: condition of the clarification loop and it must not depend on wording.
    needs_person: list[str] = field(default_factory=list)

    @property
    def gaps(self) -> list[Capability]:
        return [c for c in self.capabilities if c.gap]

    @property
    def actionable_items(self) -> list[str]:
        return [i for i in self.items if i not in self.needs_person]

    def render(self) -> str:
        lines = ["  what the game needs, and whether this system can do it:"]
        for c in self.capabilities:
            mark = "--" if c.gap else "ok"
            lines.append(f"    [{mark}] {c.action:<24} {c.covered_by or 'NOTHING COVERS THIS'}")
        lines.append("\n  agenda, in order:")
        lines += [f"    {i + 1}. {item}" for i, item in enumerate(self.items)]
        return "\n".join(lines)


def capabilities(frame: GameFrame, memory, toolbox=None) -> list[Capability]:
    """Match every action the game requires against something that can perform it.

    Matching is by embedding against the primitive descriptions and the toolbox,
    not by name, so "tidy up the rows" finds a tool that advertises cleaning a
    ragged CSV. An action nothing covers is returned as a gap with
    `covered_by=None`, and those are the items the agenda attacks first -- they
    are the moves that are not yet legal for this player.
    """
    out: list[Capability] = []
    prim_vectors = {name: memory.embedder.embed(f"{name}: {desc}", learn=False)
                    for name, desc in PRIMITIVES.items()}
    prim_words = {name: set(desc.split()) for name, desc in PRIMITIVES.items()}
    for action in frame.actions or [frame.objective]:
        tokens = {w.lower().strip(".,") for w in action.split()}

        # Exact surface-form match first, and it is not a shortcut. The 512-dim
        # signed hash has a collision noise floor around 0.17 while genuine
        # matches land at 0.25-0.37, so a one-word action like "build" is simply
        # not separable by cosine -- it scored 0.17 against `distil` (a collision)
        # and 0.148 against `forge` (the right answer). The synonym lists in
        # PRIMITIVES enumerate exactly these surface forms, so set membership is
        # both exact and the intended use of them. Cosine still handles the
        # phrasings nobody listed.
        hit = next((n for n, words in prim_words.items() if tokens & words), None)
        if hit:
            out.append(Capability(action, f"primitive:{hit}", 1.0))
            continue

        v = memory.embedder.embed(action, learn=False)
        best_name, best_sim = None, 0.20          # above the collision noise floor
        for name, pv in prim_vectors.items():
            sim = cosine(v, pv)
            if sim > best_sim:
                best_name, best_sim = f"primitive:{name}", sim
        if toolbox is not None:
            for spec, _score, sim in toolbox.find(action, k=3):
                if sim > best_sim:
                    best_name, best_sim = f"tool:{spec.name}", sim
        out.append(Capability(action, best_name, round(best_sim, 3)))
    return out


def agenda(frame: GameFrame, caps: list[Capability]) -> Agenda:
    """The ordered list to attack.

    The order is not arbitrary and is the operational content of "understand the
    game first":

    1. **If the game is not understood, establish it.** Nothing else is worth
       doing while the objective or the move set is unknown -- work done against
       the wrong frame is the most expensive kind.
    2. **Close the capability gaps.** A move you cannot make is not a plan.
    3. **Then play**, in an order the solution concept chooses.

    A gap becomes a `forge` item, because writing and verifying a tool is exactly
    how this system acquires a move it did not have.
    """
    items: list[str] = []
    blocking: list[str] = []
    if not frame.objective:
        item = "establish what winning means -- no objective could be read from the task"
        items.append(item)
        blocking.append(item)
    if not frame.actions:
        item = "establish the move set -- no actions could be identified"
        items.append(item)
        blocking.append(item)
    # A missing referee and unknown payoffs are listed but are NOT blocking.
    # They say the work cannot be *self-graded*, not that it cannot be *started*
    # -- and gating on them refused perfectly clear instructions like "dedupe the
    # records" because no verifier could be named for them in advance. What
    # blocks a first step is narrower: not being able to state what done means,
    # or having no move to make.
    if not frame.referee:
        items.append("find the referee: what would say no, and how fast? "
                     "(without one, nothing here can be self-graded)")
    if frame.information == Information.INCOMPLETE:
        items.append("the payoffs are unknown: form beliefs before optimising")
    for cap in caps:
        if cap.gap:
            items.append(f"forge a capability for {cap.action!r} -- nothing covers it yet")
    for cap in caps:
        if not cap.gap:
            items.append(f"{cap.action} (via {cap.covered_by})")
    if frame.horizon == Horizon.REPEATED:
        items.append("this game repeats: prefer a move that survives being played again")
    return Agenda(frame, caps, items, blocking)
