"""When the game is not understood, ask. Do not proceed.

`frame.py` computes whether a game is understood well enough to play
(`GameFrame.understood`) and `GREN/DESIGN.md` 22.1 makes that a hard boundary
rather than a warning. This module is what happens on the wrong side of it.

The rule:

> **If the objective is not understood, ask clarifying questions until the first
> step is actionable. Then take it.**

Two halves, and the second is the one that keeps this from becoming an
interrogation nobody wanted:

**Ask about what is actually missing.** Not generic premise attacks -- those are
`challenge.py`, and they are for a task that *is* understood. These questions are
derived from the specific axes the frame could not fill: no objective, no move
set, no referee, unknown payoffs. Each gap has its own question, phrased to be
answerable in a sentence, and each names what it unblocks so the person being
asked can see why it matters.

**Stop as soon as the first step is actionable.** Not when every axis is filled --
a frame can stay imperfect forever and most do. The loop terminates when the
agenda's first item is executable, because that is the point at which asking more
questions costs more than trying. "Understand the game first" does not mean
"understand the game completely first"; it means do not swing at a ball you
cannot see.

Three things keep the asking honest:

1. **Memory answers before the person does.** Every question is first put to the
   store -- a previous frame, a case, a recorded fact. Asking someone what they
   already told you is how a system teaches people to stop answering it.
2. **Questions are ranked by what they unblock**, and the ranking is not
   negotiable: objective before referee before actions before everything else. A
   system that cannot state what winning means should not be asking about input
   formats.
3. **A question is asked once.** If an answer does not raise confidence, the gap
   is recorded as *contested* and the loop moves on rather than rephrasing the
   same question at someone who has already tried to answer it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .frame import Framer, GameFrame, Information, agenda, capabilities
from .game import entropy
from .goals import checkability
from .memory import Kind, Source


class Gap:
    """A specific thing the frame could not establish. Ordered by what it blocks:
    every later gap is answerable only once the earlier ones are."""
    OBJECTIVE = "objective"      # what does winning mean?
    REFEREE = "referee"          # what says no? (decides if anything is gradeable)
    ACTIONS = "actions"          # what moves exist?
    PAYOFF = "payoff"            # whose gain is whose loss?
    INPUTS = "inputs"            # what is observable?
    OUTPUTS = "outputs"          # what is produced?
    CAPABILITY = "capability"    # a required move nothing covers

    #: Strict priority. Asking about output formats while the objective is
    #: unknown is the most common way a clarifying conversation wastes everyone's
    #: time, and a fixed order is the cheapest possible fix for it.
    ORDER = (OBJECTIVE, REFEREE, ACTIONS, PAYOFF, CAPABILITY, INPUTS, OUTPUTS)

    #: The two that stop work outright. Everything else is a limitation to note,
    #: not a reason to refuse to start -- see `first_step_actionable`.
    BLOCKING = (OBJECTIVE, ACTIONS)


_QUESTIONS = {
    Gap.OBJECTIVE: ("What would count as this being done? Describe the finished "
                    "state, not the activity.",
                    "nothing can be selected or verified without it"),
    Gap.REFEREE: ("What would tell you this is wrong -- a test, a person, a "
                  "measurement? If nothing would, say so.",
                  "with no referee nothing here can be self-graded"),
    Gap.ACTIONS: ("What are you actually able to do here? List the moves "
                  "available, even the bad ones.",
                  "a move set is what the plan is chosen from"),
    Gap.PAYOFF: ("Is anyone else's interest at stake, and does your gain cost "
                 "them? Or are you all after the same outcome?",
                 "it decides whether to compete, cooperate, or ignore others"),
    Gap.CAPABILITY: ("For {detail} -- is there an existing way to do this, or "
                     "does it need building from scratch?",
                     "this move has no implementation yet"),
    Gap.INPUTS: ("What information do you have going in, and what is hidden "
                 "from you?", "it separates a known problem from a guess"),
    Gap.OUTPUTS: ("What should exist at the end that does not exist now?",
                  "it is what gets checked"),
}


@dataclass
class Question:
    gap: str
    text: str
    unblocks: str
    value: float                    # bits: how much of the frame this settles
    answered_by_memory: str | None = None

    def render(self) -> str:
        prefix = f"[{self.gap}]"
        if self.answered_by_memory:
            return f"  {prefix} {self.text}\n      (memory already says: {self.answered_by_memory[:100]})"
        return f"  {prefix} {self.text}\n      -> {self.unblocks}"


@dataclass
class Clarification:
    frame: GameFrame
    plan: object
    questions: list[Question] = field(default_factory=list)
    rounds: int = 0
    asked: list[str] = field(default_factory=list)
    contested: list[str] = field(default_factory=list)   # asked, still unresolved
    resolved: list[str] = field(default_factory=list)
    actionable: bool = False
    reason: str = ""

    def render(self) -> str:
        lines = [self.frame.render(), ""]
        if self.actionable:
            lines.append(f"  ready: {self.reason}")
            if self.plan and self.plan.items:
                lines.append(f"  first step: {self.plan.items[0]}")
        else:
            lines.append(f"  not ready: {self.reason}")
        if self.questions:
            lines.append("\n  questions, most blocking first:")
            lines += [q.render() for q in self.questions]
        if self.contested:
            lines.append(f"\n  asked and still open: {', '.join(self.contested)}")
        return "\n".join(lines)


def objective_unclear(frame: GameFrame) -> bool:
    """Is the objective genuinely not understood?

    The tempting test -- "the objective is just the task restated" -- is wrong,
    and gating on it broke eight working tasks. `Framer` falls back to the task
    text whenever no explicit purpose clause is present, which is most of the
    time, and for an imperative like *build a csv parser that passes the test
    suite* that fallback is **correct**: the task states its own objective.

    What separates that from *make the thing better* is not whether the objective
    was restated but whether the verb admits a check. `goals.checkability`
    already draws that line -- observable verbs (compute, parse, build) score
    high, judgement verbs (improve, optimise, clean, better) score low -- and it
    is the same line distillation uses to decide where it may stop. Reusing it
    keeps one definition of "checkable" in the system instead of two that drift.
    """
    if not frame.objective:
        return True
    if frame.objective.strip() != frame.task.strip():
        return False           # an objective was stated separately; take it
    return checkability(frame.task) < 0.5


def gaps(frame: GameFrame, plan) -> list[str]:
    """What the frame could not establish, in blocking order."""
    found: list[str] = []
    if objective_unclear(frame):
        found.append(Gap.OBJECTIVE)
    if not frame.referee:
        found.append(Gap.REFEREE)
    if not frame.actions:
        found.append(Gap.ACTIONS)
    if frame.information == Information.INCOMPLETE:
        found.append(Gap.PAYOFF)
    if plan is not None and plan.gaps:
        found.append(Gap.CAPABILITY)
    if not frame.inputs:
        found.append(Gap.INPUTS)
    if not frame.outputs:
        found.append(Gap.OUTPUTS)
    return [g for g in Gap.ORDER if g in found]


def first_step_actionable(frame: GameFrame, plan) -> tuple[bool, str]:
    """Can the first agenda item actually be attempted?

    This is the termination condition, and it is deliberately weaker than
    "the frame is complete". Three things must hold: the system can say what
    winning means, it has at least one move, and the first item on the agenda is
    covered by a primitive or a tool. Everything else -- input formats, payoff
    structure, who else is playing -- can be discovered by attempting the step,
    and discovering it that way is cheaper than asking.
    """
    if objective_unclear(frame):
        return False, "no objective: cannot say what would count as done"
    if not frame.actions:
        return False, "no move set: nothing to choose between"
    if plan is None or not plan.items:
        return False, "no agenda: nothing to attempt"
    first = plan.items[0]
    if first in plan.needs_person:
        return False, f"first agenda item needs a person: {first}"
    # A capability gap is not a blocker -- forging is itself a primitive, and
    # "write the tool you are missing" is exactly the step to take. What blocks
    # is an item nothing can discharge without an answer from someone.
    return True, f"first agenda item is executable: {first}"


class Clarifier:
    def __init__(self, memory, framer: Framer, toolbox=None) -> None:
        self.memory = memory
        self.framer = framer
        self.toolbox = toolbox

    # -- asking ---------------------------------------------------------------

    def questions(self, frame: GameFrame, plan, skip: set[str] | None = None,
                  limit: int = 4) -> list[Question]:
        skip = skip or set()
        out: list[Question] = []
        for gap in gaps(frame, plan):
            if gap in skip:
                continue
            template, unblocks = _QUESTIONS[gap]
            detail = ""
            if gap == Gap.CAPABILITY and plan is not None and plan.gaps:
                detail = ", ".join(c.action for c in plan.gaps[:3])
            text = template.format(detail=detail) if "{detail}" in template else template
            # Value falls with position in the blocking order: the first unknown
            # is worth a full bit, each later one less, because later answers are
            # partly determined by earlier ones.
            rank = Gap.ORDER.index(gap)
            out.append(Question(gap, text, unblocks,
                                value=round(entropy(0.5) / (1.0 + rank), 4),
                                answered_by_memory=self._memory_answer(gap, frame)))
            if len(out) >= limit:
                break
        return out

    def _memory_answer(self, gap: str, frame: GameFrame) -> str | None:
        """Has this exact gap already been answered for this exact game?

        Matched on the `gap` and `task` keys `absorb` writes, not by similarity.
        Two earlier versions tried similarity and both were wrong in opposite
        directions: querying with the question text never cleared the threshold
        (the question is generic prose, the stored fact is a short labelled
        assertion), and querying with the task text cleared it for *every* gap,
        because the task dominates the vector and one gap token cannot separate
        them. The second failure is the dangerous one -- it silently answered
        questions from unrelated facts and stopped asking them.

        Pre-empting a question is only safe when the match is exact. Skipping a
        question the system merely *suspects* was answered is strictly worse than
        asking it again.
        """
        for trace in self.memory.of_kind(Kind.FACT):
            if trace.meta.get("gap") == gap and trace.meta.get("task") == frame.task:
                return trace.text
        return None

    # -- absorbing ------------------------------------------------------------

    def absorb(self, frame: GameFrame, answers: dict[str, str]) -> GameFrame:
        """Fold answers back into the frame and re-derive its confidence.

        Answers are written into memory as facts about this game, so the next
        time something structurally similar comes up the question does not need
        asking again -- which is the whole reason the frame is a memory record
        rather than a local variable.
        """
        for gap, answer in answers.items():
            answer = (answer or "").strip()
            if not answer:
                continue
            if gap == Gap.OBJECTIVE:
                frame.objective = answer[:120]
            elif gap == Gap.REFEREE:
                frame.referee = answer[:60] if answer.lower() not in ("none", "nothing") else ""
            elif gap == Gap.ACTIONS:
                frame.actions = [a.strip() for a in answer.replace(";", ",").split(",") if a.strip()][:8]
            elif gap == Gap.INPUTS:
                frame.inputs = [a.strip() for a in answer.replace(";", ",").split(",") if a.strip()][:6]
            elif gap == Gap.OUTPUTS:
                frame.outputs = [a.strip() for a in answer.replace(";", ",").split(",") if a.strip()][:6]
            elif gap == Gap.PAYOFF and answer:
                frame.information = Information.IMPERFECT     # payoffs are now known
            frame.evidence.append(f"{gap} supplied by the user")
            self.memory.remember(
                Kind.FACT, f"about the game {frame.task!r}: {gap} is {answer}",
                meta={"task": frame.task, "gap": gap}, grade=1.0, source=Source.USER)
        frame.confidence = self._confidence(frame)
        return frame

    @staticmethod
    def _confidence(frame: GameFrame) -> float:
        """Recomputed from what is now filled, over the same seven axes `Framer`
        scores. Kept here rather than mutating `Framer` so a user-supplied answer
        and a lexically-guessed one are scored on the same scale."""
        filled = sum([
            bool(frame.objective and frame.objective.strip() != frame.task.strip()),
            bool(frame.actions), bool(frame.referee), bool(frame.inputs),
            bool(frame.outputs), frame.information != Information.INCOMPLETE,
            bool(frame.players),
        ])
        return round(min(1.0, filled / 7.0), 3)

    # -- the loop -------------------------------------------------------------

    def clarify(self, task: str, ask=None, max_rounds: int = 3) -> Clarification:
        """Frame, then ask until the first step is actionable.

        `ask(questions) -> dict[gap, answer]` is anything that answers: a human at
        a prompt, a caller, a test. **With no `ask`, this returns the questions
        instead of guessing** -- which is the correct behaviour for a system that
        does not understand the objective, and the reason `solve()` can hand them
        upward rather than proceeding on an assumption.
        """
        frame = self.framer.frame(task)
        caps = capabilities(frame, self.memory, self.toolbox)
        plan = agenda(frame, caps)
        asked: set[str] = set()
        contested: list[str] = []
        resolved: list[str] = []

        for round_no in range(max_rounds + 1):
            ok, reason = first_step_actionable(frame, plan)
            if ok:
                self.framer.remember(frame)
                return Clarification(frame, plan, [], round_no, sorted(asked),
                                     contested, resolved, True, reason)
            # Incidentals are asked once; a *blocking* gap is asked again while
            # it remains unanswered. Skipping everything already asked was tidier
            # and wrong: with the objective unanswered and nothing else left to
            # ask, the loop ran out of questions and gave up on the one thing
            # that was actually preventing progress. "Ask until the first step is
            # done" means exactly this gap, and only this gap, is worth repeating.
            skip = {g for g in asked if g not in Gap.BLOCKING}
            pending = self.questions(frame, plan, skip=skip)
            if not pending or ask is None or round_no == max_rounds:
                self.framer.remember(frame)
                why = (reason if not pending else
                       "waiting on answers" if ask is None else
                       f"still not actionable after {round_no} round(s): {reason}")
                return Clarification(frame, plan, pending, round_no, sorted(asked),
                                     contested, resolved, False, why)

            # Anything memory already answers is applied without asking.
            free = {q.gap: q.answered_by_memory for q in pending if q.answered_by_memory}
            to_ask = [q for q in pending if not q.answered_by_memory]
            answers = dict(free)
            if to_ask:
                answers.update(ask(to_ask) or {})
            asked.update(q.gap for q in pending)
            repeated = {q.gap for q in pending if q.gap in Gap.BLOCKING}

            before = frame.confidence
            frame = self.absorb(frame, answers)
            # A gap that was asked about and did not move the frame is contested,
            # not forgotten. Re-asking it would be asking the same question of
            # someone who has already tried to answer it.
            for q in pending:
                target = resolved if answers.get(q.gap) else contested
                if q.gap not in target:
                    target.append(q.gap)
            # An answer can arrive for a gap raised in an earlier round. Without
            # this, the gap stays on the contested list forever and the report
            # says a question is open that the person already answered.
            for gap in answers:
                if answers[gap] and gap in contested:
                    contested.remove(gap)
                    if gap not in resolved:
                        resolved.append(gap)
            caps = capabilities(frame, self.memory, self.toolbox)
            plan = agenda(frame, caps)
            if frame.confidence <= before and not answers:
                break

        ok, reason = first_step_actionable(frame, plan)
        self.framer.remember(frame)
        return Clarification(frame, plan, [] if ok else self.questions(frame, plan, skip=asked),
                             max_rounds, sorted(asked), contested, resolved, ok, reason)
