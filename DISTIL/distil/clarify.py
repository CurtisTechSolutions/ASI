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
from .goals import _CHECKABLE, _UNCHECKABLE, checkability
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
            # The item the gate actually judged. Advisories lead `items`, so
            # printing items[0] named a sentence as the first step.
            doable = self.plan.actionable_items if self.plan else []
            if doable:
                lines.append(f"  first step: {doable[0]}")
        else:
            lines.append(f"  not ready: {self.reason}")
        if self.questions:
            lines.append("\n  questions, most blocking first:")
            lines += [q.render() for q in self.questions]
        if self.contested:
            lines.append(f"\n  asked and still open: {', '.join(self.contested)}")
        return "\n".join(lines)


#: Words that stand in for a subject instead of naming one.
_PLACEHOLDERS = {
    "thing", "things", "stuff", "it", "this", "that", "these", "those",
    "everything", "something", "anything", "nothing", "all", "them",
    "work", "item", "items", "issue", "issues", "problem", "problems",
    "bit", "bits", "part", "parts", "code", "system", "everything",
}
#: Words that add urgency or politeness and no content.
_FILLERS = {
    "now", "please", "somehow", "just", "really", "very", "quickly", "asap",
    "soon", "properly", "again", "up", "out", "better", "more", "less",
    "a", "an", "the", "some", "any", "with", "for", "of", "in", "on", "to",
    "and", "or", "my", "our", "its", "their", "is", "be", "should", "must",
}


def _content_words(task: str) -> set[str]:
    """What the task is actually about, after removing verbs and filler.

    Known verbs are stripped because the verb is a separate signal; what remains
    should be the artefact. If nothing remains, the task named no subject.
    """
    import re as _re
    words = [w for w in _re.findall(r"[a-z][a-z'-]*", task.lower())]
    verbs = set(_CHECKABLE.findall(task.lower())) | set(_UNCHECKABLE.findall(task.lower()))
    # Generic verbs neither pattern lists. Without them "fix everything" kept
    # "fix" as its content word and read as naming a subject, so it passed while
    # the identical "fix the thing" did not.
    verbs |= {"make", "do", "get", "have", "give", "take", "handle", "deal",
              "fix", "sort", "tidy", "update", "change", "ensure", "help",
              "try", "keep", "put", "set", "run", "use", "add", "remove"}
    return {w for w in words
            if w not in _FILLERS and w not in _PLACEHOLDERS and w not in verbs}


def objective_unclear(frame: GameFrame) -> bool:
    """Is the objective genuinely not understood?

    Two signals, and both are needed. `goals.checkability` is used by
    distillation and its own docstring calls it "a prior that only has to be
    right on average" -- so using it directly as a boundary was the mistake, not
    the number it returns. As a gate its errors became refusals and false green
    lights in both directions at once: its no-signal answer is exactly 0.5 and
    the test was `< 0.5`, so "unknown" resolved to "clear" and *handle it
    somehow please* sailed through while *write a parser that produces clean
    output* was refused for containing the word "clean". One filler word decided
    it -- *fix everything* was gated and *fix everything now* was not.

    The two signals:

    **Does the task name something?** A task whose content words are all
    placeholders -- *the thing*, *everything*, *it*, *the stuff* -- has no
    subject, whatever its verb. This is what separates *handle it somehow* from
    *dedupe the records*.

    **Is the verb's result observable?** A judgement verb with no observable verb
    anywhere is a preference, not an outcome. *Improve the design* names a
    subject and still does not say what done looks like; *write a parser that
    produces clean output* contains a judgement word but also says write and
    produce, so it does.

    A referee deliberately does not rescue either. This function is reached only
    when `Framer` could read no objective at all, so letting a judge stand in for
    knowing what winning is confuses two different axes.
    """
    if not frame.objective:
        return True
    if frame.objective.strip() != frame.task.strip():
        return False           # an objective was stated separately; take it
    task = frame.task
    if not _content_words(task):
        return True            # nothing is named, only pointed at
    judgement = bool(_UNCHECKABLE.search(task))
    observable = bool(_CHECKABLE.search(task))
    return judgement and not observable


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
    blocking = [i for i in plan.items if i in plan.needs_person]
    if blocking:
        return False, f"an agenda item needs a person: {blocking[0]}"
    if not plan.capabilities:
        return False, "no capability was evaluated: nothing is known to be playable"
    # The first item that is actual work. Advisories ("nothing here can be
    # self-graded") lead the list because they change how the work should be
    # read, so answering "is the first step executable?" from items[0] asked
    # about a sentence rather than about a step. A capability gap is not a
    # blocker either -- forging is itself a primitive, and "write the tool you
    # are missing" is exactly the step to take.
    doable = plan.actionable_items
    if not doable:
        return False, "the agenda holds no executable step"
    first = doable[0]
    if not (first.startswith("forge a capability") or "(via " in first):
        return False, f"first agenda item is not something this system can execute: {first}"
    return True, f"first agenda item is executable: {first}"


def _skippable(asked: set[str]) -> set[str]:
    """Gaps already asked about that are not worth repeating.

    Incidentals are asked once; a blocking gap stays askable while it is
    unresolved, because it is the only thing standing between the caller and a
    first step.
    """
    return {g for g in asked if g not in Gap.BLOCKING}


#: Phrases that mean "I have no answer" wherever they appear. Each is several
#: words and unambiguous.
_DENIAL_PHRASES = ("don't know", "dont know", "do not know", "no idea",
                   "not known", "not sure", "no clue", "who knows")
#: Single words that mean it ONLY when they are the entire answer. "none",
#: "nothing" and "unknown" are ordinary vocabulary -- "return none when the list
#: is empty" is a perfectly good objective, and matching them anywhere threw it
#: away and reported the question as still open.
_DENIAL_WORDS = ("unknown", "unclear", "unsure", "n/a", "na", "tbd",
                 "none", "nothing", "no", "?", "-")


def _denies_knowledge(answer: str) -> bool:
    """Does this answer say "I don't know" rather than answer the question?

    Two classes, because one rule cannot serve both. Multi-word phrases are
    unambiguous and match anywhere, which is what catches "I don't know" -- the
    phrasing an earlier anchored version missed. Bare words are matched only as
    the whole answer, because they are also ordinary content.
    """
    low = " ".join((answer or "").lower().split())
    if not low:
        return True
    if any(phrase in low for phrase in _DENIAL_PHRASES):
        return True
    return low.strip(" .!") in _DENIAL_WORDS


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
                # The stored ANSWER, not the sentence that wraps it. Returning
                # `trace.text` fed "about the game 'x': objective is y" straight
                # back into `absorb`, which set that whole string as the
                # objective -- and because it no longer equalled the task, the
                # frame then read as understood. A rejected objective laundered
                # itself into an accepted one through the memory layer.
                answer = trace.meta.get("answer")
                return answer if answer else None
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
            if _denies_knowledge(answer):
                # Only PAYOFF was guarded. "I don't know" supplied for the
                # objective was written straight into `frame.objective`, and
                # because it then differed from the task, `objective_unclear`
                # returned False and solve() proceeded on an objective that
                # says nobody has one.
                frame.evidence.append(f"{gap}: asked, and the answer was that it is not known")
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
            elif gap == Gap.PAYOFF:
                # "I don't know" is an answer, and it is not the answer that
                # makes the payoffs known. Treating any non-empty reply as
                # knowledge flipped the game out of INCOMPLETE on the strength of
                # someone saying they had no idea.
                if not _denies_knowledge(answer):
                    frame.information = Information.IMPERFECT
            frame.evidence.append(f"{gap} supplied by the user")
            self.memory.remember(
                Kind.FACT, f"about the game {frame.task!r}: {gap} is {answer}",
                meta={"task": frame.task, "gap": gap, "answer": answer},
                grade=1.0, source=Source.USER)
        frame.confidence = self._confidence(frame)
        return frame

    @staticmethod
    def _confidence(frame: GameFrame) -> float:
        """Recomputed from what is now filled, over the axes that can vary.

        Two corrections. The objective is scored with `objective_unclear`, the
        same test the gate uses -- it previously used the "is it the task
        restated?" comparison that test was written to replace, so a frame could
        be judged actionable and unconfident at once. And `bool(frame.players)`
        is gone: `players` always has a value (it defaults to vs-nature), so it
        contributed a constant 1 to every score and one of the seven axes
        measured nothing.
        """
        filled = sum([
            not objective_unclear(frame),
            bool(frame.actions), bool(frame.referee), bool(frame.inputs),
            bool(frame.outputs), frame.information != Information.INCOMPLETE,
        ])
        return round(min(1.0, filled / 6.0), 3)

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
            pending = self.questions(frame, plan, skip=_skippable(asked))

            # Apply anything memory already answers BEFORE deciding to stop.
            # This used to run after the early return, so on the `ask=None` path
            # -- which is how `solve` calls it -- a question the store had
            # already answered was reported as outstanding and the task was
            # gated again on its own recorded answer.
            known = {q.gap: q.answered_by_memory for q in pending if q.answered_by_memory}
            if known:
                frame = self.absorb(frame, known)
                caps = capabilities(frame, self.memory, self.toolbox)
                plan = agenda(frame, caps)
                # Resolved means the gap is GONE, not that a string arrived --
                # the invariant the ask path enforces. Filing them
                # unconditionally let one Clarification report a gap resolved
                # and ask about it in the same breath.
                open_now = set(gaps(frame, plan))
                for gap in known:
                    target, other = ((contested, resolved) if gap in open_now
                                     else (resolved, contested))
                    if gap in other:
                        other.remove(gap)
                    if gap not in target:
                        target.append(gap)
                ok, reason = first_step_actionable(frame, plan)
                if ok:
                    self.framer.remember(frame)
                    return Clarification(frame, plan, [], round_no, sorted(asked),
                                         contested, resolved, True, reason)
                pending = self.questions(frame, plan, skip=_skippable(asked))

            if not pending or ask is None or round_no == max_rounds:
                self.framer.remember(frame)
                why = (reason if not pending else
                       "waiting on answers" if ask is None else
                       f"still not actionable after {round_no} round(s): {reason}")
                return Clarification(frame, plan, pending, round_no, sorted(asked),
                                     contested, resolved, False, why)

            # Skip a question only when memory's answer actually closed the gap.
            # `answered_by_memory` alone dropped it from `to_ask` every round, so
            # a blocking gap whose stored answer the frame could not use was
            # never asked again -- silently undoing the rule that blockers repeat.
            to_ask = [q for q in pending
                      if not q.answered_by_memory or q.gap in Gap.BLOCKING]
            answers = dict(ask(to_ask) or {}) if to_ask else {}
            asked.update(q.gap for q in pending)

            before = set(gaps(frame, plan))
            frame = self.absorb(frame, answers)
            # A gap that was asked about and did not move the frame is contested,
            # not forgotten. Re-asking it would be asking the same question of
            # someone who has already tried to answer it.
            caps = capabilities(frame, self.memory, self.toolbox)
            plan = agenda(frame, caps)

            # Resolved means the gap is GONE from the frame, not that a string
            # was supplied for it. An answer the frame could not use -- an empty
            # actions list, a referee of "nothing" -- was being filed as resolved
            # and the report claimed progress that had not happened.
            still_open = set(gaps(frame, plan))
            for gap in {q.gap for q in pending} | set(answers):
                target, other = ((contested, resolved) if gap in still_open
                                 else (resolved, contested))
                if gap in other:
                    other.remove(gap)      # a gap is in exactly one of the two
                if gap not in target:
                    target.append(gap)

            if still_open == before and not answers:
                # `answers` now holds only what `ask` returned; memory-supplied
                # answers are applied above and can no longer disguise a round in
                # which nobody told us anything.
                # No answers and no movement. Comparing confidence here was
                # wrong: `absorb` rescales it to a different denominator than
                # `Framer` uses, so the first pass through always "improved" by
                # a change of scale rather than by learning anything.
                rounds_run = round_no + 1
                self.framer.remember(frame)
                return Clarification(frame, plan, self.questions(frame, plan, skip=_skippable(asked)),
                                     rounds_run, sorted(asked), contested, resolved,
                                     False, "asked, and nothing came back that moved the frame")

        ok, reason = first_step_actionable(frame, plan)
        self.framer.remember(frame)
        # `skip=asked` here withheld the blocking question from the caller on
        # exactly the path where it is the only thing worth asking -- the loop
        # ran out of rounds with the objective still unknown and then returned no
        # questions at all.
        return Clarification(frame, plan,
                             [] if ok else self.questions(frame, plan, skip=_skippable(asked)),
                             max_rounds, sorted(asked), contested, resolved, ok, reason)
