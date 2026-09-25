"""Thinking - the fourth sentinel at work: what makes the model think, what it thinks, and what happens when it stops.

The graph's ``THINK`` sentinel (:data:`radixnet.graph.THINK`) faces both ways.
Its **in-edges** are where the model has learned to stop and think: an edge
``p -> THINK`` is taught by experience, the way ``BACK``'s are, whenever an
*event* at ``p`` called for a thought.  Its **out-edges** are how thoughts
begin: a thought is a text whose walk starts at ``THINK`` instead of ``START``,
observed from an LLM's thinking the way texts are observed from a corpus
(:func:`think_on`, fed by ``radixnet ollama think`` -
:func:`radixnet.ollama.thoughts_from_prompt`).

:func:`think` is one thought.  It is **triggered** by an event - a voice that
caught itself repeating (:func:`radixnet.dialogue.backtrack`), a question asked
of the model (``radixnet think --about ...``), a thought questioning itself - and
it does four things, in order:

1. it **teaches where** it had to think: ``observe_think(p)`` on the node the
   event happened at (unless the thought is one the model asked itself);
2. it **thinks**: the prediction search run from ``THINK`` to the end of a
   text, so the thought is in the language of the thoughts it was taught
   (nothing, when it was taught none - a model cannot think in words it has
   not learned);
3. it may **question itself**: along the thought's own path, at a node where
   the model has learned to think (:meth:`~radixnet.graph.RadixCyclicGraph.thinks_at`),
   a nested thought is triggered - one that must say something the chain of
   thoughts above it has not - up to ``max_depth`` deep and ``max_questions``
   per thought;
4. when it **stops**, it triggers the sentinel the event calls for: a thought
   that was thinking its way out of a repeat hands over to ``BACK``
   (``observe_back`` on the node, with the step it was about to loop through
   and the one it took instead - the rethink's lesson, now taught *after*
   thinking rather than instead of it); a thought the model asked itself
   returns to the thought that asked (``"think"``); a thought that was merely
   asked for ends (``"end"``).

Every thought is a :class:`Thought` record - the trigger, the node, the text,
its questions, how it stopped and what it triggered - and rides on the turn's
:class:`~radixnet.dialogue.Rethink` as ``thought`` when a conversation thought.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .graph import BACK, END, FIRST, START, THINK

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import GraphModel
    from .search import PathResult

__all__ = [
    "ASKED",
    "BACKING",
    "QUESTIONED",
    "REPEAT",
    "STUTTER",
    "THEN_BACK",
    "THEN_END",
    "THEN_THINK",
    "THINK_DEPTH",
    "THINK_LENGTH",
    "THINK_QUESTIONS",
    "Thought",
    "place",
    "questions_in",
    "think",
    "think_on",
]

ASKED = "asked"
"""The trigger of a thought somebody asked for (``radixnet think``, ``POST /api/think``)."""
QUESTIONED = "questioned"
"""The trigger of a thought a thought asked itself."""
STUTTER, REPEAT = "stutter", "repeat"
"""The triggers of a thought a voice had on catching itself repeating (:class:`radixnet.dialogue.Rethink`)."""
BACKING = (STUTTER, REPEAT)
"""The triggers whose thought hands over to ``BACK`` when it stops: it was thinking its way out of a repeat."""

THEN_BACK, THEN_END, THEN_THINK = "back", "end", "think"
"""What a thought triggers when it stops: the BACK sentinel, the END of it, or the thought it questioned."""
STOPPED_END, STOPPED_LENGTH, STOPPED_NOTHING = "end", "length", "nothing"
"""How a thought stopped: it reached END, it ran out of length, or it had nothing (new) to think."""

THINK_LENGTH = 60
"""Units a thought may run to by default."""
THINK_DEPTH = 2
"""How deep a thought may question itself by default: a thought, and a question about it, and one about that."""
THINK_QUESTIONS = 1
"""Questions one thought may ask itself by default - one, so a thought cannot spend itself questioning."""

_TERMINATORS = ".!?"
_SPACES = " \t\n\r"


@dataclass
class Thought:
    """One thought: what triggered it, what it thought, what it questioned and what it triggered when it stopped."""

    trigger: str = ASKED
    at: int = -1  # the node it thought at (-1: nowhere in particular)
    about: str = ""  # the text it was thinking about, when there was one
    text: str = ""  # the thought itself ("" when it had nothing to think with)
    depth: int = 0  # 0 for a thought, 1 for a question it asked itself, 2 for a question about that
    stopped: str = STOPPED_NOTHING
    then: str = THEN_END  # the sentinel it triggered when it stopped
    taught: int = -1  # the node taught to think here (p -> THINK), or -1
    handed_over: int = -1  # the node taught to hand over when it stopped (p -> BACK), or -1
    cost: float = 0.0
    probability: float = 1.0
    expanded: int = 0  # search states the thought weighed
    questions: list["Thought"] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    step_costs: list[float] = field(default_factory=list)

    @property
    def questioned(self) -> int:
        """How many times the thought questioned itself."""
        return len(self.questions)

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger, "at": self.at, "about": self.about, "text": self.text, "depth": self.depth,
            "stopped": self.stopped, "then": self.then, "taught": self.taught, "handed_over": self.handed_over,
            "cost": self.cost, "probability": self.probability, "expanded": self.expanded,
            "questioned": self.questioned, "questions": [q.to_dict() for q in self.questions],
            "labels": list(self.labels), "node_ids": list(self.node_ids), "step_costs": list(self.step_costs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Thought":
        """A thought read back from :meth:`to_dict` (what a streamed turn carries), questions and all."""
        return cls(
            trigger=str(d.get("trigger", ASKED)), at=int(d.get("at", -1)), about=str(d.get("about", "")),
            text=str(d.get("text", "")), depth=int(d.get("depth", 0)), stopped=str(d.get("stopped", STOPPED_NOTHING)),
            then=str(d.get("then", THEN_END)), taught=int(d.get("taught", -1)), handed_over=int(d.get("handed_over", -1)),
            cost=float(d.get("cost", 0.0)), probability=float(d.get("probability", 1.0)),
            expanded=int(d.get("expanded", 0)),
            questions=[cls.from_dict(q) for q in (d.get("questions") or []) if isinstance(q, dict)],
            labels=[str(x) for x in (d.get("labels") or [])], node_ids=[int(x) for x in (d.get("node_ids") or [])],
            step_costs=[float(x) for x in (d.get("step_costs") or [])],
        )


def questions_in(text: str) -> list[tuple[int, str]]:
    """The sentences of ``text`` that end in a question mark, each with the index it starts at.

    A sentence runs from the first character after the previous sentence's
    terminators (``.``, ``!``, ``?``) to the end of its own; a run of
    terminators containing ``?`` makes it a question.  ``"Hmm. Is that right?
    Yes."`` holds one, at index 5.  This is how a thought is found to have
    questioned itself, in the thinking an LLM wrote as much as in the model's
    own (:func:`think_on`).
    """
    out: list[tuple[int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i] in _SPACES:
            i += 1
        if i >= n:
            break
        start = i
        while i < n and text[i] not in _TERMINATORS:
            i += 1
        j = i
        while j < n and text[j] in _TERMINATORS:
            j += 1
        if i > start and "?" in text[i:j]:
            out.append((start, text[start:j].strip()))
        i = j
    return out


def place(model: "GraphModel", prefix: str) -> int:
    """The node the walk of ``prefix`` ends at, cut so that it ends there, or ``-1`` when the graph cannot place it.

    The last gram of ``prefix`` is located; when it sits inside a compressed
    node the node is split after it, so that an edge taught from the node
    (``observe_think``) fires exactly where the prefix ends and not at the end
    of whatever the node went on to say.  A prefix the graph does not know
    whole - a guessed partial gram, a text it never saw - has no node to
    return, and ``-1`` says so.
    """
    graph = model.graph
    node, offset, lead = model._prefix_start(prefix)
    if node < FIRST or lead or not graph.alive[node]:
        return -1
    enc = graph.encoding
    if offset + enc.n < graph.label_len(node):
        graph.split(node, offset + enc.stride)  # the gram becomes the last of its node
    return node


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _candidates(
    model: "GraphModel", mode: str, k: int, beam: int | None, max_length: int, step_penalty: float,
    temperature: float, rng: random.Random | None,
) -> tuple[list["PathResult"], int]:
    """Up to ``k`` thoughts from THINK, most likely first (beam), or ``k`` walks (sample), and the states weighed."""
    if mode == "sample":
        drawn: list[PathResult] = []
        expanded = 0
        for _ in range(max(1, k)):
            walk = model._search(
                "", max_length, "sample", 0, None, 0.0, temperature, False, max_length, rng=rng, origin=THINK,
            )
            expanded += walk.expanded
            drawn.append(walk)
        return drawn, expanded
    found = model._search("", 0, "beam", k, beam, step_penalty, temperature, True, max_length, origin=THINK)
    return list(found.top), found.expanded


def think(
    model: "GraphModel",
    *,
    about: str = "",
    at: int = -1,
    trigger: str = ASKED,
    went: int | None = None,
    instead: int | None = None,
    mode: str = "beam",
    k: int = 5,
    beam: int | None = None,
    max_length: int = THINK_LENGTH,
    step_penalty: float = 0.0,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    seed: int | None = None,
    depth: int = 0,
    max_depth: int = THINK_DEPTH,
    max_questions: int = THINK_QUESTIONS,
    learn: bool = True,
    amount: float = 1.0,
    _thought: frozenset[str] = frozenset(),
) -> Thought:
    """One thought, triggered by ``trigger`` at node ``at`` (or at the end of ``about``).

    * ``trigger`` says what happened: :data:`ASKED` (somebody asked), :data:`STUTTER` / :data:`REPEAT` (a voice
      caught itself repeating - :func:`radixnet.dialogue.backtrack`), :data:`QUESTIONED` (a thought asked itself),
      or a caller's own word for its event.  It decides what the thought triggers when it stops: :data:`BACKING`
      triggers hand over to ``BACK`` (``went`` / ``instead`` as :meth:`~radixnet.graph.RadixCyclicGraph.observe_back`
      takes them), a question returns to its thought, anything else ends.
    * ``at`` is the node the event happened at; with ``about`` and no ``at`` the node is where ``about`` ends
      (:func:`place`).  With ``learn`` the node is taught to think (``observe_think``) unless the thought is a
      question the model asked itself - it already knows to think there, that is why it asked.
    * ``mode`` / ``k`` / ``beam`` / ``max_length`` / ``step_penalty`` / ``temperature`` / ``rng`` / ``seed`` are the
      prediction search's, run from ``THINK``: ``"beam"`` thinks the most likely thought, ``"sample"`` draws one.
    * ``max_depth`` and ``max_questions`` bound how the thought questions itself (:data:`THINK_DEPTH`,
      :data:`THINK_QUESTIONS`); ``0`` turns it off.

    Returns the :class:`Thought`.  ``learn=False`` thinks without writing anything into the model.
    """
    mode = (mode or "beam").lower()
    if mode == "dijkstra":
        mode = "beam"
    if mode not in ("beam", "sample"):
        raise ValueError(f"unknown mode {mode!r}; expected 'beam' or 'sample'")
    if not isinstance(about, str):
        raise TypeError("about must be a string")
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if beam is not None and beam < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")
    if max_depth < 0 or max_questions < 0:
        raise ValueError("max_depth and max_questions must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if amount < 0:
        raise ValueError(f"amount must be >= 0, got {amount}")
    if rng is None and seed is not None:
        rng = random.Random(seed)
    graph = model.graph
    if at < 0 and about.strip():
        at = place(model, about)
    if at >= 0 and (at < FIRST or at >= len(graph.labels) or not graph.alive[at]):
        raise ValueError(f"node {at} is not a real node to think at")
    record = Thought(trigger=trigger, at=at, about=about, depth=depth)

    # 1. the event teaches where to think - from experience, as BACK is taught
    if learn and at >= FIRST and trigger != QUESTIONED:
        graph.observe_think(at, amount)
        record.taught = at

    # 2. the thought: the search from THINK, in the language of the thoughts it was taught
    candidates, expanded = _candidates(model, mode, k, beam, max_length, step_penalty, temperature, rng)
    record.expanded = expanded
    chosen = None
    for cand in candidates:
        if not cand.text.strip():
            continue
        if _normalize(cand.text) in _thought:
            continue  # a question that only repeats the thought above it is no question
        chosen = cand
        break
    if chosen is not None:
        record.text = chosen.text
        record.cost = chosen.cost
        record.probability = math.exp(-chosen.cost) if math.isfinite(chosen.cost) else 0.0
        record.labels = list(chosen.labels)
        record.node_ids = list(chosen.node_ids)
        record.step_costs = list(chosen.step_costs)
        record.stopped = STOPPED_END if chosen.reached_end else STOPPED_LENGTH

        # 3. questioning itself: where, along its own path, the model has learned to think
        if depth < max_depth and max_questions > 0:
            thought_so_far = _thought | {_normalize(record.text)}
            for node in record.node_ids:
                if node < FIRST or not graph.thinks_at(node):
                    continue
                question = think(
                    model, at=node, trigger=QUESTIONED, mode=mode, k=k, beam=beam, max_length=max_length,
                    step_penalty=step_penalty, temperature=temperature, rng=rng, depth=depth + 1,
                    max_depth=max_depth, max_questions=max_questions, learn=learn, amount=amount,
                    _thought=thought_so_far,
                )
                record.questions.append(question)
                thought_so_far = thought_so_far | {_normalize(question.text)}
                if len(record.questions) >= max_questions:
                    break

    # 4. when thinking stops, it triggers the sentinel the event calls for
    if trigger == QUESTIONED:
        record.then = THEN_THINK
    elif trigger in BACKING and at >= FIRST:
        record.then = THEN_BACK
        if learn:
            graph.observe_back(
                at,
                went=None if went is None or went == BACK or went not in graph.children[at] else went,
                instead=None if instead is None or instead not in graph.children[at] else instead,
                amount=amount,
            )
            record.handed_over = at
    else:
        record.then = THEN_END
    return record


def think_on(
    model: "GraphModel",
    thoughts: Iterable[str] | str,
    *,
    questions: bool = True,
    amount: float = 1.0,
    progress: Any = None,
    stop_event: Any = None,
    **train_options: Any,
) -> dict:
    """Teach the model thoughts: texts whose walk begins at ``THINK``.

    Every thought is trained the way this model trains texts - the same
    structure, the same counting or gradient, the same compression - from the
    other sentinel, so the model learns how *thoughts* begin and go on without
    a word of them leaking into what it says from START (:meth:`GraphModel.train`
    with ``origin=THINK``; ``train_options`` are its ``epochs``, ``lr``, ...).

    With ``questions``, a thought that **questions itself** teaches that too: every
    sentence ending in ``?`` (:func:`questions_in`) marks the node before it as a
    place the model stops to think (``observe_think`` by ``amount``, on the node
    cut to end exactly there - :func:`place`) and is trained as a thought of its
    own, so a thought passing through that node later may question itself with
    it (:func:`think`).  Returns ``{"thoughts", "questions", "taught", "epochs"}``:
    what was learned and the epoch records.
    """
    if model.kind == "negative":
        raise ValueError("the negative network judges; it does not think")
    items = [thoughts] if isinstance(thoughts, str) else list(thoughts)
    cleaned: list[str] = []
    for t in items:
        if not isinstance(t, str):
            raise TypeError(f"thoughts must be strings, got {type(t).__name__}")
        one_line = " ".join(t.split())
        if one_line:
            cleaned.append(one_line)
    records = model.train(cleaned, origin=THINK, progress=progress, stop_event=stop_event, **train_options)
    asked: list[str] = []
    taught = 0
    if questions and cleaned and not (stop_event is not None and stop_event.is_set()):
        graph = model.graph
        for t in cleaned:
            for start, question in questions_in(t):
                node = place(model, t[:start])
                if node >= FIRST:
                    graph.observe_think(node, amount)
                    taught += 1
                asked.append(question)
        # a question is a thought of its own: the model may begin a thought with it
        asked = [q for q in asked if model.encoding.length(q) >= model.encoding.n]
        if asked:
            records = records + model.train(asked, origin=THINK, progress=progress, stop_event=stop_event, **train_options)
    return {"thoughts": len(cleaned), "questions": len(asked), "taught": taught, "epochs": records}


def summarize(thought: Thought) -> str:
    """One line saying what a thought did, for transcripts: *thought "..."; questioned itself once; then backed up*."""
    if thought.text:
        said = f"thought {_quote(thought.text)}"
    elif thought.stopped == STOPPED_NOTHING and thought.depth:
        said = "had nothing new to think"
    else:
        said = "had nothing to think with yet"
    parts = [said]
    if thought.questions:
        times = "once" if thought.questioned == 1 else f"{thought.questioned} times"
        parts.append(f"questioned itself {times}")
    then = {THEN_BACK: "then backed up", THEN_THINK: "then went back to the thought", THEN_END: "then went on"}
    parts.append(then.get(thought.then, f"then {thought.then}"))
    return "; ".join(parts)


def _quote(text: str) -> str:
    return '"' + text.replace('"', '\\"') + '"'


def thoughts_of(turns: Sequence[Any]) -> list[Thought]:
    """Every thought a conversation had, in order (the turns' rethinks' thoughts)."""
    out: list[Thought] = []
    for turn in turns:
        rethink = getattr(turn, "rethink", None)
        thought = getattr(rethink, "thought", None) if rethink is not None else None
        if thought is not None:
            out.append(thought)
    return out


_SENTINELS = (START, END, BACK, THINK)
