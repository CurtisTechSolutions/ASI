"""A model conversing with itself.

Two voices take turns.  Every reply is the prediction search run from the
end of what was just said: the tail of the previous line (its last
``context`` characters, snapped to a word boundary) is located in the graph
and the search continues it to the end of a text, so each speaker picks up
the other's last words and carries on from them.  ``"beam"`` offers the
``k`` most likely continuations and the first one the conversation has not
heard yet is spoken - a conversation that never repeats itself, and a
deterministic one.  An utterance counts as heard when it was said before,
when it adds the words some earlier reply already added (after a different
context), or when it is a mere echo (a piece of a line already spoken); a
reply that repeats itself - the same run of words twice in a row, "say
morning morning" - is skipped the same way;
``"sample"`` draws stochastic walks.  When nothing follows the tail of a
line, or the graph only knows part of its last word, the context is
shortened a word at a time (what else could follow "on the"?) and, if the
graph still has nothing to add, the voice changes the subject with a fresh
text from START.  The two voices may be two different models (``partner``),
e.g. the radix and the count model talking to each other.

When every candidate is a repeat the best one is spoken anyway and the turn
is flagged ``repeat=True``; speaking that same duplicate a second time would
only go round in circles, so the conversation ends there instead.
:func:`repeats` collects the utterances spoken twice so they can be punished
(the thumbs down of the Converse tab, the 2NRL negative phase), which is how
the model is taught out of the duplicates it cannot avoid by itself.

Both kinds of repetition are settings: ``avoid_repeats`` for what the
conversation has already heard, ``avoid_word_repeats`` for what one utterance
says twice in a row.  With a setting off that kind is neither skipped nor
flagged, and so is never punished.

A conversation can be **streamed** as it happens (``stream=``, a callable
that takes one event dict; :data:`StreamFn`).  The stream has two layers.
``turn`` events are the response: a turn is spoken once and never taken back,
so they can be appended to a transcript as they arrive.  Everything between
two of them is the *window* - what the voice is doing before it commits, and
what a backtrack may still rewrite: ``look`` (the context it continues, losing
a word at a time), ``draft`` (what it was about to say), ``caught`` (the words
it caught itself on and what it keeps), ``backtrack`` (each step back, and the
cut it explores from), ``found`` (another way on) or ``stuck`` (nothing new
from any cut).  Streaming changes nothing about what is said: the turns are
the ones :func:`converse` returns, event for event.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .beam import path_probability
from .graph import FIRST, START
from .search import PathResult
from .thinking import THINK_DEPTH, Thought, think

if TYPE_CHECKING:  # pragma: no cover
    from .model import GraphModel

DEFAULT_SPEAKERS: tuple[str, ...] = ("A", "B")
MODES = ("beam", "sample")

Veto = Callable[[str], bool]
"""What a voice may not say: ``veto(utterance)`` is true for a candidate the speaker must not speak.

The conversation knows nothing about *why* - :class:`radixnet.duo.NegativeFilter` passes its own judgement in
(the negative network guarding the positive one), and a candidate it refuses is skipped exactly like one that
had been said before, except that it may not even be the fallback."""


EXPLORE = 3
"""Times a voice may back up out of a repeat by default (0 turns the exploring off)."""

StreamFn = Callable[[dict], Any]
"""Where a conversation streams what it is doing, one event dict at a time, as it happens.

Every event carries ``event`` (its kind), ``index`` and ``speaker`` (whose turn it is).  The committed layer is
``turn`` (``turn``: the :class:`Turn` as :meth:`Turn.to_dict` writes it) - a turn is spoken once and never taken
back.  The window between two turns is ``look`` (``from``: the context it continues; ``""`` for a fresh text from
START), ``draft`` (``text``, ``cost``: what it was about to say before it caught itself), ``caught`` (``kind``,
``noticed``, ``cut``: what it keeps, ``""`` when it cannot back up), ``backtrack`` (``step``, ``cut``, ``wider``:
one step back and the candidates it weighs from there), ``found`` (``text``, ``cost``, ``explored``) or ``stuck``
(``explored``: nothing new from any cut).  Only a ``turn`` is the model's answer; the rest is what a backtrack
may still rewrite, which is exactly why it is streamed apart.
"""

STREAM_EVENTS: tuple[str, ...] = ("look", "draft", "caught", "backtrack", "found", "stuck", "turn")
"""Every kind of event a streamed conversation emits, in the order a turn goes through them."""


def _tagged(stream: StreamFn | None, index: int, speaker: str) -> StreamFn | None:
    """``stream`` with the turn's ``index`` and ``speaker`` written into every event (``None`` stays ``None``)."""
    if stream is None:
        return None

    def tag(event: dict) -> None:
        stream({"event": event["event"], "index": index, "speaker": speaker,
                **{key: value for key, value in event.items() if key != "event"}})

    return tag


@dataclass
class Rethink:
    """A voice catching itself repeating, and what it did about it.

    The metacognition of a turn.  ``kind`` is what it caught: ``"stutter"`` - its own words, twice in a row - or
    ``"repeat"``, something the conversation had already heard.  ``noticed`` is the words themselves, ``cut``
    what it kept of that attempt (everything up to where it would have started saying them again), ``steps`` how
    many times it backed up, ``explored`` the paths it weighed from there and ``found`` whether one of them said
    something new.  A turn that never had to think twice has no record at all.
    """

    kind: str = ""
    noticed: str = ""
    cut: str = ""
    steps: int = 0
    explored: int = 0
    found: bool = False
    taught: int = -1  # the node it taught to hand over here, or -1 when it taught nothing
    thought: Thought | None = None  # what it thought before backing up (:func:`radixnet.thinking.think`)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "noticed": self.noticed, "cut": self.cut, "steps": self.steps,
                "explored": self.explored, "found": self.found, "taught": self.taught,
                "thought": self.thought.to_dict() if self.thought is not None else None}


@dataclass
class Turn:
    """One utterance of a conversation."""

    index: int
    speaker: str
    text: str
    context: str = ""  # the tail of the previous line this reply picked up ("" for a fresh start)
    reply: str = ""  # what the search added after the context
    cost: float = 0.0
    probability: float = 1.0
    reached_end: bool = False
    fresh: bool = False  # spoken from START: the opening, or nothing followed the previous line
    given: bool = False  # supplied by the caller (the opening), not generated
    repeat: bool = False  # every candidate repeated something; the best one was spoken anyway
    stutter: bool = False  # the utterance says the same run of words twice in a row ("say morning morning")
    rethink: "Rethink | None" = None  # what the voice noticed about a repeat of its own, and how it backed out
    candidates: int = 0  # continuations the search offered for this turn
    skipped: int = 0  # candidates rejected (empty, or already said) before the spoken one
    vetoed: int = 0  # candidates the guard (the negative network) refused for this turn
    labels: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    step_costs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "speaker": self.speaker,
            "text": self.text,
            "context": self.context,
            "reply": self.reply,
            "cost": self.cost,
            "probability": self.probability,
            "reached_end": self.reached_end,
            "fresh": self.fresh,
            "given": self.given,
            "repeat": self.repeat,
            "stutter": self.stutter,
            "rethink": self.rethink.to_dict() if self.rethink is not None else None,
            "candidates": self.candidates,
            "skipped": self.skipped,
            "vetoed": self.vetoed,
            "labels": list(self.labels),
            "node_ids": list(self.node_ids),
            "step_costs": list(self.step_costs),
        }


def tail_context(text: str, chars: int) -> str:
    """The last ``chars`` characters of ``text``, cut forward to a word boundary when a word straddles the cut."""
    text = text.rstrip()
    if chars <= 0 or not text:
        return ""
    if len(text) <= chars:
        return text.lstrip()
    tail = text[-chars:]
    cut = tail.find(" ")
    if cut >= 0 and tail[cut + 1 :].strip():
        tail = tail[cut + 1 :]
    return tail.lstrip()


def normalize(text: str) -> str:
    """The key two utterances are compared by: whitespace-collapsed, case-folded."""
    return " ".join(text.split()).casefold()


LONGEST_STUTTER = 4
"""Words a run may be long for :func:`stutter` to call its immediate repetition a stutter."""


def _caught(text: str, longest: int) -> tuple[str, int]:
    """``(the run said twice in a row, where in ``text`` it starts saying it again)``, or ``("", -1)``."""
    found = [(m.start(), m.group().casefold()) for m in re.finditer(r"\S+", text)]
    words = [word for _at, word in found]
    for i in range(len(words)):
        for run in range(1, min(longest, (len(words) - i) // 2) + 1):
            if words[i:i + run] == words[i + run:i + 2 * run]:
                return " ".join(words[i:i + run]), found[i + run][0]
    return "", -1


def stutter(text: str, longest: int = LONGEST_STUTTER) -> str:
    """The words an utterance says twice in a row, or ``""`` when it says each thing once.

    A *stutter* is a run of one to ``longest`` words repeated immediately after itself - "the **the** west",
    "say morning **morning**", "**the cat** the cat sat" - the shape a cyclic graph falls into when it walks a
    loop instead of going somewhere.  Words that come back later in the line are not a stutter: "where there is
    a will there is a way" says its words again, and says something with them.
    """
    return _caught(text, longest)[0]


def _last_word_at(text: str) -> int:
    """Where the last word of ``text`` starts (a character index), or ``-1`` when it has fewer than two.

    Where a voice repeating a whole utterance has to differ: everything before it can be said again, and only
    the end of the line is the voice's to change.
    """
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    return starts[-1] if len(starts) > 1 else -1


def stutter_at(text: str, longest: int = LONGEST_STUTTER) -> int:
    """Where a :func:`stutter` starts saying itself again (a character index into ``text``), or ``-1``.

    This is where the walk went round: everything before it was said once, and a voice backing out of the loop
    keeps exactly that much (:func:`backtrack`).
    """
    return _caught(text, longest)[1]


def transcript(turns: Sequence[Turn]) -> str:
    """``speaker: text`` lines."""
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def repeats(turns: Sequence[Turn]) -> list[str]:
    """The duplicates a conversation could not avoid: the utterances of the turns flagged ``repeat``, each once.

    These are the texts to punish - the thumbs down of a 2NRL negative phase - so the model stops offering them.
    An utterance that repeats its own words is among them when ``avoid_word_repeats`` was on: every candidate
    then repeated something, and the one spoken anyway carries the flag like any other.
    """
    out: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        key = normalize(turn.text)
        if turn.repeat and key and key not in seen:
            seen.add(key)
            out.append(turn.text)
    return out


class Heard:
    """What a conversation has already heard, and what therefore counts as a duplicate.

    An utterance duplicates the conversation when

    * it was *said* before (the same words, whitespace and case aside), or
    * its reply *adds* what some earlier reply already added - the same continuation reached from another
      context ("blowers" answered with "blower" twice), or
    * it merely *echoes* a line already spoken: the whole utterance sits inside one of them ("the west" after
      "on the west" adds nothing).

    A longer utterance that happens to contain an earlier one is not a duplicate: it says more than was heard.
    """

    def __init__(self, texts: Sequence[str] = ()) -> None:
        self.keys: list[str] = []  # the utterances heard, in order
        self.said: set[str] = set()
        self.added: set[str] = set()  # the words the replies added
        for text in texts:
            self.remember(text)

    def remember(self, text: str, reply: str = "") -> None:
        """Take an utterance (and the words its reply added) into the conversation."""
        key = normalize(text)
        if key and key not in self.said:
            self.said.add(key)
            self.keys.append(key)
        added = normalize(reply)
        if added:
            self.added.add(added)

    def match(self, text: str, reply: str = "") -> str:
        """What speaking ``text`` (a reply adding ``reply``) would repeat, or ``""`` when it says something new.

        The words it clashes with: the utterance it was going to say again, the words an earlier reply already
        added, or the line it would only echo - what a voice thinking twice about it has caught itself doing.
        """
        key = normalize(text)
        if not key:
            return ""
        if key in self.said:
            return key
        added = normalize(reply)
        if added and added in self.added:
            return added
        for heard in self.keys:
            if key in heard:
                return heard
        return ""

    def duplicate(self, text: str, reply: str = "") -> bool:
        """Whether speaking ``text`` (a reply adding ``reply``) would repeat the conversation."""
        return bool(self.match(text, reply))


def _check(turns: int, max_length: int, context: int, k: int, beam: int | None, temperature: float,
           step_penalty: float, speakers: Sequence[str]) -> None:
    if turns < 0:
        raise ValueError(f"turns must be >= 0, got {turns}")
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    if context < 0:
        raise ValueError(f"context must be >= 0, got {context}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if beam is not None and beam < 1:
        raise ValueError(f"beam must be >= 1, got {beam}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if step_penalty < 0:
        raise ValueError(f"step_penalty must be >= 0, got {step_penalty}")
    if not speakers or not all(isinstance(s, str) and s.strip() for s in speakers):
        raise ValueError("speakers must be one or more non-empty names")


def _candidates(
    voice: "GraphModel", context: str, mode: str, k: int, beam: int | None, max_length: int,
    step_penalty: float, temperature: float, rng: random.Random | None,
) -> list[PathResult]:
    """Continuations of ``context`` (``""``: texts from START), most likely first; ``text`` is the continuation and
    ``full_text`` the whole utterance."""
    if mode == "sample":
        walk = voice._search(context, 0, "sample", 0, None, step_penalty, temperature, False, max_length, rng=rng)
        return [walk]
    found = voice.predict(
        context, length=0, mode="beam", k=k, beam=beam, to_end=True, max_length=max_length, step_penalty=step_penalty,
    )
    return list(found.top)


def _offer(
    voice: "GraphModel", context: str, mode: str, count: int, beam: int | None, max_length: int,
    step_penalty: float, temperature: float, rng: random.Random | None,
) -> list[PathResult]:
    """Up to ``count`` continuations of ``context``: the ``count`` most likely (beam), or that many walks (sample)."""
    if mode != "sample":
        return _candidates(voice, context, mode, count, beam, max_length, step_penalty, temperature, rng)
    drawn: list[PathResult] = []
    for _draw in range(count):
        drawn.extend(_candidates(voice, context, mode, 1, beam, max_length, step_penalty, temperature, rng))
    return drawn


def teach_back(voice: "GraphModel", text: str, at: int, found: PathResult | None, amount: float = 1.0) -> int:
    """Teach the graph what a rethink just found out, and return the node it was taught at (``-1``: none).

    A voice that has to back up has learned something no corpus could tell it: *this* is where its walks go
    round.  The node it backed up to gets an edge into ``BACK`` (:meth:`RadixCyclicGraph.observe_back`), the
    step it was about to loop through gets dearer, and the step it took instead - when it found one - gets
    cheaper.  Nothing else in the model is touched, and the whole of it is one bump of three ordinary edges.

    From then on the search itself hands over at that node (:func:`radixnet.search.onward`), wherever it is
    walking: the trait is the model's, not the conversation's.
    """
    node, went, instead = _backing(voice, text, at, found)
    if node < 0:
        return -1  # nothing of its own to mark: the repeat started where the graph could not place it
    voice.graph.observe_back(node, went=went, instead=instead, amount=amount)
    return node


def _backing(voice: "GraphModel", text: str, at: int, found: PathResult | None) -> tuple[int, int | None, int | None]:
    """Where a rethink backs up to, and what it teaches: ``(node, went, instead)``, ``node=-1`` when unplaceable."""
    graph = voice.graph
    node, _offset, lead = voice._prefix_start(text[:at])
    if node < FIRST or lead or not graph.alive[node]:
        return -1, None, None
    went, _o, went_lead = voice._prefix_start(text[:at + len(text[at:].split(" ")[0])])
    instead = found.node_ids[1] if found is not None and len(found.node_ids) > 1 else None
    return (
        node,
        None if went_lead or went == node or went not in graph.children[node] else went,
        None if instead is None or instead not in graph.children[node] else instead,
    )


def think_back(
    voice: "GraphModel",
    text: str,
    at: int,
    found: PathResult | None,
    kind: str,
    *,
    learn: bool = True,
    think_depth: int = THINK_DEPTH,
    mode: str = "beam",
    k: int = 5,
    beam: int | None = None,
    max_length: int = 60,
    step_penalty: float = 0.0,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> Thought | None:
    """A voice that caught itself repeating **thinks** before it backs up (:func:`radixnet.thinking.think`).

    The event is the rethink itself (``kind``: ``"stutter"`` or ``"repeat"``), the node it thinks at is the
    one it backed up to, and when the thought stops it hands over to ``BACK`` - the lesson :func:`teach_back`
    used to write directly: the node hands over, the step it was about to loop through gets dearer, the step it
    took instead cheaper.  With ``learn`` off the voice still thinks, and teaches nothing.  ``None`` when the
    graph cannot place the repeat: nothing of its own to think at.
    """
    node, went, instead = _backing(voice, text, at, found)
    if node < 0:
        return None
    return think(
        voice, at=node, about=text, trigger=kind, went=went, instead=instead, mode=mode, k=k, beam=beam,
        max_length=max_length, step_penalty=step_penalty, temperature=temperature, rng=rng,
        max_depth=think_depth, learn=learn,
    )


@dataclass
class _Pick:
    """What one look through the candidates turned up."""

    spoken: PathResult | None = None  # what to say, or the best repeat when everything repeated
    skipped: int = 0
    repeat: bool = False  # ``spoken`` repeats something: nothing else was left
    vetoed: int = 0
    caught: PathResult | None = None  # the best candidate rejected for repeating: the one worth backing out of

def backtrack(
    voice: "GraphModel",
    text: str,
    keep: str = "",
    heard: Heard | None = None,
    *,
    added: str = "",
    explore: int = EXPLORE,
    mode: str = "beam",
    k: int = 5,
    beam: int | None = None,
    max_length: int = 60,
    step_penalty: float = 0.0,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    avoid_repeats: bool = True,
    avoid_word_repeats: bool = True,
    learn: bool = True,
    veto: "Veto | None" = None,
    stream: StreamFn | None = None,
    think: bool = True,
    think_depth: int = THINK_DEPTH,
) -> tuple[PathResult | None, Rethink]:
    """A voice that caught itself repeating goes back to where it would have started saying it again, and looks
    for another way on.

    ``text`` is the utterance it was about to say (``added`` the words its reply would add to the context), and
    what it caught itself doing decides where it backs up to:

    * a **stutter** - its own words twice in a row - is cut at :func:`stutter_at`, where the walk went round:
      everything before that was said once, so it is kept;
    * a **repeat** of something the conversation has already heard (:meth:`Heard.match`) is cut at
      :func:`_last_word_at`: the whole line is a retread, so it keeps as much of it as it can and differs at
      the end - the latest point at which it still can.

    The search then runs again from the cut - a longer prefix than the turn started with, which *forces* the
    walk to leave the line at exactly that point; asking the same question again from the context would only
    rank the same answers.  Nothing is found, or everything found repeats too?  Then it backs up one word
    further and looks wider, ``explore`` times over.

    ``keep`` is what it may not rewrite - the context it picked up from the other voice - so a voice rethinks
    what it said, never what it heard.  ``(candidate, rethink)``: the candidate continues the kept words (its
    ``full_text`` is the whole utterance, its ``cost`` the path it explored from where it backed up), and the
    :class:`Rethink` says what it noticed and did, found or not.

    With ``learn`` the voice does not only get out of this one: :func:`teach_back` writes what it found out
    into the graph, so the *model* learns where it goes round and the search hands over there by itself from
    then on, in any walk it makes.  That is the whole point of doing it here - a procedure that has to be
    re-run every time has learned nothing.

    With ``think`` (the default) catching itself is an **event the voice thinks about** first
    (:func:`think_back`): the node it backed up to is taught as a place to stop and think, the thought is
    walked from ``THINK`` - questioning itself up to ``think_depth`` deep where it has learned to - and when
    the thought stops it hands over to ``BACK`` with the same lesson :func:`teach_back` teaches.  The
    ``Rethink`` carries the thought.
    ``stream`` watches it happen (:data:`StreamFn`): ``caught`` the moment it notices (with the ``cut`` it will
    keep, or ``""`` when it cannot back up), ``backtrack`` for every step back, then ``found`` or ``stuck``.  The
    events are the same whatever they are written to, and they change nothing about what is found.
    """
    heard = Heard() if heard is None else heard
    kind, noticed, at = "", "", -1
    if avoid_word_repeats:
        noticed, at = _caught(text, LONGEST_STUTTER)
        kind = "stutter" if noticed else ""
    if not noticed and avoid_repeats:
        noticed = heard.match(text, added)
        if noticed:
            kind, at = "repeat", _last_word_at(text)
    record = Rethink(kind=kind, noticed=noticed)
    if noticed and stream is not None:
        # what it will keep - "" when there is no backing up from here: nowhere to cut, the exploring off, the
        # repeat inside the words it picked up, or nothing of its own before it
        cut = text[:at] if at >= 0 else ""
        kept = cut if at >= 0 and explore > 0 and len(cut) >= len(keep) and cut.strip() else ""
        stream({"event": "caught", "kind": kind, "noticed": noticed, "cut": kept})
    if not noticed or at < 0 or explore <= 0:
        return None, record
    cut = text[:at]
    if len(cut) < len(keep):  # the other voice's words: not this one's to rethink
        return None, record
    for step in range(explore):
        if not cut.strip():  # nothing of its own left to keep: changing the subject is the caller's move
            break
        record.cut, record.steps = cut, step + 1
        wider = k * (step + 2)  # the further back it goes, the wider it looks
        if stream is not None:
            stream({"event": "backtrack", "step": step + 1, "cut": cut, "wider": wider})
        for cand in _offer(voice, cut, mode, wider, beam, max_length, step_penalty, temperature, rng):
            record.explored += 1
            if not cand.text.strip() or (veto is not None and veto(cand.full_text)):
                continue
            if (avoid_word_repeats and stutter(cand.full_text)) or \
                    (avoid_repeats and heard.duplicate(cand.full_text, cand.text)):
                continue
            record.found = True
            record.taught = _teach(voice, text, at, cand, record, learn, think, think_depth, mode, k, beam,
                                   max_length, step_penalty, temperature, rng)
            if stream is not None:
                stream({"event": "found", "text": cand.full_text, "cost": cand.cost, "explored": record.explored})
            return cand, record
        shorter = _shorter(cut)
        if len(shorter) < len(keep) or shorter == cut:
            break
        cut = shorter if shorter.endswith(" ") or not shorter else shorter + " "
    # it goes round here even if it found no way out
    record.taught = _teach(voice, text, at, None, record, learn, think, think_depth, mode, k, beam, max_length,
                           step_penalty, temperature, rng)
    if stream is not None and record.steps:
        stream({"event": "stuck", "explored": record.explored})
    return None, record


def _teach(
    voice: "GraphModel", text: str, at: int, found: PathResult | None, record: Rethink, learn: bool, think: bool,
    think_depth: int, mode: str, k: int, beam: int | None, max_length: int, step_penalty: float,
    temperature: float, rng: random.Random | None,
) -> int:
    """What a rethink teaches: a thought that hands over to BACK when it stops (``think``), else BACK directly."""
    if think:
        record.thought = think_back(
            voice, text, at, found, record.kind, learn=learn, think_depth=think_depth, mode=mode, k=k, beam=beam,
            max_length=max_length, step_penalty=step_penalty, temperature=temperature, rng=rng,
        )
        return record.thought.handed_over if record.thought is not None else -1
    return teach_back(voice, text, at, found) if learn else -1




def _pick(
    candidates: Sequence[PathResult], heard: Heard, avoid_repeats: bool, veto: "Veto | None" = None,
    avoid_word_repeats: bool = True,
) -> _Pick:
    """The first candidate that adds something, is not vetoed and repeats nothing - neither what the
    conversation has heard (``avoid_repeats``) nor its own words (``avoid_word_repeats``: a :func:`stutter`).

    When they all repeat something, the best of them is the fallback - the cheapest one that was never said word
    for word (an echo says at least something new about where the voice is), else the cheapest of all; speaking
    it flags the turn a ``repeat``.  A vetoed candidate is never the fallback - that is the whole point of the
    veto.  ``caught`` is the best candidate rejected for repeating - its own words or the conversation's - the
    one worth backing out of (:func:`backtrack`) rather than dropping for a lesser answer.
    """
    out = _Pick()
    fallback_word_for_word = True
    for cand in candidates:
        if not cand.text.strip():
            out.skipped += 1
            continue
        if veto is not None and veto(cand.full_text):
            out.vetoed += 1
            out.skipped += 1
            continue
        heard_before = avoid_repeats and heard.duplicate(cand.full_text, cand.text)
        went_round = avoid_word_repeats and bool(stutter(cand.full_text))
        if heard_before or went_round:
            if out.caught is None:
                out.caught = cand
            word_for_word = normalize(cand.full_text) in heard.said
            if out.spoken is None or (fallback_word_for_word and not word_for_word):
                out.spoken, fallback_word_for_word = cand, word_for_word
            out.skipped += 1
            continue
        out.spoken = cand
        return out
    out.repeat = out.spoken is not None
    return out


def _usable(voice: "GraphModel", context: str) -> bool:
    """Whether the graph knows the end of ``context`` whole: its last trigram is a node (no guessed partial match)."""
    node, _offset, lead = voice._prefix_start(context)
    return node != START and not lead


def _shorter(context: str) -> str:
    """The context without its last word."""
    context = context.rstrip()
    return context.rsplit(" ", 1)[0].rstrip() if " " in context else ""


def converse(
    model: "GraphModel",
    opening: str = "",
    turns: int = 6,
    mode: str = "beam",
    max_length: int = 60,
    context: int = 12,
    temperature: float = 1.0,
    k: int = 5,
    beam: int | None = None,
    step_penalty: float = 0.0,
    seed: int | None = None,
    speakers: Sequence[str] = DEFAULT_SPEAKERS,
    history: Sequence[str] = (),
    partner: "GraphModel | None" = None,
    avoid_repeats: bool = True,
    avoid_word_repeats: bool = True,
    explore: int = EXPLORE,
    learn: bool = True,
    veto: Veto | None = None,
    stream: StreamFn | None = None,
    think: bool = True,
    think_depth: int = THINK_DEPTH,
) -> list[Turn]:
    """``model`` talks to itself (or to ``partner``) for ``turns`` new turns.

    * ``opening`` - the first line, spoken by the first voice as given (returned as turn 0, ``given=True``);
      without one the first voice speaks a fresh text from START.
    * ``history`` - earlier utterances of a conversation being continued (not returned; the reply picks up
      the last one, and the speakers keep alternating from where they left off).
    * ``context`` - characters of the previous line a reply picks up (a word boundary is respected; when
      nothing follows them, or their last word is only partly known, the context loses a word at a time).
    * ``mode`` - ``"beam"``: the ``k`` most likely continuations, the first unheard one is spoken;
      ``"sample"``: stochastic walks (up to ``k`` draws for an unheard one; ``seed`` makes them reproducible).
    * ``max_length`` - characters a voice may add to the context in one turn.
    * ``partner`` - a second model speaking the even-numbered voice (``speakers[1]``); default the same model.
    * ``avoid_repeats`` - skip the candidates that duplicate the conversation (:class:`Heard`: an utterance
      said before, a reply adding what an earlier reply added, an echo of a line already spoken).
    * ``avoid_word_repeats`` - skip the candidates that repeat themselves: a :func:`stutter`, the same run of
      words twice in a row inside the one utterance.
    * ``explore`` - times a voice that caught itself repeating - its own words, or the conversation's - may
      back up to where it would have started saying it again and look for another way on before it gives up on
      the line (:func:`backtrack`; 0 turns the exploring off).
    * ``learn`` - teach the model what each rethink found out (:func:`teach_back`), so the graph itself learns
      where it goes round.  **A conversation with this on changes the model**: two runs of the same
      conversation differ because the first one taught it something.
    * ``veto`` - a candidate the voice may not speak (the guard: see :data:`Veto`).  A turn whose every
      candidate is vetoed falls back like any other dead end - a shorter context, then a fresh text - and the
      conversation stops when there is nothing left that may be said.
    * ``think`` - a voice that catches itself repeating thinks about it before it backs up
      (:func:`think_back`, :mod:`radixnet.thinking`), questioning itself up to ``think_depth`` deep; the
      thought rides on the turn's rethink.  With ``learn`` it also teaches the graph where it stopped to
      think.
    * ``stream`` - where the conversation is streamed as it happens (:data:`StreamFn`): a ``turn`` event for
      every turn spoken, the opening included, and between them what each voice does before it commits.  The
      turns returned are exactly the ones streamed.

    Every generated turn records the context it picked up, its cost and probability, whether it started fresh
    from START (nothing followed the previous line), and whether it had to repeat something already said
    (:func:`repeats` collects those utterances, the ones to punish).
    Generation stops early when a voice has nothing at all to say (an untrained model) and when it can only
    say a duplicate it has already repeated - that conversation would go round in circles.
    """
    if not isinstance(opening, str):
        raise TypeError("opening must be a string")
    if not all(isinstance(h, str) for h in history):
        raise TypeError("history must be a list of strings")
    mode = (mode or "beam").lower()
    if mode == "dijkstra":
        mode = "beam"
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected 'beam' or 'sample'")
    speakers = tuple(speakers)
    _check(turns, max_length, context, k, beam, temperature, step_penalty, speakers)
    rng = random.Random(seed) if seed is not None else None
    voices = (model, partner or model)

    said_list = [h for h in history]
    repeated: set[str] = set()  # the duplicates already spoken: saying one of them again would only loop
    result: list[Turn] = []
    index = len(said_list)
    if opening.strip():
        score = model.score(opening)
        result.append(Turn(
            index=index, speaker=speakers[index % len(speakers)], text=opening, context="", reply=opening,
            cost=-score["log_prob"], probability=path_probability(PathResult(cost=-score["log_prob"])),
            reached_end=True, fresh=True, given=True, candidates=1,
        ))
        _spoken(stream, result[-1])
        said_list.append(opening)
        index += 1
    heard = Heard(said_list)

    for _ in range(turns):
        voice = voices[index % 2]
        turn = reply(
            voice, said_list[-1] if said_list else "", heard=heard, index=index,
            speaker=speakers[index % len(speakers)], mode=mode, max_length=max_length, context=context,
            temperature=temperature, k=k, beam=beam, step_penalty=step_penalty, rng=rng,
            avoid_repeats=avoid_repeats, avoid_word_repeats=avoid_word_repeats, explore=explore, learn=learn,
            veto=veto, think=think, think_depth=think_depth, stream=stream,
        )
        if turn is None:
            break
        if turn.repeat and normalize(turn.text) in repeated:
            break  # the voice can only say a duplicate it has already repeated: the conversation is over
        result.append(turn)
        _spoken(stream, turn)
        said_list.append(turn.text)
        heard.remember(turn.text, turn.reply if turn.context else "")
        if turn.repeat:
            repeated.add(normalize(turn.text))
        index += 1
    return result


def _spoken(stream: StreamFn | None, turn: Turn) -> None:
    """The committed layer of the stream: a turn that has been spoken, and will not be taken back."""
    if stream is not None:
        stream({"event": "turn", "index": turn.index, "speaker": turn.speaker, "turn": turn.to_dict()})


def reply(
    voice: "GraphModel",
    previous: str,
    *,
    heard: Heard | None = None,
    index: int = 0,
    speaker: str = "B",
    mode: str = "beam",
    max_length: int = 60,
    context: int = 12,
    temperature: float = 1.0,
    k: int = 5,
    beam: int | None = None,
    step_penalty: float = 0.0,
    rng: random.Random | None = None,
    avoid_repeats: bool = True,
    avoid_word_repeats: bool = True,
    explore: int = EXPLORE,
    learn: bool = True,
    veto: Veto | None = None,
    stream: StreamFn | None = None,
    think: bool = True,
    think_depth: int = THINK_DEPTH,
) -> Turn | None:
    """What ``voice`` says next after ``previous`` - one turn, or ``None`` when it has nothing to say.

    This is the whole of a conversational turn, and :func:`converse` is a loop
    over it: the tail of ``previous`` is located in the graph and continued
    (``mode``), the context loses a word at a time while nothing follows it,
    and a voice with nothing left to add changes the subject with a fresh text
    from START.  ``heard`` is what the conversation has already heard (so a
    reply does not duplicate it - :class:`Heard`), ``avoid_word_repeats``
    whether a reply may repeat its own words (a :func:`stutter`) and ``veto``
    what the speaker may not say.  Remember the turn in ``heard`` before asking
    for the next one, or the same reply comes back.

    A voice about to repeat - its own words, or something the conversation has
    already heard - does not simply give up on the line: ``explore`` times over
    it backs up to where it would have started saying it again and looks for
    another way on (:func:`backtrack`), and only a voice that cannot think of
    one drops to a lesser answer or changes the subject.  What it noticed and
    did is the turn's :class:`Rethink`, and with ``learn`` what it found out is
    written into the graph (:func:`teach_back`) - so the model, not just this
    turn, comes out of it knowing where it goes round.

    It is public because the other voice need not be a model at all: the chat
    loop (:mod:`radixnet.chat`) has an LLM speak every other line and calls
    this for the model's own.

    ``stream`` (:data:`StreamFn`) watches the turn being found: ``look`` for
    every context it continues (and ``""`` for a fresh text), then - when a
    candidate is caught repeating - ``draft``, and what :func:`backtrack` does
    about it.  The turn itself is not an event of this function's: it is the
    caller's to speak (:func:`converse` streams it as ``turn``), since a reply
    may still be refused for repeating a duplicate already repeated.
    """
    mode = "beam" if mode in ("", "dijkstra") else mode
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected 'beam' or 'sample'")
    _check(0, max_length, context, k, beam, temperature, step_penalty, (speaker,))
    heard = Heard([previous]) if heard is None else heard
    ctx = tail_context(previous, context)
    spoken: PathResult | None = None
    offered = skipped = vetoed = 0
    repeat = False
    rethought: Rethink | None = None
    draws = k if mode == "sample" else 1
    watch = _tagged(stream, index, speaker)

    def think_again(pick: _Pick, keep: str) -> tuple[PathResult | None, bool]:
        """A candidate rejected for repeating - its own words, or the conversation's - is worth backing out of:
        keep what it said up to the repetition and look for another way on, once per turn."""
        nonlocal rethought, offered
        if pick.caught is None or explore <= 0 or rethought is not None:
            return pick.spoken, pick.repeat
        if watch is not None:
            watch({"event": "draft", "text": pick.caught.full_text, "cost": pick.caught.cost})
        found, rethought = backtrack(
            voice, pick.caught.full_text, keep, heard, added=pick.caught.text, explore=explore, mode=mode, k=k,
            beam=beam, max_length=max_length, step_penalty=step_penalty, temperature=temperature, rng=rng,
            avoid_repeats=avoid_repeats, avoid_word_repeats=avoid_word_repeats, learn=learn, veto=veto,
            think=think, think_depth=think_depth,
            stream=watch,
        )
        offered += rethought.explored
        return (found, False) if found is not None else (pick.spoken, pick.repeat)

    while ctx:
        if _usable(voice, ctx):
            if watch is not None:
                watch({"event": "look", "from": ctx})
            for _draw in range(draws):
                cands = _candidates(voice, ctx, mode, k, beam, max_length, step_penalty, temperature, rng)
                offered += len(cands)
                pick = _pick(cands, heard, avoid_repeats, veto, avoid_word_repeats)
                skipped += pick.skipped
                vetoed += pick.vetoed
                spoken, repeat = think_again(pick, ctx)
                if spoken is not None and not repeat:
                    break
            if spoken is not None and not repeat:
                break
        ctx = _shorter(ctx)
    if spoken is None or repeat:
        # nothing (new) follows the previous line: change the subject with a fresh text
        fresh_pick: PathResult | None = None
        fresh_repeat = False
        if watch is not None:
            watch({"event": "look", "from": ""})
        for _draw in range(draws):
            cands = _candidates(voice, "", mode, k, beam, max_length, step_penalty, temperature, rng)
            offered += len(cands)
            pick = _pick(cands, heard, avoid_repeats, veto, avoid_word_repeats)
            skipped += pick.skipped
            vetoed += pick.vetoed
            fresh_pick, fresh_repeat = think_again(pick, "")
            if fresh_pick is not None and not fresh_repeat:
                break
        if fresh_pick is not None and (spoken is None or not fresh_repeat):
            spoken, repeat, ctx = fresh_pick, fresh_repeat, ""
    if spoken is None:
        return None
    # full_text is the whole utterance whatever it was continued from - the context, or the words a rethink kept
    text = spoken.full_text
    return Turn(
        index=index, speaker=speaker, text=text, context=ctx,
        reply=text[len(ctx):], cost=spoken.cost, probability=path_probability(spoken),
        reached_end=spoken.reached_end, fresh=not ctx, repeat=repeat, stutter=bool(stutter(text)),
        rethink=rethought, candidates=offered, skipped=skipped,
        vetoed=vetoed, labels=list(spoken.labels), node_ids=list(spoken.node_ids),
        step_costs=list(spoken.step_costs),
    )
