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
context), or when it is a mere echo (a piece of a line already spoken);
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
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from .beam import path_probability
from .graph import START
from .search import PathResult

if TYPE_CHECKING:  # pragma: no cover
    from .model import GraphModel

DEFAULT_SPEAKERS: tuple[str, ...] = ("A", "B")
MODES = ("beam", "sample")

Veto = Callable[[str], bool]
"""What a voice may not say: ``veto(utterance)`` is true for a candidate the speaker must not speak.

The conversation knows nothing about *why* - :class:`radixnet.duo.NegativeFilter` passes its own judgement in
(the negative network guarding the positive one), and a candidate it refuses is skipped exactly like one that
had been said before, except that it may not even be the fallback."""


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
    repeat: bool = False  # every candidate had been said before; the best one was spoken anyway
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


def transcript(turns: Sequence[Turn]) -> str:
    """``speaker: text`` lines."""
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def repeats(turns: Sequence[Turn]) -> list[str]:
    """The duplicates a conversation could not avoid: the utterances of the turns flagged ``repeat``, each once.

    These are the texts to punish - the thumbs down of a 2NRL negative phase - so the model stops offering them.
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

    def duplicate(self, text: str, reply: str = "") -> bool:
        """Whether speaking ``text`` (a reply adding ``reply``) would repeat the conversation."""
        key = normalize(text)
        if not key:
            return False
        if key in self.said:
            return True
        added = normalize(reply)
        if added and added in self.added:
            return True
        return any(key in heard for heard in self.keys)


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


def _pick(
    candidates: Sequence[PathResult], heard: Heard, avoid_repeats: bool, veto: "Veto | None" = None,
) -> tuple[PathResult | None, int, bool, int]:
    """``(candidate, skipped, repeat, vetoed)``: the first candidate that adds something, is not vetoed and (when
    asked) does not duplicate what the conversation has heard.

    When they all duplicate it, the best duplicate is the fallback - the cheapest one that was never said word
    for word (an echo says at least something new about where the voice is), else the cheapest of all; speaking
    it flags the turn a ``repeat``.  A vetoed candidate is never the fallback - that is the whole point of the
    veto.
    """
    skipped = vetoed = 0
    fallback: PathResult | None = None
    fallback_word_for_word = True
    for cand in candidates:
        if not cand.text.strip():
            skipped += 1
            continue
        if veto is not None and veto(cand.full_text):
            vetoed += 1
            skipped += 1
            continue
        if avoid_repeats and heard.duplicate(cand.full_text, cand.text):
            word_for_word = normalize(cand.full_text) in heard.said
            if fallback is None or (fallback_word_for_word and not word_for_word):
                fallback, fallback_word_for_word = cand, word_for_word
            skipped += 1
            continue
        return cand, skipped, False, vetoed
    return fallback, skipped, fallback is not None, vetoed


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
    veto: Veto | None = None,
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
    * ``veto`` - a candidate the voice may not speak (the guard: see :data:`Veto`).  A turn whose every
      candidate is vetoed falls back like any other dead end - a shorter context, then a fresh text - and the
      conversation stops when there is nothing left that may be said.

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
        said_list.append(opening)
        index += 1
    heard = Heard(said_list)

    for _ in range(turns):
        voice = voices[index % 2]
        turn = reply(
            voice, said_list[-1] if said_list else "", heard=heard, index=index,
            speaker=speakers[index % len(speakers)], mode=mode, max_length=max_length, context=context,
            temperature=temperature, k=k, beam=beam, step_penalty=step_penalty, rng=rng,
            avoid_repeats=avoid_repeats, veto=veto,
        )
        if turn is None:
            break
        if turn.repeat and normalize(turn.text) in repeated:
            break  # the voice can only say a duplicate it has already repeated: the conversation is over
        result.append(turn)
        said_list.append(turn.text)
        heard.remember(turn.text, turn.reply if turn.context else "")
        if turn.repeat:
            repeated.add(normalize(turn.text))
        index += 1
    return result


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
    veto: Veto | None = None,
) -> Turn | None:
    """What ``voice`` says next after ``previous`` - one turn, or ``None`` when it has nothing to say.

    This is the whole of a conversational turn, and :func:`converse` is a loop
    over it: the tail of ``previous`` is located in the graph and continued
    (``mode``), the context loses a word at a time while nothing follows it,
    and a voice with nothing left to add changes the subject with a fresh text
    from START.  ``heard`` is what the conversation has already heard (so a
    reply does not duplicate it - :class:`Heard`) and ``veto`` what the speaker
    may not say.  Remember the turn in ``heard`` before asking for the next one,
    or the same reply comes back.

    It is public because the other voice need not be a model at all: the chat
    loop (:mod:`radixnet.chat`) has an LLM speak every other line and calls
    this for the model's own.
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
    draws = k if mode == "sample" else 1
    while ctx:
        if _usable(voice, ctx):
            for _draw in range(draws):
                cands = _candidates(voice, ctx, mode, k, beam, max_length, step_penalty, temperature, rng)
                offered += len(cands)
                spoken, dropped, repeat, refused = _pick(cands, heard, avoid_repeats, veto)
                skipped += dropped
                vetoed += refused
                if spoken is not None and not repeat:
                    break
            if spoken is not None and not repeat:
                break
        ctx = _shorter(ctx)
    if spoken is None or repeat:
        # nothing (new) follows the previous line: change the subject with a fresh text
        fresh_pick: PathResult | None = None
        fresh_repeat = False
        for _draw in range(draws):
            cands = _candidates(voice, "", mode, k, beam, max_length, step_penalty, temperature, rng)
            offered += len(cands)
            fresh_pick, dropped, fresh_repeat, refused = _pick(cands, heard, avoid_repeats, veto)
            skipped += dropped
            vetoed += refused
            if fresh_pick is not None and not fresh_repeat:
                break
        if fresh_pick is not None and (spoken is None or not fresh_repeat):
            spoken, repeat, ctx = fresh_pick, fresh_repeat, ""
    if spoken is None:
        return None
    return Turn(
        index=index, speaker=speaker, text=spoken.full_text if ctx else spoken.text, context=ctx,
        reply=spoken.text, cost=spoken.cost, probability=path_probability(spoken),
        reached_end=spoken.reached_end, fresh=not ctx, repeat=repeat, candidates=offered, skipped=skipped,
        vetoed=vetoed, labels=list(spoken.labels), node_ids=list(spoken.node_ids),
        step_costs=list(spoken.step_costs),
    )
