"""A model conversing with itself.

Two voices take turns.  Every reply is the prediction search run from the
end of what was just said: the tail of the previous line (its last
``context`` characters, snapped to a word boundary) is located in the graph
and the search continues it to the end of a text, so each speaker picks up
the other's last words and carries on from them.  ``"beam"`` offers the
``k`` most likely continuations and the first one the conversation has not
heard yet (and that is not a mere echo of the previous line) is spoken - a
conversation that never repeats itself, and a deterministic one;
``"sample"`` draws stochastic walks.  When nothing follows the tail of a
line, or the graph only knows part of its last word, the context is
shortened a word at a time (what else could follow "on the"?) and, if the
graph still has nothing to add, the voice changes the subject with a fresh
text from START.  The two voices may be two different models (``partner``),
e.g. the radix and the count model talking to each other.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from .beam import path_probability
from .graph import START
from .search import PathResult

if TYPE_CHECKING:  # pragma: no cover
    from .model import GraphModel

DEFAULT_SPEAKERS: tuple[str, ...] = ("A", "B")
MODES = ("beam", "sample")


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
    candidates: Sequence[PathResult], said: set[str], previous: str, avoid_repeats: bool,
) -> tuple[PathResult | None, int, bool]:
    """``(candidate, skipped, repeat)``: the first candidate that adds something and (when asked) is neither an
    utterance heard before nor an echo (a piece of the previous line); the best repeat is the fallback."""
    skipped = 0
    fallback: PathResult | None = None
    for cand in candidates:
        if not cand.text.strip():
            skipped += 1
            continue
        key = normalize(cand.full_text)
        if avoid_repeats and (key in said or (previous and key in previous)):
            if fallback is None:
                fallback = cand
            skipped += 1
            continue
        return cand, skipped, False
    return fallback, skipped, fallback is not None


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

    Every generated turn records the context it picked up, its cost and probability, whether it started fresh
    from START (nothing followed the previous line), and whether it had to repeat something already said.
    Generation stops early when a voice has nothing at all to say (an untrained model).
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
    said = {normalize(t) for t in said_list if t.strip()}

    for _ in range(turns):
        voice = voices[index % 2]
        speaker = speakers[index % len(speakers)]
        previous = said_list[-1] if said_list else ""
        previous_key = normalize(previous)
        ctx = tail_context(previous, context)
        spoken: PathResult | None = None
        offered = skipped = 0
        repeat = False
        draws = k if mode == "sample" else 1
        while ctx:
            if _usable(voice, ctx):
                for _draw in range(draws):
                    cands = _candidates(voice, ctx, mode, k, beam, max_length, step_penalty, temperature, rng)
                    offered += len(cands)
                    spoken, dropped, repeat = _pick(cands, said, previous_key, avoid_repeats)
                    skipped += dropped
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
                fresh_pick, dropped, fresh_repeat = _pick(cands, said, previous_key, avoid_repeats)
                skipped += dropped
                if fresh_pick is not None and not fresh_repeat:
                    break
            if fresh_pick is not None and (spoken is None or not fresh_repeat):
                spoken, repeat, ctx = fresh_pick, fresh_repeat, ""
        if spoken is None:
            break
        text = spoken.full_text if ctx else spoken.text
        turn = Turn(
            index=index, speaker=speaker, text=text, context=ctx, reply=spoken.text, cost=spoken.cost,
            probability=path_probability(spoken), reached_end=spoken.reached_end, fresh=not ctx, repeat=repeat,
            candidates=offered, skipped=skipped, labels=list(spoken.labels), node_ids=list(spoken.node_ids),
            step_costs=list(spoken.step_costs),
        )
        result.append(turn)
        said_list.append(text)
        said.add(normalize(text))
        index += 1
    return result
