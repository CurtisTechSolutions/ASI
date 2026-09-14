"""The model in conversation with an LLM, and the LLM marking the conversation.

Every other teacher in this project talks *at* the network: the English tutor
writes a prefix and marks the completion (:mod:`radixnet.tutor`), the critic
reviews texts the model wrote alone (:mod:`radixnet.critic`), the agent judges
a transcript of tool calls (:mod:`radixnet.agent`).  The one thing a language
model is actually for - holding up its end of a conversation - was only ever
tested against *itself* (:mod:`radixnet.dialogue`, two voices of the same
network), where nothing can tell it that its answer did not follow on.

This module puts a second party on the other side of the line.  One
conversation is:

1. the **partner** - a local Ollama model by default, ChatGPT with
   ``provider="chatgpt"`` - says a line, told to keep it short, plain and easy
   to carry on from, because that is what a character-level model can reply to
   (:func:`radixnet.ollama.chat_line`);
2. the **model** replies to it the way it replies to anything: the tail of
   that line is located in the graph and continued
   (:func:`radixnet.dialogue.reply`), so a reply is a real walk of the network
   and not a prompt trick.  With a negative network in hand the pair vetoes a
   reply before it is spoken (:mod:`radixnet.duo`);
3. they take turns for ``turns`` exchanges;
4. the **judge** then marks every reply the model gave out of 10 against the
   line it was replying to, and the conversation as a whole
   (:func:`radixnet.ollama.review_conversation`).

What the marks buy:

* the failures **blame** the negative network, the passes **clear** it
  (:func:`radixnet.blame.teach_reviews`) - the section 24.2 path, as usual;
* and, unlike the critic, this loop **trains the positive model too**: the
  replies that passed are rewarded, the ones that failed are punished, and the
  partner's own lines join the positive phase (``teach_partner``) because they
  are exactly what a good reply in this conversation would have looked like.

That last point is the whole idea.  A model that only ever learns from a
corpus has no way of finding out that what it said did not answer the
question; here something answers back, and says so.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from .llm import DEFAULT_PROVIDER, PROVIDERS, normalise_provider

__all__ = ["Chat", "ChatConfig", "DEFAULT_SPEAKERS", "report_card"]

ProgressFn = Callable[[dict], Any]

DEFAULT_SPEAKERS: tuple[str, str] = ("Partner", "Model")
"""Who is who in the transcript: the LLM, then the network."""


@dataclasses.dataclass
class ChatConfig:
    """How the conversation is held, marked and learned from."""

    conversations: int = 1
    """Conversations to hold; 0 means keep going until something stops it."""
    turns: int = 4
    """Replies the model gives per conversation (the partner speaks between them)."""
    topic: str = ""
    """What to talk about; empty lets the partner choose."""
    opening: str = ""
    """The first line, given; empty lets the partner open."""
    persona: str = ""
    """Who the partner is being ("a curious child", "a vet", ...)."""
    context: int = 12
    """Characters of the previous line a reply picks up (a word boundary is respected)."""
    max_length: int = 60
    """Characters the model may add to a context in one reply."""
    mode: str = "beam"
    """How a reply is found: ``beam`` (the most likely unheard one) or ``sample``."""
    k: int = 5
    """Candidates considered per reply."""
    temperature: float = 1.0
    """Sampling temperature of the model's replies."""
    partner_temperature: float = 0.8
    """Sampling temperature of the partner's lines."""
    threshold: float = 6.0
    """Pass mark out of 10: below it a reply is a failure."""
    provider: str = DEFAULT_PROVIDER
    partner_model: str = ""
    """The partner's model name; empty means the client's own default."""
    judge_model: str = ""
    """A different model for marking; empty means the partner's."""
    guard: bool = True
    """Let the negative network veto a reply before it is spoken (:mod:`radixnet.duo`)."""
    blame: bool = True
    """Blame the failed replies (and clear with the passed ones)."""
    clear_passes: bool = True
    learn: bool = True
    """Train the positive model on the marked replies (2NRL), and let its rethinks teach the graph where it
    goes round (:func:`radixnet.dialogue.teach_back`)."""
    teach_partner: bool = True
    """Put the partner's own lines in the positive phase: they are what a good reply looked like."""
    avoid_repeats: bool = True
    """Do not let the model say something the conversation has already heard."""
    avoid_word_repeats: bool = True
    """Do not let one reply repeat its own words (a stutter: "say morning morning")."""
    explore: int = 3
    """Times a reply that caught itself repeating may back up and look for another way on (0 = not at all)."""
    neg_epochs: int = 2
    pos_epochs: int = 3
    neg_lr: float = 0.5
    pos_lr: float = 0.1
    batch_size: int = 4
    strength: float | None = None
    """Count / reward model: the magnitude of a penalty or reward (the learning rates are ignored there)."""
    epochs: int = 1
    """Blame epochs per conversation."""
    seed: int | None = None
    """Seed of the first conversation's sampling; later ones advance it, so they differ."""

    def validate(self) -> None:
        if self.conversations < 0:
            raise ValueError("conversations must be >= 0 (0 = until stopped)")
        if self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.context < 0:
            raise ValueError("context must be >= 0")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.mode not in ("beam", "sample"):
            raise ValueError("mode must be 'beam' or 'sample'")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.temperature < 0 or self.partner_temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.threshold <= 10:
            raise ValueError("threshold must lie in [0, 10]")
        if self.neg_epochs < 0 or self.pos_epochs < 0 or self.epochs < 0:
            raise ValueError("epochs must be >= 0")
        if self.neg_lr < 0 or self.pos_lr < 0:
            raise ValueError("learning rates must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.strength is not None and self.strength < 0:
            raise ValueError("strength must be >= 0")
        if normalise_provider(self.provider) not in PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Chat:
    """The conversation loop: the partner talks, the model replies, the judge marks, both networks learn.

    ``model`` is the network doing the talking, ``client`` any
    :class:`~radixnet.llm.LLMClient` (Ollama or ChatGPT) playing the partner
    and the judge, and ``negative`` an optional
    :class:`~radixnet.negative.NegativeNet` that learns what went wrong - and,
    with ``config.guard``, vetoes a bad reply before it is ever spoken.
    ``external`` wraps the slow LLM calls so a server can release its model
    lock around them; the model's own replies are walks of the graph and are
    made with the lock held.
    """

    def __init__(
        self,
        model: Any,
        client: Any,
        config: ChatConfig | None = None,
        negative: Any = None,
        external: Callable[[], Any] | None = None,
        judge_client: Any = None,
    ) -> None:
        if model is None:
            raise ValueError("a model to converse with is required")
        if client is None:
            raise ValueError("an LLM client to converse with is required")
        self.model = model
        self.client = client
        self.judge_client = judge_client or client
        self.negative = negative
        self.config = config or ChatConfig()
        self.config.validate()
        self._external = external or nullcontext
        self.history: list[dict] = []
        self.number = 0
        self._stop = threading.Event()
        self._pair = None

    # -- the pair ------------------------------------------------------------

    def pair(self):
        """The negative network guarding this conversation, or ``None`` when there is nothing to guard with."""
        if not (self.config.guard and self.negative is not None):
            return None
        if self._pair is None:
            from .duo import FilterConfig, NegativeFilter

            self._pair = NegativeFilter(self.model, self.negative, FilterConfig())
        return self._pair if self._pair.ready else None

    def _veto(self) -> "_Veto | None":
        """What the model may not say: the pair's verdict, or nothing when it is not guarding."""
        pair = self.pair()
        return None if pair is None else _Veto(pair)

    # -- one conversation ----------------------------------------------------

    def _partner_line(self, transcript: list[tuple[str, str]]) -> str:
        from .ollama import chat_line

        cfg = self.config
        with self._external():  # only the partner's thinking happens outside the lock
            return chat_line(
                self.client, transcript, topic=cfg.topic, persona=cfg.persona, speakers=DEFAULT_SPEAKERS,
                model=cfg.partner_model or None, temperature=cfg.partner_temperature,
            )

    def converse(self, progress: ProgressFn | None = None) -> dict:
        """Hold one conversation: ``{"transcript", "exchanges", "turns", "repeats", "stalled", "vetoed"}``.

        ``transcript`` is every line as ``(speaker, text)``, ``exchanges`` the
        ``(said to it, its reply)`` pairs the judge marks, ``repeats`` the
        replies the model could only repeat (:func:`radixnet.dialogue.repeats`:
        punished with the failures).  ``stalled`` is true when it ended early -
        the model had nothing left to say, or the partner went quiet.
        """
        from .dialogue import Heard, repeats, reply

        cfg = self.config
        seed = None if cfg.seed is None else int(cfg.seed) + self.number
        rng = None
        if cfg.mode == "sample" and seed is not None:
            import random

            rng = random.Random(seed)
        veto = self._veto()
        transcript: list[tuple[str, str]] = []
        exchanges: list[tuple[str, str]] = []
        heard = Heard()  # what has been said, on both sides: the model does not repeat any of it
        stalled = ""

        line = cfg.opening.strip() or self._partner_line(transcript)
        if not line:
            return {"transcript": [], "exchanges": [], "turns": [], "repeats": [],
                    "stalled": "the partner said nothing", "vetoed": 0}
        transcript.append((DEFAULT_SPEAKERS[0], line))
        heard.remember(line)
        turns: list[Any] = []
        for index in range(cfg.turns):
            if self._stop.is_set():
                stalled = "stopped"
                break
            turn = reply(
                self.model, line, heard=heard, index=len(transcript), speaker=DEFAULT_SPEAKERS[1], mode=cfg.mode,
                max_length=cfg.max_length, context=cfg.context, temperature=cfg.temperature, k=cfg.k, rng=rng,
                avoid_repeats=cfg.avoid_repeats, avoid_word_repeats=cfg.avoid_word_repeats,
                explore=cfg.explore, learn=cfg.learn, veto=veto,
            )
            if turn is None:
                # the guard can silence it outright, and that is worth saying
                stalled = "the guard vetoed everything it could say" if veto is not None and veto.refused \
                    else "the model had nothing to say"
                break
            turns.append(turn)
            transcript.append((DEFAULT_SPEAKERS[1], turn.text))
            exchanges.append((line, turn.text))
            heard.remember(turn.text, turn.reply if turn.context else "")
            if progress is not None:
                progress({
                    "kind": "exchange", "conversation": self.number, "exchange": index + 1,
                    "said": line, "reply": turn.text, "context": turn.context, "fresh": turn.fresh,
                    "repeat": turn.repeat, "stutter": turn.stutter, "vetoed": turn.vetoed, "cost": turn.cost,
                    "probability": turn.probability,
                    "rethink": turn.rethink.to_dict() if turn.rethink is not None else None,
                })
            if index == cfg.turns - 1:
                break  # the last word is the model's: no line after it to reply to
            line = self._partner_line(transcript)
            if not line:
                stalled = "the partner went quiet"
                break
            transcript.append((DEFAULT_SPEAKERS[0], line))
            heard.remember(line)
        return {"transcript": transcript, "exchanges": exchanges, "turns": turns,
                # the replies it could only repeat: punished with the failures, whatever the judge made of them
                "repeats": repeats(turns), "stalled": stalled,
                "vetoed": veto.refused if veto is not None else 0}

    def run_conversation(self, progress: ProgressFn | None = None) -> dict:
        """One conversation, marked and learned from; returns its record."""
        from . import blame as blame_module
        from .ollama import review_conversation

        cfg = self.config
        self.number += 1
        started = time.perf_counter()
        held = self.converse(progress)
        review: dict = {"reviews": [], "good": [], "bad": [], "mean_rating": None, "pass_rate": None,
                        "overall": None, "model": cfg.judge_model or cfg.partner_model or None}
        if held["exchanges"]:
            with self._external():  # the judge's thinking, also outside the lock
                review = review_conversation(
                    self.judge_client, held["exchanges"], topic=cfg.topic,
                    model=cfg.judge_model or cfg.partner_model or None, threshold=cfg.threshold,
                    speakers=DEFAULT_SPEAKERS,
                )
        taught: dict = {}
        if self.negative is not None and cfg.blame and review["reviews"]:
            taught = blame_module.teach_reviews(
                self.negative, review, threshold=cfg.threshold, clear_passes=cfg.clear_passes,
                source="chat", epochs=cfg.epochs,
            )
        learned = self._learn(review, held) if cfg.learn else {}
        overall = review.get("overall") or {}
        record = {
            "kind": "conversation",
            "conversation": self.number,
            "topic": cfg.topic,
            "judge": review.get("model"),
            "threshold": float(cfg.threshold),
            "transcript": [{"speaker": speaker, "text": text} for speaker, text in held["transcript"]],
            "exchanges": len(held["exchanges"]),
            "reviews": review.get("reviews") or [],
            "mean_rating": review.get("mean_rating"),
            "pass_rate": review.get("pass_rate"),
            "passed": len(review.get("good") or []),
            "failed": len(review.get("bad") or []),
            "overall_rating": overall.get("rating"),
            "overall_critique": overall.get("critique"),
            "stalled": held["stalled"],
            "vetoed": held["vetoed"],
            "blamed": taught.get("blamed", 0),
            "cleared": taught.get("cleared", 0),
            "edges": taught.get("edges", 0),
            "reasons": taught.get("reasons") or {},
            "repeats": len(held["repeats"]),  # replies the model could only repeat, punished with the failures
            # the learning keys are always present, whether or not it learned, so a record has one shape
            "action": None, "bad": 0, "good": 0, "neg_loss": None, "pos_loss": None,
            "seconds": time.perf_counter() - started,
            **learned,
        }
        self.history.append(record)
        return record

    # -- learning ------------------------------------------------------------

    def _learn(self, review: dict, held: dict) -> dict:
        """2NRL on the marked replies: what failed is punished, what passed (and the partner's lines) rewarded.

        A reply the model could only repeat is punished whatever the judge made of it: it said nothing new, and
        rewarding it would only make the duplicate likelier next time.
        """
        cfg = self.config
        bad = [text for text in (review.get("bad") or []) if text.strip()]
        good = [text for text in (review.get("good") or []) if text.strip()]
        if cfg.teach_partner:
            # the partner's own lines are what a good reply in this conversation would have looked like
            good += [text for speaker, text in held["transcript"] if speaker == DEFAULT_SPEAKERS[0] and text.strip()]
        said_twice = [text for text in held["repeats"] if text.strip()]
        good = [text for text in dict.fromkeys(good) if text not in said_twice]
        bad = [text for text in dict.fromkeys(bad + said_twice) if text not in good]
        result: dict[str, Any] = {"action": None, "bad": len(bad), "good": len(good),
                                  "repeats": len(said_twice), "neg_loss": None, "pos_loss": None}
        if not bad and not good:
            return result
        shared = {"batch_size": cfg.batch_size, "stop_event": self._stop}
        if bad and good:
            outcome = self.model.two_nrl(
                bad, good, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs, neg_lr=cfg.neg_lr,
                pos_lr=cfg.pos_lr, strength=cfg.strength, **shared,
            )
            result.update(action="2nrl", neg_loss=_last_loss(outcome["negative"]),
                          pos_loss=_last_loss(outcome["positive"]))
        elif good:
            records = self.model.reward(good, epochs=cfg.pos_epochs, lr=cfg.pos_lr, strength=cfg.strength, **shared)
            result.update(action="reward", pos_loss=_last_loss(records))
        else:
            records = self.model.punish(bad, epochs=cfg.neg_epochs, lr=cfg.neg_lr, strength=cfg.strength, **shared)
            result.update(action="punish", neg_loss=_last_loss(records))
        return result

    # -- the loop ------------------------------------------------------------

    def run(
        self,
        conversations: int | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict]:
        """Hold ``conversations`` conversations (``None`` uses the config; 0 there means until stopped).

        The stop event is checked before every conversation, so a loop left
        running ends cleanly after the one it is in.  The records are the
        conversation records, followed by one ``report`` record.
        """
        limit = self.config.conversations if conversations is None else conversations
        if limit is not None and limit < 0:
            raise ValueError(f"conversations must be >= 0 or None, got {limit}")
        self._stop = stop_event if stop_event is not None else threading.Event()
        forever = not limit
        records: list[dict] = []
        while forever or len(records) < limit:
            if self._stop.is_set():
                break
            record = self.run_conversation(progress)
            records.append(record)
            if progress is not None:
                progress(record)
        card = report_card(records)
        records.append(card)
        if progress is not None:
            progress(card)
        return records

    def __repr__(self) -> str:
        return f"Chat(conversation={self.number}, config={self.config!r})"


class _Veto:
    """The pair's verdict as a conversational veto, counting the distinct replies it stopped.

    A candidate can be offered again after the context is shortened, so the
    judgements are remembered rather than repeated, and ``refused`` counts
    texts rather than refusals.
    """

    __slots__ = ("pair", "seen")

    def __init__(self, pair: Any) -> None:
        self.pair = pair
        self.seen: dict[str, bool] = {}

    def __call__(self, text: str) -> bool:
        if text not in self.seen:
            self.seen[text] = self.pair.judge(text)["decision"] == "reject"
        return self.seen[text]

    @property
    def refused(self) -> int:
        return sum(1 for rejected in self.seen.values() if rejected)


def _last_loss(records: list[dict] | None) -> float | None:
    return records[-1].get("loss") if records else None


def report_card(records: list[dict]) -> dict:
    """What a run of conversations came to: how well the model held them up, and what it was taught.

    ``mean_rating`` averages the per-reply marks over the conversations that
    produced one, ``overall_rating`` the judge's verdict on each conversation
    as a whole, and ``trend`` is the last conversation's mean minus the
    first's - positive when the model is answering better than it did at the
    start.
    """
    held = [r for r in records if r.get("kind") == "conversation"]
    ratings = [float(r["mean_rating"]) for r in held if isinstance(r.get("mean_rating"), (int, float))]
    rates = [float(r["pass_rate"]) for r in held if isinstance(r.get("pass_rate"), (int, float))]
    overall = [float(r["overall_rating"]) for r in held if isinstance(r.get("overall_rating"), (int, float))]
    reasons: dict[str, int] = {}
    for record in held:
        for reason, count in (record.get("reasons") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return {
        "kind": "report",
        "conversations": len(held),
        "exchanges": sum(int(r.get("exchanges") or 0) for r in held),
        "passed": sum(int(r.get("passed") or 0) for r in held),
        "failed": sum(int(r.get("failed") or 0) for r in held),
        "vetoed": sum(int(r.get("vetoed") or 0) for r in held),
        "stalled": sum(1 for r in held if r.get("stalled")),
        "blamed": sum(int(r.get("blamed") or 0) for r in held),
        "cleared": sum(int(r.get("cleared") or 0) for r in held),
        "edges": sum(int(r.get("edges") or 0) for r in held),
        "mean_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "pass_rate": (sum(rates) / len(rates)) if rates else None,
        "overall_rating": (sum(overall) / len(overall)) if overall else None,
        "trend": (ratings[-1] - ratings[0]) if len(ratings) > 1 else None,
        "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }
