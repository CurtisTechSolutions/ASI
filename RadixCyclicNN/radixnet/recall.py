"""What the network remembers of what it was shown: the recall tutor for speech and images.

The English tutor (:mod:`radixnet.tutor`) needs an LLM because nobody knows in
advance what the right sentence is.  Speech and images need no LLM at all,
because **the right answer is on file**: an utterance or a picture was encoded
into text (:mod:`radixnet.speech`, :mod:`radixnet.vision`) and trained on, so
asking the network to write that text again is an exercise whose correction
already exists.

One round is the same three steps the English tutor takes:

1. **the exercise** - the opening of a text the network was taught: the
   utterance's own token and the waveform header (``<speech:9f2a1c7d>
   aud:mu:8000x1:``), or an image header and a few characters of its payload
   (``img:tiny:128x128:AAEC…``);
2. **the completion** - the network writes the rest;
3. **the marking** - the completion is run back through the codec and compared
   with the original.  It is marked out of 10 and, when it fails, the single
   worst thing wrong with it is named:

   ==============  =========================================================
   ``unreadable``  not an encoded waveform / image at all
   ``truncated``   it stopped well short of the payload
   ``overrun``     it rambled past the end
   ``garbled``     the base64 had to be repaired to be read
   ``silence``     a waveform that decodes to nothing (``blank``: a flat image)
   ``clipping``    a waveform railed against the limits (``noise``: an image of extremes)
   ``mishearing``  it decodes to speech, but to different words than were said
   ``distortion``  readable and the right length, but simply wrong (``drift``: an image)
   ==============  =========================================================

The mark is the severity and the original text is the correction, which is
exactly what the negative network wants: :func:`radixnet.blame.teach_recall`
blames only the characters the network actually got wrong and clears the rest.

Nothing here needs Pillow, torch or a transcriber: the comparison is over the
payload bytes both codecs already produce.  Decoding a *picture* needs Pillow
and re-transcribing a waveform needs an ASR backend, but grading does not.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Iterable, Sequence
from typing import Any

from .blame import RECALL_OVERRUN, recall_reason

__all__ = [
    "DEFAULT_LEAD",
    "MODALITIES",
    "RecallExercise",
    "RecallGrade",
    "RecallLesson",
    "TOLERANCE",
    "check",
    "check_image",
    "check_speech",
    "cue_of",
    "exercises",
    "grade",
    "mark",
    "modality_of",
    "quiz",
    "report_card",
]

MODALITIES = ("speech", "image")
"""The two things a network is taught that it can be asked to remember exactly."""

DEFAULT_LEAD = {"speech": 0, "image": 16}
"""Payload characters the exercise gives away.

A spoken text carries a token of its own, so the exercise can be the token
alone - "what did *that* utterance sound like?".  An image header
(``img:tiny:128x128:``) is the same for every picture of that size, so without
a few characters of the payload there is nothing to identify which one is
wanted.
"""

TOLERANCE = 32.0
"""Byte distance at which two payloads stop agreeing at all (one eighth of the range).

Agreement is per byte, ``max(0, 1 - |written - true| / TOLERANCE)``, averaged
over the payload that was asked for - so a byte the network never wrote counts
as total disagreement and truncation needs no separate penalty.  An exact
recall scores 1, a recall off by a quantisation step or two scores ~0.95, and
random bytes score ~0.12: the metric has to be generous about the last bits of
a quantised sample and unforgiving about noise.
"""

_FLAT_PEAK = 0.02
"""Below this peak amplitude a decoded waveform is silence."""

_FLAT_SPREAD = 2.0
"""Below this byte standard deviation an image payload has no picture in it."""

_RAILED = 0.5
"""Fraction of railed samples / bytes above which a completion is clipping or noise."""


# ---------------------------------------------------------------------------
# the exercise
# ---------------------------------------------------------------------------


def modality_of(text: str) -> str:
    """``"speech"`` or ``"image"`` from the text's own header; raises :class:`ValueError` for anything else."""
    body = str(text or "")
    if "aud:" in body:
        return "speech"
    if "img:" in body:
        return "image"
    raise ValueError("not an encoded utterance or image: expected an 'aud:' or 'img:' header")


def _payload_at(text: str) -> int:
    """Index where the base64 payload starts (just past the codec header)."""
    body = str(text or "")
    for header in ("aud:", "img:"):
        at = body.find(header)
        if at < 0:
            continue
        colons = 0
        for i in range(at, len(body)):
            if body[i] == ":":
                colons += 1
                if colons == 3:
                    return i + 1
        return len(body)
    raise ValueError("not an encoded utterance or image: expected an 'aud:' or 'img:' header")


def cue_of(text: str, lead: int | None = None) -> str:
    """The opening the network is given: everything up to the payload, plus ``lead`` characters of it.

    ``lead`` defaults to :data:`DEFAULT_LEAD` for the text's modality.  The cue
    never runs past the text itself, so a very short payload simply gives less
    away.
    """
    body = str(text or "")
    start = _payload_at(body)
    if lead is None:
        lead = DEFAULT_LEAD.get(modality_of(body), 0)
    return body[: min(len(body), start + max(0, int(lead)))]


@dataclasses.dataclass(slots=True)
class RecallExercise:
    """One text the network was taught, and the opening it is asked to continue."""

    id: str
    modality: str
    reference: str  # the whole true text
    cue: str  # what the prediction search is given
    label: str = ""  # where it came from: a file name, an utterance token
    index: int = 0  # which of the texts it came from (texts that are not encoded are skipped, so this is not its position)

    @property
    def answer(self) -> str:
        """The continuation that would be perfect."""
        return self.reference[len(self.cue) :]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "modality": self.modality, "label": self.label, "index": self.index, "cue": self.cue,
            "cue_chars": len(self.cue), "answer_chars": len(self.answer),
        }


def exercises(texts: Iterable[str], lead: int | None = None, labels: Sequence[str] = ()) -> list[RecallExercise]:
    """Turn taught texts into recall exercises (texts that are not encoded are skipped)."""
    out: list[RecallExercise] = []
    names = list(labels)
    for i, text in enumerate(texts or []):
        body = str(text or "").strip()
        if not body:
            continue
        try:
            kind = modality_of(body)
            cue = cue_of(body, lead)
        except ValueError:
            continue
        label = names[i] if i < len(names) else ""
        out.append(RecallExercise(id=f"r{i + 1}", modality=kind, reference=body, cue=cue, label=label, index=i))
    return out


# ---------------------------------------------------------------------------
# the marking
# ---------------------------------------------------------------------------


def _bytes_of(text: str, modality: str) -> tuple[bytes, bool, str]:
    """``(payload, repaired, error)`` of an encoded text; ``error`` is set when it cannot be read at all."""
    try:
        if modality == "speech":
            from .speech import parse_text as parse_speech

            _, _, _, payload, repaired = parse_speech(text)
        else:
            from .vision import parse_text as parse_image

            _, _, _, payload, repaired = parse_image(text)
    except Exception as exc:  # noqa: BLE001 - SpeechError / VisionError, both ValueErrors
        return b"", False, str(exc)
    return payload, bool(repaired), ""


def _agreement(written: bytes, truth: bytes) -> float:
    """Mean per-byte agreement over ``truth`` (see :data:`TOLERANCE`); bytes never written count as 0."""
    if not truth:
        return 1.0 if not written else 0.0
    total = 0.0
    for i, expected in enumerate(truth):
        if i >= len(written):
            break  # the rest was never written: no agreement to add
        total += max(0.0, 1.0 - abs(written[i] - expected) / TOLERANCE)
    return total / len(truth)


def _spread(payload: bytes) -> float:
    """Standard deviation of the payload bytes."""
    if not payload:
        return 0.0
    mean = sum(payload) / len(payload)
    return math.sqrt(sum((b - mean) ** 2 for b in payload) / len(payload))


def _railed(payload: bytes) -> float:
    """Fraction of bytes sitting against either end of the byte range."""
    if not payload:
        return 0.0
    return sum(1 for b in payload if b <= 2 or b >= 253) / len(payload)


def _peak(payload: bytes, codec: str) -> float:
    """Peak amplitude of a decoded waveform payload, in ``[0, 1]``."""
    if not payload:
        return 0.0
    from .speech import get_codec

    _, _, decode = get_codec(codec)
    samples = decode(payload)
    return max((abs(float(v)) for v in samples), default=0.0)


def _common(payload: bytes, truth: bytes, repaired: bool, error: str, modality: str) -> dict:
    """The facts both modalities share: readable, repaired, the lengths, the agreement and the extremes."""
    readable = not error
    return {
        "modality": modality,
        "readable": readable,
        "repaired": repaired,
        "error": error,
        "written_bytes": len(payload),
        "expected_bytes": len(truth),
        "length_ratio": (len(payload) / len(truth)) if truth else (1.0 if not payload else 2.0),
        "agreement": _agreement(payload, truth) if readable else 0.0,
        "flat": False,
        "reference_flat": False,
        "extreme": readable and _railed(payload) > _RAILED,
        "reference_extreme": _railed(truth) > _RAILED,
    }


def _codec_of(text: str) -> str:
    """The codec a waveform text declares (``auto`` when it declares none the reader understands)."""
    try:
        from .speech import parse_text as parse_speech

        return parse_speech(text)[0]
    except Exception:  # noqa: BLE001 - not a waveform text at all: the default codec decodes it as well as any
        return "auto"


def _silent(payload: bytes, codec: str) -> bool:
    """Does this waveform payload decode to (near) nothing?"""
    try:
        return _peak(payload, codec) < _FLAT_PEAK
    except Exception:  # noqa: BLE001 - a codec the reader does not know: fall back to the raw byte spread
        return _spread(payload) < _FLAT_SPREAD


def check_speech(written: str, reference: str, *, heard: str = "", said: str = "") -> dict:
    """Facts about a recalled waveform, against the waveform it should have been.

    ``heard`` / ``said`` are the round-trip transcript and the words that were
    actually spoken; when both are given the completion is also marked on
    whether it still says the same thing.
    """
    truth, _, _ = _bytes_of(reference, "speech")
    payload, repaired, error = _bytes_of(written, "speech")
    facts = _common(payload, truth, repaired, error, "speech")
    if facts["readable"]:
        facts["flat"] = _silent(payload, _codec_of(written))
    facts["reference_flat"] = bool(truth) and _silent(truth, _codec_of(reference))
    said, heard = " ".join(str(said or "").split()), " ".join(str(heard or "").split())
    facts["said"], facts["heard"] = said, heard
    facts["match"] = (heard.lower() == said.lower()) if (said and heard) else None
    return facts


def check_image(written: str, reference: str) -> dict:
    """Facts about a recalled image text, against the image text it should have been."""
    truth, _, _ = _bytes_of(reference, "image")
    payload, repaired, error = _bytes_of(written, "image")
    facts = _common(payload, truth, repaired, error, "image")
    if facts["readable"]:
        facts["flat"] = _spread(payload) < _FLAT_SPREAD
    facts["reference_flat"] = bool(truth) and _spread(truth) < _FLAT_SPREAD
    facts["said"] = facts["heard"] = ""
    facts["match"] = None
    return facts


def check(written: str, reference: str, **options: Any) -> dict:
    """:func:`check_speech` or :func:`check_image`, decided by the reference's own header."""
    if modality_of(reference) == "speech":
        return check_speech(written, reference, **options)
    return check_image(written, reference)


def mark(facts: dict) -> float:
    """The mark out of 10 a set of :func:`check` facts earns.

    The agreement over the payload is the mark, thinned by whatever the network
    wrote *past* the end - agreement only asks whether the true bytes came back,
    so without that a completion that rambles on after a perfect recall would
    score full marks.  Repairing the base64 costs a point, and a completion that
    is silent, railed or says the wrong words is capped however well its bytes
    happen to line up: those are failures of a different kind from being
    slightly off.
    """
    if not facts.get("readable"):
        return 0.0
    score = 10.0 * float(facts.get("agreement") or 0.0)
    written, expected = int(facts.get("written_bytes") or 0), int(facts.get("expected_bytes") or 0)
    if written > expected > 0:
        score *= expected / written  # the excess is waste: it was never asked for
    if facts.get("repaired"):
        score -= 1.0
    if facts.get("flat") and not facts.get("reference_flat"):
        score = min(score, 2.0)
    if facts.get("extreme") and not facts.get("reference_extreme"):
        score = min(score, 3.0)
    if facts.get("match") is False:
        score = min(score, 4.0)
    return max(0.0, min(10.0, score))


@dataclasses.dataclass(slots=True)
class RecallGrade:
    """The marking of one recalled text: the same shape the English tutor's grade has."""

    score: float | None = None
    passed: bool = False
    error: str = "none"  # the reason tag (radixnet.blame.recall_reason)
    correction: str = ""  # the text it should have written
    comment: str = ""  # one sentence saying what went wrong
    graded_by: str = "recall"
    facts: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score, "passed": self.passed, "error": self.error, "correction": self.correction,
            "comment": self.comment, "graded_by": self.graded_by, "facts": dict(self.facts),
        }


def _comment(facts: dict, reason: str, score: float) -> str:
    """One sentence of teaching, in the plain words the journal keeps."""
    thing = "waveform" if facts.get("modality") == "speech" else "image"
    if reason == "unreadable":
        return f"the completion is not an encoded {thing}: {facts.get('error') or 'the header is gone'}"
    if reason == "truncated":
        return (f"the {thing} stops after {facts['written_bytes']} of {facts['expected_bytes']} bytes "
                f"({facts['length_ratio'] * 100:.0f}% of it)")
    if reason == "overrun":
        return (f"the {thing} runs on to {facts['written_bytes']} bytes where {facts['expected_bytes']} "
                f"were wanted")
    if reason == "garbled":
        return f"the base64 had to be repaired before the {thing} could be read"
    if reason == "silence":
        return "it decodes to silence: nothing was said back"
    if reason == "blank":
        return "it decodes to a flat image: nothing was drawn"
    if reason == "clipping":
        return "the waveform is railed against the limits rather than shaped"
    if reason == "noise":
        return "the image is all extremes: noise rather than a picture"
    if reason == "mishearing":
        return f"it says {facts.get('heard')!r} where {facts.get('said')!r} was said"
    if reason in ("distortion", "drift"):
        return (f"the {thing} is the right shape but wrong: {facts['agreement'] * 100:.0f}% agreement "
                f"with the original")
    return f"recalled at {score:.1f}/10"


def grade(facts: dict, correction: str = "", threshold: float = 6.0) -> RecallGrade:
    """Facts -> a grade: the mark, the reason, the correction and one sentence of teaching."""
    score = mark(facts)
    passed = score >= float(threshold)
    reason = "none" if passed else recall_reason(facts)
    return RecallGrade(
        score=score,
        passed=passed,
        error=reason,
        correction="" if passed else str(correction or ""),
        comment="" if passed else _comment(facts, reason, score),
        facts=dict(facts),
    )


# ---------------------------------------------------------------------------
# the lesson
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class RecallLesson:
    """One exercise, one completion by the network, one grade - the shape :mod:`radixnet.blame` reads."""

    exercise: RecallExercise
    attempt: int
    mode: str
    continuation: str
    text: str  # cue + continuation: what was marked
    cost: float = 0.0
    reached_end: bool = False
    seconds: float = 0.0
    grade: RecallGrade = dataclasses.field(default_factory=RecallGrade)

    @property
    def sentence(self) -> str:
        """What was marked (the name :mod:`radixnet.blame` and the English tutor both use)."""
        return self.text

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise.to_dict(), "attempt": self.attempt, "mode": self.mode,
            "sentence": self.text, "text": self.text, "continuation_chars": len(self.continuation),
            "cost": self.cost, "reached_end": self.reached_end, "seconds": self.seconds,
            "grade": self.grade.to_dict(),
        }


def _transcript_of(text: str) -> str:
    """The words a decoded waveform text says, when a transcriber is reachable; ``""`` otherwise."""
    try:
        from .speech import SpeechError, decode_text, transcribe

        wav = decode_text(text)["wav"]
        return " ".join(str(transcribe(wav, backend="auto").get("transcript") or "").split())
    except Exception:  # noqa: BLE001 - no backend, no Pillow-equivalent, an unreadable prediction: not a failure
        return ""


def quiz(
    model: Any,
    texts: Iterable[str],
    *,
    lead: int | None = None,
    labels: Sequence[str] = (),
    attempts: int = 1,
    mode: str = "beam",
    temperature: float = 1.0,
    length: int = 0,
    threshold: float = 6.0,
    listen_back: bool = False,
    said: Sequence[str] = (),
    progress: Any = None,
    stop_event: Any = None,
) -> list[RecallLesson]:
    """Ask the network to write out texts it was taught, and mark what comes back.

    ``texts`` are encoded utterances / images (:func:`radixnet.speech.teach`,
    :func:`radixnet.vision.encode_image`).  ``length`` caps how many payload
    characters are asked for - 0 asks for the whole thing - and the comparison
    is against exactly that much of the original, so a capped quiz is a fair
    one.  ``listen_back`` decodes each recalled waveform and transcribes it, so
    a completion that sounds like different words is marked as a
    ``mishearing``; it needs an ASR backend and is off by default.
    """
    items = exercises(texts, lead, labels)
    spoken = list(said)
    lessons: list[RecallLesson] = []
    for item in items:
        reference = item.reference
        wanted = len(reference) - len(item.cue)
        if length and length < wanted:
            wanted = int(length)
            reference = item.cue + item.answer[:wanted]  # a capped quiz is marked against what it asked for
        budget = max(1, int(wanted * RECALL_OVERRUN) + 8)
        for attempt in range(max(1, int(attempts))):
            if stop_event is not None and stop_event.is_set():
                return lessons
            t0 = time.perf_counter()
            result = model.predict(
                item.cue, length=budget, mode=(mode if attempt == 0 else "sample"),
                temperature=temperature, max_length=budget, to_end=False,
            )
            written = item.cue + result.text
            facts = (
                check_speech(
                    written, reference,
                    said=(spoken[item.index] if item.index < len(spoken) else ""),
                    heard=_transcript_of(written) if listen_back else "",
                )
                if item.modality == "speech"
                else check_image(written, reference)
            )
            lesson = RecallLesson(
                exercise=item, attempt=attempt, mode=(mode if attempt == 0 else "sample"),
                continuation=result.text, text=written, cost=float(getattr(result, "cost", 0.0) or 0.0),
                reached_end=bool(getattr(result, "reached_end", False)),
                seconds=time.perf_counter() - t0,
                grade=grade(facts, reference, threshold),
            )
            lessons.append(lesson)
            if progress is not None:
                progress(lesson.to_dict())
            if lesson.grade.passed:
                break  # it remembered: no need to ask again
    return lessons


def report_card(lessons: Sequence[RecallLesson]) -> dict:
    """``{"lessons", "passed", "mean_score", "mean_agreement", "reasons", "modality"}`` over a quiz."""
    graded = [lesson for lesson in lessons if lesson.grade.score is not None]
    reasons: dict[str, int] = {}
    for lesson in lessons:
        if not lesson.grade.passed and lesson.grade.error != "none":
            reasons[lesson.grade.error] = reasons.get(lesson.grade.error, 0) + 1
    scores = [float(lesson.grade.score) for lesson in graded]
    agreements = [float(lesson.grade.facts.get("agreement") or 0.0) for lesson in lessons]
    kinds = {lesson.exercise.modality for lesson in lessons}
    return {
        "lessons": len(lessons),
        "passed": sum(1 for lesson in lessons if lesson.grade.passed),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "mean_agreement": (sum(agreements) / len(agreements)) if agreements else 0.0,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "modality": kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else ""),
    }
