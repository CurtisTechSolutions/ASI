"""Where the negative network's data comes from: the tutor's verdicts, turned into blame.

Nothing in :class:`~radixnet.negative.NegativeNet` is invented.  Every failure
it knows was handed to it by something outside the network that looked at an
output and said it was wrong, and why:

* the **English tutor** (:mod:`radixnet.tutor`): the LLM sets an exercise, the
  network completes it and the LLM marks the sentence - a mark out of 10, the
  single worst mistake named out of :data:`radixnet.tutor.ERROR_TYPES`
  (``agreement``, ``tense``, ``article``, ``plural``, ...), one sentence of
  teaching and the *correction*, the same sentence written out in correct
  English.  The mistake is the reason, the mark is the severity, and the diff
  against the correction says which characters were wrong
  (:func:`faults_from_lessons`, :meth:`radixnet.negative.NegativeNet.correct`);
* the **adversarial reviewer** (:mod:`radixnet.ollama`): an Ollama model rates
  the network's own texts 0-10, passes or fails each one and writes a
  one-sentence critique;
* the **code-generation teacher and judge** (:mod:`radixnet.codegen`): the
  sandbox says a program crashed, timed out or printed the wrong thing, the
  style checker names its issues and the LLM judge says whether the task was
  accomplished at all;
* a **person** pressing thumbs down in the frontend, and the evolve loop's
  discriminator, which is the network's own critic.

This module turns those verdicts into **faults**.  A fault is a text, a
**reason** (a short tag - the tutor's own error type, or one out of
:data:`REASONS` / :data:`CODE_REASONS` picked from the tutor's words by
:func:`classify`), a **severity** (how badly it failed) and a **note** (the
tutor's sentence, kept verbatim for the journal).  :func:`teach` hands faults
to the negative network; the texts the tutor *passed* are cleared in the same
call, so a fragment that shows up in good and bad output alike stops carrying
the verdict on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "CODE_REASONS",
    "CODE_SEVERITY",
    "DEFAULT_REASON",
    "REASONS",
    "classify",
    "code_reason",
    "faults_from_attempts",
    "faults_from_lessons",
    "faults_from_reviews",
    "severity_from_rating",
    "teach",
    "teach_attempts",
    "teach_lessons",
    "teach_reviews",
]

DEFAULT_REASON = "other"

REASONS = (
    "empty",
    "gibberish",
    "repetition",
    "truncated",
    "grammar",
    "spelling",
    "contradiction",
    "false",
    "incoherent",
    "off-topic",
    DEFAULT_REASON,
)
"""Reason tags for reviewed *text* (the order is the order :func:`classify` tries them in)."""

CODE_REASONS = (
    "timeout",
    "crash",
    "wrong-output",
    "task-not-done",
    "style",
    "naming",
    DEFAULT_REASON,
)
"""Reason tags for reviewed *programs*."""

CODE_SEVERITY = {
    "timeout": 1.5,
    "crash": 1.5,
    "wrong-output": 1.25,
    "task-not-done": 1.0,
    "style": 0.5,
    "naming": 0.5,
    DEFAULT_REASON: 1.0,
}
"""How heavily each code failure is blamed (1 = one ordinary failure)."""

_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("empty", ("empty output", "no output", "produced nothing", "blank")),
    ("gibberish", ("gibberish", "nonsense word", "word salad", "garbled", "random character", "not words", "noise")),
    ("repetition", ("repeat", "repetit", "duplicat", "over and over", "loops")),
    ("truncated", ("truncat", "cut off", "cut short", "incomplete", "unfinished", "mid-sentence", "mid sentence")),
    ("grammar", ("grammar", "grammatic", "syntax", "word order", "punctuation", "agreement", "tense", "malformed sentence")),
    ("spelling", ("spelling", "misspell", "typo")),
    ("contradiction", ("contradict", "inconsisten", "conflicts with")),
    ("false", ("false", "factual", "inaccurate", "untrue", "wrong fact", "not true", "misleading")),
    ("incoherent", ("incoheren", "meaningless", "make no sense", "does not make sense", "doesn't make sense",
                    "nonsensical", "confusing", "unintelligible")),
    ("off-topic", ("off-topic", "off topic", "irrelevant", "unrelated", "does not answer", "ignores the prompt")),
)


def classify(critique: str | None, *, verdict: str | None = None, rating: float | None = None,
             default: str = DEFAULT_REASON) -> str:
    """The reason tag behind a tutor's critique (its own words decide; ``default`` when nothing matches).

    ``"unrated"`` comes back when the tutor failed a text without a critique
    the vocabulary recognises - it still failed, and the network records that
    nobody said why.
    """
    text = " ".join(str(critique or "").split()).lower()
    for reason, needles in _PATTERNS:
        if any(needle in text for needle in needles):
            return reason
    if (verdict or "").strip().lower() == "unrated":
        return "unrated"  # the tutor failed it without saying anything the vocabulary knows
    if rating is not None and rating <= 0 and not text:
        return "gibberish"
    return default


def severity_from_rating(rating: float | None, threshold: float = 6.0, floor: float = 0.25,
                         ceiling: float = 2.0) -> float:
    """How heavily a rated failure is blamed: ``ceiling`` at rating 0, ``floor`` at the pass threshold.

    An unrated failure (``rating`` ``None``) is blamed like one ordinary
    failure (1.0).
    """
    if rating is None:
        return 1.0
    limit = float(threshold) if threshold > 0 else 1.0
    share = max(0.0, min(1.0, (limit - float(rating)) / limit))
    return floor + (ceiling - floor) * share


def code_reason(attempt: Any) -> str:
    """The reason a code attempt was rejected, from the sandbox, the style report and the judge."""
    data = attempt.to_dict() if hasattr(attempt, "to_dict") else dict(attempt or {})
    run = data.get("run") or {}
    style = data.get("style") or {}
    verdict = data.get("verdict") or {}
    if run.get("timed_out"):
        return "timeout"
    if run.get("ok") is False:
        return "crash"
    if run.get("expected_ok") is False:
        return "wrong-output"
    if verdict.get("task") is False:
        return "task-not-done"
    if verdict.get("naming") is False or style.get("naming_ok") is False:
        return "naming"
    if verdict.get("pep8") is False or style.get("pep8_ok") is False:
        return "style"
    critique = verdict.get("critique") or ""
    issues = verdict.get("issues") or []
    return classify(critique or "; ".join(str(i) for i in issues[:3]))


def _fault(text: str, reason: str, severity: float, note: str, source: str) -> dict:
    return {"text": text, "reason": reason, "severity": float(severity), "note": note, "source": source}


def faults_from_reviews(reviews: Iterable[dict], threshold: float = 6.0, source: str = "review") -> tuple[list[dict], list[str]]:
    """``(faults, passed_texts)`` from :func:`radixnet.ollama.review_texts` entries.

    Everything the reviewer did not pass becomes a fault whose reason comes
    from its critique and whose severity comes from its rating; the texts it
    passed come back separately so they can clear blame.
    """
    faults: list[dict] = []
    passed: list[str] = []
    for entry in reviews or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "")
        if not text:
            continue
        verdict = str(entry.get("verdict") or "").strip().lower()
        rating = entry.get("rating")
        rating = float(rating) if isinstance(rating, (int, float)) else None
        if verdict == "pass":
            passed.append(text)
            continue
        critique = str(entry.get("critique") or "")
        faults.append(_fault(
            text,
            classify(critique, verdict=verdict, rating=rating),
            severity_from_rating(rating, threshold),
            critique,
            source,
        ))
    return faults, passed


def faults_from_attempts(attempts: Iterable[Any], source: str = "codegen") -> tuple[list[dict], list[str]]:
    """``(faults, correct_texts)`` from :class:`radixnet.codegen.Attempt` objects (or their dicts).

    A rejected program is blamed for what the sandbox, the style checker or
    the judge found (:func:`code_reason`), with the tutor's feedback as the
    note; the programs that were accepted come back to clear blame.
    """
    faults: list[dict] = []
    correct: list[str] = []
    for attempt in attempts or []:
        data = attempt.to_dict() if hasattr(attempt, "to_dict") else dict(attempt or {})
        text = getattr(attempt, "text", None) or data.get("text") or data.get("code") or ""
        text = str(text)
        if not text:
            continue
        if data.get("correct") or (data.get("verdict") or {}).get("correct"):
            correct.append(text)
            continue
        reason = code_reason(attempt)
        note = attempt.feedback() if hasattr(attempt, "feedback") else "; ".join(
            str(i) for i in ((data.get("verdict") or {}).get("issues") or [])[:4]
        )
        faults.append(_fault(text, reason, CODE_SEVERITY.get(reason, 1.0), note, source))
    return faults, correct


def faults_from_lessons(lessons: Iterable[Any], threshold: float = 6.0, source: str = "tutor") -> tuple[list[dict], list[str]]:
    """``(faults, passed_texts)`` from :class:`radixnet.tutor.Lesson` objects (or their dicts).

    The English tutor is the richest source of negatives there is: the mistake
    it named (``grade.error``, one of :data:`radixnet.tutor.ERROR_TYPES`) is
    the reason, its mark is the severity, its sentence of teaching is the note
    and its **correction** rides along in the fault, so :func:`teach` can blame
    only the characters the teacher actually changed
    (:meth:`radixnet.negative.NegativeNet.correct`).  The sentences that
    passed, the corrections themselves and the teacher's own model answers all
    come back as cleared text.
    """
    faults: list[dict] = []
    passed: list[str] = []
    for lesson in lessons or []:
        data = lesson.to_dict() if hasattr(lesson, "to_dict") else dict(lesson or {})
        sentence = " ".join(str(data.get("sentence") or "").split())
        grade = data.get("grade") or {}
        exercise = data.get("exercise") or {}
        correction = " ".join(str(grade.get("correction") or "").split())
        answer = " ".join(str(exercise.get("answer") or "").split())
        if sentence and grade.get("passed"):
            passed.append(sentence)
        elif sentence:
            error = str(grade.get("error") or "").strip().lower()
            comment = str(grade.get("comment") or "")
            reason = error if error and error != "none" else classify(comment, verdict=grade.get("graded_by"))
            score = grade.get("score")
            fault = _fault(
                sentence,
                reason,
                severity_from_rating(score if isinstance(score, (int, float)) else None, threshold),
                comment,
                source,
            )
            if correction and correction != sentence:
                fault["correction"] = correction
            faults.append(fault)
        for text in (correction, answer):
            if text and text not in passed:
                passed.append(text)
    return faults, passed


def teach(
    negative: Any,
    faults: Sequence[dict],
    passed: Sequence[str] = (),
    *,
    epochs: int = 1,
    clear_epochs: int = 1,
    progress: Any = None,
    stop_event: Any = None,
) -> dict:
    """Hand the tutor's faults to the negative network; ``passed`` texts clear blame afterwards.

    Returns ``{"blamed", "cleared", "unmatched", "edges", "reasons",
    "severity_mean", "records"}``.
    """
    records: list[dict] = []
    reasons: dict[str, int] = {}
    edges = 0
    severities: list[float] = []
    for fault in faults:
        if stop_event is not None and stop_event.is_set():
            break
        text = str(fault.get("text") or "")
        if not text:
            continue
        reason = str(fault.get("reason") or DEFAULT_REASON)
        severity = float(fault.get("severity", 1.0))
        correction = str(fault.get("correction") or "")
        source = str(fault.get("source") or "tutor")
        note = str(fault.get("note") or "")
        if correction and hasattr(negative, "correct"):
            # the tutor wrote the sentence out correctly: blame only the characters it changed
            outcome = negative.correct(
                text, correction, reason=reason, severity=severity, source=source, note=note,
            )
            edges += outcome["blamed"]
            reasons[reason] = reasons.get(reason, 0) + 1
            severities.append(severity)
            records.append(outcome)
            continue
        out = negative.blame(
            [text], reason=reason, severity=severity, source=source, note=note,
            epochs=epochs, progress=progress, stop_event=stop_event,
        )
        if out:
            edges += out[-1].get("edges_touched", 0)
        records.extend(out)
        reasons[reason] = reasons.get(reason, 0) + 1
        severities.append(severity)
    cleared = 0
    unmatched = 0
    texts = [str(t) for t in passed if str(t)]
    if texts and not (stop_event is not None and stop_event.is_set()):
        out = negative.clear(texts, epochs=clear_epochs, progress=progress, stop_event=stop_event)
        records.extend(out)
        if out:
            cleared = out[-1].get("matched", 0)
            unmatched = out[-1].get("unmatched", 0)
    return {
        "blamed": len(severities),
        "cleared": cleared,
        "unmatched": unmatched,
        "edges": edges,
        "reasons": reasons,
        "severity_mean": sum(severities) / len(severities) if severities else 0.0,
        "records": records,
    }


def _reviews_of(reviews: Any) -> tuple[list[dict], float | None]:
    """Accept the ``reviews`` list or the whole :func:`radixnet.ollama.adversarial_review` result."""
    if isinstance(reviews, dict):
        return list(reviews.get("reviews") or []), reviews.get("threshold")
    return list(reviews or []), None


def teach_reviews(negative: Any, reviews: Any, *, threshold: float = 6.0, clear_passes: bool = True,
                  source: str = "review", **options: Any) -> dict:
    """Feed an adversarial review straight into the negative network (see :func:`teach`)."""
    entries, own = _reviews_of(reviews)
    limit = float(own) if isinstance(own, (int, float)) else float(threshold)
    faults, passed = faults_from_reviews(entries, limit, source)
    report = teach(negative, faults, passed if clear_passes else (), **options)
    report.update(source=source, threshold=limit, faults=faults, passed=len(passed))
    return report


def teach_lessons(negative: Any, lessons: Iterable[Any], *, threshold: float = 6.0, clear_passes: bool = True,
                  source: str = "tutor", **options: Any) -> dict:
    """Feed a round of English lessons into the negative network (see :func:`faults_from_lessons`)."""
    faults, passed = faults_from_lessons(lessons, threshold, source)
    report = teach(negative, faults, passed if clear_passes else (), **options)
    report.update(source=source, threshold=float(threshold), faults=faults, passed=len(passed))
    return report


def teach_attempts(negative: Any, attempts: Iterable[Any], *, clear_passes: bool = True, source: str = "codegen",
                   **options: Any) -> dict:
    """Feed the code-generation teacher's rejected attempts into the negative network (see :func:`teach`)."""
    faults, correct = faults_from_attempts(attempts, source)
    report = teach(negative, faults, correct if clear_passes else (), **options)
    report.update(source=source, faults=faults, passed=len(correct))
    return report
