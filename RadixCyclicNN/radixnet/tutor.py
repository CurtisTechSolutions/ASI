"""Automated English lessons: the tutor sets the exercise, the network completes it, the tutor marks it.

    topic -> prefix (LLM) -> completion (the prediction search) -> grade (LLM) -> 2NRL

The Predict tab with nobody at the keyboard.  One *lesson* is the whole
prediction process run end to end without a human:

1. **The exercise** — the LLM writes a handful of sentence openings about a
   topic (:func:`write_exercises`), each drilling one point of English
   grammar (``focus``) and each with the teacher's own model answer
   (``answer``) so a lesson can teach even when the network says nothing.
2. **The completion** — the network continues the prefix with the ordinary
   prediction search (cheapest path, beam or sampling), exactly what
   ``predict`` does for a human-typed prefix.
3. **The grade** — the LLM marks the finished sentence as an English teacher
   (:func:`grade_completions`): grammar, spelling and fluency out of 10, the
   single worst error named from a fixed list (:data:`ERROR_TYPES`), one
   sentence of teaching, and — the part the network learns from — the
   *correction*: the same sentence written out in correct English, keeping
   the prefix word for word.
4. **The lesson learned** — a correction is taught *as a correction*: the
   sentence the network wrote and the sentence the teacher wrote instead are
   aligned character by character (:mod:`radixnet.diff`), and only the
   trigram nodes they disagree on move — the step that wrote the wrong
   character is penalised, the step that writes the right one is rewarded,
   and the words both sentences share keep what they earned
   (:meth:`radixnet.countnet.CountRewardNet.correct`,
   ``diff_corrections``).  Everything else is 2NRL as before: sentences with
   no correction to diff are garbage weighted by how bad the mark was, the
   passed sentences and the teacher's own English are the fine-tune pass.

Grammar is what is being taught, so grammar is what the overall score mostly
is: ``score = grammar_weight * grammar + (1 - grammar_weight) * mean(spelling,
fluency)``, ``grammar_weight`` = 0.6 by default.  The error types of a round
add up to a **report card** (:func:`report_card`); with ``adapt`` on, the
weakest points of one round are handed to the LLM as the syllabus of the
next one, and ``drills`` asks it for extra correct example sentences about
them - a teacher noticing that the class keeps failing plurals and setting
plural exercises.

The teacher is a local Ollama model by default (``OLLAMA_HOST``,
``RADIXNET_TUTOR_MODEL``) or ChatGPT (``tutor_provider="chatgpt"``, see
:mod:`radixnet.chatgpt`, which needs ``OPENAI_API_KEY``); the marking follows
the tutor unless ``grader_provider`` names the other one, so the lessons can
be set by one and marked by the other.  A hosted tutor sends the topic, the
exercises and everything the network writes to OpenAI, and costs money per
call.

Cost, per round: one call for the exercises, one per grading batch, one more
when ``drills`` is on.
"""

from __future__ import annotations

import dataclasses
import os
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import Any

from . import diff
from .beam import path_probability
from .llm import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    LLMClient,
    LLMError,
    default_model,
    loads_lenient,
    normalise_provider,
    provider_of,
)
from .ollama import DEFAULT_MODEL, parse_lines

__all__ = [
    "DEFAULT_TUTOR_MODEL",
    "ERROR_TYPES",
    "PROVIDERS",
    "TEACHER_WEIGHT",
    "MODES",
    "TWONRL_PER",
    "Correction",
    "Exercise",
    "Grade",
    "Lesson",
    "TutorConfig",
    "TutorTrainer",
    "cue",
    "default_tutor_model",
    "drill_sentences",
    "grade_completions",
    "overall_score",
    "parse_exercises",
    "report_card",
    "write_exercises",
]

DEFAULT_TUTOR_MODEL = os.environ.get("RADIXNET_TUTOR_MODEL", "").strip() or DEFAULT_MODEL
"""Ollama model that sets and marks the exercises (``RADIXNET_TUTOR_MODEL``, else the Ollama default)."""


def default_tutor_model(provider: str | None = None) -> str:
    """The model a teacher uses when none is named: :data:`DEFAULT_TUTOR_MODEL` for Ollama, else the provider's own default."""
    provider = normalise_provider(provider)
    return DEFAULT_TUTOR_MODEL if provider == "ollama" else default_model(provider)

ERROR_TYPES = (
    "none",
    "agreement",
    "tense",
    "article",
    "preposition",
    "plural",
    "pronoun",
    "word-order",
    "spelling",
    "punctuation",
    "vocabulary",
    "fragment",
    "nonsense",
)
"""The grammar / English mistakes a completion is marked with; ``"none"`` is a clean sentence."""

MODES = ("dijkstra", "beam", "sample")
"""How the network completes a prefix: the cheapest path, the beam search, or a stochastic walk."""

TWONRL_PER = ("round", "lesson")
"""Apply 2NRL once per round over every graded sentence, or after each lesson."""

TEACHER_WEIGHT = 1.0
"""Positive-phase weight of text the teacher wrote (a correction, a model answer, a drill): correct by construction."""

MAX_COMMENT_CHARS = 300
_SCHEMA = (
    '{"grades": [{"index": <int>, "grammar": <0-10>, "spelling": <0-10>, "fluency": <0-10>, '
    '"error": "<one of the error types>", "correction": "<the whole sentence in correct English>", '
    '"comment": "<one sentence of teaching>"}, ...]}'
)


def cue(prefix: str) -> str:
    """What is actually fed to the search: the prefix with exactly one trailing space, so a new word follows."""
    text = str(prefix or "").rstrip()
    return f"{text} " if text else ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _score(value: Any, default: float | None = None) -> float | None:
    """A 0-10 mark from an LLM answer (numbers, ``"7/10"``, ``"7"``); ``default`` when unusable."""
    if isinstance(value, str):
        value = value.split("/")[0].strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(10.0, number))


def _error_type(value: Any) -> str:
    """The reported mistake mapped onto :data:`ERROR_TYPES` (``"other"`` for anything unrecognised)."""
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text or text in ("no-error", "no-errors", "correct", "ok", "n/a"):
        return "none"
    if text in ERROR_TYPES:
        return text
    for known in ERROR_TYPES[1:]:  # "subject-verb-agreement" -> "agreement", "verb-tense" -> "tense"
        if known in text:
            return known
    return "other"


def overall_score(grammar: float | None, spelling: float | None, fluency: float | None, grammar_weight: float = 0.6) -> float | None:
    """One mark out of 10 from the three sub-marks, grammar carrying ``grammar_weight`` of it.

    Missing sub-marks are dropped: with only ``grammar`` the score *is* the
    grammar mark, with only spelling and fluency it is their mean.
    """
    rest = [v for v in (spelling, fluency) if v is not None]
    mean_rest = statistics.fmean(rest) if rest else None
    if grammar is None:
        return mean_rest
    if mean_rest is None:
        return grammar
    return grammar_weight * grammar + (1.0 - grammar_weight) * mean_rest


# ---------------------------------------------------------------------------
# the exercise
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Exercise:
    """One sentence opening for the network to finish, with the point it drills and a model answer."""

    id: str
    prefix: str
    focus: str = ""  # the grammar point: "past tense", "plural nouns", ...
    answer: str = ""  # the teacher's own complete sentence, taught when the network fails

    @property
    def cue(self) -> str:
        """:func:`cue` of the prefix - what the prediction search is given."""
        return cue(self.prefix)

    def to_dict(self) -> dict:
        return {"id": self.id, "prefix": self.prefix, "focus": self.focus, "answer": self.answer}


_EXERCISE_SYSTEM = (
    "You are an English teacher writing exercises for a beginner student who completes sentences: you give the "
    "opening words, the student writes the rest. Every exercise is one unfinished sentence in plain, simple, "
    "modern English. The prefix must be {words} words, must NOT end with punctuation and must be genuinely "
    "unfinished, so that finishing it correctly needs the point of grammar you are drilling. Each exercise also "
    "carries 'focus' (the point of grammar in two or three words, e.g. 'subject-verb agreement', 'past tense', "
    "'plural nouns', 'articles', 'prepositions of place') and 'answer' (the whole sentence, prefix included, "
    "finished correctly by you - short, natural and factual). Vary the focus and the vocabulary across the "
    "exercises. Reply with JSON only, no prose, exactly of the form "
    '{{"exercises": [{{"prefix": "...", "focus": "...", "answer": "..."}}, ...]}} with exactly {n} entries.'
)


def parse_exercises(raw: str, count: int, round_no: int = 1) -> list[Exercise]:
    """LLM answer -> at most ``count`` :class:`Exercise` objects; unusable and duplicate entries are dropped.

    Tolerates the shapes an LLM drifts into: ``{"exercises": [...]}``, a bare
    list, plain strings instead of objects, ``sentence`` / ``opening`` /
    ``stem`` instead of ``prefix``, and a model answer that repeats the prefix
    or omits it.
    """
    data = loads_lenient(raw)
    items: Any = None
    if isinstance(data, dict):
        for key in ("exercises", "prefixes", "items", "results"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if items is None and any(k in data for k in ("prefix", "opening", "stem")):
            items = [data]
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        items = [{"prefix": line} for line in parse_lines(raw, limit=count)]  # plain lines: still usable
    exercises: list[Exercise] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            item = {"prefix": item}
        if not isinstance(item, dict):
            continue
        prefix = ""
        for key in ("prefix", "opening", "stem", "start", "sentence"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                prefix = " ".join(value.split()).strip()
                break
        prefix = prefix.rstrip(" .,:;!?-–—_")  # an opening, never a finished sentence
        if not prefix:
            continue
        key = prefix.lower()
        if key in seen:
            continue
        seen.add(key)
        answer = ""
        for name in ("answer", "solution", "completion", "full_sentence", "example"):
            value = item.get(name)
            if isinstance(value, str) and value.strip():
                answer = " ".join(value.split()).strip()
                break
        if answer and not answer.lower().startswith(prefix.lower()):
            answer = cue(prefix) + answer.lstrip()  # the teacher answered with the continuation only
        focus = item.get("focus", item.get("point", item.get("skill", "")))
        exercises.append(Exercise(
            id=f"r{round_no}e{len(exercises) + 1}",
            prefix=prefix,
            focus=_clip(focus if isinstance(focus, str) else "", 60),
            answer=answer,
        ))
        if len(exercises) >= count:
            break
    return exercises


def write_exercises(
    client: LLMClient,
    topic: str,
    count: int = 5,
    *,
    focus: str | None = None,
    level: str = "beginner",
    weak: Sequence[str] = (),
    words: str = "3 to 6",
    model: str | None = None,
    temperature: float = 0.9,
) -> list[Exercise]:
    """Ask the teacher for ``count`` sentence openings about ``topic``.

    ``focus`` pins the point of grammar for every exercise, ``weak`` lists the
    mistakes the student has been making (the previous round's report card),
    which the teacher is asked to drill.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    topic = str(topic or "").strip()
    if not topic:
        raise ValueError("topic must be a non-empty string")
    system = _EXERCISE_SYSTEM.format(n=count, words=words)
    lines = [f"Topic: {topic}", f"Level: {(level or 'beginner').strip()}"]
    if focus and focus.strip():
        lines.append(f"Every exercise must drill: {focus.strip()}")
    weak = [w for w in weak if w and w != "none"]
    if weak:
        lines.append(
            "The student keeps making these mistakes, so drill them: " + ", ".join(weak[:5]) + "."
        )
    lines.append(f"Write the {count} exercises now.")
    raw = client.generate(
        "\n".join(lines), system=system, model=model, json_mode=True, options={"temperature": temperature}
    )
    exercises = parse_exercises(raw, count)
    if not exercises:
        raise LLMError(f"the teacher model {model or client.model!r} returned no usable exercises")
    return exercises


_DRILL_SYSTEM = (
    "You are an English teacher writing model sentences for a beginner student to imitate. Answer with exactly "
    "{n} lines and nothing else: one short, correct, natural sentence per line, plain text, no numbering, no "
    "quotes, no commentary. Every sentence must be simple, factual and grammatically perfect, because the "
    "student learns English by copying them."
)


def drill_sentences(
    client: LLMClient,
    topic: str,
    count: int = 5,
    *,
    weak: Sequence[str] = (),
    model: str | None = None,
    temperature: float = 0.8,
) -> list[str]:
    """Correct example sentences about ``topic`` for the fine-tune pass, drilling the ``weak`` points."""
    if count < 1:
        return []
    weak = [w for w in weak if w and w != "none"]
    user = f"Topic: {topic.strip()}\n"
    if weak:
        user += "Each sentence must clearly demonstrate the correct use of: " + ", ".join(weak[:5]) + ".\n"
    user += f"Write the {count} sentences now."
    raw = client.generate(user, system=_DRILL_SYSTEM.format(n=count), model=model, options={"temperature": temperature})
    return parse_lines(raw, limit=count)


# ---------------------------------------------------------------------------
# the grade
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Grade:
    """The teacher's marking of one completed sentence."""

    score: float | None = None  # 0-10 overall (grammar-weighted), None when the answer could not be read
    grammar: float | None = None
    spelling: float | None = None
    fluency: float | None = None
    passed: bool = False
    error: str = "none"  # one of ERROR_TYPES, or "other"
    correction: str = ""  # the whole sentence in correct English: what the network is taught
    comment: str = ""  # one sentence of teaching
    graded_by: str = DEFAULT_PROVIDER  # the marking provider ("ollama" | "chatgpt"), "empty" (nothing to mark) or "unrated"

    def to_dict(self) -> dict:
        return {
            "score": self.score, "grammar": self.grammar, "spelling": self.spelling, "fluency": self.fluency,
            "passed": self.passed, "error": self.error, "correction": self.correction, "comment": self.comment,
            "graded_by": self.graded_by,
        }


@dataclasses.dataclass(slots=True)
class Lesson:
    """One exercise, one completion by the network, one grade."""

    exercise: Exercise
    attempt: int  # 0-based: attempt 0 is the cheapest path, later ones are samples
    mode: str
    continuation: str
    sentence: str  # cue(prefix) + continuation - what is graded and, when it passes, learned
    cost: float = 0.0
    probability: float = 1.0
    reached_end: bool = False
    seconds: float = 0.0
    grade: Grade = dataclasses.field(default_factory=Grade)

    @property
    def empty(self) -> bool:
        return not self.continuation.strip()

    @property
    def changes(self) -> list[dict]:
        """What the teacher changed, span by span (empty when the sentence passed or was left uncorrected)."""
        if self.grade.passed or not self.grade.correction.strip() or not self.sentence.strip():
            return []
        return diff.summary(self.sentence.strip(), self.grade.correction.strip())

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise.to_dict(), "attempt": self.attempt, "mode": self.mode,
            "continuation": self.continuation, "sentence": self.sentence, "cost": self.cost,
            "probability": self.probability, "reached_end": self.reached_end, "seconds": self.seconds,
            "grade": self.grade.to_dict(), "changes": self.changes,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Correction:
    """A sentence the network wrote, the sentence the teacher wrote instead, and how bad the mark was."""

    wrong: str
    right: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"wrong": self.wrong, "right": self.right, "weight": self.weight}


_GRADE_SYSTEM = (
    "You are a strict but constructive English teacher marking sentence completions. The student is a beginner "
    "learning English: it is given the opening words of a sentence (the prefix, shown in <<>>) and writes the "
    "rest. Mark what the student wrote, judged as part of the whole sentence. For each completion give three "
    "marks out of 10 - grammar (agreement, tense, articles, prepositions, word order, sentence structure), "
    "spelling and fluency (does it read like natural English) - and name the single most important mistake as "
    "one of: {types}. Use \"none\" only for a sentence a teacher would accept as it stands. A completion that "
    "is empty, cut off, gibberish or not English scores 0 with the error \"nonsense\" or \"fragment\". "
    "'correction' is the whole sentence written out in correct English: keep the prefix word for word, change "
    "only what follows it, stay as close to what the student wrote as the mistake allows, and finish the "
    "sentence properly - the student learns English by being shown this sentence, so it must be correct and "
    "natural on its own. 'comment' is one short sentence of teaching addressed to the student, naming the rule "
    "that was broken. Reply with JSON only, no prose, exactly of the form {schema} with one entry per "
    "completion, in the given order and with the given index."
)


def _parse_grades(
    raw: str, count: int, grammar_weight: float, threshold: float, graded_by: str = DEFAULT_PROVIDER
) -> dict[int, Grade]:
    """``{index: Grade}`` for the entries of an LLM answer that could be understood."""
    data = loads_lenient(raw)
    items: Any = None
    if isinstance(data, dict):
        for key in ("grades", "reviews", "results", "items", "marks"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if items is None and any(k in data for k in ("grammar", "score", "correction")):
            items = [data]
    elif isinstance(data, list):
        items = data
    grades: dict[int, Grade] = {}
    if not isinstance(items, list):
        return grades
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        if not (0 <= index < count) or index in grades:
            continue
        grammar = _score(item.get("grammar", item.get("grammar_score")))
        spelling = _score(item.get("spelling", item.get("spelling_score")))
        fluency = _score(item.get("fluency", item.get("fluency_score", item.get("naturalness"))))
        score = overall_score(grammar, spelling, fluency, grammar_weight)
        if score is None:
            score = _score(item.get("score", item.get("rating")))
        if score is None:
            continue  # nothing to mark with: leave it unrated
        correction = item.get("correction", item.get("corrected", item.get("fixed", "")))
        comment = item.get("comment", item.get("critique", item.get("feedback", item.get("reason", ""))))
        grades[index] = Grade(
            score=score, grammar=grammar, spelling=spelling, fluency=fluency, passed=score >= threshold,
            error=_error_type(item.get("error", item.get("error_type", item.get("mistake")))),
            correction=" ".join(str(correction or "").split()),
            comment=_clip(comment if isinstance(comment, str) else "", MAX_COMMENT_CHARS),
            graded_by=graded_by,
        )
    return grades


def grade_completions(
    client: LLMClient,
    lessons: Sequence[Lesson],
    *,
    topic: str = "",
    threshold: float = 6.0,
    grammar_weight: float = 0.6,
    model: str | None = None,
    batch: int = 10,
    temperature: float = 0.2,
    external: Callable[[], Any] | None = None,
    graded_by: str | None = None,
) -> list[Lesson]:
    """Mark every lesson in place (``lesson.grade``) in batches of ``batch``; returns ``lessons``.

    ``graded_by`` names the provider doing the marking (the client's own by
    default) and ends up in every grade it gives.

    An empty completion is failed without asking (``graded_by="empty"``), with
    the teacher's model answer as the correction when the exercise has one; a
    lesson the LLM said nothing usable about keeps ``score=None`` and
    ``graded_by="unrated"`` and is treated as a failure by the trainer.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    graded_by = graded_by or provider_of(client)
    system = _GRADE_SYSTEM.format(types=", ".join(f'"{t}"' for t in ERROR_TYPES), schema=_SCHEMA)
    hold = external or nullcontext
    for start in range(0, len(lessons), batch):
        chunk = list(lessons[start : start + batch])
        asked = [(i, lesson) for i, lesson in enumerate(chunk) if not lesson.empty]
        for i, lesson in enumerate(chunk):
            if lesson.empty:
                lesson.grade = Grade(
                    score=0.0, grammar=0.0, spelling=0.0, fluency=0.0, passed=False, error="nonsense",
                    correction=lesson.exercise.answer,
                    comment="Nothing was written: the sentence has to be finished.", graded_by="empty",
                )
        if not asked:
            continue
        body = "\n".join(
            f"[{i}] <<{lesson.exercise.cue}>>{lesson.continuation}"
            + (f"   (drilling: {lesson.exercise.focus})" if lesson.exercise.focus else "")
            for i, lesson in asked
        )
        user = (
            (f"Topic of the lesson: {topic.strip()}\n\n" if topic and topic.strip() else "")
            + f"Mark these {len(asked)} completions:\n{body}\n\nReturn the JSON now."
        )
        with hold():
            raw = client.generate(user, system=system, model=model, json_mode=True, options={"temperature": temperature})
        parsed = _parse_grades(raw, len(chunk), grammar_weight, threshold, graded_by)
        for i, lesson in asked:
            grade = parsed.get(i)
            if grade is None:
                lesson.grade = Grade(
                    score=None, passed=False, error="other", correction=lesson.exercise.answer,
                    comment="no grade returned", graded_by="unrated",
                )
                continue
            if not grade.correction:
                grade.correction = lesson.exercise.answer if not grade.passed else lesson.sentence
            lesson.grade = grade
    return list(lessons)


def report_card(lessons: Iterable[Lesson]) -> dict:
    """Marks and mistakes of a set of lessons: means, pass rate, the error histogram and the weakest points."""
    lessons = list(lessons)
    graded = [l for l in lessons if l.grade.score is not None]
    errors = Counter(l.grade.error for l in lessons if l.grade.error not in ("none", ""))
    means = {
        name: (statistics.fmean(values) if values else None)
        for name, values in (
            ("score", [l.grade.score for l in graded]),
            ("grammar", [l.grade.grammar for l in graded if l.grade.grammar is not None]),
            ("spelling", [l.grade.spelling for l in graded if l.grade.spelling is not None]),
            ("fluency", [l.grade.fluency for l in graded if l.grade.fluency is not None]),
        )
    }
    passed = sum(1 for l in lessons if l.grade.passed)
    return {
        "lessons": len(lessons),
        "graded": len(graded),
        "passed": passed,
        "failed": len(lessons) - passed,
        "pass_rate": passed / len(lessons) if lessons else None,
        "mean_score": means["score"],
        "mean_grammar": means["grammar"],
        "mean_spelling": means["spelling"],
        "mean_fluency": means["fluency"],
        "errors": dict(errors.most_common()),
        "weakest": [name for name, _count in errors.most_common(3)],
    }


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TutorConfig:
    """Settings of a tutoring run (see :class:`TutorTrainer`)."""

    topic: str = "everyday life"
    rounds: int = 3
    exercises: int = 5  # sentence openings per round
    attempts: int = 1  # completions the network writes per exercise
    focus: str | None = None  # pin every exercise to one point of grammar
    level: str = "beginner"
    words: str = "3 to 6"  # how long a prefix the teacher writes
    tutor_provider: str = DEFAULT_PROVIDER  # "ollama" | "chatgpt": who teaches
    tutor_model: str = ""  # "" = the teacher provider's default model
    grader_provider: str = ""  # "" = the teacher's provider
    grader_model: str | None = None  # a different model for the marking (default: the tutor model)
    # the completion
    mode: str = "dijkstra"
    length: int = 20
    max_length: int = 80
    temperature: float = 1.0
    to_end: bool = True
    beam: int | None = None
    # the marking
    threshold: float = 6.0
    grammar_weight: float = 0.6
    batch: int = 10
    adapt: bool = True  # drill the previous round's weakest points
    drills: int = 0  # extra correct example sentences per round
    teach_answer: bool = True  # a failed lesson also learns the teacher's model answer
    # 2NRL
    learn: bool = True  # False: a dry run - the grades are reported, the network is left alone
    twonrl_per: str = "round"
    diff_corrections: bool = True  # teach a correction from its diff with the sentence, not as two whole sentences
    keep_weight: float = 0.25  # what the unchanged part of a correction still earns (1 = the whole sentence)
    min_weight: float = 0.25  # negative-phase weight of a near miss (a hopeless answer weighs 1)
    neg_epochs: int = 2
    pos_epochs: int = 3
    neg_lr: float = 0.5
    pos_lr: float = 0.1
    batch_size: int = 4
    strength: float | None = None  # count / reward model: magnitude of a penalty / reward
    replay: bool = True  # keep teaching earlier corrections
    replay_limit: int = 64  # how many of them to keep (0 = no limit)
    checkpoint_every: int = 0  # rounds

    def __post_init__(self) -> None:
        """Resolve the providers and the model names they imply (so reports name the real model)."""
        self.tutor_provider = normalise_provider(self.tutor_provider)
        self.grader_provider = normalise_provider(self.grader_provider) if str(self.grader_provider).strip() else self.tutor_provider
        if not str(self.tutor_model).strip():
            self.tutor_model = default_tutor_model(self.tutor_provider)
        if self.grader_model is not None and not str(self.grader_model).strip():
            self.grader_model = None

    @property
    def resolved_grader_model(self) -> str:
        """The model the marking runs on: ``grader_model``, else the teacher's model on the teacher's provider."""
        if self.grader_model:
            return self.grader_model
        if self.grader_provider == self.tutor_provider:
            return self.tutor_model
        return default_tutor_model(self.grader_provider)

    def validate(self) -> None:
        if self.tutor_provider not in PROVIDERS or self.grader_provider not in PROVIDERS:
            raise ValueError(f"tutor_provider and grader_provider must be one of {', '.join(PROVIDERS)}")
        if not str(self.topic or "").strip():
            raise ValueError("topic must be a non-empty string")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.exercises < 1:
            raise ValueError("exercises must be >= 1")
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.twonrl_per not in TWONRL_PER:
            raise ValueError(f"twonrl_per must be one of {', '.join(TWONRL_PER)}")
        if self.length < 0 or self.max_length < 1:
            raise ValueError("length must be >= 0 and max_length >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.beam is not None and self.beam < 1:
            raise ValueError("beam must be >= 1")
        if not (0.0 <= self.threshold <= 10.0):
            raise ValueError("threshold must lie in [0, 10]")
        if not (0.0 <= self.grammar_weight <= 1.0):
            raise ValueError("grammar_weight must lie in [0, 1]")
        if not (0.0 <= self.min_weight <= 1.0):
            raise ValueError("min_weight must lie in [0, 1]")
        if not (0.0 <= self.keep_weight <= 1.0):
            raise ValueError("keep_weight must lie in [0, 1]")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.drills < 0 or self.replay_limit < 0 or self.checkpoint_every < 0:
            raise ValueError("drills, replay_limit and checkpoint_every must be >= 0")
        if self.neg_epochs < 0 or self.pos_epochs < 0:
            raise ValueError("epochs must be >= 0")
        if self.neg_lr < 0 or self.pos_lr < 0:
            raise ValueError("learning rates must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.strength is not None and self.strength < 0:
            raise ValueError("strength must be >= 0")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


ProgressFn = Callable[[dict], None]


class TutorTrainer:
    """Runs the lessons: the LLM sets and marks the exercises, the network completes them and learns from the grades.

    ``client`` is the teacher: any provider client (Ollama or ChatGPT).  The
    marking shares it unless ``grader_client`` is given or the configuration
    names a different ``grader_provider``, in which case a client for that
    provider is built from the environment; every grade records which one
    marked it.

    ``external`` is an optional zero-argument callable returning a context
    manager entered around the LLM calls; the API passes its lock-releasing
    one so the server stays responsive while the teacher thinks.
    """

    def __init__(
        self,
        model: Any,
        client: LLMClient,
        config: TutorConfig | None = None,
        external: Callable[[], Any] | None = None,
        grader_client: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.config = config or TutorConfig()
        self.config.validate()
        if grader_client is None and self.config.grader_provider != provider_of(client):
            from .llm import make_client

            grader_client = make_client(self.config.grader_provider, model=self.config.resolved_grader_model)
        self.grader_client = grader_client if grader_client is not None else client
        self.tutor_provider = provider_of(client)
        self.grader_provider = provider_of(self.grader_client)
        self._external = external or nullcontext
        self.history: list[dict] = []
        self.lessons: list[Lesson] = []
        self.replay_buffer: list[str] = []
        self.weak: list[str] = []
        self._stop = threading.Event()

    # -- the three steps of a lesson -----------------------------------------

    def _stopped(self) -> bool:
        return self._stop.is_set()

    def set_exercises(self, round_no: int) -> list[Exercise]:
        """Step 1: the teacher writes this round's sentence openings."""
        cfg = self.config
        with self._external():
            exercises = write_exercises(
                self.client, cfg.topic, cfg.exercises, focus=cfg.focus, level=cfg.level,
                weak=self.weak if cfg.adapt else (), words=cfg.words, model=cfg.tutor_model,
            )
        for position, exercise in enumerate(exercises, 1):
            exercise.id = f"r{round_no}e{position}"
        return exercises

    def complete(self, exercise: Exercise, attempt: int) -> Lesson:
        """Step 2: the network continues the prefix (attempt 0 in the configured mode, later ones sampled)."""
        cfg = self.config
        mode = cfg.mode if attempt == 0 else "sample"
        t0 = time.perf_counter()
        result = self.model.predict(
            exercise.cue, length=cfg.length, mode=mode, temperature=cfg.temperature,
            max_length=cfg.max_length, to_end=cfg.to_end and mode != "sample", beam=cfg.beam,
        )
        continuation = result.text
        return Lesson(
            exercise=exercise, attempt=attempt, mode=mode, continuation=continuation,
            sentence=exercise.cue + continuation, cost=result.cost,
            probability=path_probability(result), reached_end=result.reached_end,
            seconds=time.perf_counter() - t0,
        )

    def grade(self, lessons: list[Lesson]) -> list[Lesson]:
        """Step 3: the teacher marks the completions (one call per :attr:`TutorConfig.batch`)."""
        cfg = self.config
        return grade_completions(
            self.grader_client, lessons, topic=cfg.topic, threshold=cfg.threshold, grammar_weight=cfg.grammar_weight,
            model=cfg.resolved_grader_model, batch=cfg.batch, external=self._external,
            graded_by=self.grader_provider,
        )

    # -- what a grade is worth -----------------------------------------------

    def weight_of(self, grade: Grade) -> float:
        """Negative-phase weight of a failed sentence: ``min_weight`` for a near miss, 1 for a hopeless one."""
        cfg = self.config
        if grade.score is None:
            return 1.0
        if cfg.threshold <= 0:
            return 1.0
        badness = max(0.0, min(1.0, (cfg.threshold - grade.score) / cfg.threshold))
        return cfg.min_weight + (1.0 - cfg.min_weight) * badness

    def reward_of(self, grade: Grade) -> float:
        """Positive-phase weight of a sentence the network wrote: its mark, ``score / 10``.

        The mark decides how much of the sentence the network keeps - a 10 out
        of 10 is learned at the full rate, a bare pass at a fraction of it -
        so passing is a rating, not a like.  What the *teacher* wrote (a
        correction, a model answer, a drill sentence) is correct by
        construction and always weighs :data:`TEACHER_WEIGHT`.
        """
        if grade.score is None:
            return 0.0
        return max(0.0, min(1.0, grade.score / 10.0))

    def diffs(self) -> bool:
        """Is a correction taught from its diff (``diff_corrections``, and a model that can learn one)?"""
        return bool(self.config.diff_corrections) and callable(getattr(self.model, "correct", None))

    def corrections_of(self, lessons: Iterable[Lesson]) -> list[Correction]:
        """The failed lessons a diff can teach: what the network wrote, what the teacher wrote, how bad it was.

        A lesson that wrote nothing has no mistake to align, and one the
        teacher left uncorrected has nothing to align it against; both go the
        old way, through :meth:`texts_of`.
        """
        if not self.diffs():
            return []
        out: list[Correction] = []
        seen: dict[tuple[str, str], int] = {}
        for lesson in lessons:
            grade = lesson.grade
            if grade.passed or not lesson.continuation.strip() or not grade.correction.strip():
                continue
            pair = (lesson.sentence.strip(), grade.correction.strip())
            weight = self.weight_of(grade)
            if pair in seen:  # the same mistake twice (several attempts) keeps its worst mark
                at = seen[pair]
                if weight > out[at].weight:
                    out[at] = Correction(*pair, weight)
                continue
            seen[pair] = len(out)
            out.append(Correction(*pair, weight))
        return out

    def texts_of(self, lessons: Iterable[Lesson]) -> tuple[list[str], list[float], list[str], list[float]]:
        """``(bad, bad_weights, good, good_weights)`` of graded lessons.

        Failures are garbage weighted by how bad the mark was; the sentences
        that passed are rewarded in proportion to their mark and the
        teacher's own English at full weight.  With ``diff_corrections`` on,
        a failure the teacher corrected is left out of both lists:
        :meth:`corrections_of` hands it to the diff instead, which punishes
        the words that were actually wrong rather than the whole sentence.
        """
        cfg = self.config
        diffed = {(c.wrong, c.right) for c in self.corrections_of(lessons)}
        bad: list[str] = []
        bad_weights: list[float] = []
        good: list[tuple[str, float]] = []
        for lesson in lessons:
            grade = lesson.grade
            if grade.passed:
                if lesson.sentence.strip():
                    good.append((lesson.sentence.strip(), self.reward_of(grade)))
                continue
            pair = (lesson.sentence.strip(), grade.correction.strip())
            if pair in diffed:  # the diff teaches this one, sentence against correction
                if cfg.teach_answer and lesson.exercise.answer.strip():
                    good.append((lesson.exercise.answer.strip(), TEACHER_WEIGHT))
                continue
            if lesson.continuation.strip():  # nothing written is nothing to punish - the prefix itself is correct
                bad.append(lesson.sentence.strip())
                bad_weights.append(self.weight_of(grade))
            if grade.correction.strip():
                good.append((grade.correction.strip(), TEACHER_WEIGHT))
            if cfg.teach_answer and lesson.exercise.answer.strip():
                good.append((lesson.exercise.answer.strip(), TEACHER_WEIGHT))
        texts, weights = _merge_weighted(good)
        return bad, bad_weights, texts, weights

    def learn(
        self,
        bad: list[str],
        bad_weights: list[float],
        good: list[str],
        good_weights: list[float] | None = None,
        corrections: Sequence[Correction] = (),
    ) -> dict:
        """One set of grades: the corrections taught from their diffs, then 2NRL over whatever is left.

        Every :class:`Correction` moves only the trigram nodes its two
        sentences disagree on (``diff_corrections``); the rest is the old
        whole-sentence pass - garbage weighted by how bad it was, good
        English by how good.  ``good_weights`` defaults to the full rate for
        every text (the teacher's own English); replayed corrections are
        taught at the full rate too.
        """
        cfg = self.config
        weights_of = dict(zip(good, good_weights)) if good_weights else {}
        good_all, good_all_weights = _merge_weighted(
            [(text, weights_of.get(text, TEACHER_WEIGHT)) for text in good]
        )
        if cfg.replay:
            extra = [t for t in self.replay_buffer if t not in good_all]
            if cfg.replay_limit:  # 0 = no limit, as for the buffer itself
                extra = extra[-cfg.replay_limit :]
            good_all += extra
            good_all_weights += [TEACHER_WEIGHT] * len(extra)
        result: dict[str, Any] = {
            "bad": len(bad), "good": len(good_all), "action": None, "neg_loss": None, "pos_loss": None,
            "mean_weight": statistics.fmean(bad_weights) if bad_weights else None,
            "mean_reward": statistics.fmean(good_all_weights) if good_all_weights else None,
            "corrections": 0, "edits": 0, "penalised": 0, "rewarded": 0,
        }
        actions: list[str] = []
        for correction in corrections:
            if self._stopped():
                break
            moved = self.model.correct(
                correction.wrong, correction.right, strength=cfg.strength, weight=correction.weight,
                reward=TEACHER_WEIGHT, keep=cfg.keep_weight,
            )
            result["corrections"] += 1
            result["edits"] += int(moved.get("edits") or 0)
            result["penalised"] += int(moved.get("penalised") or 0)
            result["rewarded"] += int(moved.get("rewarded") or 0)
            if moved.get("loss") is not None:
                result["pos_loss"] = moved["loss"]
            if correction.right not in self.replay_buffer:
                self.replay_buffer.append(correction.right)
        if result["corrections"]:
            actions.append("correct")
        if not bad and not good_all:
            result["action"] = "+".join(actions) or None
            self._trim_replay()
            return result
        shared = {"batch_size": cfg.batch_size, "stop_event": self._stop}
        if bad and good_all:
            outcome = self.model.two_nrl(
                bad, good_all, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs, neg_lr=cfg.neg_lr,
                pos_lr=cfg.pos_lr, strength=cfg.strength, bad_weights=bad_weights or None,
                good_weights=good_all_weights or None, **shared,
            )
            actions.append("2nrl")
            result.update(neg_loss=_last_loss(outcome["negative"]), pos_loss=_last_loss(outcome["positive"]))
        elif good_all:
            records = self.model.reward(
                good_all, epochs=cfg.pos_epochs, lr=cfg.pos_lr, strength=cfg.strength,
                weights=good_all_weights or None, **shared,
            )
            actions.append("reward")
            result.update(pos_loss=_last_loss(records))
        else:
            records = self.model.punish(
                bad, epochs=cfg.neg_epochs, lr=cfg.neg_lr, strength=cfg.strength, weights=bad_weights or None, **shared,
            )
            actions.append("punish")
            result.update(neg_loss=_last_loss(records))
        result["action"] = "+".join(dict.fromkeys(actions)) or None
        for text in good:
            if text not in self.replay_buffer:
                self.replay_buffer.append(text)
        self._trim_replay()
        return result

    def _trim_replay(self) -> None:
        """Keep the replay buffer to ``replay_limit`` texts (0 = no limit)."""
        limit = self.config.replay_limit
        if limit and len(self.replay_buffer) > limit:
            del self.replay_buffer[: len(self.replay_buffer) - limit]

    # -- driving ---------------------------------------------------------------

    def _emit(self, progress: ProgressFn | None, record: dict) -> None:
        self.history.append(record)
        if progress is not None:
            progress(record)

    def _emit_lesson(self, progress: ProgressFn | None, round_no: int, lesson: Lesson) -> None:
        grade = lesson.grade
        self._emit(progress, {
            "kind": "lesson", "round": round_no, "exercise": lesson.exercise.id, "prefix": lesson.exercise.prefix,
            "focus": lesson.exercise.focus, "attempt": lesson.attempt + 1, "mode": lesson.mode,
            "continuation": _clip(lesson.continuation, 400), "sentence": _clip(lesson.sentence, 400),
            "score": grade.score, "grammar": grade.grammar, "spelling": grade.spelling, "fluency": grade.fluency,
            "passed": grade.passed, "error": grade.error, "correction": _clip(grade.correction, 400),
            "comment": grade.comment, "graded_by": grade.graded_by, "probability": lesson.probability,
            "seconds": lesson.seconds, "changes": self.changes_of(lesson),
        })

    def changes_of(self, lesson: Lesson) -> list[dict]:
        """What the teacher changed, span by span - ``[{"op", "wrong", "right"}]`` - or ``[]``."""
        return lesson.changes

    def run_round(self, round_no: int, progress: ProgressFn | None = None) -> tuple[dict, list[Lesson]]:
        """One round: exercises, completions, grades, and the 2NRL they lead to."""
        cfg = self.config
        t0 = time.perf_counter()
        exercises = self.set_exercises(round_no)
        lessons: list[Lesson] = []
        for exercise in exercises:
            if self._stopped():
                break
            for attempt in range(cfg.attempts):
                lessons.append(self.complete(exercise, attempt))
        if lessons and not self._stopped():
            self.grade(lessons)
        for lesson in lessons:
            self._emit_lesson(progress, round_no, lesson)
        self.lessons.extend(lessons)
        card = report_card(lessons)
        drills: list[str] = []
        if cfg.drills and lessons and not self._stopped():
            with self._external():
                try:
                    drills = drill_sentences(
                        self.client, cfg.topic, cfg.drills, weak=card["weakest"], model=cfg.tutor_model
                    )
                except LLMError as exc:  # the lesson stands without its drill sentences
                    self._emit(progress, {"kind": "note", "round": round_no, "message": f"no drill sentences: {exc}"})
        learned = {} if self._stopped() else self._learn_lessons(lessons, drills)
        if cfg.adapt:
            self.weak = card["weakest"]
        record = {
            "kind": "round", "round": round_no, "topic": cfg.topic, "focus": cfg.focus,
            "exercises": len(exercises), "drills": len(drills), "seconds": time.perf_counter() - t0,
            **card, **learned,
        }
        return record, lessons

    def _learn_lessons(self, lessons: list[Lesson], drills: list[str]) -> dict:
        """2NRL over the whole round, or lesson by lesson (the drill sentences are taught once either way)."""
        if not self.config.learn:  # a dry run: what would have been learned, without touching the network
            bad, bad_weights, good, good_weights = self.texts_of(lessons)
            corrections = self.corrections_of(lessons)
            good, good_weights = _merge_weighted(
                list(zip(good, good_weights)) + [(t, TEACHER_WEIGHT) for t in drills]
            )
            return {
                "bad": len(bad), "good": len(good), "action": None, "neg_loss": None, "pos_loss": None,
                "mean_weight": statistics.fmean(bad_weights) if bad_weights else None,
                "mean_reward": statistics.fmean(good_weights) if good_weights else None,
                "corrections": len(corrections),
                "edits": sum(len(diff.summary(c.wrong, c.right, limit=0)) for c in corrections),
                "penalised": 0, "rewarded": 0,
            }
        if self.config.twonrl_per != "lesson":
            bad, bad_weights, good, good_weights = self.texts_of(lessons)
            return self.learn(
                bad, bad_weights, good + drills, good_weights + [TEACHER_WEIGHT] * len(drills),
                self.corrections_of(lessons),
            )
        merged: dict[str, Any] = {
            "bad": 0, "good": 0, "action": None, "neg_loss": None, "pos_loss": None,
            "mean_weight": None, "mean_reward": None, "corrections": 0, "edits": 0, "penalised": 0, "rewarded": 0,
        }
        actions: list[str] = []
        bad_seen: list[float] = []
        good_seen: list[float] = []
        for lesson in lessons:
            if self._stopped():
                break
            bad, bad_weights, good, good_weights = self.texts_of([lesson])
            outcome = self.learn(
                bad, bad_weights, good + drills, good_weights + [TEACHER_WEIGHT] * len(drills),
                self.corrections_of([lesson]),
            )
            drills = []
            bad_seen += bad_weights
            good_seen += good_weights
            if outcome["action"]:
                actions.extend(outcome["action"].split("+"))
            merged["bad"] += outcome["bad"]
            merged["good"] += outcome["good"]
            for key in ("corrections", "edits", "penalised", "rewarded"):
                merged[key] += outcome.get(key, 0)
            for key in ("neg_loss", "pos_loss"):
                if outcome[key] is not None:
                    merged[key] = outcome[key]
        merged["action"] = "+".join(dict.fromkeys(actions)) or None
        merged["mean_weight"] = statistics.fmean(bad_seen) if bad_seen else None
        merged["mean_reward"] = statistics.fmean(good_seen) if good_seen else None
        return merged

    def run(
        self,
        rounds: int | None = None,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager: Any = None,
    ) -> list[dict]:
        """Every round, then a final report card; the lesson records go to ``progress`` as they happen."""
        cfg = self.config
        self._stop = stop_event if stop_event is not None else threading.Event()
        total = cfg.rounds if rounds is None else rounds
        if total < 1:
            raise ValueError("rounds must be >= 1")
        records: list[dict] = []
        for round_no in range(1, total + 1):
            if self._stopped():
                break
            record, _lessons = self.run_round(round_no, progress)
            self._emit(progress, record)
            records.append(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and round_no % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self.model, round_no, "tutor", {
                    "round": round_no, "mean_score": record.get("mean_score"), "pass_rate": record.get("pass_rate"),
                })
        summary = {"kind": "report", "rounds": len(records), "topic": cfg.topic, **report_card(self.lessons)}
        self._emit(progress, summary)
        records.append(summary)
        return records


def _unique(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(t for t in texts if t and t.strip()))


def _merge_weighted(pairs: Iterable[tuple[str, float]]) -> tuple[list[str], list[float]]:
    """``(texts, weights)`` in first-seen order; a text offered twice keeps its largest weight."""
    merged: dict[str, float] = {}
    for text, weight in pairs:
        if not text or not text.strip():
            continue
        text = text.strip()
        merged[text] = max(merged.get(text, 0.0), float(weight))
    return list(merged), list(merged.values())


def _last_loss(records: list[dict]) -> float | None:
    return records[-1].get("loss") if records else None

