"""Tool use: the network browses and solves on its own, Ollama sets the bar and teaches.

    task -> acceptance criteria (Ollama) -> the network works with tools -> judge (Ollama)
         -> teach when it failed (Ollama) -> 2NRL (punish the failures, reward the correct run)

The network only ever emits characters, so a tool call is text it writes
(``<tool>web_fetch {"url": "..."}</tool>``, see :mod:`radixnet.tools`) and the
answer comes back as text it reads.  Around that, the LLM plays four roles and
never the one of solving the task *for* the network unless it has to:

* **criteria** — :func:`write_criteria` asks for the handful of checkable
  statements a correct answer must satisfy, *before* anything is attempted, so
  the bar is set independently of what the network happens to produce.
* **mediator** — :func:`mediate_call` turns an unusable emission (an untrained
  network writes noise) into one valid call against the real tool schemas,
  using Ollama's own tool-calling API where the model supports it.  The
  repaired call is executed *and written into the transcript*, so what the
  network learns is always well-formed, and the share of calls it got right by
  itself (``autonomy``) is what the records track.
* **judge** — :func:`judge_attempt` marks the finished transcript against the
  criteria written up front: every criterion met or not, a score, a critique.
* **teacher** — :func:`teach_task` solves the task with the same real tools
  when the network failed; its transcript is the positive phase of 2NRL.

:meth:`AgentTrainer.run` does that over a list of tasks.
:meth:`AgentTrainer.explore` does it over tasks the network *chooses*: it
continues ``TASK: `` into whatever it is reaching for, the LLM turns that into
one concrete answerable question (:func:`propose_task`), preferring pages the
network has not seen yet, and the same criteria / solve / judge / teach / 2NRL
cycle runs on it.  Links and results found along the way become the frontier,
so the exploration compounds and can run indefinitely.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import Any

from .gan import BLATANT_MODES
from .llm import LLMClient, LLMError, loads_lenient
from .ollama import DEFAULT_MODEL, parse_lines
from .tools import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    CALL_CLOSE,
    CALL_OPEN,
    ToolBox,
    ToolCall,
    ToolResult,
    answer_text,
    call_text,
    find_answer,
    format_observation,
    parse_call,
    task_header,
    transcript_text,
)

__all__ = [
    "BLATANT_MODES",
    "DEFAULT_AGENT_MODEL",
    "MEDIATION",
    "PHASES",
    "Attempt",
    "AgentConfig",
    "AgentTrainer",
    "Failure",
    "Step",
    "Task",
    "Verdict",
    "judge_attempt",
    "load_tasks",
    "mediate_call",
    "parse_task_file",
    "parse_tasks",
    "propose_task",
    "teach_task",
    "write_criteria",
]

DEFAULT_AGENT_MODEL = os.environ.get("RADIXNET_AGENT_MODEL", "").strip() or DEFAULT_MODEL
"""Model that writes the criteria, mediates, judges and teaches (``RADIXNET_AGENT_MODEL`` overrides).

Any :class:`~radixnet.llm.LLMClient` does: Ollama's tool-calling API is used
where the provider has it, and a provider without it answers the same questions
in JSON mode instead.
"""

PHASES = ("teacher", "model")
MEDIATION = ("repair", "always", "never")
MAX_CRITERIA = 8
MAX_TRANSCRIPT_CHARS = 6000
MIN_EMISSION = 48
"""Characters the cheapest path is asked for: enough for a whole ``<tool>name {...}</tool>`` line."""


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Task:
    """Something to find out: a question, optionally with criteria and a known answer already written down."""

    id: str
    prompt: str
    criteria: tuple[str, ...] = ()
    answer: str | None = None
    seeds: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "prompt": self.prompt, "criteria": list(self.criteria),
            "answer": self.answer, "seeds": list(self.seeds),
        }

    @classmethod
    def from_any(cls, item: Any, index: int) -> "Task":
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"task {index}: empty prompt")
            return cls(f"t{index}", item.strip())
        if not isinstance(item, dict):
            raise ValueError(f"task {index}: must be a string or an object, got {type(item).__name__}")
        prompt = None
        for key in ("prompt", "task", "question", "goal"):
            if isinstance(item.get(key), str) and item[key].strip():
                prompt = item[key].strip()
                break
        if prompt is None:
            raise ValueError(f"task {index}: missing 'prompt'")
        criteria = item.get("criteria") or ()
        if isinstance(criteria, str):
            criteria = [line for line in criteria.splitlines() if line.strip()]
        if not isinstance(criteria, (list, tuple)) or any(not isinstance(c, str) for c in criteria):
            raise ValueError(f"task {index}: 'criteria' must be a list of strings")
        seeds = item.get("seeds") or item.get("urls") or ()
        if isinstance(seeds, str):
            seeds = [seeds]
        if not isinstance(seeds, (list, tuple)) or any(not isinstance(s, str) for s in seeds):
            raise ValueError(f"task {index}: 'seeds' must be a list of URLs")
        answer = item.get("answer", item.get("expected"))
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"task {index}: 'answer' must be a string")
        raw_id = item.get("id")
        return cls(
            str(raw_id).strip() if raw_id not in (None, "") else f"t{index}",
            prompt, tuple(c.strip() for c in criteria if c.strip()), answer,
            tuple(s.strip() for s in seeds if s.strip()),
        )


def parse_tasks(items: Iterable[Any]) -> list[Task]:
    """Strings or ``{prompt, criteria, answer, seeds}`` objects -> :class:`Task` list (ids default to ``t1, t2, ...``)."""
    tasks = [Task.from_any(item, i) for i, item in enumerate(items, 1)]
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
    return tasks


def parse_task_file(text: str, ext: str = ".txt") -> list[Task]:
    """``.txt`` one question per line (``#`` comments), ``.jsonl`` objects, ``.json`` a list or ``{"tasks": [...]}``."""
    ext = ext.lower()
    if ext == ".jsonl":
        items: list[Any] = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif ext == ".json":
        data = json.loads(text)
        items = data.get("tasks", data.get("problems")) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("a JSON task file must hold a list or an object with a 'tasks' list")
    else:
        items = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not items:
        raise ValueError("no tasks in the file")
    return parse_tasks(items)


def load_tasks(path: str) -> list[Task]:
    with open(path, encoding="utf-8") as fh:
        return parse_task_file(fh.read(), os.path.splitext(path)[1])


# ---------------------------------------------------------------------------
# one step, one attempt, one verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Step:
    """One tool call inside an attempt, and where the call came from."""

    index: int
    call: ToolCall
    result: ToolResult
    source: str  # "model" | "mediator" | "teacher"
    emission: str = ""

    @property
    def repaired(self) -> bool:
        return self.source == "mediator"

    def to_dict(self) -> dict:
        return {
            "index": self.index, "tool": self.call.name, "arguments": self.call.arguments, "source": self.source,
            "ok": self.result.ok, "output": self.result.output[:500], "error": self.result.error,
            "seconds": round(self.result.seconds, 4), "emission": self.emission[:200],
        }


@dataclasses.dataclass(slots=True)
class Verdict:
    """What the judge decided about a finished attempt."""

    correct: bool
    score: float | None = None
    met: tuple[bool, ...] = ()
    critique: str = ""
    issues: tuple[str, ...] = ()
    judged_by: str = "none"  # "ollama" | "answer" | "none"

    def to_dict(self) -> dict:
        return {
            "correct": self.correct, "score": self.score, "met": list(self.met),
            "critique": self.critique, "issues": list(self.issues), "judged_by": self.judged_by,
        }


@dataclasses.dataclass(slots=True)
class Failure:
    """A failed transcript and how badly it failed (``gap`` in ``[0, 1]``, 1 = it got nothing right)."""

    text: str
    gap: float
    task: str = ""
    source: str = "model"

    def to_dict(self) -> dict:
        return {"task": self.task, "source": self.source, "gap": round(self.gap, 4), "chars": len(self.text)}


@dataclasses.dataclass(slots=True)
class Attempt:
    """One run at a task: the steps taken, the answer given, the transcript learned and the verdict."""

    index: int
    source: str  # "model" | "teacher"
    steps: list[Step]
    answer: str | None
    text: str
    verdict: Verdict
    seconds: float = 0.0

    @property
    def calls(self) -> int:
        return len(self.steps)

    @property
    def own_calls(self) -> int:
        """Calls the network wrote by itself (the mediator repaired the rest)."""
        return sum(1 for s in self.steps if s.source == "model")

    @property
    def autonomy(self) -> float | None:
        return self.own_calls / self.calls if self.steps else None

    def feedback(self) -> str:
        """Why the attempt was rejected, in the words the teacher gets to see."""
        parts: list[str] = []
        if self.answer is None:
            parts.append("It never gave an answer.")
        failed = [s for s in self.steps if not s.result.ok]
        if failed:
            parts.append("Failed tool calls: " + "; ".join(f"{s.call.name}: {s.result.error}" for s in failed[:4]))
        if self.verdict.issues:
            parts.append("Unmet criteria: " + "; ".join(self.verdict.issues[:6]))
        if self.verdict.critique:
            parts.append("Judge: " + self.verdict.critique)
        return "\n".join(parts) or "It was judged incorrect."

    def to_dict(self) -> dict:
        return {
            "index": self.index, "source": self.source, "answer": self.answer, "calls": self.calls,
            "own_calls": self.own_calls, "autonomy": self.autonomy, "steps": [s.to_dict() for s in self.steps],
            "verdict": self.verdict.to_dict(), "correct": self.verdict.correct, "text_chars": len(self.text),
            "seconds": self.seconds,
        }


def _clip(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """The tail of a long transcript (what matters for the next step)."""
    return text if len(text) <= limit else "... " + text[-limit:]


# ---------------------------------------------------------------------------
# the four LLM roles
# ---------------------------------------------------------------------------

_CRITERIA_SYSTEM = (
    "You write acceptance criteria for a task that will be solved by browsing the web with tools. Write at most "
    "{count} short, checkable statements that any correct answer must satisfy: what the answer has to contain, how "
    "specific it has to be, and what would make it wrong. Judge only the answer, never the wording or the effort. "
    "Do not solve the task and do not state the answer. Reply with JSON only, of the form "
    "{{\"criteria\": [\"<statement>\", ...]}}."
)


def write_criteria(client: LLMClient, task: Task, *, model: str | None = None, count: int = 4) -> list[str]:
    """The statements a correct answer must satisfy, written before anything is attempted.

    A task that already carries criteria keeps them; an LLM that answers with
    nothing usable falls back to one criterion naming the task itself, so the
    judge always has something objective to mark against.
    """
    if task.criteria:
        return list(task.criteria)
    count = max(1, min(int(count), MAX_CRITERIA))
    raw = client.generate(
        f"Task: {task.prompt}\n\nWrite the criteria now.",
        system=_CRITERIA_SYSTEM.format(count=count), model=model, json_mode=True, options={"temperature": 0.2},
    )
    data = loads_lenient(raw)
    items: Any = data.get("criteria") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    criteria: list[str] = []
    if isinstance(items, list):
        for item in items:
            text = item if isinstance(item, str) else (item.get("criterion") or item.get("text") or "") if isinstance(item, dict) else ""
            text = " ".join(str(text).split())
            if text and text not in criteria:
                criteria.append(text)
    if not criteria:
        criteria = parse_lines(raw, limit=count)
    if not criteria:
        criteria = [f"The answer states, correctly and specifically, what the task asked for: {task.prompt}"]
    return criteria[:count]


_MEDIATOR_SYSTEM = (
    "You are the tool mediator for a very small character-level neural network that is learning to use tools. It "
    "writes its intentions as raw text, which is often garbled, truncated or empty. Your job is to turn what it "
    "wrote into exactly ONE valid tool call that makes progress on the task — never to solve the task yourself. "
    "Read its text charitably: if it reaches for a search, search for what it seems to want; if it names a URL, open "
    "that URL; if it is unreadable, choose the call that the task and the transcript so far make obvious. Never "
    "repeat a call that is already in the transcript. Reply with JSON only, of the form "
    "{\"tool\": \"<name>\", \"arguments\": {...}}."
)


def _tools_block(toolbox: ToolBox) -> str:
    return "The tools are:\n" + toolbox.catalogue()


def _chat_message(client: LLMClient, messages: list[dict], *, tools: list[dict] | None = None, **options: Any) -> dict:
    """One assistant message, with ``tools`` when the provider supports tool calling.

    Ollama answers with ``{"tool_calls": [...]}``; a provider whose client has
    no ``chat_message`` (or refuses the tools) answers in JSON instead, which
    :func:`_call_from_message` reads just the same.
    """
    chat_message = getattr(client, "chat_message", None)
    if chat_message is not None:
        try:
            return chat_message(messages, tools=tools, **options)
        except LLMError:
            if not tools:
                raise
    return {"role": "assistant", "content": client.chat(messages, json_mode=True, **options)}


def _call_from_message(message: dict, toolbox: ToolBox) -> ToolCall | None:
    """One call out of Ollama's native ``tool_calls``, or out of a JSON answer in the content."""
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for entry in calls:
            function = entry.get("function") if isinstance(entry, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = loads_lenient(arguments)
            if name and toolbox.has(name):
                return ToolCall(name, arguments if isinstance(arguments, dict) else {})
    data = loads_lenient(message.get("content") or "")
    if isinstance(data, dict):
        name = str(data.get("tool") or data.get("name") or data.get("function") or "").strip()
        arguments = data.get("arguments", data.get("args", data.get("parameters")))
        if isinstance(arguments, str):
            arguments = loads_lenient(arguments)
        if name and toolbox.has(name):
            return ToolCall(name, arguments if isinstance(arguments, dict) else {})
    return None


def mediate_call(
    client: LLMClient,
    emission: str,
    toolbox: ToolBox,
    task: Task,
    transcript: str = "",
    *,
    model: str | None = None,
    hint: str | None = None,
) -> ToolCall:
    """Turn the network's unusable emission into one valid call — Ollama's tool-calling API, then JSON, then a guess.

    Raises :class:`~radixnet.llm.LLMError` only when the LLM cannot be
    reached; an answer that names no known tool falls back to the first network
    tool with the task as its argument, so the loop always makes a move.
    """
    user = (
        f"Task: {task.prompt}\n\n{_tools_block(toolbox)}\n\n"
        f"Transcript so far:\n{_clip(transcript, 2000) or '(nothing yet)'}\n\n"
        f"What the network just wrote:\n{(emission or '').strip()[:500] or '(nothing)'}\n\n"
        + (f"{hint}\n\n" if hint else "")
        + "Give the one tool call to make now."
    )
    messages = [{"role": "system", "content": _MEDIATOR_SYSTEM}, {"role": "user", "content": user}]
    message: dict = {}
    try:
        message = client.chat_message(messages, model=model, tools=toolbox.schemas(), options={"temperature": 0.2})
    except LLMError:
        message = {}
    call = _call_from_message(message, toolbox) if message else None
    if call is None:  # a model without tool calling: ask for the JSON directly
        raw = client.chat(messages, model=model, json_mode=True, options={"temperature": 0.2})
        call = _call_from_message({"content": raw}, toolbox)
    if call is None:
        fallback = next((t for t in toolbox.tools() if t.network), None) or next(iter(toolbox.tools()), None)
        if fallback is None:
            return ToolCall("", {}, error="no tools are installed")
        required = [p for p in fallback.params if p.required]
        call = ToolCall(fallback.name, {required[0].name: task.prompt} if required else {})
    tool = toolbox.get(call.name)
    try:
        call.arguments = tool.coerce(call.arguments)
    except Exception as exc:  # noqa: BLE001 - reported on the call, executed as a failure
        call.error = str(exc)
    return call


_JUDGE_SYSTEM = (
    "You are the judge of an attempt at a task by a very small neural network that browses with tools. You are given "
    "the acceptance criteria written before the attempt, the transcript of the tool calls and their results, and the "
    "answer. Mark each criterion strictly: met only if the answer really satisfies it — an answer that is empty, "
    "vague, off-topic, or contradicted by the transcript meets nothing. Never reward effort, only the answer. Reply "
    "with JSON only, of the form {\"met\": [true, false, ...] with one entry per criterion in order, "
    "\"score\": <0-10>, \"correct\": true or false, \"critique\": \"<one sentence naming the worst flaw, or 'none'>\"}."
)


_TRUE_WORDS = frozenset("true yes y pass passed met ok correct 1".split())
_FALSE_WORDS = frozenset("false no n fail failed unmet not_met missing incorrect 0".split())


def _mark(value: Any) -> bool | None:
    """One criterion's mark: a bool, a number, or a word the LLM used for it (``None`` = unreadable)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, dict):
        for key in ("met", "ok", "pass", "passed", "result", "verdict"):
            if key in value:
                return _mark(value[key])
        return None
    if isinstance(value, str):
        word = value.strip().strip(".").lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


def _marks(value: Any) -> list[bool]:
    """The per-criterion marks of a judge's answer, or ``[]`` when any of them cannot be read.

    All or nothing on purpose: a partly readable list would silently misalign
    with the criteria it is supposed to mark, and a criterion counted as met
    because its text happened to be a non-empty string is worse than no mark.
    """
    if not isinstance(value, list) or not value:
        return []
    marks = [_mark(item) for item in value]
    return [] if any(mark is None for mark in marks) else marks  # type: ignore[misc]


def judge_attempt(
    client: LLMClient,
    task: Task,
    criteria: list[str],
    transcript: str,
    answer: str | None,
    *,
    model: str | None = None,
) -> dict:
    """Mark a finished attempt against ``criteria``: ``{"met", "score", "correct", "critique"}``."""
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1)) or "1. The answer is correct."
    user = (
        f"Task: {task.prompt}\n\nAcceptance criteria:\n{numbered}\n\n"
        f"Transcript:\n{_clip(transcript)}\n\nAnswer: {answer if answer else '(no answer was given)'}\n\n"
        + (f"The known correct answer is: {task.answer}\n\n" if task.answer else "")
        + f"Return the JSON with exactly {len(criteria) or 1} entries in \"met\"."
    )
    raw = client.generate(user, system=_JUDGE_SYSTEM, model=model, json_mode=True, options={"temperature": 0.1})
    data = loads_lenient(raw)
    if not isinstance(data, dict):
        return {"met": [], "score": None, "correct": None, "critique": "the judge returned nothing usable"}
    met = _marks(data.get("met", data.get("criteria_met", data.get("results"))))
    score: float | None
    try:
        score = max(0.0, min(10.0, float(data.get("score", data.get("rating")))))
    except (TypeError, ValueError):
        score = None
    correct = data.get("correct", data.get("verdict"))
    if isinstance(correct, str):
        correct = correct.strip().lower() in ("true", "pass", "yes", "correct")
    elif not isinstance(correct, bool):
        correct = None
    return {
        "met": met, "score": score, "correct": correct,
        "critique": " ".join(str(data.get("critique") or data.get("reason") or "").split()),
    }


def _decide(judged: dict, criteria: list[str], answer: str | None, strict: bool) -> Verdict:
    """The judge's answer as a :class:`Verdict`; no answer is never correct, and ``strict`` needs every criterion met."""
    met = tuple(judged.get("met") or ())
    issues = tuple(c for c, ok in zip(criteria, met, strict=False) if not ok)
    if not answer:
        return Verdict(False, judged.get("score"), met, judged.get("critique") or "no answer was given",
                       issues or tuple(criteria), "answer")
    correct = judged.get("correct")
    all_met = bool(met) and all(met) and len(met) >= len(criteria)
    if correct is None:
        correct = all_met if met else bool(judged.get("score") is not None and judged["score"] >= 6.0)
    if strict:
        correct = bool(correct) and (all_met if met else True)
    return Verdict(bool(correct), judged.get("score"), met, judged.get("critique") or "", issues,
                   "ollama" if judged.get("met") or judged.get("score") is not None else "none")


_TEACHER_SYSTEM = (
    "You solve a task with the tools you are given, so that a very small neural network can learn from your example. "
    "Work the way it must: call one tool at a time, read the result, and call another if you need to. Use the tools "
    "for anything factual — never answer from memory alone. When you know the answer, stop calling tools and reply "
    "with the answer itself, one short sentence, nothing else."
)


def teach_task(
    client: LLMClient,
    task: Task,
    toolbox: ToolBox,
    *,
    model: str | None = None,
    max_steps: int = 6,
    criteria: list[str] | None = None,
    feedback: str | None = None,
    external: Callable[[], Any] = nullcontext,
    observation_chars: int = 600,
) -> tuple[list[Step], str | None]:
    """The LLM solves the task with the *real* tools; returns the steps it took and its answer.

    Every call is executed for real, so the demonstration the network learns
    from is a transcript of things that actually happened — not an invented one.
    """
    numbered = "\n".join(f"- {c}" for c in (criteria or []))
    user = (
        f"Task: {task.prompt}\n\n{_tools_block(toolbox)}\n"
        + (f"\nA correct answer must satisfy:\n{numbered}\n" if numbered else "")
        + (f"\nAn earlier attempt failed:\n{feedback}\n" if feedback else "")
        + (f"\nStart from: {', '.join(task.seeds)}\n" if task.seeds else "")
        + "\nMake the first tool call now."
    )
    messages: list[dict] = [{"role": "system", "content": _TEACHER_SYSTEM}, {"role": "user", "content": user}]
    schemas = toolbox.schemas()
    steps: list[Step] = []
    native = True
    for index in range(max(1, int(max_steps))):
        with external():
            if native:
                try:
                    message = client.chat_message(messages, model=model, tools=schemas, options={"temperature": 0.3})
                except LLMError:
                    native, message = False, {}
            if not native:
                message = {"role": "assistant", "content": client.chat(messages, model=model, options={"temperature": 0.3})}
        call = _call_from_message(message, toolbox)
        content = (message.get("content") or "").strip()
        if call is None:
            if content:
                return steps, " ".join(content.split())
            if steps:
                break
            messages.append({"role": "user", "content": "Make a tool call now, or give the answer."})
            continue
        with external():
            result = toolbox.run(call)
        steps.append(Step(index, call, result, "teacher"))
        messages.append(message if message.get("role") else {"role": "assistant", "content": content})
        messages.append({
            "role": "tool",
            "content": (result.output if result.ok else f"ERROR: {result.error}")[: max(100, observation_chars * 2)],
        })
    with external():
        answer = client.chat(
            messages + [{"role": "user", "content": "Give the answer now, one short sentence, nothing else."}],
            model=model, options={"temperature": 0.2},
        )
    return steps, " ".join(answer.split()) or None


_PROPOSE_SYSTEM = (
    "A very small neural network that browses the web is choosing what to look into next, and it writes its "
    "intentions as raw, often garbled text. Turn what it wrote into ONE concrete question that can be settled by "
    "browsing and answered in a sentence. Follow whatever it seems to be reaching for; when it is unreadable, ask "
    "something that follows on from what it has already looked at, not a repeat of it. Prefer a question about one "
    "of the unvisited pages when they fit. Reply with JSON only, of the form "
    "{\"question\": \"<the question>\", \"seed\": \"<a URL to start from, or empty>\"}."
)


def propose_task(
    client: LLMClient,
    emission: str,
    *,
    model: str | None = None,
    frontier: Iterable[str] = (),
    visited: Iterable[str] = (),
    index: int = 1,
) -> Task:
    """Turn the network's own emission into one concrete, answerable question (exploration picks its own tasks)."""
    unvisited = list(frontier)[:12]
    seen = list(visited)[-8:]

    def listed(title: str, urls: list[str]) -> str:
        return f"{title}:\n" + "\n".join(f"- {url}" for url in urls) + "\n\n" if urls else ""

    user = (
        f"What the network wrote:\n{(emission or '').strip()[:500] or '(nothing)'}\n\n"
        + listed("Unvisited pages it has come across", unvisited)
        + listed("Pages it has already read", seen)
        + "Give the question now."
    )
    raw = client.generate(user, system=_PROPOSE_SYSTEM, model=model, json_mode=True, options={"temperature": 0.8})
    data = loads_lenient(raw)
    question = seed = ""
    if isinstance(data, dict):
        question = " ".join(str(data.get("question") or data.get("task") or data.get("prompt") or "").split())
        seed = str(data.get("seed") or data.get("url") or "").strip()
    if not question:
        lines = parse_lines(raw, limit=1)
        question = lines[0] if lines else ""
    if not question:
        question = f"What can be found out about {unvisited[0]}?" if unvisited else "What is on the front page of a well-known encyclopaedia?"
    return Task(f"x{index}", question, seeds=(seed,) if seed.startswith("http") else ())


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AgentConfig:
    """How the loop runs: who attempts, how far, how strictly it is judged, and how hard it is taught."""

    agent_model: str = DEFAULT_AGENT_MODEL
    judge_model: str | None = None
    phases: tuple[str, ...] = ("model",)
    rounds: int = 1
    max_steps: int = 6            # tool calls per attempt
    model_attempts: int = 2       # attempts by the network per task
    teacher_attempts: int = 1
    first_attempt_dijkstra: bool = True   # the first attempt is the beam search, not a sample
    candidates: int = 5           # continuations it may offer per step before the mediator steps in
    temperature: float = 1.0
    max_length: int = 200         # characters the network writes per step
    mediation: str = "repair"     # "repair" (only unusable emissions) | "always" | "never"
    criteria_count: int = 4
    strict: bool = True           # every criterion must be met
    use_judge: bool = True
    teach_on_failure: bool = True
    observation_chars: int = 600
    twonrl_per: str = "task"      # "task" | "round"
    replay: bool = True
    replay_limit: int = 64
    read_reward: bool = False     # also fine-tune on the text of the pages that were read
    blatant_mode: str = "fail_invert"  # "none" | "fail_invert" | "activation" | "state" (section 9.1)
    blatant_margin: float = 0.5   # the gap at which a failure counts as blatant (1 = missed everything)
    blatant_boost: float = 4.0    # the largest learning-rate multiplier a failure can earn
    pass_score: float = 6.0       # the judge's 0-10 score a correct answer is expected to reach
    neg_epochs: int = 2
    pos_epochs: int = 3
    neg_lr: float = 0.5
    pos_lr: float = 0.1
    batch_size: int = 4
    checkpoint_every: int = 0     # tasks
    seed: int = 0

    def validate(self) -> None:
        if not self.phases or any(p not in PHASES for p in self.phases):
            raise ValueError(f"phases must be a non-empty subset of {', '.join(PHASES)}")
        if self.mediation not in MEDIATION:
            raise ValueError(f"mediation must be one of {', '.join(MEDIATION)} (got {self.mediation!r})")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.model_attempts < 1 or self.teacher_attempts < 1:
            raise ValueError("model_attempts and teacher_attempts must be >= 1")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.candidates < 1:
            raise ValueError("candidates must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 1 <= self.criteria_count <= MAX_CRITERIA:
            raise ValueError(f"criteria_count must be between 1 and {MAX_CRITERIA}")
        if self.twonrl_per not in ("task", "round"):
            raise ValueError("twonrl_per must be 'task' or 'round'")
        if self.blatant_mode not in BLATANT_MODES:
            raise ValueError(f"blatant_mode must be one of {', '.join(BLATANT_MODES)} (got {self.blatant_mode!r})")
        if not self.blatant_margin > 0:
            raise ValueError("blatant_margin must be > 0")
        if not self.blatant_boost >= 1:
            raise ValueError("blatant_boost must be >= 1")
        if not 0 < self.pass_score <= 10:
            raise ValueError("pass_score must be in (0, 10]")
        if self.observation_chars < 0 or self.replay_limit < 0 or self.checkpoint_every < 0:
            raise ValueError("observation_chars, replay_limit and checkpoint_every must be >= 0")
        if self.neg_epochs < 0 or self.pos_epochs < 0 or self.neg_lr < 0 or self.pos_lr < 0:
            raise ValueError("epochs and learning rates must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["phases"] = list(self.phases)
        return data


ProgressFn = Callable[[dict], None]


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class AgentTrainer:
    """Runs tasks (or self-chosen exploration) through tools and teaches the network with 2NRL.

    ``external`` is a zero-argument callable returning a context manager that is
    entered around slow outside work (tool calls, LLM calls); the API passes
    ``ModelService.pause_lock`` so readers are served while the network browses.
    """

    def __init__(
        self,
        model: Any,
        client: LLMClient,
        toolbox: ToolBox,
        config: AgentConfig | None = None,
        external: Callable[[], Any] | None = None,
    ) -> None:
        if not len(toolbox):
            raise ValueError("the toolbox is empty: there is nothing for the network to call")
        self.model = model
        self.client = client
        self.toolbox = toolbox
        self.config = config or AgentConfig()
        self.config.validate()
        self._external = external or nullcontext
        self.history: list[dict] = []
        self.replay_buffer: list[str] = []
        self.solved: dict[str, str] = {}
        self.criteria: dict[str, list[str]] = {}
        self.frontier: list[str] = []
        self.visited: list[str] = []
        self.pages: list[str] = []
        self.rng = random.Random(self.config.seed)
        self._stop = threading.Event()

    # -- helpers -------------------------------------------------------------

    def _stopped(self) -> bool:
        return self._stop.is_set()

    def _emit(self, progress: ProgressFn | None, record: dict) -> None:
        self.history.append(record)
        if progress is not None:
            progress(record)

    def _model_name(self, judge: bool = False) -> str:
        return (self.config.judge_model or self.config.agent_model) if judge else self.config.agent_model

    def _note_urls(self, result: ToolResult) -> None:
        """Keep the pages a tool turned up as the frontier of the exploration."""
        found: list[str] = []
        for key in ("results", "links"):
            items = result.meta.get(key)
            for item in items if isinstance(items, (list, tuple)) else ():
                url = item.get("url") if isinstance(item, dict) else None
                if isinstance(url, str) and url.startswith("http"):
                    found.append(url)
        url = result.meta.get("url")
        if isinstance(url, str) and url.startswith("http") and url not in self.visited:
            self.visited.append(url)
        for candidate in found:
            if candidate not in self.visited and candidate not in self.frontier:
                self.frontier.append(candidate)
        if len(self.frontier) > 200:
            del self.frontier[: len(self.frontier) - 200]
        if self.config.read_reward and result.ok and result.meta.get("chars"):
            page = result.output.strip()
            if len(page) >= 3:
                self.pages.append(page)
                del self.pages[:-32]

    def criteria_for(self, task: Task) -> list[str]:
        """The acceptance criteria of a task, written once by the LLM and remembered."""
        if task.id in self.criteria:
            return self.criteria[task.id]
        if task.criteria:
            criteria = list(task.criteria)
        elif not self.config.use_judge:
            criteria = [f"The answer states what the task asked for: {task.prompt}"]
        else:
            with self._external():
                criteria = write_criteria(
                    self.client, task, model=self._model_name(), count=self.config.criteria_count
                )
        self.criteria[task.id] = criteria
        return criteria

    # -- the network's own attempt -------------------------------------------

    def _emissions(self, transcript: str, attempt_index: int) -> list[str]:
        """What the network offers to write next, most likely first.

        The first attempt is deterministic: the beam search returns its
        ``candidates`` most likely continuations, and the caller takes the
        first one that is a usable call — the same "what else could follow
        this?" the dialogue module uses.  A single cheapest path is no good
        here: Dijkstra accepts reaching END as a goal, so it happily answers
        with a couple of characters that can never be a whole call.  Later
        attempts sample instead, which is where the variety comes from.
        """
        cfg = self.config
        prefix = _clip(transcript, 2000)
        wanted = min(MIN_EMISSION, cfg.max_length)
        if attempt_index == 0 and cfg.first_attempt_dijkstra:
            prediction = self.model.predict(
                prefix, length=wanted, mode="beam", k=cfg.candidates, max_length=cfg.max_length
            )
            texts = [r.text for r in (getattr(prediction, "top", None) or [prediction])]
        else:
            texts = [
                self.model.predict(
                    prefix, length=1, mode="sample", temperature=cfg.temperature, max_length=cfg.max_length
                ).text
                for _ in range(cfg.candidates)
            ]
        offered = [text for text in dict.fromkeys(texts) if text.strip()]
        # a candidate that closed its tag wrote a whole call; an unterminated one only started one
        return sorted(offered, key=lambda text: CALL_CLOSE not in text and ANSWER_CLOSE not in text)

    def _next_call(self, task: Task, transcript: str, attempt_index: int, step_index: int) -> tuple[ToolCall | None, str, str, str | None]:
        """``(call, source, emission, answer)`` for one step: what the network wrote, repaired if it has to be.

        Every candidate it offers is tried in turn; the first usable call (or
        an answer it gives instead) is the network's own move.  Only when none
        of them can be used does the mediator step in, on the best candidate.
        """
        cfg = self.config
        candidates = [] if cfg.mediation == "always" else self._emissions(transcript, attempt_index)
        broken: ToolCall | None = None
        for emission in candidates:
            answer_at = emission.find(ANSWER_OPEN)
            call_at = emission.find(CALL_OPEN)
            if answer_at >= 0 and (call_at < 0 or answer_at < call_at):
                answer = find_answer(emission)
                if answer:
                    return None, "model", emission, answer
            call = parse_call(emission, self.toolbox)
            if call is not None and call.ok:
                return call, "model", emission, None
            if broken is None and call is not None:
                broken = call
        emission = candidates[0] if candidates else ""
        if cfg.mediation == "never":
            # a broken call is executed as a failure and learned from; nothing at all ends the attempt
            return broken, "model", emission, None
        with self._external():
            mediated = mediate_call(
                self.client, emission, self.toolbox, task, transcript, model=self._model_name(),
                hint=f"Its call could not be read: {broken.error}" if broken is not None and broken.error else None,
            )
        return mediated, "mediator", emission, None

    def solve_with_model(self, task: Task, index: int, progress: ProgressFn | None, phase: str) -> Attempt:
        """One attempt by the network: it writes calls, the toolbox runs them, it answers."""
        t0 = time.perf_counter()
        cfg = self.config
        transcript = task_header(task.prompt)
        if task.seeds:
            transcript += f"START: {', '.join(task.seeds)}\n"
        steps: list[Step] = []
        answer: str | None = None
        for step_index in range(cfg.max_steps):
            if self._stopped():
                break
            call, source, emission, said = self._next_call(task, transcript, index, step_index)
            if call is None:
                answer = said
                break
            with self._external():
                result = self.toolbox.run(call)
            self._note_urls(result)
            step = Step(step_index, call, result, source, emission)
            steps.append(step)
            transcript += call_text(call.name, call.arguments) + format_observation(result, cfg.observation_chars)
            self._emit_step(progress, phase, task, index, step)
        if answer is None and not self._stopped():
            # out of steps: whatever it writes now is its answer, tagged or not - an untrained network
            # answers with noise, and that noise is the failure the judge rejects and 2NRL trains on
            offered = self._emissions(transcript, index)
            for emission in offered:
                answer = find_answer(emission)
                if answer:
                    break
            else:
                answer = " ".join(offered[0].split())[:200] if offered else None
        text = transcript + (answer_text(answer) if answer else "")
        return Attempt(index, "model", steps, answer, text, Verdict(False), time.perf_counter() - t0)

    # -- the teacher's attempt -------------------------------------------------

    def solve_with_teacher(self, task: Task, criteria: list[str], index: int, feedback: str | None = None) -> Attempt:
        """The LLM demonstrates: the same tools, really called, and the answer it reaches."""
        t0 = time.perf_counter()
        cfg = self.config
        steps, answer = teach_task(
            self.client, task, self.toolbox, model=self._model_name(), max_steps=cfg.max_steps, criteria=criteria,
            feedback=feedback, external=self._external, observation_chars=cfg.observation_chars,
        )
        for step in steps:
            self._note_urls(step.result)
        text = transcript_text(
            task.prompt, [(s.call.name, s.call.arguments, s.result) for s in steps], answer, cfg.observation_chars
        )
        return Attempt(index, "teacher", steps, answer, text, Verdict(False), time.perf_counter() - t0)

    # -- judging ----------------------------------------------------------------

    def judge(self, task: Task, criteria: list[str], attempt: Attempt) -> Verdict:
        """Mark an attempt against the criteria written before it started."""
        if not attempt.answer:
            return Verdict(False, None, (), "no answer was given", tuple(criteria), "answer")
        if not self.config.use_judge:
            correct = bool(task.answer) and task.answer.strip().lower() in attempt.answer.lower()
            return Verdict(correct, 10.0 if correct else 0.0, (), "" if correct else "the known answer is not in it",
                           () if correct else tuple(criteria), "answer" if task.answer else "none")
        with self._external():
            judged = judge_attempt(
                self.client, task, criteria, attempt.text, attempt.answer, model=self._model_name(judge=True)
            )
        return _decide(judged, criteria, attempt.answer, self.config.strict)

    # -- learning: failures first, then invert -----------------------------------

    def gap_of(self, attempt: Attempt, criteria: list[str]) -> float:
        """How badly an attempt failed, in ``[0, 1]``: 1 = it answered nothing or satisfied nothing.

        The judge gives a 0-10 score and marks each criterion, so the gap is
        the worse of "how far below :attr:`AgentConfig.pass_score` it scored"
        and "what share of the criteria it missed".  Every failure keeps a
        floor of ``0.1`` — a failure that is only just a failure still trains.
        """
        if attempt.verdict.correct:
            return 0.0
        if not attempt.answer:
            return 1.0
        gaps = []
        score = attempt.verdict.score
        if score is not None:
            gaps.append(max(0.0, (self.config.pass_score - score) / self.config.pass_score))
        met = attempt.verdict.met
        if met:
            gaps.append(sum(1 for ok in met if not ok) / len(met))
        elif criteria:
            gaps.append(1.0)
        return min(1.0, max(0.1, max(gaps) if gaps else 1.0))

    def learn(self, failures: list["Failure"], good: list[str]) -> dict:
        """Train on the failures — the worse, the harder — then invert, then fine-tune on what was right.

        This is section 9.1 applied to tool use: with
        ``blatant_mode="fail_invert"`` every failed transcript is a negative
        pass whose learning rates are multiplied by
        ``min(blatant_boost, 1 + gap / blatant_margin)``, so the network is
        pushed to reproduce its worst attempts *on purpose* before
        :meth:`~radixnet.model.RadixNet.invert` turns that into avoidance.  The
        local modes (``activation`` / ``state``) instead negate the nodes on a
        failed path, and blatant failures then leave the 2NRL garbage set.
        Nothing failed -> the correct run is rewarded and nothing is inverted.
        """
        cfg = self.config
        good_all = list(dict.fromkeys(good))
        if cfg.read_reward and self.pages:
            good_all += [p for p in self.pages if p not in good_all]
        if cfg.replay:
            extra = [t for t in self.replay_buffer if t not in good_all]
            good_all += extra[-cfg.replay_limit:] if cfg.replay_limit else []
        seen: dict[str, float] = {}
        for failure in failures:  # a transcript that failed twice keeps its worst gap
            if failure.text and failure.text not in good_all:
                seen[failure.text] = max(seen.get(failure.text, 0.0), failure.gap)
        bad = list(seen)
        gaps = [seen[text] for text in bad]
        blatant = [text for text, gap in seen.items() if gap > cfg.blatant_margin]
        result: dict[str, Any] = {
            "bad": len(bad), "good": len(good_all), "action": None, "neg_loss": None, "pos_loss": None,
            "failures": len(bad), "blatant": len(blatant), "mode": cfg.blatant_mode, "flipped": 0,
            "boost_mean": None, "boost_max": None,
        }
        if not bad and not good_all:
            return result
        weights: list[float] = []
        if bad and cfg.blatant_mode == "fail_invert":
            weights = [min(cfg.blatant_boost, 1.0 + gap / cfg.blatant_margin) for gap in gaps]
        elif bad and cfg.blatant_mode in ("activation", "state"):
            # the local variant: move every other node of a failed path toward its negation, the worse the more
            amounts = [min(1.0, gap / (2.0 * cfg.blatant_margin)) for gap in gaps]
            outcome = self.model.invert_paths(bad, mode=cfg.blatant_mode, amounts=amounts)
            result.update(flipped=outcome["flipped"], boost_mean=outcome["amount_mean"], boost_max=max(amounts))
            blatant_set = set(blatant)
            keep = [(text, gap) for text, gap in zip(bad, gaps) if text not in blatant_set]
            bad, gaps = [t for t, _ in keep], [g for _, g in keep]
        if weights:
            result.update(boost_mean=sum(weights) / len(weights), boost_max=max(weights))
        if bad and good_all:
            outcome = self.model.two_nrl(
                bad, good_all, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs, neg_lr=cfg.neg_lr,
                pos_lr=cfg.pos_lr, batch_size=cfg.batch_size, bad_weights=weights or None, stop_event=self._stop,
            )
            result.update(action="2nrl", neg_loss=_last_loss(outcome["negative"]), pos_loss=_last_loss(outcome["positive"]))
        elif good_all:
            records = self.model.reward(good_all, epochs=cfg.pos_epochs, lr=cfg.pos_lr, batch_size=cfg.batch_size, stop_event=self._stop)
            result.update(action="reward", pos_loss=_last_loss(records))
        elif bad:
            records = self._punish_weighted(bad, weights)
            result.update(action="punish", neg_loss=_last_loss(records))
        for text in good:
            if text not in self.replay_buffer:
                self.replay_buffer.append(text)
        if cfg.replay_limit and len(self.replay_buffer) > cfg.replay_limit:
            del self.replay_buffer[: len(self.replay_buffer) - cfg.replay_limit]
        self.pages.clear()
        return result

    def _punish_weighted(self, bad: list[str], weights: list[float]) -> list[dict]:
        """Nothing went right: the negative passes (heaviest failure first), then the inversion."""
        cfg = self.config
        if not weights:
            return self.model.punish(bad, epochs=cfg.neg_epochs, lr=cfg.neg_lr, batch_size=cfg.batch_size, stop_event=self._stop)
        records: list[dict] = []
        groups: dict[float, list[str]] = {}
        for text, weight in zip(bad, weights):
            groups.setdefault(round(weight, 3), []).append(text)
        for weight, group in sorted(groups.items(), key=lambda item: -item[0]):
            if self._stopped():
                break
            written = self.model.train(
                group, epochs=cfg.neg_epochs, lr=cfg.neg_lr * weight, phase="negative",
                batch_size=cfg.batch_size, stop_event=self._stop,
            )
            for record in written:
                record["weight"] = weight
            records.extend(written)
        if not self._stopped():
            self.model.invert()
        return records

    # -- one task ----------------------------------------------------------------

    def _emit_step(self, progress: ProgressFn | None, phase: str, task: Task, attempt: int, step: Step) -> None:
        self._emit(progress, {
            "kind": "step", "phase": phase, "task": task.id, "attempt": attempt + 1, "step": step.index + 1,
            "tool": step.call.name, "arguments": step.call.arguments, "source": step.source,
            "ok": step.result.ok, "error": step.result.error, "output": step.result.output[:300],
            "seconds": round(step.result.seconds, 3),
        })

    def _emit_attempt(self, progress: ProgressFn | None, phase: str, round_no: int, task: Task, attempt: Attempt) -> None:
        self._emit(progress, {
            "kind": "attempt", "phase": phase, "round": round_no, "task": task.id, "attempt": attempt.index + 1,
            "source": attempt.source, "calls": attempt.calls, "own_calls": attempt.own_calls,
            "autonomy": attempt.autonomy, "answer": attempt.answer, "correct": attempt.verdict.correct,
            "score": attempt.verdict.score, "met": list(attempt.verdict.met), "critique": attempt.verdict.critique,
            "seconds": attempt.seconds,
        })

    def run_task(
        self, task: Task, phase: str, round_no: int, progress: ProgressFn | None
    ) -> tuple[dict, list[Failure], list[str]]:
        """Criteria, attempts, judging and (on failure) the teacher's demonstration for one task."""
        t0 = time.perf_counter()
        cfg = self.config
        criteria = self.criteria_for(task)
        self._emit(progress, {
            "kind": "criteria", "phase": phase, "round": round_no, "task": task.id,
            "prompt": task.prompt[:300], "criteria": criteria,
        })
        attempts: list[Attempt] = []
        if phase == "model":
            for index in range(cfg.model_attempts):
                if self._stopped():
                    break
                attempt = self.solve_with_model(task, index, progress, phase)
                attempt.verdict = self.judge(task, criteria, attempt)
                attempts.append(attempt)
                self._emit_attempt(progress, phase, round_no, task, attempt)
                if attempt.verdict.correct:
                    break
        taught = False
        if (phase == "teacher" or (cfg.teach_on_failure and not any(a.verdict.correct for a in attempts))) and not self._stopped():
            feedback = attempts[-1].feedback() if attempts else None
            for _try in range(cfg.teacher_attempts):
                attempt = self.solve_with_teacher(task, criteria, len(attempts), feedback)
                attempt.verdict = self.judge(task, criteria, attempt)
                attempts.append(attempt)
                taught = True
                self._emit_attempt(progress, phase, round_no, task, attempt)
                if attempt.verdict.correct or self._stopped():
                    break
                feedback = attempt.feedback()
        correct = next((a for a in attempts if a.verdict.correct), None)
        good = [a.text for a in attempts if a.verdict.correct]
        failures = [
            Failure(a.text, self.gap_of(a, criteria), task.id, a.source) for a in attempts if not a.verdict.correct
        ]
        if correct is not None:
            self.solved[task.id] = correct.text
        model_attempts = [a for a in attempts if a.source == "model"]
        own = sum(a.own_calls for a in model_attempts)
        calls = sum(a.calls for a in model_attempts)
        shown = correct if correct is not None else (attempts[-1] if attempts else None)
        record = {
            "kind": "task", "phase": phase, "round": round_no, "task": task.id, "prompt": task.prompt[:200],
            "criteria": len(criteria), "attempts": len(attempts), "correct": correct is not None,
            "solved_by": correct.source if correct else None,
            "model_solved": any(a.source == "model" and a.verdict.correct for a in attempts),
            "taught": taught, "calls": calls, "own_calls": own, "autonomy": (own / calls) if calls else None,
            "answer": shown.answer if shown else None, "score": shown.verdict.score if shown else None,
            "issues": list(shown.verdict.issues[:4]) if shown else [], "bad": len(failures), "good": len(good),
            "gap_max": max((f.gap for f in failures), default=None),
            "seconds": time.perf_counter() - t0,
        }
        return record, failures, good

    # -- driving -------------------------------------------------------------------

    def run(
        self,
        tasks: list[Task],
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager: Any = None,
    ) -> list[dict]:
        """Every round and phase over ``tasks``; returns the task / round records."""
        if not tasks:
            raise ValueError("no tasks to solve")
        cfg = self.config
        self._stop = stop_event if stop_event is not None else threading.Event()
        records: list[dict] = []
        step = 0
        for round_no in range(1, cfg.rounds + 1):
            for phase in cfg.phases:
                t0 = time.perf_counter()
                round_bad: list[Failure] = []
                round_good: list[str] = []
                solved = model_solved = done = 0
                for task in tasks:
                    if self._stopped():
                        break
                    t_task = time.perf_counter()
                    record, bad, good = self.run_task(task, phase, round_no, progress)
                    if cfg.twonrl_per == "task":
                        record.update(self.learn(bad, good))
                    else:
                        round_bad += bad
                        round_good += good
                    done += 1
                    solved += int(record["correct"])
                    model_solved += int(record["model_solved"])
                    record["seconds"] = time.perf_counter() - t_task
                    self._emit(progress, record)
                    records.append(record)
                    step += 1
                    if checkpoint_manager is not None and cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                        checkpoint_manager.save(self.model, step, "agent", {k: record[k] for k in ("phase", "round", "task", "correct")})
                learned: dict = {}
                if cfg.twonrl_per == "round" and (round_bad or round_good) and not self._stopped():
                    learned = self.learn(round_bad, round_good)
                summary = {
                    "kind": "round", "phase": phase, "round": round_no, "tasks": done, "solved": solved,
                    "model_solved": model_solved, "seconds": time.perf_counter() - t0, **learned,
                }
                self._emit(progress, summary)
                records.append(summary)
                if self._stopped():
                    return records
        return records

    # -- exploring on its own ---------------------------------------------------

    def propose(self, index: int) -> tuple[Task, str]:
        """The network chooses what to look into next; the LLM turns that into an answerable question."""
        seed = task_header("").strip()  # "TASK:" - what every transcript it has learned starts with
        emission = self.model.predict(
            seed, length=1, mode="sample", temperature=max(0.8, self.config.temperature),
            max_length=self.config.max_length,
        ).text
        with self._external():
            task = propose_task(
                self.client, emission, model=self._model_name(), frontier=self.frontier,
                visited=self.visited, index=index,
            )
        if task.seeds:
            self.frontier = [u for u in self.frontier if u not in task.seeds]
        return task, emission

    def explore(
        self,
        steps: int | None = 10,
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager: Any = None,
    ) -> list[dict]:
        """Browse on the network's own initiative: it picks each task, the loop judges and teaches it.

        ``steps`` is how many tasks to go through; ``None`` (or 0) runs until
        ``stop_event`` is set, so it can be left running like the GAN loop.
        """
        cfg = self.config
        self._stop = stop_event if stop_event is not None else threading.Event()
        records: list[dict] = []
        index = 0
        while not self._stopped() and (not steps or index < steps):
            index += 1
            t0 = time.perf_counter()
            task, emission = self.propose(index)
            self._emit(progress, {
                "kind": "proposal", "step": index, "task": task.id, "prompt": task.prompt,
                "emission": " ".join(emission.split())[:200], "seeds": list(task.seeds),
                "frontier": len(self.frontier), "visited": len(self.visited),
            })
            if self._stopped():
                break
            record, bad, good = self.run_task(task, "model", index, progress)
            record.update(self.learn(bad, good))
            record.update(
                kind="explore", step=index, frontier=len(self.frontier), visited=len(self.visited),
                seconds=time.perf_counter() - t0,
            )
            self._emit(progress, record)
            records.append(record)
            if checkpoint_manager is not None and cfg.checkpoint_every and index % cfg.checkpoint_every == 0:
                checkpoint_manager.save(self.model, index, "explore", {k: record[k] for k in ("task", "correct", "autonomy")})
        return records


def _last_loss(records: list[dict]) -> float | None:
    return records[-1].get("loss") if records else None
