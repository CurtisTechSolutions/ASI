"""Ollama integration: prompt-driven training corpora and an adversarial LLM review.

Two ways of hooking the network into a local large language model served by
`Ollama <https://ollama.com>`_ (HTTP API, standard library only):

* **Corpus from a prompt** — :func:`corpus_from_prompt` asks the LLM for *N*
  lines of text about a prompt, either correct (``style="good"``) or
  deliberately wrong (``style="garbage"``), which is exactly the pair of
  inputs 2NRL wants (train on garbage, invert, fine-tune on correct data).
* **Adversarial review** — :func:`review_texts` makes the LLM play the harsh
  critic: every text gets a rating from 0 to 10, a pass/fail verdict and a
  one-sentence critique.  :func:`adversarial_review` runs that over the
  network's own generations (or given texts) and partitions them into ``bad``
  (failed) and ``good`` (passed) sets, ready for a 2NRL pass — an external
  discriminator in the spirit of the GAN loop.
* **Letter-level correction** — :func:`correct_texts` makes the LLM play the
  copy editor instead: every text comes back written out correctly with as few
  characters changed as possible, and the *diff* between the two
  (:mod:`radixnet.diff`) says exactly which characters were the mistake.
  ``"Hi howe are you??"`` corrected to ``"Hi, how are you?"`` blames the
  ``e`` and the second ``?`` (and the missing comma), nothing else: that is
  what :func:`radixnet.blame.teach_corrections` hands the negative network
  (:meth:`radixnet.negative.NegativeNet.correct`), while a text the editor
  handed back unchanged clears blame.  :func:`adversarial_correction` runs it
  over the network's own generations (or given texts).

The Ollama endpoint is taken from the ``OLLAMA_HOST`` environment variable
(Ollama's own convention; ``host:port`` without a scheme is accepted), else
``http://127.0.0.1:11434``; the default model from ``RADIXNET_OLLAMA_MODEL``,
else ``llama3.2``.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from .llm import LLMError, loads_lenient as _loads_lenient

__all__ = [
    "CORRECTION_VERDICTS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL",
    "OllamaClient",
    "OllamaError",
    "adversarial_correction",
    "adversarial_review",
    "chat_line",
    "correct_texts",
    "corpus_from_prompt",
    "normalise_url",
    "parse_lines",
    "review_conversation",
    "review_texts",
    "sample_texts",
    "summarise_corrections",
    "summarise_reviews",
]

DEFAULT_TIMEOUT = 120.0
"""Seconds to wait for one Ollama answer (local models can be slow)."""


def normalise_url(url: str | None) -> str:
    """``host:port`` or a full URL -> ``http://host:port`` without a trailing slash."""
    text = (url or "").strip().rstrip("/")
    if not text:
        raise ValueError("the Ollama URL is empty")
    if "://" not in text:
        text = "http://" + text
    return text


def _default_url() -> str:
    return normalise_url(os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434")


DEFAULT_URL = _default_url()
DEFAULT_MODEL = os.environ.get("RADIXNET_OLLAMA_MODEL", "").strip() or "llama3.2"


class OllamaError(LLMError):
    """Ollama is unreachable, answered an error, or returned something unusable."""


class OllamaClient:
    """Minimal client for Ollama's HTTP API (``/api/tags``, ``/api/generate``, ``/api/chat``)."""

    provider = "ollama"

    __slots__ = ("url", "model", "timeout")

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float | None = None) -> None:
        self.url = normalise_url(url or DEFAULT_URL)
        self.model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.timeout = float(timeout) if timeout else DEFAULT_TIMEOUT

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.url + path, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a local server: never via a proxy
        try:
            with opener.open(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500].strip()
            try:
                detail = str(json.loads(detail).get("error", detail))
            except (ValueError, AttributeError):
                pass
            raise OllamaError(f"Ollama {method} {path} failed with HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or exc
            raise OllamaError(f"cannot reach Ollama at {self.url}: {reason}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise OllamaError(f"Ollama returned invalid JSON for {path}: {exc}") from exc

    # -- API -----------------------------------------------------------------

    def models(self) -> list[dict]:
        """Installed models (``GET /api/tags``)."""
        data = self._request("GET", "/api/tags")
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise OllamaError("unexpected /api/tags response (no 'models' list)")
        return [m for m in models if isinstance(m, dict)]

    def available(self) -> bool:
        try:
            self.models()
        except OllamaError:
            return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """One completion (``POST /api/generate``, non-streaming); ``json_mode`` asks for a JSON answer."""
        body: dict[str, Any] = {"model": model or self.model, "prompt": prompt, "stream": False}
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"
        if options:
            body["options"] = dict(options)
        data = self._request("POST", "/api/generate", body, timeout)
        text = data.get("response") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise OllamaError("unexpected /api/generate response (no 'response' text)")
        return text

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        timeout: float | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """One chat turn (``POST /api/chat``); ``messages`` are ``{"role", "content"}`` dicts."""
        message = self.chat_message(
            messages, model=model, json_mode=json_mode, options=options, timeout=timeout, tools=tools
        )
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("unexpected /api/chat response (no message content)")
        return content

    def chat_message(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        timeout: float | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """The whole assistant message of one chat turn.

        ``tools`` are JSON-schema function definitions (Ollama's own ``tools``
        format, see :meth:`radixnet.tools.ToolBox.schemas`); a model that
        supports tool calling answers with ``{"tool_calls": [...]}`` beside (or
        instead of) ``content``.  The message is returned as it came, with
        ``content`` guaranteed to be a string.
        """
        body: dict[str, Any] = {"model": model or self.model, "messages": list(messages), "stream": False}
        if json_mode:
            body["format"] = "json"
        if options:
            body["options"] = dict(options)
        if tools:
            body["tools"] = list(tools)
        data = self._request("POST", "/api/chat", body, timeout)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise OllamaError("unexpected /api/chat response (no message)")
        if not isinstance(message.get("content"), str):
            message["content"] = ""
        return message


# ---------------------------------------------------------------------------
# corpus from a prompt
# ---------------------------------------------------------------------------

_CORPUS_SYSTEM = (
    "You produce training data for a small character-level language model. Answer with exactly {n} lines "
    "and nothing else: one short, self-contained sentence per line, plain text, no numbering, no bullets, "
    "no quotes, no blank lines, no headings and no commentary. "
)
_STYLE_GOOD = "Every line must be correct, natural and factual, in the style and about the topic requested."
_STYLE_GARBAGE = (
    "Every line must be deliberately WRONG in a way a careful reader would reject: false facts, scrambled "
    "word order, broken grammar, nonsense words, contradictions. Keep the requested topic and vocabulary so "
    "the errors are the only difference from good text."
)
_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*")

STYLES = ("good", "garbage")


def parse_lines(text: str, limit: int | None = None) -> list[str]:
    """LLM output -> clean list of lines: numbering / bullets / quotes stripped, blanks, fences and duplicates dropped."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        line = _LINE_PREFIX.sub("", line).strip().strip("\"'`“”").strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if limit is not None and len(lines) >= limit:
            break
    return lines


def corpus_from_prompt(
    client: OllamaClient, prompt: str, lines: int = 20, style: str = "good", model: str | None = None
) -> list[str]:
    """Ask the LLM for ``lines`` lines of text about ``prompt``; ``style`` is ``"good"`` or ``"garbage"``."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if lines < 1:
        raise ValueError("lines must be >= 1")
    style = (style or "good").strip().lower()
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)} (got {style!r})")
    system = _CORPUS_SYSTEM.format(n=lines) + (_STYLE_GOOD if style == "good" else _STYLE_GARBAGE)
    user = f"Topic / instructions: {prompt.strip()}\n\nWrite the {lines} lines now."
    text = client.generate(user, system=system, model=model, options={"temperature": 0.9 if style == "good" else 1.1})
    return parse_lines(text, limit=lines)


# ---------------------------------------------------------------------------
# adversarial review
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = (
    "You are an adversarial reviewer of text produced by a small experimental language model. Assume every "
    "text is flawed and hunt for the flaws: gibberish, broken grammar, wrong word order, truncated or repeated "
    "fragments, contradictions, false statements, incoherence. Rate each text from 0 to 10: 10 = "
    "indistinguishable from correct, fluent, factual human writing; 6 = acceptable with minor flaws; 0 = pure "
    "gibberish. The verdict is \"pass\" for a rating of {threshold:g} or more, otherwise \"fail\". Reply with "
    "JSON only, no prose, exactly of the form {{\"reviews\": [{{\"index\": <int>, \"rating\": <number>, "
    "\"verdict\": \"pass\" or \"fail\", \"critique\": \"<one sentence naming the worst flaw, or 'no flaw "
    "found'>\"}}, ...]}} with one entry per text, in the given order and with the given index."
)


def _parse_reviews(raw: str, count: int) -> dict[int, dict]:
    """``{index: {"rating", "critique"}}`` for the entries that could be understood."""
    data = _loads_lenient(raw)
    items: Any = None
    if isinstance(data, dict):
        items = data.get("reviews")
        if items is None:
            items = data.get("results", data.get("items"))
        if items is None and ("rating" in data or "score" in data):
            items = [data]
    elif isinstance(data, list):
        items = data
    parsed: dict[int, dict] = {}
    if not isinstance(items, list):
        return parsed
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        index = item.get("index", position)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = position
        rating = item.get("rating", item.get("score"))
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            continue
        if rating != rating:  # NaN
            continue
        rating = max(0.0, min(10.0, rating))
        critique = str(item.get("critique") or item.get("reason") or item.get("comment") or "").strip()
        if 0 <= index < count and index not in parsed:
            parsed[index] = {"rating": rating, "critique": critique or "no critique given"}
    return parsed


def review_texts(
    client: OllamaClient,
    texts: list[str],
    *,
    context: str | None = None,
    model: str | None = None,
    threshold: float = 6.0,
    batch: int = 20,
) -> list[dict]:
    """Adversarial ratings for ``texts`` (input order): ``{index, text, rating, verdict, critique}`` each.

    ``rating`` is ``None`` with verdict ``"unrated"`` when the LLM answer could
    not be understood for that text.  Blank texts are failed without asking.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    results: list[dict] = []
    system = _REVIEW_SYSTEM.format(threshold=threshold)
    for start in range(0, len(texts), batch):
        chunk = [str(t) for t in texts[start : start + batch]]
        asked = [(i, t) for i, t in enumerate(chunk) if t.strip()]
        parsed: dict[int, dict] = {}
        if asked:
            numbered = "\n".join(f"[{i}] {t}" for i, t in asked)
            user = (
                (f"Context: {context.strip()}\n\n" if context and context.strip() else "")
                + f"Review these {len(asked)} texts:\n{numbered}\n\nReturn the JSON now."
            )
            raw = client.generate(user, system=system, model=model, json_mode=True, options={"temperature": 0.2})
            parsed = _parse_reviews(raw, len(chunk))
        for i, text in enumerate(chunk):
            entry = {"index": start + i, "text": text}
            if not text.strip():
                entry.update(rating=0.0, verdict="fail", critique="empty output")
            elif i in parsed:
                rating = parsed[i]["rating"]
                entry.update(rating=rating, verdict="pass" if rating >= threshold else "fail", critique=parsed[i]["critique"])
            else:
                entry.update(rating=None, verdict="unrated", critique="no review returned")
            results.append(entry)
    return results


def sample_texts(
    model: Any, count: int, prefix: str = "", max_length: int = 60, temperature: float = 1.0, seed: int | None = None
) -> list[str]:
    """``count`` stochastic texts from a :class:`~radixnet.model.RadixNet` (continuations of ``prefix`` when given)."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if prefix:
        return [
            model.predict(prefix, length=max_length, mode="sample", temperature=temperature, max_length=max_length).full_text
            for _ in range(count)
        ]
    return [r.text for r in model.generate(max_length=max_length, mode="sample", temperature=temperature, count=count, seed=seed)]


def adversarial_review(
    model: Any,
    client: OllamaClient,
    *,
    count: int = 8,
    prefix: str = "",
    max_length: int = 60,
    temperature: float = 1.0,
    texts: list[str] | None = None,
    threshold: float = 6.0,
    context: str | None = None,
    ollama_model: str | None = None,
    seed: int | None = None,
) -> dict:
    """Let the LLM judge the network's own output (or ``texts``) and split it into ``good`` / ``bad`` sets.

    Returns ``{"source", "model", "threshold", "texts", "reviews", "mean_rating",
    "pass_rate", "good", "bad"}``; ``bad`` holds failed and unrated texts.
    """
    if texts is None:
        if model is None:
            raise ValueError("either a model to sample from or texts to review is required")
        samples = sample_texts(model, count, prefix=prefix, max_length=max_length, temperature=temperature, seed=seed)
        source = "model"
    else:
        samples = [str(t) for t in texts]
        source = "given"
    reviews = review_texts(client, samples, context=context, model=ollama_model, threshold=threshold)
    return summarise_reviews(source, ollama_model or client.model, threshold, samples, reviews)


def summarise_reviews(source: str, model: str, threshold: float, texts: list[str], reviews: list[dict]) -> dict:
    """Split a marked set into what passed and what did not, and average the marks.

    It is separate from :func:`adversarial_review` because a caller holding a
    model lock has to sample and review in two steps: the sampling walks the
    graph and must hold the lock, the reviewing is a network call and must not.
    """
    ratings = [r["rating"] for r in reviews if r["rating"] is not None]
    passed = [r["text"] for r in reviews if r["verdict"] == "pass"]
    failed = [r["text"] for r in reviews if r["verdict"] != "pass"]
    return {
        "source": source,
        "model": model,
        "threshold": float(threshold),
        "texts": list(texts),
        "reviews": reviews,
        "mean_rating": statistics.fmean(ratings) if ratings else None,
        "pass_rate": len(passed) / len(reviews) if reviews else None,
        "good": passed,
        "bad": failed,
    }

# ---------------------------------------------------------------------------
# letter-level correction
# ---------------------------------------------------------------------------

_CORRECT_SYSTEM = (
    "You are a meticulous copy editor correcting short texts written by a small experimental character-level "
    "language model. For each text write the corrected text: the same text in correct, natural English with the "
    "SMALLEST possible change. Keep every character that is already right, keep the wording, the meaning and the "
    "length as they are, and change only what is actually wrong: a misspelt letter, a missing or doubled "
    "punctuation mark, a wrong ending, a missing word. Never rewrite, never add commentary, never quote. If a text "
    "is already correct, return it exactly as it is. Name the kind of mistake with one word from this list: "
    "{reasons}; use \"none\" for a text you did not change. Reply with JSON only, no prose, exactly of the form "
    "{{\"corrections\": [{{\"index\": <int>, \"correction\": \"<the corrected text>\", \"reason\": "
    "\"<one word from the list>\", \"note\": \"<one short sentence saying what was wrong, or 'nothing'>\"}}, "
    "...]}} with one entry per text, in the given order and with the given index."
)

CORRECTION_VERDICTS = ("corrected", "unchanged", "uncorrected")
"""What the editor did with a text: changed it, handed it back as it was, or could not be understood about it."""


def _parse_corrections(raw: str, count: int) -> dict[int, dict]:
    """``{index: {"correction", "reason", "note"}}`` for the entries that could be understood."""
    data = _loads_lenient(raw)
    items: Any = None
    if isinstance(data, dict):
        items = data.get("corrections")
        if items is None:
            items = data.get("results", data.get("items"))
        if items is None and ("correction" in data or "corrected" in data):
            items = [data]
    elif isinstance(data, list):
        items = data
    parsed: dict[int, dict] = {}
    if not isinstance(items, list):
        return parsed
    for position, item in enumerate(items):
        if isinstance(item, str):  # a bare list of corrected lines
            item = {"correction": item}
        if not isinstance(item, dict):
            continue
        index = item.get("index", position)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = position
        correction = item.get("correction", item.get("corrected", item.get("text")))
        if not isinstance(correction, str):
            continue
        correction = correction.strip().strip("\"'`\u201c\u201d").strip()
        reason = str(item.get("reason") or item.get("error") or "").strip().lower()
        note = " ".join(str(item.get("note") or item.get("critique") or item.get("comment") or "").split())
        if 0 <= index < count and index not in parsed:
            parsed[index] = {"correction": correction, "reason": reason, "note": note}
    return parsed


def _correction_entry(index: int, text: str, correction: str | None, reason: str, note: str) -> dict:
    """One :func:`correct_texts` result: the diff against the correction, and what the editor said."""
    from . import blame as blame_module
    from . import diff

    entry: dict[str, Any] = {"index": index, "text": text}
    if correction is None:
        entry.update(
            correction=None, verdict="uncorrected", reason="", note=note or "no correction returned",
            changes=[], edits=0, wrong_chars=0, right_chars=0,
        )
        return entry
    changes = [
        {"op": e.op, "wrong": e.wrong, "right": e.right, "at": [e.a0, e.a1], "to": [e.b0, e.b1]}
        for e in diff.edits(text, correction) if e.op != "equal"
    ]
    changed = correction != text
    entry.update(
        correction=correction,
        verdict="corrected" if changed else "unchanged",
        reason=blame_module.correction_reason(reason, note, changes) if changed else "none",
        note=note or ("nothing" if not changed else "corrected"),
        changes=changes,
        edits=len(changes),
        wrong_chars=sum(e["at"][1] - e["at"][0] for e in changes),
        right_chars=sum(e["to"][1] - e["to"][0] for e in changes),
    )
    return entry


def correct_texts(
    client: OllamaClient,
    texts: list[str],
    *,
    context: str | None = None,
    model: str | None = None,
    batch: int = 20,
) -> list[dict]:
    """Letter-level corrections of ``texts`` (input order): ``{index, text, correction, verdict, reason, note,
    changes, edits, wrong_chars, right_chars}`` each.

    ``verdict`` is ``"corrected"`` when the editor changed something,
    ``"unchanged"`` when it handed the text back as it was (it is correct, so
    it clears blame), and ``"uncorrected"`` when its answer could not be
    understood for that text (``correction`` is ``None``: nothing is known
    about it, so it neither blames nor clears).  ``changes`` are the edits of
    the alignment (:func:`radixnet.diff.edits`), ``{"op", "wrong", "right",
    "at": [a0, a1], "to": [b0, b1]}``, with the equal runs left out; ``reason``
    is the editor's word for the mistake, mapped onto
    :data:`radixnet.blame.CORRECTION_REASONS` (the shape of the diff decides
    when it gave none).  Blank texts are ``uncorrected`` without asking.
    """
    from .blame import CORRECTION_REASONS

    if batch < 1:
        raise ValueError("batch must be >= 1")
    results: list[dict] = []
    system = _CORRECT_SYSTEM.format(reasons=", ".join(r for r in CORRECTION_REASONS if r != "none"))
    for start in range(0, len(texts), batch):
        chunk = [str(t) for t in texts[start : start + batch]]
        asked = [(i, t) for i, t in enumerate(chunk) if t.strip()]
        parsed: dict[int, dict] = {}
        if asked:
            numbered = "\n".join(f"[{i}] {t}" for i, t in asked)
            user = (
                (f"Context: {context.strip()}\n\n" if context and context.strip() else "")
                + f"Correct these {len(asked)} texts:\n{numbered}\n\nReturn the JSON now."
            )
            raw = client.generate(user, system=system, model=model, json_mode=True, options={"temperature": 0.0})
            parsed = _parse_corrections(raw, len(chunk))
        for i, text in enumerate(chunk):
            if not text.strip():
                results.append(_correction_entry(start + i, text, None, "", "empty output"))
            elif i in parsed:
                item = parsed[i]
                results.append(_correction_entry(start + i, text, item["correction"], item["reason"], item["note"]))
            else:
                results.append(_correction_entry(start + i, text, None, "", ""))
    return results


def adversarial_correction(
    model: Any,
    client: OllamaClient,
    *,
    count: int = 8,
    prefix: str = "",
    max_length: int = 60,
    temperature: float = 1.0,
    texts: list[str] | None = None,
    context: str | None = None,
    ollama_model: str | None = None,
    seed: int | None = None,
) -> dict:
    """Let the LLM copy-edit the network's own output (or ``texts``), letter by letter.

    Returns the :func:`summarise_corrections` shape: ``{"source", "model",
    "texts", "corrections", "corrected", "unchanged", "uncorrected", "edits",
    "wrong_chars", "right_chars", "change_rate"}``.
    """
    if texts is None:
        if model is None:
            raise ValueError("either a model to sample from or texts to correct is required")
        samples = sample_texts(model, count, prefix=prefix, max_length=max_length, temperature=temperature, seed=seed)
        source = "model"
    else:
        samples = [str(t) for t in texts]
        source = "given"
    corrections = correct_texts(client, samples, context=context, model=ollama_model)
    return summarise_corrections(source, ollama_model or client.model, samples, corrections)


def summarise_corrections(source: str, model: str, texts: list[str], corrections: list[dict]) -> dict:
    """Split a copy-edited set into what was changed, what was right as it was, and what got no answer.

    ``change_rate`` is the share of the answered texts the editor changed
    (``None`` when it answered none): the copy editor's counterpart of the
    reviewer's pass rate, falling as the model's writing improves.  Separate
    from :func:`adversarial_correction` for the same reason
    :func:`summarise_reviews` is: a server samples under its model lock and
    corrects outside it.
    """
    corrected = [c["text"] for c in corrections if c["verdict"] == "corrected"]
    unchanged = [c["text"] for c in corrections if c["verdict"] == "unchanged"]
    uncorrected = [c["text"] for c in corrections if c["verdict"] == "uncorrected"]
    answered = len(corrected) + len(unchanged)
    return {
        "source": source,
        "model": model,
        "texts": list(texts),
        "corrections": corrections,
        "corrected": corrected,
        "unchanged": unchanged,
        "uncorrected": uncorrected,
        "edits": sum(int(c.get("edits") or 0) for c in corrections),
        "wrong_chars": sum(int(c.get("wrong_chars") or 0) for c in corrections),
        "right_chars": sum(int(c.get("right_chars") or 0) for c in corrections),
        "change_rate": len(corrected) / answered if answered else None,
    }


# ---------------------------------------------------------------------------
# conversing with the network, and marking the conversation
# ---------------------------------------------------------------------------

_CHAT_SYSTEM = (
    "You are having a short, ordinary conversation with a very small character-level neural network that is "
    "learning to talk. It answers by continuing the last few words you wrote, so every line you write must be "
    "short, plain and concrete, and must end on words that are easy to carry on from. Never mention that it is a "
    "model, never explain yourself, never ask more than one thing at a time, and never write more than one "
    "sentence. Reply with the next thing you would say and nothing else."
)

_CONVERSATION_SYSTEM = (
    "You are marking a conversation between a person and a very small character-level neural network that is "
    "learning to talk. Mark each of the network's lines out of 10 for one thing only: is it a real reply to the "
    "line before it - does it follow on, is it about the same thing, is it a sentence at all. Ignore style, "
    "length and ambition: a short plain line that follows on is a 10. A line that merely repeats what was just "
    "said, that is gibberish, or that answers something nobody asked is 0 to 3. {threshold:g} out of 10 is a "
    "pass. Reply with JSON only, of the form {{\"reviews\": [{{\"index\": <n>, \"rating\": <0-10>, "
    "\"critique\": \"<one sentence>\"}}, ...], \"overall\": {{\"rating\": <0-10>, \"critique\": "
    "\"<one sentence about the conversation as a whole>\"}}}}."
)


def chat_line(
    client: OllamaClient,
    transcript: Sequence[tuple[str, str]] = (),
    *,
    topic: str = "",
    persona: str = "",
    speakers: Sequence[str] = ("Partner", "Model"),
    model: str | None = None,
    temperature: float = 0.8,
) -> str:
    """The next line of the LLM's side of a conversation with the network.

    ``transcript`` is what has been said so far as ``(speaker, text)`` pairs;
    empty opens the conversation.  The system prompt asks for short, plain
    lines ending on words that are easy to carry on from, because that is what
    a character-level model can actually reply to.
    """
    system = _CHAT_SYSTEM
    if topic.strip():
        system += f" The conversation is about {topic.strip()}."
    if persona.strip():
        system += f" You are {persona.strip()}."
    said = "\n".join(f"{speaker}: {text}" for speaker, text in transcript if str(text).strip())
    if said:
        user = f"The conversation so far:\n{said}\n\nWrite your next line."
    else:
        opener = f" about {topic.strip()}" if topic.strip() else ""
        user = f"Open the conversation{opener} with one short, plain line."
    raw = client.generate(user, system=system, model=model, options={"temperature": temperature})
    return _first_line(raw)


def _first_line(raw: str) -> str:
    """One line out of an LLM answer that may have written several (or quoted itself)."""
    for line in str(raw or "").splitlines():
        text = " ".join(line.split()).strip()
        if not text:
            continue
        for speaker in ("Partner:", "Model:", "You:", "Me:"):
            if text.lower().startswith(speaker.lower()):
                text = text[len(speaker):].strip()
        text = text.strip('"\u201c\u201d')
        if text:
            return text
    return ""


def review_conversation(
    client: OllamaClient,
    exchanges: Sequence[tuple[str, str]],
    *,
    topic: str = "",
    model: str | None = None,
    threshold: float = 6.0,
    speakers: Sequence[str] = ("Partner", "Model"),
) -> dict:
    """Mark every reply the network gave in a conversation, and the conversation as a whole.

    ``exchanges`` are ``(what was said to it, what it replied)`` pairs in
    order.  The result is the :func:`summarise_reviews` shape - so
    :func:`radixnet.blame.teach_reviews` takes it as it is - plus ``overall``,
    the judge's verdict on the conversation itself (``None`` when it did not
    give one).
    """
    pairs = [(str(said), str(reply)) for said, reply in exchanges]
    replies = [reply for _said, reply in pairs]
    asked = [(i, said, reply) for i, (said, reply) in enumerate(pairs) if reply.strip()]
    parsed: dict[int, dict] = {}
    overall: dict | None = None
    if asked:
        numbered = "\n\n".join(
            f"[{i}] {speakers[0]}: {said}\n    {speakers[1]}: {reply}" for i, said, reply in asked
        )
        user = (
            (f"Topic: {topic.strip()}\n\n" if topic.strip() else "")
            + f"The conversation:\n{numbered}\n\nReturn the JSON now."
        )
        raw = client.generate(
            user, system=_CONVERSATION_SYSTEM.format(threshold=threshold), model=model, json_mode=True,
            options={"temperature": 0.2},
        )
        parsed = _parse_reviews(raw, len(pairs))
        overall = _parse_overall(raw)
    reviews: list[dict] = []
    for i, (said, reply) in enumerate(pairs):
        entry: dict[str, Any] = {"index": i, "text": reply, "said": said}
        if not reply.strip():
            entry.update(rating=0.0, verdict="fail", critique="it said nothing")
        elif i in parsed:
            rating = parsed[i]["rating"]
            entry.update(
                rating=rating, verdict="pass" if rating >= threshold else "fail", critique=parsed[i]["critique"],
            )
        else:
            entry.update(rating=None, verdict="unrated", critique="no review returned")
        reviews.append(entry)
    summary = summarise_reviews("chat", model or client.model, threshold, replies, reviews)
    summary["overall"] = overall
    return summary


def _parse_overall(raw: str) -> dict | None:
    """The judge's verdict on the conversation as a whole, when it gave one."""
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        return None
    item = data.get("overall") or data.get("conversation") or data.get("summary")
    if not isinstance(item, dict):
        return None
    rating = item.get("rating", item.get("score"))
    try:
        rating = max(0.0, min(10.0, float(rating)))
    except (TypeError, ValueError):
        rating = None
    critique = str(item.get("critique") or item.get("reason") or item.get("comment") or "").strip()
    if rating is None and not critique:
        return None
    return {"rating": rating, "critique": critique or "no critique given"}
