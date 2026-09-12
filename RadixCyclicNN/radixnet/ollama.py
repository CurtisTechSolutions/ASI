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
from typing import Any

from .llm import LLMError, loads_lenient as _loads_lenient

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL",
    "OllamaClient",
    "OllamaError",
    "adversarial_review",
    "corpus_from_prompt",
    "normalise_url",
    "parse_lines",
    "review_texts",
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
    ) -> str:
        """One chat turn (``POST /api/chat``); ``messages`` are ``{"role", "content"}`` dicts."""
        body: dict[str, Any] = {"model": model or self.model, "messages": list(messages), "stream": False}
        if json_mode:
            body["format"] = "json"
        if options:
            body["options"] = dict(options)
        data = self._request("POST", "/api/chat", body, timeout)
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("unexpected /api/chat response (no message content)")
        return content


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
    ratings = [r["rating"] for r in reviews if r["rating"] is not None]
    passed = [r["text"] for r in reviews if r["verdict"] == "pass"]
    failed = [r["text"] for r in reviews if r["verdict"] != "pass"]
    return {
        "source": source,
        "model": ollama_model or client.model,
        "threshold": float(threshold),
        "texts": samples,
        "reviews": reviews,
        "mean_rating": statistics.fmean(ratings) if ratings else None,
        "pass_rate": len(passed) / len(reviews) if reviews else None,
        "good": passed,
        "bad": failed,
    }
