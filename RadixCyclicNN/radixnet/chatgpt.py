"""ChatGPT (OpenAI) as a tutor: the same client interface as Ollama, over the hosted API.

:class:`ChatGPTClient` speaks OpenAI's chat-completions API (HTTP, standard
library only) and offers exactly what :class:`~radixnet.ollama.OllamaClient`
offers — :meth:`~ChatGPTClient.models`, :meth:`~ChatGPTClient.available`,
:meth:`~ChatGPTClient.generate`, :meth:`~ChatGPTClient.chat` — so anything that
takes a client (the code-generation tutor and its judge above all) can be
pointed at ChatGPT instead of a local model without further changes.

Configuration comes from the environment, the way the OpenAI tools expect it:

* ``OPENAI_API_KEY`` — the key (or ``OPENAI_API_KEY_FILE``, a file holding it,
  for Docker / Compose secrets).  There is no default and no way to pass a key
  through the HTTP API: the server uses the key of the process it runs in.
* ``OPENAI_BASE_URL`` — the endpoint, ``https://api.openai.com/v1`` by default;
  point it at any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, a
  gateway) to use that instead.
* ``RADIXNET_OPENAI_MODEL`` — the default model, else ``gpt-4o-mini``.
* ``OPENAI_ORG_ID`` / ``OPENAI_PROJECT_ID`` — optional account headers.

Ollama's ``options`` names are translated (``num_predict`` ->
``max_completion_tokens``, and so on), and a model that rejects an optional
field — the reasoning models refuse ``temperature``, older ones refuse
``response_format`` — is retried without it instead of failing, so the tutor
works across model families.

Unlike the local provider this sends every prompt (problem statements, the
network's own programs) to OpenAI and costs money per call.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .llm import LLMError

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL",
    "ChatGPTClient",
    "ChatGPTError",
    "api_key",
    "api_key_configured",
    "normalise_url",
]

DEFAULT_TIMEOUT = 120.0
"""Seconds to wait for one ChatGPT answer."""

OPENAI_URL = "https://api.openai.com/v1"
"""The hosted API; ``OPENAI_BASE_URL`` overrides it (any OpenAI-compatible server)."""

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0", "host.docker.internal")


def _is_local(host: str) -> bool:
    """True for an OpenAI-compatible server on this machine (llama.cpp, vLLM, LM Studio, a test double)."""
    host = (host or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".local")


def normalise_url(url: str | None) -> str:
    """``host:port`` or a full URL -> a base URL without a trailing slash (``/v1`` added when there is no path).

    Refuses to carry a key over plain ``http://`` to anything but a local
    server unless ``RADIXNET_OPENAI_ALLOW_INSECURE`` is set.
    """
    text = (url or "").strip()
    if not text:
        raise ValueError("the OpenAI base URL is empty")
    if "://" not in text:
        text = "https://" + text
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"the OpenAI base URL must be http:// or https:// (got {url!r})")
    if not parts.netloc:
        raise ValueError(f"the OpenAI base URL has no host (got {url!r})")
    host = (parts.hostname or "").lower()
    if parts.scheme == "http" and not _is_local(host) and not _flag("RADIXNET_OPENAI_ALLOW_INSECURE"):
        raise ValueError(
            f"refusing to send the API key unencrypted to {host!r}: use https://, a local server, "
            "or set RADIXNET_OPENAI_ALLOW_INSECURE=1"
        )
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/') or '/v1'}"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_url() -> str:
    configured = os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("OPENAI_API_BASE", "").strip()
    try:
        return normalise_url(configured) if configured else OPENAI_URL
    except ValueError:  # a broken environment must not stop the package from importing
        return OPENAI_URL


DEFAULT_URL = _default_url()
DEFAULT_MODEL = os.environ.get("RADIXNET_OPENAI_MODEL", "").strip() or "gpt-4o-mini"


def api_key(explicit: str | None = None) -> str | None:
    """The API key: ``explicit``, else ``$OPENAI_API_KEY``, else the content of ``$OPENAI_API_KEY_FILE``."""
    if explicit and explicit.strip():
        return explicit.strip()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("OPENAI_API_KEY_FILE", "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip() or None
        except OSError:
            return None
    return None


def api_key_configured(explicit: str | None = None) -> bool:
    """True when a key is available (its value is never returned or logged)."""
    return bool(api_key(explicit))


class ChatGPTError(LLMError):
    """The OpenAI API is unreachable, refused the request, or answered something unusable."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.param = param


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

_OPTION_NAMES = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "num_predict": "max_completion_tokens",  # Ollama's name for the answer length
    "max_tokens": "max_completion_tokens",
    "max_completion_tokens": "max_completion_tokens",
}

_OPTIONAL_FIELDS = (
    "temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty",
    "max_completion_tokens", "response_format",
)
"""Body fields worth dropping and retrying when a model rejects them."""

_REJECTED = re.compile(r"unsupported|unrecognized|not supported|does not support|unknown parameter", re.IGNORECASE)
_QUOTED_NAME = re.compile(r"['\"`]([A-Za-z_][A-Za-z0-9_]*)['\"`]")
_ARGUMENT_NAME = re.compile(r"argument(?: supplied)?:\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _options_body(options: dict | None) -> dict:
    """Ollama-style ``options`` -> chat-completions fields (unknown names are dropped)."""
    body: dict[str, Any] = {}
    for name, value in (options or {}).items():
        field = _OPTION_NAMES.get(name)
        if field is None or value is None:
            continue
        if field == "max_completion_tokens":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:  # Ollama's "no limit"
                continue
        body[field] = value
    return body


def _rejected_field(error: ChatGPTError, body: dict) -> str | None:
    """The optional body field an error blames, when dropping it is worth one retry."""
    if error.status != 400 or not _REJECTED.search(str(error)):
        return None
    message = str(error)
    names = [error.param] if isinstance(error.param, str) else []
    names += _ARGUMENT_NAME.findall(message)
    names += _QUOTED_NAME.findall(message)
    for name in names:
        if name in _OPTIONAL_FIELDS and name in body:
            return name
    return None


def _answer(data: Any) -> str:
    """The assistant text of a chat-completions response."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ChatGPTError("unexpected chat completion (no 'choices')")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):  # content parts instead of one string
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(content, str) and content.strip():
        return content
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise ChatGPTError(f"the model refused to answer: {refusal.strip()}")
    raise ChatGPTError(f"the model returned an empty answer (finish_reason={choice.get('finish_reason')!r})")


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


class ChatGPTClient:
    """Minimal client for OpenAI's API (``GET /models``, ``POST /chat/completions``).

    ``url``, ``model`` and ``timeout`` fall back to the module defaults; the
    key comes from ``api_key`` or the environment and is read at request time,
    never stored in a report, a job record or a log line.
    """

    provider = "chatgpt"

    __slots__ = ("url", "model", "timeout", "_api_key", "_local", "organization", "project")

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self.url = normalise_url(url or DEFAULT_URL)
        self._local = _is_local(urllib.parse.urlsplit(self.url).hostname or "")
        self.model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.timeout = float(timeout) if timeout else DEFAULT_TIMEOUT
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.organization = os.environ.get("OPENAI_ORG_ID", "").strip() or None
        self.project = os.environ.get("OPENAI_PROJECT_ID", "").strip() or None

    def __repr__(self) -> str:  # never shows the key
        return f"ChatGPTClient(url={self.url!r}, model={self.model!r})"

    @property
    def configured(self) -> bool:
        """True when a key is available for this client."""
        return api_key_configured(self._api_key)

    # -- transport -----------------------------------------------------------

    def _key(self) -> str:
        key = api_key(self._api_key)
        if not key:
            raise ChatGPTError(
                "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) in the environment of the "
                "process that talks to ChatGPT"
            )
        return key

    def _headers(self, body: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._key()}", "Accept": "application/json"}
        if body:
            headers["Content-Type"] = "application/json"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    def _http_error(self, exc: urllib.error.HTTPError, path: str) -> ChatGPTError:
        raw = exc.read().decode("utf-8", errors="replace")[:1000].strip()
        detail, code, param = raw, None, None
        try:
            error = json.loads(raw).get("error")
        except (ValueError, AttributeError):
            error = None
        if isinstance(error, dict):
            detail = str(error.get("message") or raw)
            code = error.get("code") if isinstance(error.get("code"), str) else None
            param = error.get("param") if isinstance(error.get("param"), str) else None
        elif isinstance(error, str):
            detail = error
        hint = {
            401: " (check OPENAI_API_KEY)",
            403: " (the key may not be allowed to use this model)",
            404: f" (does the key have access to model {self.model!r}?)",
            429: " (rate limit or exhausted quota)",
        }.get(exc.code, "")
        return ChatGPTError(
            f"OpenAI {path} failed with HTTP {exc.code}: {detail or exc.reason}{hint}",
            status=exc.code, code=code, param=param,
        )

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=self._headers(data is not None))
        # The hosted API goes through the environment's proxy; a server on this machine never does.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if self._local else urllib.request.build_opener()
        try:
            with opener.open(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, f"{method} {path}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or exc
            raise ChatGPTError(f"cannot reach the OpenAI API at {self.url}: {reason}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ChatGPTError(f"the OpenAI API returned invalid JSON for {path}: {exc}") from exc

    # -- API -----------------------------------------------------------------

    def models(self) -> list[dict]:
        """Models the key can use (``GET /models``), as ``{"name", "id", "owned_by", "created"}`` sorted by name."""
        data = self._request("GET", "/models")
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ChatGPTError("unexpected /models response (no 'data' list)")
        models = [
            {"name": str(m.get("id")), "id": str(m.get("id")), "owned_by": m.get("owned_by"), "created": m.get("created")}
            for m in items
            if isinstance(m, dict) and m.get("id")
        ]
        return sorted(models, key=lambda m: m["name"])

    def available(self) -> bool:
        """True when a key is configured and the API answers (never raises)."""
        try:
            self.models()
        except LLMError:
            return False
        return True

    def _complete(
        self, messages: list[dict], model: str | None, json_mode: bool, options: dict | None, timeout: float | None
    ) -> str:
        body: dict[str, Any] = {"model": model or self.model, "messages": messages}
        body.update(_options_body(options))
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        while True:
            try:
                return _answer(self._request("POST", "/chat/completions", body, timeout))
            except ChatGPTError as exc:
                rejected = _rejected_field(exc, body)
                if rejected is None:
                    raise
                body.pop(rejected)  # this model does not take that field: ask again without it

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
        """One completion (``POST /chat/completions`` with a single user turn); ``json_mode`` asks for JSON."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, model, json_mode, options, timeout)

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """One chat turn; ``messages`` are ``{"role", "content"}`` dicts."""
        return self._complete([dict(m) for m in messages], model, json_mode, options, timeout)
