"""Provider-independent plumbing for the large language models the network talks to.

Everything that asks an LLM a question — the code-generation tutor and its
judge, the corpus and review helpers — needs the same three things: a list of
models, one completion, one chat turn.  Two providers offer them:

* ``ollama`` — a model served locally by `Ollama <https://ollama.com>`_
  (:mod:`radixnet.ollama`): no key, no cost, nothing leaves the machine.
* ``chatgpt`` — OpenAI's hosted ChatGPT models (:mod:`radixnet.chatgpt`):
  much stronger tutoring, but it needs ``OPENAI_API_KEY`` and every prompt
  (problem statements and generated programs included) is sent to OpenAI.

:class:`LLMClient` is the shape both clients share, :func:`make_client` builds
one from a provider name, and :class:`LLMError` is the failure both raise, so
``except LLMError`` means "the LLM did not work" whichever provider is in use.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "LLMClient",
    "LLMError",
    "default_model",
    "default_url",
    "loads_lenient",
    "make_client",
    "normalise_provider",
    "provider_of",
]

PROVIDERS = ("ollama", "chatgpt")
"""Providers that can answer as the tutor, the judge or the reviewer."""

DEFAULT_PROVIDER = "ollama"
"""The local provider: chosen when nothing else is asked for."""

_ALIASES = {"openai": "chatgpt", "gpt": "chatgpt", "chat-gpt": "chatgpt", "chat_gpt": "chatgpt", "local": "ollama"}


class LLMError(Exception):
    """An LLM is unreachable, refused the request, or answered something unusable."""


class LLMClient(Protocol):
    """What the rest of the package uses a provider client for.

    :class:`~radixnet.ollama.OllamaClient` and
    :class:`~radixnet.chatgpt.ChatGPTClient` both implement it, which is why
    the tutor, the judge and the reviewer never look at which one they got.
    """

    provider: str
    url: str
    model: str

    def models(self) -> list[dict]:
        """Models this provider can answer with, ``{"name", ...}`` each."""

    def available(self) -> bool:
        """True when the provider answers right now (never raises)."""

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
        """One completion of ``prompt`` (with an optional ``system`` instruction)."""

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """One chat turn over ``{"role", "content"}`` messages."""


def normalise_provider(name: str | None) -> str:
    """``"OpenAI"`` / ``"gpt"`` -> ``"chatgpt"``, ``""`` -> the default provider; unknown names raise."""
    text = (name or "").strip().lower().replace(" ", "")
    if not text:
        return DEFAULT_PROVIDER
    text = _ALIASES.get(text, text)
    if text not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)} (got {name!r})")
    return text


def default_model(provider: str | None = None) -> str:
    """The model a provider answers with when none is named."""
    if normalise_provider(provider) == "chatgpt":
        from .chatgpt import DEFAULT_MODEL

        return DEFAULT_MODEL
    from .ollama import DEFAULT_MODEL

    return DEFAULT_MODEL


def default_url(provider: str | None = None) -> str:
    """The endpoint a provider is reached at when none is given."""
    if normalise_provider(provider) == "chatgpt":
        from .chatgpt import DEFAULT_URL

        return DEFAULT_URL
    from .ollama import DEFAULT_URL

    return DEFAULT_URL


def make_client(
    provider: str | None = None,
    url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """A client for ``provider``; blank arguments fall back to that provider's defaults.

    ``api_key`` is only meaningful for ``chatgpt`` (where it defaults to
    ``$OPENAI_API_KEY``) and is ignored by the keyless local provider.
    """
    if normalise_provider(provider) == "chatgpt":
        from .chatgpt import ChatGPTClient

        return ChatGPTClient(url, model, timeout, api_key=api_key)
    from .ollama import OllamaClient

    return OllamaClient(url, model, timeout)


def provider_of(client: Any) -> str:
    """The provider name a client belongs to (``DEFAULT_PROVIDER`` for anything unlabelled)."""
    provider = getattr(client, "provider", None)
    return provider if isinstance(provider, str) and provider in PROVIDERS else DEFAULT_PROVIDER


def loads_lenient(raw: str) -> Any:
    """JSON from an LLM answer: tolerates code fences and prose around the object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    return None
