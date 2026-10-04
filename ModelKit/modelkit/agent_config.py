"""Load an ``agent.json`` chat descriptor and call its Chat Completions endpoint.

This is a client configuration, separate from a saved graph and from the
tool-using training agent. Standard library only, with no model dependency.
"""

from __future__ import annotations

import ipaddress
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from os import PathLike
from typing import Any

__all__ = ["AgentConfig", "load_agent_config"]


@dataclass(frozen=True)
class AgentConfig:
    """Connection settings and request defaults for an unauthenticated chat server.

    ``id`` is a display label; ``model`` is the name sent to the server.
    ``structured`` requests JSON-object output, whose support is up to the
    server. It does not change or wrap the returned assistant text.
    """

    id: str
    type: str
    base_url: str
    model: str
    timeout: float = 60
    max_tokens: int = 128
    temperature: float = 0
    structured: bool = True

    def __post_init__(self) -> None:
        for name in ("id", "type", "base_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"agent config {name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if self.type != "chat":
            raise ValueError("agent config type must be 'chat'")
        for name, positive in (("timeout", True), ("temperature", False)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                bound = "positive" if positive else "non-negative"
                raise ValueError(f"agent config {name} must be a finite {bound} number")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("agent config max_tokens must be a positive integer")
        if not isinstance(self.structured, bool):
            raise ValueError("agent config structured must be a boolean")
        parts = urllib.parse.urlsplit(self.base_url)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or any(c.isspace() for c in self.base_url)):
            raise ValueError("agent config base_url must be an http(s) URL without credentials, query or fragment")
        # Accessing port also validates malformed / out-of-range ports.
        parts.port
        path = parts.path.rstrip("/") or "/v1"
        object.__setattr__(self, "base_url", urllib.parse.urlunsplit(parts._replace(path=path)))

    def request(self, body: dict[str, Any]) -> dict:
        """Send one non-streaming request; explicit body fields override the file's defaults.

        No API keys or account headers are read from the environment. The
        complete response is retained, including usage, thinking and tool calls.
        """
        if not isinstance(body, dict):
            raise ValueError("the agent request must be a JSON object")
        if body.get("stream") not in (None, False):
            raise ValueError("agent config requests do not support streaming; set stream to false")
        payload: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
        }
        if self.structured:
            payload["response_format"] = {"type": "json_object"}
        payload.update(body)
        if "max_completion_tokens" in body:
            payload.pop("max_tokens", None)
        payload["stream"] = False
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST",
        )
        host = urllib.parse.urlsplit(self.base_url).hostname
        local = host == "localhost"
        try:
            local = local or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({})) if local
                  else urllib.request.build_opener())
        try:
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            try:
                error = json.loads(detail).get("error")
                if isinstance(error, dict):
                    detail = error.get("message") or detail
                elif isinstance(error, str):
                    detail = error
            except (ValueError, AttributeError):
                pass
            raise ValueError(f"agent {self.id!r} returned HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ValueError(f"cannot reach agent {self.id!r} at {self.base_url}: {exc}") from exc
        try:
            doc = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"agent {self.id!r} returned invalid JSON") from exc
        choices = doc.get("choices") if isinstance(doc, dict) else None
        if (not isinstance(choices, list) or not choices
                or any(not isinstance(c, dict) or not isinstance(c.get("message"), dict) for c in choices)):
            raise ValueError(f"agent {self.id!r} returned no chat completion choices")
        return doc


def load_agent_config(path: str | PathLike[str]) -> AgentConfig:
    """Read a UTF-8 JSON object, reporting bad files and fields before contacting a server."""
    try:
        with open(path, encoding="utf-8-sig") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return AgentConfig(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot load agent config {str(path)!r}: {exc}") from exc
