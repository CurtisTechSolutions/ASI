"""The model, whichever one it is.

Ollama, OpenAI and Anthropic over `urllib` -- no vendor SDKs. Three reasons,
and the last is the real one:

1. The SDKs disagree about transitive dependencies and this repository has none.
2. The surface actually used here is two endpoints and a JSON body.
3. A provider seam this thin cannot leak provider concepts into the rest of the
   system. Everything above this file sees `complete()` and `embed()`, so
   swapping a local 3B model for a frontier API is a config change, and
   comparing them on the same task is a loop over this list (DESIGN 3.2).

`LocalProvider` is not a mock. It is a deterministic, offline decomposer that
implements the same interface with hand-written heuristics, and it is what the
test suite and `--offline` run against. The system must be runnable, gradeable
and improvable with no key and no network, because a self-improvement loop whose
every iteration costs money and latency will never be run long enough to improve
anything.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Overridable per provider; these are defaults, not pins.
OLLAMA_MODEL = os.environ.get("DISTIL_OLLAMA_MODEL", "gemma4")
OLLAMA_EMBED = os.environ.get("DISTIL_OLLAMA_EMBED", "nomic-embed-text")
OPENAI_MODEL = os.environ.get("DISTIL_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED = os.environ.get("DISTIL_OPENAI_EMBED", "text-embedding-3-small")
ANTHROPIC_MODEL = os.environ.get("DISTIL_ANTHROPIC_MODEL", "claude-sonnet-5")

TIMEOUT = float(os.environ.get("DISTIL_TIMEOUT", "60"))


class ProviderError(RuntimeError):
    pass


@dataclass
class Message:
    role: str
    content: str

    def to_json(self) -> dict:
        return {"role": self.role, "content": self.content}


def _post(url: str, payload: dict, headers: dict, timeout: float = TIMEOUT) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:                      # surface the body: the
        detail = exc.read().decode("utf-8", "replace")[:500]    # message is the diagnosis
        raise ProviderError(f"{url} -> {exc.code}: {detail}") from exc
    except Exception as exc:
        raise ProviderError(f"{url} -> {exc}") from exc


class Provider:
    """Interface. `complete` returns text; `embed` returns one vector per input.

    Every call is counted, and so is every failure. Callers all over this package
    wrap provider calls in `except Exception` and degrade -- correctly, because a
    reasoning step that cannot reach a model should fall back to a heuristic
    rather than take down the run. But that made a completely dead provider
    indistinguishable from a working one: the agent reported `ollama`, ran
    entirely on offline rules, and said nothing. The counters are what turn that
    from a mystery into a number, and `health` is what the UI shows.
    """

    name = "abstract"
    can_embed = False

    #: Bumped by `record`. Class-level defaults so a subclass that forgets to
    #: call super().__init__() still has them.
    calls = 0
    failures = 0
    last_error: str | None = None

    def available(self) -> bool:
        raise NotImplementedError

    def why_unavailable(self) -> str:
        """One line a person can act on. Overridden where there is a real answer."""
        return "" if self.available() else f"{self.name} is not reachable or not configured"

    def record(self, error: Exception | None = None) -> None:
        self.calls += 1
        if error is not None:
            self.failures += 1
            self.last_error = str(error)[:300]

    def health(self) -> dict:
        return {"name": self.name, "calls": self.calls, "failures": self.failures,
                "last_error": self.last_error,
                # The number that matters. A provider answering nothing while the
                # agent claims to be using it is the failure this exists to show.
                "failing": bool(self.calls and self.failures == self.calls)}

    def complete(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderError(f"{self.name} has no embedding endpoint")


class OllamaProvider(Provider):
    name = "ollama"
    can_embed = True

    def __init__(self, host: str | None = None, model: str = OLLAMA_MODEL, embed_model: str = OLLAMA_EMBED) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.embed_model = embed_model

    def installed(self) -> list[str]:
        """Model tags the daemon actually has, or [] if it cannot be asked."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                if r.status != 200:
                    return []
                return [m.get("name", "") for m in
                        (json.loads(r.read().decode("utf-8")).get("models") or [])]
        except Exception:
            return []

    @staticmethod
    def _matches(wanted: str, tags: list[str]) -> bool:
        """`gemma4` matches `gemma4:latest`; `gemma4:27b` matches only itself.

        Ollama reports fully-qualified tags and people type the short form, so a
        literal comparison called a present model missing.
        """
        if wanted in tags:
            return True
        if ":" in wanted:
            return False
        return any(t.split(":", 1)[0] == wanted for t in tags)

    def available(self) -> bool:
        """Reachable AND holding the model it is configured to use.

        Checking only that the daemon answers was the bug behind "auto mode is
        not querying ollama": /api/tags returns 200 whatever is installed, so an
        un-pulled model meant ollama was selected, every /api/chat 404ed, every
        caller swallowed the error and degraded, and the agent reported `ollama`
        while running on offline rules. A provider that cannot answer is not
        available, and saying so here is what makes `auto()` skip it and
        `--provider ollama` fail loudly.
        """
        tags = self.installed()
        return bool(tags) and self._matches(self.model, tags)

    def why_unavailable(self) -> str:
        tags = self.installed()
        if not tags:
            return (f"no ollama at {self.host} -- start it with `ollama serve`, "
                    f"or set OLLAMA_HOST")
        if not self._matches(self.model, tags):
            return (f"ollama is running but has no {self.model!r}: pull it with "
                    f"`ollama pull {self.model}`, or set DISTIL_OLLAMA_MODEL to one of "
                    f"{', '.join(sorted(tags)[:6])}")
        if self.can_embed and not self._matches(self.embed_model, tags):
            return (f"chat model is present but the embedder is not: "
                    f"`ollama pull {self.embed_model}`")
        return ""

    def complete(self, messages, temperature=0.7, max_tokens=1024) -> str:
        try:
            out = _post(f"{self.host}/api/chat", {
                "model": self.model,
                "messages": [m.to_json() for m in messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }, {})
        except Exception as exc:
            self.record(exc)
            raise
        self.record()
        return out.get("message", {}).get("content", "")

    def embed(self, texts):
        vectors = []
        for t in texts:                       # /api/embeddings is one text per call
            try:
                out = _post(f"{self.host}/api/embeddings",
                            {"model": self.embed_model, "prompt": t}, {})
            except Exception as exc:
                self.record(exc)
                raise
            self.record()
            vectors.append(out["embedding"])
        return vectors


class OpenAIProvider(Provider):
    name = "openai"
    can_embed = True

    def __init__(self, key: str | None = None, model: str = OPENAI_MODEL, embed_model: str = OPENAI_EMBED,
                 base: str | None = None) -> None:
        self.key = key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.embed_model = embed_model
        # base_url so any OpenAI-compatible server (vLLM, llama.cpp, LM Studio,
        # OpenRouter, Together) is reachable through this same class
        self.base = (base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    def available(self) -> bool:
        return bool(self.key)

    def why_unavailable(self) -> str:
        return "" if self.key else "OPENAI_API_KEY is not set"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def complete(self, messages, temperature=0.7, max_tokens=1024) -> str:
        try:
            out = _post(f"{self.base}/chat/completions", {
                "model": self.model,
                "messages": [m.to_json() for m in messages],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }, self._headers())
        except Exception as exc:
            self.record(exc)
            raise
        self.record()
        return out["choices"][0]["message"]["content"] or ""

    def embed(self, texts):
        out = _post(f"{self.base}/embeddings", {"model": self.embed_model, "input": texts}, self._headers())
        rows = sorted(out["data"], key=lambda d: d["index"])     # order is not promised
        return [r["embedding"] for r in rows]


class AnthropicProvider(Provider):
    """Claude. No embedding endpoint exists, so `can_embed` is False and the
    memory layer keeps its lexical embedder -- stated plainly rather than
    discovered as a 404 halfway through a run."""

    name = "anthropic"
    can_embed = False

    def __init__(self, key: str | None = None, model: str = ANTHROPIC_MODEL) -> None:
        self.key = key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")

    def available(self) -> bool:
        return bool(self.key)

    def why_unavailable(self) -> str:
        return "" if self.key else "ANTHROPIC_API_KEY is not set"

    def complete(self, messages, temperature=0.7, max_tokens=1024) -> str:
        system = " ".join(m.content for m in messages if m.role == "system")
        turns = [m.to_json() for m in messages if m.role != "system"]
        payload = {"model": self.model, "messages": turns, "max_tokens": max_tokens,
                   "temperature": temperature}
        if system:
            payload["system"] = system            # a top-level field, not a turn
        try:
            out = _post(f"{self.base}/messages", payload,
                        {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        except Exception as exc:
            self.record(exc)
            raise
        self.record()
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")


@dataclass
class LocalProvider(Provider):
    """Offline, deterministic, and honest about what it is.

    It answers the three prompts the system actually depends on -- decompose a
    task, propose an experiment, write a tool -- with rules rather than a model.
    Decomposition is real (verb/object parsing against a goal grammar). Tool
    synthesis covers a fixed library of goal shapes and returns nothing outside
    it, which is the correct behaviour for a synthesiser that cannot generalise:
    `None` routes the goal to a real provider instead of emitting plausible
    code that was never going to run.

    The point is that the loop closes with no key. Every result in the README
    was produced this way.
    """

    name: str = "local"
    can_embed: bool = False
    seen: list[str] = field(default_factory=list)

    def available(self) -> bool:
        return True

    def complete(self, messages, temperature=0.7, max_tokens=1024) -> str:
        prompt = messages[-1].content if messages else ""
        self.seen.append(prompt)
        self.record()
        if "DECOMPOSE" in prompt:
            return "\n".join(_decompose(_payload(prompt)))
        if "EXPERIMENT" in prompt:
            return "\n".join(_experiments(_payload(prompt)))
        if "WHY" in prompt:
            return _why(_payload(prompt))
        return ""


def _payload(prompt: str) -> str:
    _, _, rest = prompt.partition("::")
    return rest.strip() or prompt


_STOP = {"the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "with", "that", "this", "it"}
_VERBS = ("write", "build", "parse", "compute", "sort", "find", "count", "check", "convert",
          "read", "fetch", "clean", "merge", "validate", "render", "sum", "reverse", "filter")


def _decompose(task: str) -> list[str]:
    """Split a task into goals along the joints the language already marks.

    Conjunctions and punctuation separate independent goals; a leading verb marks
    an action; a bare noun phrase marks an artefact that must first be defined.
    Crude, and it does not need to be more than crude -- its job is to keep the
    control loop exercisable offline, and the loop treats every proposal as a
    hypothesis to be verified regardless of who proposed it.
    """
    # Already reduced to a goal by an earlier pass: a rule-based splitter has
    # nothing further to say about "define X", and saying it anyway produces
    # "define define X". Returning nothing leaves the goal a leaf that still
    # needs a check, which is the true state of affairs.
    if re.match(r"^(define|verify)\b", task.strip(), re.I):
        return []
    parts = [p.strip() for p in re.split(r"[;,.]|\bthen\b|\band\b|\bafter that\b", task) if p.strip()]
    goals: list[str] = []
    for part in parts:
        words = [w for w in re.findall(r"[\w'-]+", part.lower()) if w not in _STOP]
        if not words:
            continue
        verb = next((w for w in words if w in _VERBS), None)
        if verb:
            obj = " ".join(w for w in words if w != verb)[:60]
            goals.append(f"{verb} {obj}".strip())
        else:
            goals.append(f"define {' '.join(words[:6])}")
    if len(goals) < 2 and goals:
        goals.append(f"verify {goals[0]}"[:70])
    return goals[:6]


def _why(question: str) -> str:
    """The offline answer to "why?".

    It does not invent a justification, because inventing one is the failure the
    why-chain exists to catch. It reports that nothing on record justifies the
    claim -- which `challenge.classify` reads as ASSUMED, correctly: with no
    model and no memory, an unexamined premise is exactly what this is.
    """
    subject = question.strip().splitlines()[0].removeprefix("why").strip(" ?")
    subject = subject.replace("no recorded justification for ", "")      # do not nest
    subject = subject.split("; assume")[0].strip()                       # nor re-wrap
    return f"no recorded justification for {subject[:80]}; assume it was inherited from an earlier decision"


def _experiments(seed: str) -> list[str]:
    return [f"try {seed} with the inputs reversed",
            f"try {seed} on the empty case",
            f"try {seed} at a scale it was not designed for"]


def catalogue() -> list[Provider]:
    return [OllamaProvider(), OpenAIProvider(), AnthropicProvider(), LocalProvider()]


def auto(prefer: str | None = None) -> Provider:
    """First reachable provider, preference first.

    `DISTIL_PROVIDER` pins one; without it, order is local-model-first. Not for
    quality -- an 8B model loses to a frontier model at every task here -- but
    because the exploration loop runs thousands of iterations unattended, and a
    default that quietly bills for every one of them is a bad default.
    """
    prefer = prefer or os.environ.get("DISTIL_PROVIDER")
    providers = catalogue()
    if prefer:
        chosen = next((p for p in providers if p.name == prefer), None)
        if chosen is None:
            raise ProviderError(f"unknown provider {prefer!r}; have {[p.name for p in providers]}")
        if not chosen.available():
            # `why_unavailable` knows the actual reason -- not pulled, wrong host,
            # no key. "key set? daemon running?" made the reader guess at what the
            # code could simply say.
            raise ProviderError(f"provider {prefer!r} is not available: "
                                f"{chosen.why_unavailable()}")
        return chosen
    for p in providers:
        if p.available():
            return p
    return LocalProvider()
