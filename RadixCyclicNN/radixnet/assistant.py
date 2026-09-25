"""The model in today's format: messages in, an assistant message out - thinking, text, tool calls, streamed.

Every language model is talked to the same way now: a list of ``{"role",
"content"}`` messages goes in, an assistant message comes back, and while it is
being written the client sees it arrive - the model's *thinking* first, then
the answer, chunk by chunk, over server-sent events.  Two dialects of that
format cover every client there is: OpenAI's Chat Completions (``POST
/v1/chat/completions``, ``reasoning_content``, ``data: [DONE]``) and
Anthropic's Messages (``POST /v1/messages``, typed content blocks, one event
per block).  This module gives this model both.

Nothing here is a prompt trick.  A reply is what :func:`radixnet.dialogue.reply`
says next after the last line of the conversation - the tail of that line is
located in the graph and continued, exactly as in the Converse and Chat tabs -
and the format is a rendering of what that search already produces:

* the **thinking** is the search's own trace, line by line as it happens: which
  tail of the line it looked for and whether the graph knew it, how many paths
  it weighed, what the negative network vetoed and why, where it caught itself
  repeating and how it backed out, and what it finally said at what cost.  It
  is never prose the model did not produce; a reader can check every line
  against the turn record that comes back beside it;
* the **text** streams one chunk per node of the walk - the radix compression
  made visible: a merged node arrives as a word, an unmerged one as a letter;
* a ``<tool>name {...}</tool>`` line the model writes (the agent's own call
  format, :mod:`radixnet.tools`) comes back as a **tool call** when the request
  offered that tool, and a tool's answer goes back in as the ``<result>`` text
  the agent trains on, so the model reads it the way it learned it;
* the **stop reason** says how the walk ended (the end of a text, the cap, a
  stop sequence, a tool call, or the guard vetoing everything), and **usage**
  counts *units of the model's encoding* - a token here is one character of a
  character model and one word of a word model, the units ``max_tokens`` caps.

A system prompt is accepted and not read: the network continues text and
cannot follow an instruction, and the thinking says so rather than pretending.
What the conversation has said, on both sides, counts as heard, so the reply
does not echo it (:class:`radixnet.dialogue.Heard`), and with ``learn`` on
(the default, as in every conversation here) what a rethink finds out is
taught to the graph - a conversation changes the model, D-068.

The same shape is served three times over: here, in ``go/radixnet/assistant.go``
and in ``rust/src/assistant.rs``, held to this one by the parity suites.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

from .dialogue import EXPLORE, Heard, reply as dialogue_reply
from .encoding import WORDS, Encoding
from .graph import FIRST
from .tools import PARAM_TYPES, Param, Tool, ToolBox, call_text, parse_call, result_text

if TYPE_CHECKING:  # pragma: no cover
    from .duo import NegativeFilter
    from .model import GraphModel

__all__ = [
    "ANTHROPIC", "AnthropicStream", "Ask", "AskError", "Choice", "DEFAULT_MAX_TOKENS", "END_TURN", "FINISH_REASONS",
    "FORMATS", "MAX_TOKENS", "MODEL_PREFIX", "Message", "OPENAI", "OpenAIStream", "REFUSAL", "ROLES", "Reply",
    "STOP_REASONS", "STOP_SEQUENCE", "TOOL_USE", "ToolCall", "complete", "deltas", "input_units", "kind_of_id", "model_id",
    "offered_toolbox",
    "parse_anthropic", "parse_openai", "parse_request", "respond", "sse", "stream", "to_anthropic", "to_openai",
    "to_format", "usage_of",
]

OPENAI, ANTHROPIC = "openai", "anthropic"
FORMATS = (OPENAI, ANTHROPIC)
"""The two dialects: OpenAI's Chat Completions and Anthropic's Messages."""

ROLES = ("system", "user", "assistant", "tool")

END_TURN, MAX_TOKENS, STOP_SEQUENCE, TOOL_USE, REFUSAL = "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"
STOP_REASONS = (END_TURN, MAX_TOKENS, STOP_SEQUENCE, TOOL_USE, REFUSAL)
"""How a reply ended, in the Messages vocabulary; :data:`FINISH_REASONS` is the same in Chat Completions'."""

FINISH_REASONS = {END_TURN: "stop", MAX_TOKENS: "length", STOP_SEQUENCE: "stop", TOOL_USE: "tool_calls", REFUSAL: "content_filter"}

MODEL_PREFIX = "radixnet-"
"""A model's id is its kind behind this prefix: ``radixnet-count``, ``radixnet-radix``, ``radixnet-resonant``."""

DEFAULT_MAX_TOKENS = 60
"""Units a reply may add when the request sets no ``max_tokens``: the dialogue's own ``max_length``."""

MAX_CHOICES = 8
"""How many alternative replies (``n``) one request may ask for."""

_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class AskError(ValueError):
    """A request that cannot be answered as it stands; ``param`` names the field when one is to blame."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """One line of the conversation as the model reads it: who said it, and the text.

    An assistant's tool calls and a tool's results are rendered into the text
    in the agent's transcript format (``<tool>…</tool>``, ``<result>…</result>``),
    which is the shape the network was trained on.
    """

    role: str
    text: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.text}


@dataclass
class Ask:
    """A request in either dialect, normalised: what to answer, and how the search should look for it."""

    messages: list[Message] = field(default_factory=list)
    """The conversation, system lines excluded (they are in ``system``); the reply continues the last one."""
    system: str = ""
    model: str = ""
    """The model asked for by id (``""`` is whichever is active)."""
    max_tokens: int = DEFAULT_MAX_TOKENS
    """Units the reply may add to the context it picks up."""
    temperature: float = 1.0
    stop: list[str] = field(default_factory=list)
    n: int = 1
    stream: bool = False
    include_usage: bool = False
    """Chat Completions: ``stream_options.include_usage`` - a usage chunk before ``[DONE]``."""
    tools: list[dict] = field(default_factory=list)
    """``[{"name", "description", "parameters"}]``: a written call to one of these comes back as a tool call."""
    thinking: bool = True
    """Whether the search's trace is returned as the thinking (it is always kept in the turn record)."""
    # the dialogue's own dials, named as POST /api/converse names them
    mode: str = "beam"
    context: int = 12
    k: int = 5
    beam: int | None = None
    step_penalty: float = 0.0
    explore: int = EXPLORE
    avoid_repeats: bool = True
    avoid_word_repeats: bool = True
    learn: bool = True
    guard: bool = True
    seed: int | None = None

    def validate(self) -> None:
        """Raise :class:`AskError` when the request cannot be answered."""
        if not self.messages:
            raise AskError("messages must hold at least one user or assistant message", "messages")
        for i, message in enumerate(self.messages):
            if message.role not in ROLES or message.role == "system":
                raise AskError(f"messages[{i}].role must be user, assistant or tool (got {message.role!r})", "messages")
        if self.max_tokens < 1:
            raise AskError("max_tokens must be >= 1", "max_tokens")
        if self.temperature < 0:
            raise AskError("temperature must be >= 0", "temperature")
        if not 1 <= self.n <= MAX_CHOICES:
            raise AskError(f"n must be between 1 and {MAX_CHOICES}", "n")
        if self.mode not in ("beam", "sample"):
            raise AskError("mode must be beam or sample", "mode")
        if self.context < 0:
            raise AskError("context must be >= 0", "context")
        if self.k < 1:
            raise AskError("k must be >= 1", "k")
        if self.beam is not None and self.beam < 1:
            raise AskError("beam must be >= 1", "beam")
        if self.step_penalty < 0:
            raise AskError("step_penalty must be >= 0", "step_penalty")
        if self.explore < 0:
            raise AskError("explore must be >= 0", "explore")
        for i, seq in enumerate(self.stop):
            if not isinstance(seq, str) or not seq:
                raise AskError(f"stop[{i}] must be a non-empty string", "stop")
        names: set[str] = set()
        for i, tool in enumerate(self.tools):
            name = tool.get("name")
            if not isinstance(name, str) or not _TOOL_NAME.match(name):
                raise AskError(f"tools[{i}].name must be an identifier (got {name!r})", "tools")
            if name in names:
                raise AskError(f"tools[{i}].name {name!r} is offered twice", "tools")
            names.add(name)

    @property
    def previous(self) -> str:
        """The line the reply continues: the last message of the conversation."""
        return self.messages[-1].text if self.messages else ""

    @property
    def prefill(self) -> bool:
        """Whether the last message is the assistant's own: the reply then continues it and returns only what it adds."""
        return bool(self.messages) and self.messages[-1].role == "assistant"

    def dialogue_options(self) -> dict:
        """The keyword arguments :func:`radixnet.dialogue.reply` takes from this request."""
        return {
            "mode": self.mode, "max_length": self.max_tokens, "context": self.context, "temperature": self.temperature,
            "k": self.k, "beam": self.beam, "step_penalty": self.step_penalty, "avoid_repeats": self.avoid_repeats,
            "avoid_word_repeats": self.avoid_word_repeats, "explore": self.explore, "learn": self.learn,
        }

    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self.messages], "system": self.system, "model": self.model,
            "max_tokens": self.max_tokens, "temperature": self.temperature, "stop": list(self.stop), "n": self.n,
            "stream": self.stream, "tools": [t["name"] for t in self.tools], "thinking": self.thinking, "mode": self.mode,
            "context": self.context, "k": self.k, "beam": self.beam, "step_penalty": self.step_penalty,
            "explore": self.explore, "avoid_repeats": self.avoid_repeats, "avoid_word_repeats": self.avoid_word_repeats,
            "learn": self.learn, "guard": self.guard, "seed": self.seed,
        }


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _field_int(body: dict, name: str, default: int | None, minimum: int | None = None) -> int | None:
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not value.is_integer()):
        raise AskError(f"{name} must be an integer (got {_json_type(value)})", name)
    value = int(value)
    if minimum is not None and value < minimum:
        raise AskError(f"{name} must be >= {minimum} (got {value})", name)
    return value


def _field_num(body: dict, name: str, default: float, minimum: float | None = None) -> float:
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise AskError(f"{name} must be a number (got {_json_type(value)})", name)
    if minimum is not None and value < minimum:
        raise AskError(f"{name} must be >= {minimum} (got {value})", name)
    return float(value)


def _field_bool(body: dict, name: str, default: bool) -> bool:
    value = body.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise AskError(f"{name} must be a boolean (got {_json_type(value)})", name)
    return value


def _field_str(body: dict, name: str, default: str) -> str:
    value = body.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise AskError(f"{name} must be a string (got {_json_type(value)})", name)
    return value


def _dials(body: dict, ask: Ask) -> None:
    """The dialogue's own dials, read from either dialect's body by the names ``/api/converse`` uses."""
    ask.mode = _field_str(body, "mode", "beam").lower() or "beam"
    if ask.mode == "dijkstra":
        ask.mode = "beam"
    ask.context = _field_int(body, "context", 12, 0)
    ask.k = _field_int(body, "k", 5, 1)
    ask.beam = _field_int(body, "beam", None, 1)
    ask.step_penalty = _field_num(body, "step_penalty", 0.0, 0.0)
    ask.explore = _field_int(body, "explore", EXPLORE, 0)
    ask.avoid_repeats = _field_bool(body, "avoid_repeats", True)
    ask.avoid_word_repeats = _field_bool(body, "avoid_word_repeats", True)
    ask.learn = _field_bool(body, "learn", True)
    ask.guard = _field_bool(body, "guard", True)
    ask.seed = _field_int(body, "seed", None)


def _text_parts(content: Any, where: str, dialect: str) -> str:
    """The text of a message's ``content``: a string, or a list of typed parts joined by newlines.

    Text parts are read; an assistant's tool calls and a tool's results are
    rendered into the transcript format; earlier thinking is skipped (it was
    never said); anything the model cannot read - an image, a document, audio -
    is refused with the reason.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise AskError(f"{where}.content must be a string or a list of content parts", "messages")
    parts: list[str] = []
    for i, part in enumerate(content):
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            raise AskError(f"{where}.content[{i}] must be an object with a type", "messages")
        kind = part.get("type")
        if kind in ("text", "refusal"):
            text = part.get("text") if kind == "text" else part.get("refusal")
            if not isinstance(text, str):
                raise AskError(f"{where}.content[{i}].text must be a string", "messages")
            parts.append(text)
        elif kind in ("thinking", "redacted_thinking"):
            continue  # what the model thought is not what it said
        elif kind == "tool_use" and dialect == ANTHROPIC:
            name, arguments = part.get("name"), part.get("input")
            if not isinstance(name, str) or not name:
                raise AskError(f"{where}.content[{i}].name must be the tool's name", "messages")
            if arguments is not None and not isinstance(arguments, dict):
                raise AskError(f"{where}.content[{i}].input must be an object", "messages")
            parts.append(call_text(name, arguments or {}).rstrip("\n"))
        elif kind == "tool_result" and dialect == ANTHROPIC:
            parts.append(result_text(_text_parts(part.get("content"), f"{where}.content[{i}]", dialect)).rstrip("\n"))
        else:
            what = kind if isinstance(kind, str) else _json_type(kind)
            raise AskError(f"{where}.content[{i}]: content of type {what!r} is not supported; this model reads text", "messages")
    return "\n".join(parts)


def _tool_calls_text(calls: Any, where: str) -> str:
    """Chat Completions: an assistant message's ``tool_calls`` as the ``<tool>`` lines the model writes."""
    if calls is None:
        return ""
    if not isinstance(calls, list):
        raise AskError(f"{where}.tool_calls must be a list", "messages")
    lines: list[str] = []
    for i, call in enumerate(calls):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise AskError(f"{where}.tool_calls[{i}] must be a function call with a name", "messages")
        raw = function.get("arguments")
        arguments: Any = {}
        if isinstance(raw, str) and raw.strip():
            try:
                arguments = json.loads(raw)
            except ValueError as exc:
                raise AskError(f"{where}.tool_calls[{i}].function.arguments is not JSON: {exc}", "messages") from None
        elif isinstance(raw, dict):
            arguments = raw
        if not isinstance(arguments, dict):
            raise AskError(f"{where}.tool_calls[{i}].function.arguments must encode an object", "messages")
        lines.append(call_text(function["name"], arguments).rstrip("\n"))
    return "\n".join(lines)


def _tools(items: Any, dialect: str) -> list[dict]:
    """The tools offered, in one shape: ``{"name", "description", "parameters"}``."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise AskError("tools must be a list", "tools")
    out: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise AskError(f"tools[{i}] must be an object", "tools")
        if dialect == OPENAI:
            if item.get("type", "function") != "function":
                raise AskError(f"tools[{i}].type must be \"function\"", "tools")
            spec = item.get("function", item)
            if not isinstance(spec, dict):
                raise AskError(f"tools[{i}].function must be an object", "tools")
            schema = spec.get("parameters")
        else:
            spec = item
            schema = item.get("input_schema")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise AskError(f"tools[{i}] has no name", "tools")
        if schema is not None and not isinstance(schema, dict):
            raise AskError(f"tools[{i}]: the parameters must be a JSON schema object", "tools")
        out.append({"name": name, "description": str(spec.get("description") or ""), "parameters": schema or {}})
    return out


def _stops(value: Any, name: str, one_string: bool) -> list[str]:
    """The stop sequences: a list of non-empty strings (Chat Completions also takes one string)."""
    if value is None:
        return []
    if isinstance(value, str) and one_string:
        value = [value]
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        what = "a string or a list of strings" if one_string else "a list of strings"
        raise AskError(f"{name} must be {what}", name)
    if any(not s for s in value):
        raise AskError(f"{name} must not hold an empty sequence", name)
    return list(value)


def _messages(items: Any, dialect: str) -> tuple[list[Message], str]:
    """The conversation and the system prompt out of a ``messages`` list."""
    if not isinstance(items, list):
        raise AskError("messages must be a list of {role, content} objects", "messages")
    system: list[str] = []
    messages: list[Message] = []
    for i, item in enumerate(items):
        where = f"messages[{i}]"
        if not isinstance(item, dict):
            raise AskError(f"{where} must be an object with a role and content", "messages")
        role = item.get("role")
        if not isinstance(role, str):
            raise AskError(f"{where}.role must be a string", "messages")
        role = role.lower()
        if role == "developer":
            role = "system"
        if role == "function":
            role = "tool"
        if dialect == ANTHROPIC and role not in ("user", "assistant"):
            raise AskError(f"{where}.role must be user or assistant (got {role!r})", "messages")
        if role not in ROLES:
            raise AskError(f"{where}.role must be one of {', '.join(ROLES)} (got {role!r})", "messages")
        text = _text_parts(item.get("content"), where, dialect)
        if role == "assistant" and dialect == OPENAI:
            calls = _tool_calls_text(item.get("tool_calls"), where)
            text = "\n".join(part for part in (text, calls) if part)
        if role == "tool" and dialect == OPENAI:
            text = result_text(text).rstrip("\n")
        if role == "system":
            if text.strip():
                system.append(text)
            continue
        if not text.strip():
            continue  # an empty line says nothing and is not heard
        messages.append(Message(role, text))
    return messages, "\n".join(system)


def parse_openai(body: dict) -> Ask:
    """A Chat Completions request (``POST /v1/chat/completions``) as an :class:`Ask`.

    Read: ``model``, ``messages`` (``system`` / ``developer`` lines join the
    system prompt; an assistant's ``tool_calls`` and a ``tool`` message's
    content are rendered into the transcript format), ``max_tokens`` /
    ``max_completion_tokens``, ``temperature``, ``stop``, ``n``, ``stream``,
    ``stream_options.include_usage``, ``tools`` and ``tool_choice`` (``"none"``
    withdraws them), ``reasoning_effort: "none"`` or ``thinking: false`` (no
    thinking in the answer), and the dialogue's dials by their ``/api/converse``
    names.  Everything else a client may send (``top_p``, ``logprobs``,
    ``user``, ``response_format``, ...) is accepted and ignored.
    """
    if not isinstance(body, dict):
        raise AskError("the request body must be a JSON object")
    ask = Ask(model=_field_str(body, "model", ""))
    if "messages" not in body:
        raise AskError("messages is required", "messages")
    ask.messages, ask.system = _messages(body["messages"], OPENAI)
    limit = _field_int(body, "max_completion_tokens", None, 1)
    if limit is None:
        limit = _field_int(body, "max_tokens", None, 1)
    ask.max_tokens = DEFAULT_MAX_TOKENS if limit is None else limit
    ask.temperature = _field_num(body, "temperature", 1.0, 0.0)
    ask.stop = _stops(body.get("stop"), "stop", one_string=True)
    ask.n = _field_int(body, "n", 1, 1)
    ask.stream = _field_bool(body, "stream", False)
    options = body.get("stream_options")
    if options is not None:
        if not isinstance(options, dict):
            raise AskError("stream_options must be an object", "stream_options")
        ask.include_usage = _field_bool(options, "include_usage", False)
    ask.tools = _tools(body.get("tools"), OPENAI)
    if body.get("tool_choice") == "none":
        ask.tools = []
    ask.thinking = _field_bool(body, "thinking", True) and _field_str(body, "reasoning_effort", "") != "none"
    _dials(body, ask)
    ask.validate()
    return ask


def parse_anthropic(body: dict) -> Ask:
    """A Messages request (``POST /v1/messages``) as an :class:`Ask`.

    Read: ``model``, ``system`` (a string or text blocks), ``messages`` with
    ``text``, ``tool_use`` and ``tool_result`` blocks (earlier ``thinking``
    blocks are skipped: they were never said), ``max_tokens``, ``temperature``,
    ``stop_sequences``, ``stream``, ``tools`` and ``tool_choice``, ``thinking``
    (``{"type": "enabled" | "disabled"}``), and the dialogue's dials.
    """
    if not isinstance(body, dict):
        raise AskError("the request body must be a JSON object")
    ask = Ask(model=_field_str(body, "model", ""))
    if "messages" not in body:
        raise AskError("messages is required", "messages")
    ask.messages, _ = _messages(body["messages"], ANTHROPIC)
    ask.system = _text_parts(body.get("system"), "system", ANTHROPIC)
    limit = _field_int(body, "max_tokens", None, 1)
    ask.max_tokens = DEFAULT_MAX_TOKENS if limit is None else limit
    ask.temperature = _field_num(body, "temperature", 1.0, 0.0)
    ask.stop = _stops(body.get("stop_sequences"), "stop_sequences", one_string=False)
    ask.stream = _field_bool(body, "stream", False)
    ask.tools = _tools(body.get("tools"), ANTHROPIC)
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "none":
        ask.tools = []
    thinking = body.get("thinking")
    if thinking is not None:
        if isinstance(thinking, bool):
            ask.thinking = thinking
        elif isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled", "adaptive"):
            ask.thinking = thinking["type"] != "disabled"
        else:
            raise AskError("thinking must be {\"type\": \"enabled\"} or {\"type\": \"disabled\"}", "thinking")
    _dials(body, ask)
    ask.validate()
    return ask


def parse_request(body: dict, dialect: str) -> Ask:
    """:func:`parse_openai` or :func:`parse_anthropic`, by dialect."""
    if dialect == OPENAI:
        return parse_openai(body)
    if dialect == ANTHROPIC:
        return parse_anthropic(body)
    raise AskError(f"format must be one of {', '.join(FORMATS)} (got {dialect!r})", "format")


# ---------------------------------------------------------------------------
# the model as a model id
# ---------------------------------------------------------------------------


def model_id(model: "GraphModel") -> str:
    """``radixnet-<kind>``: what a model is called in ``/v1/models`` and in every answer."""
    return MODEL_PREFIX + model.kind


def kind_of_id(name: str | None) -> str:
    """The kind a requested model id names (``""``: whichever model is active).

    ``"radixnet-count"``, ``"count"``, ``"RadixNet-Count"`` all name the count
    model; ``""``, ``"radixnet"`` and ``"default"`` name the active one.
    """
    text = (name or "").strip().lower()
    if text.startswith(MODEL_PREFIX):
        text = text[len(MODEL_PREFIX):]
    if text in ("", "radixnet", "default", "active"):
        return ""
    return text


def _token() -> str:
    return uuid.uuid4().hex[:24]


# ---------------------------------------------------------------------------
# the reply
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A call the model wrote to a tool the request offered."""

    token: str
    name: str
    input: dict

    def to_dict(self) -> dict:
        return {"id": self.token, "name": self.name, "input": self.input}


@dataclass
class Choice:
    """One reply: the thinking, the text, the calls, how it ended, and the turn record behind it all."""

    index: int = 0
    thinking: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = END_TURN
    stop_sequence: str | None = None
    turn: dict | None = None
    """The :class:`radixnet.dialogue.Turn` the reply came from (``None`` when the model had nothing to say)."""
    guard: dict | None = None
    """What the negative network vetoed on the way, with the reason behind each veto."""
    output_units: int = 0
    thinking_units: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index, "thinking": self.thinking, "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls], "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence, "turn": self.turn, "guard": self.guard,
            "output_units": self.output_units, "thinking_units": self.thinking_units,
        }


@dataclass
class Reply:
    """What :func:`respond` returns: the choices, the model they came from, and the usage in its units."""

    token: str
    model: str
    created: int
    kind: str
    units: str
    input_units: int = 0
    choices: list[Choice] = field(default_factory=list)
    thinking: bool = True

    @property
    def usage(self) -> dict:
        """``{"input", "output", "thinking", "total"}`` in the model's units (:func:`usage_of`)."""
        return usage_of(self)

    def to_dict(self) -> dict:
        return {
            "id": self.token, "model": self.model, "created": self.created, "kind": self.kind, "units": self.units,
            "choices": [c.to_dict() for c in self.choices], "usage": self.usage,
        }


def input_units(encoding: Encoding, ask: Ask) -> int:
    """What the request holds, in the model's units: the system prompt and every message (``count_tokens``)."""
    return encoding.length(ask.system) + sum(encoding.length(m.text) for m in ask.messages)


def usage_of(reply: Reply) -> dict:
    """The units read and written: a token is one unit of the model's encoding - a character, or a word."""
    output = sum(c.output_units for c in reply.choices)
    thinking = sum(c.thinking_units for c in reply.choices)
    return {"input": reply.input_units, "output": output, "thinking": thinking, "total": reply.input_units + output}


Emit = Callable[[dict], Any]
"""What :func:`respond` streams to: one event at a time, as they happen (see the module notes on events)."""


def _q(text: str) -> str:
    """Text in double quotes, JSON style - the quoting every port's thinking uses."""
    return json.dumps(text, ensure_ascii=False)


def _count(n: int, what: str) -> str:
    return f"{n} {what}(s)"


class _Narrator:
    """Turns the search's trace into the lines of the thinking, one event at a time."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)
        self._emit(text)

    def trace(self, event: dict) -> None:
        kind = event["kind"]
        if kind == "context":
            if not event["usable"]:
                self.line(f"looking for {_q(event['context'])} in the graph: not there whole; dropping a word")
        elif kind == "candidates":
            n = event["offered"]
            what = _count(n, "walk") + " drawn" if event["mode"] == "sample" else _count(n, "path") + " weighed"
            if event["context"]:
                self.line(f"looking for {_q(event['context'])} in the graph: found, {what}")
            else:
                self.line(f"starting a fresh text from the beginning: {what}")
        elif kind == "pick":
            skipped = event["skipped"] - event["vetoed"]
            if event["repeat"]:
                self.line("every path repeats something already said; the best of them is kept as a last resort")
            elif skipped > 0:
                self.line(f"skipped {skipped} (empty, or already said)")
        elif kind == "rethink":
            if event["kind_of"] == "stutter":
                caught = f"caught itself saying {_q(event['noticed'])} twice"
            else:
                caught = f"caught itself repeating {_q(event['noticed'])}"
            if not event["steps"]:
                text = caught + "; the words it picked up, not its own"
            elif event["found"]:
                text = caught + f"; kept {_q(event['cut'])} and found another way on in {_count(event['explored'], 'path')}"
            else:
                text = caught + f"; kept {_q(event['cut'])}, weighed {_count(event['explored'], 'path')}, found nothing new"
            if event["taught"] >= 0:
                text += " (and learned to hand over there)"
            self.line(text)
        elif kind == "fresh":
            self.line("nothing new follows the line; changing the subject with a fresh text")


def deltas(encoding: Encoding, labels: Sequence[str], node_ids: Sequence[int], text: str) -> list[str]:
    """``text`` cut into what each node of the walk added, so ``"".join(deltas) == text``.

    The first piece is the context the reply picked up plus the located node's
    remainder; every later piece is what its node adds beyond the overlap - a
    merged node's word, an unmerged node's letter.  A text the walk does not
    line up with (one cut at the cap) comes back whole, as one piece.
    """
    if not text:
        return []
    real = [label for node, label in zip(node_ids, labels) if node >= FIRST]
    if len(real) < 2:
        return [text]
    pieces = [encoding.piece(label, encoding.overlap) for label in real[1:]]
    tail = encoding.join(*pieces)
    if not tail:
        return [text]
    words = encoding.unit == WORDS
    if words:
        if text == tail:
            head = ""
        elif text.endswith(" " + tail):
            head = text[: len(text) - len(tail) - 1]
        else:
            return [text]
    elif text.endswith(tail):
        head = text[: len(text) - len(tail)]
    else:
        return [text]
    out = [head] if head else []
    for piece in pieces:
        if not piece:
            continue
        out.append((" " + piece) if words and out else piece)
    return out


def _windowed(pieces: Sequence[str], start: int, end: int) -> list[str]:
    """The pieces cut to the characters ``[start, end)`` of their concatenation."""
    out: list[str] = []
    at = 0
    for piece in pieces:
        lo, hi = max(start, at), min(end, at + len(piece))
        if hi > lo:
            out.append(piece[lo - at: hi - at])
        at += len(piece)
    return out


def _first_stop(text: str, stops: Sequence[str]) -> tuple[int, str | None]:
    """Where the earliest stop sequence begins (and which), or ``(len(text), None)``."""
    best, which = len(text), None
    for seq in stops:
        at = text.find(seq)
        if at >= 0 and at < best:
            best, which = at, seq
    return best, which


def offered_toolbox(tools: Sequence[dict]) -> ToolBox:
    """The tools a request offered, as a :class:`radixnet.tools.ToolBox` (with no handlers: the client runs them).

    The registry is what lets a written call's arguments be read the way the
    agent reads them - JSON, ``key=value`` pairs, or a bare value for a tool
    with one argument - from the little schema each tool came with.
    """
    box = ToolBox()
    for tool in tools:
        schema = tool.get("parameters") or {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        params = []
        for name, spec in properties.items():
            kind = spec.get("type") if isinstance(spec, dict) else None
            if kind not in PARAM_TYPES:
                kind = "string"
            description = spec.get("description", "") if isinstance(spec, dict) else ""
            params.append(Param(str(name), kind, str(description or ""), required=name in required))
        box.register(Tool(tool["name"], str(tool.get("description") or ""), tuple(params)))
    return box


def _guard_report(pair: "NegativeFilter", verdicts: list[dict]) -> dict:
    rejected = [v for v in verdicts if v["decision"] == "reject"]
    return {
        "on": True, "vetoed": len(rejected), "rejected": rejected, "verdicts": verdicts,
        "negative": pair.negative.stats(), "config": pair.describe()["config"],
    }


def respond(
    model: "GraphModel",
    ask: Ask,
    *,
    pair: "NegativeFilter | None" = None,
    on_event: Emit | None = None,
    name: str | None = None,
) -> Reply:
    """Answer ``ask`` with ``model``, streaming every event to ``on_event`` as it happens, and return the whole reply.

    ``pair`` is the guard: the negative network vetoes a candidate before it
    is spoken, and every veto is a line of the thinking with the reason.  It
    is the caller's to supply (``ModelService.guard`` on the server, the CLI's
    ``open_guard``) and to leave out with ``ask.guard`` off.

    The events, in order, per choice: ``start``; ``thinking`` deltas (a line
    each, the second onwards led by a newline, so the deltas join into the
    thinking); ``text`` deltas (one per node of the walk); ``tool_use`` when the
    reply wrote a call to an offered tool; ``done`` with the stop reason, the
    stop sequence, the units and the turn record.  After the last choice one
    ``end`` carries the usage of the whole reply.  Every event carries the
    ``index`` of its choice.

    Runs under whatever lock the caller holds; with ``ask.learn`` the model
    may change (a rethink teaches the graph where it goes round).
    """
    import random

    ask.validate()
    enc = model.encoding
    emit: Emit = on_event if on_event is not None else (lambda event: None)
    reply = Reply(
        token=_token(), model=name or model_id(model), created=int(time.time()), kind=model.kind,
        units=enc.units_name, thinking=ask.thinking, input_units=input_units(enc, ask),
    )
    heard = Heard([m.text for m in ask.messages])
    previous = ask.previous
    rng = random.Random(ask.seed) if ask.seed is not None else None
    options = ask.dialogue_options()
    offered = {t["name"] for t in ask.tools}
    toolbox = offered_toolbox(ask.tools) if ask.tools else None
    earlier = len(ask.messages) - 1
    for index in range(ask.n):
        emit({"type": "start", "index": index, "id": reply.token, "model": reply.model, "created": reply.created,
              "input_units": reply.input_units, "thinking": ask.thinking})
        first = [True]

        def think(line: str, index: int = index) -> None:
            emit({"type": "thinking", "index": index, "text": ("" if first[0] else "\n") + line})
            first[0] = False

        narrator = _Narrator(think)
        opening = f"continuing its own last line {_q(previous)}" if ask.prefill else f"answering {_q(previous)}"
        if earlier:
            opening += f" ({_count(earlier, 'earlier line')} heard)"
        narrator.line(opening)
        if ask.system.strip():
            narrator.line("a system prompt was given; the network continues text and cannot follow instructions, so it is not read")
        if ask.tools:
            names = ", ".join(t["name"] for t in ask.tools)
            narrator.line(f"{_count(len(ask.tools), 'tool')} offered ({names}); a reply that writes one comes back as a tool call")
        verdicts: list[dict] = []
        seen: dict[str, bool] = {}
        veto = None
        if pair is not None:
            judge = pair

            def veto(text: str) -> bool:
                # a candidate offered again after a shorter context is judged once
                if text not in seen:
                    verdict = judge.judge(text)
                    seen[text] = verdict["decision"] == "reject"
                    verdicts.append(verdict)
                    if seen[text]:
                        narrator.line(f"the negative network vetoed {_q(text)}: {verdict['why']}")
                return seen[text]

        turn = dialogue_reply(
            model, previous, heard=heard, index=earlier + 1, speaker="assistant", rng=rng, veto=veto,
            trace=narrator.trace, **options,
        )
        guard = None
        if pair is not None:
            rejected = [v["text"] for v in verdicts if v["decision"] == "reject"]
            if rejected and pair.config.learn:
                pair.negative.blame(
                    [t for t in rejected if pair.negative.encoding.encode(t)], reason=pair.config.reason,
                    source="filter", note="vetoed in conversation",
                )
            guard = _guard_report(pair, verdicts)
        choice = Choice(index=index, guard=guard)
        if turn is None:
            if any(seen.values()):
                narrator.line("nothing to say: the guard vetoed everything it could say")
                choice.stop_reason = REFUSAL
            else:
                narrator.line("nothing to say: the graph has no way on from here")
                choice.stop_reason = END_TURN
        else:
            ending = "reached the end of a text" if turn.reached_end else f"cut at {ask.max_tokens} {enc.units_name}"
            said = f"saying {_q(turn.text)}: cost {turn.cost:.4f}, probability {turn.probability:.4f}, {ending}"
            if turn.repeat:
                said += "; every path repeated something, so this is a repeat"
            narrator.line(said)
            whole = turn.text
            pieces = deltas(enc, turn.labels, turn.node_ids, whole)
            start = len(turn.context) if ask.prefill and turn.context else 0
            end = len(whole)
            choice.stop_reason = END_TURN if turn.reached_end else MAX_TOKENS
            spoken = whole[start:]
            cut, seq = _first_stop(spoken, ask.stop)
            if seq is not None:
                end = start + cut
                choice.stop_reason, choice.stop_sequence = STOP_SEQUENCE, seq
                narrator.line(f"stopped at the stop sequence {_q(seq)}")
            call = parse_call(whole[start:end])
            if call is not None and call.name in offered:
                call = parse_call(whole[start:end], toolbox)  # the offered tool's schema reads the arguments
            if call is not None:
                if not call.name:
                    narrator.line(f"wrote a tool call it could not finish ({call.error}); it stays text")
                elif call.name not in offered:
                    if offered:
                        narrator.line(f"wrote a call to {call.name}, which was not offered; it stays text")
                    else:
                        narrator.line(f"wrote a call to {call.name}, but no tools were offered; it stays text")
                elif call.error:
                    narrator.line(f"wrote a call to {call.name} it could not finish ({call.error}); it stays text")
                else:
                    narrator.line(f"wrote a tool call: {call_text(call.name, call.arguments).strip()}")
                    end = start + call.span[0]
                    choice.tool_calls.append(ToolCall(_token(), call.name, dict(call.arguments)))
                    choice.stop_reason, choice.stop_sequence = TOOL_USE, None
            choice.text = whole[start:end]
            choice.turn = turn.to_dict()
            for piece in _windowed(pieces, start, end):
                emit({"type": "text", "index": index, "text": piece})
            for tool_call in choice.tool_calls:
                emit({"type": "tool_use", "index": index, "id": tool_call.token, "name": tool_call.name, "input": tool_call.input})
            heard.remember(turn.text, turn.reply if turn.context else "")
        choice.thinking = "\n".join(narrator.lines)
        choice.thinking_units = enc.length(choice.thinking)
        choice.output_units = enc.length(choice.text) + choice.thinking_units
        reply.choices.append(choice)
        emit({
            "type": "done", "index": index, "stop_reason": choice.stop_reason, "stop_sequence": choice.stop_sequence,
            "output_units": choice.output_units, "thinking_units": choice.thinking_units, "turn": choice.turn,
            "guard": choice.guard,
        })
    emit({"type": "end", "usage": reply.usage})
    return reply


def complete(model: "GraphModel", ask: Ask, **options: Any) -> Reply:
    """:func:`respond` without streaming: the whole reply at once."""
    return respond(model, ask, **options)


def stream(model: "GraphModel", ask: Ask, **options: Any) -> Iterator[dict]:
    """The events of :func:`respond` as they happen, on a thread of their own.

    The search runs on the worker thread and this generator yields each event
    the moment it is emitted; the last event is ``end``.  An exception on the
    worker is raised here.  Nothing locks the model: a caller holding a lock
    must hold it for the whole iteration.
    """
    box: queue.Queue = queue.Queue()
    stop = object()

    def work() -> None:
        try:
            respond(model, ask, on_event=box.put, **options)
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            box.put(exc)
        finally:
            box.put(stop)

    threading.Thread(target=work, name="radixnet-assistant", daemon=True).start()
    while True:
        item = box.get()
        if item is stop:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


# ---------------------------------------------------------------------------
# the two dialects
# ---------------------------------------------------------------------------


def _openai_tool_call(call: ToolCall) -> dict:
    return {
        "id": "call_" + call.token, "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.input, ensure_ascii=False, sort_keys=True)},
    }


def _record(reply: Reply) -> dict:
    """The model's own account of the answer, beside the standard fields: the kind, the units, every turn."""
    return {
        "kind": reply.kind, "units": reply.units,
        "choices": [{"index": c.index, "stop_reason": c.stop_reason, "turn": c.turn, "guard": c.guard} for c in reply.choices],
    }


def to_openai(reply: Reply) -> dict:
    """The reply as a Chat Completions ``chat.completion`` object.

    The thinking is ``message.reasoning_content`` (the field the open-weight
    servers agree on), a tool call ``message.tool_calls``; ``usage`` counts in
    the model's units with the thinking in ``completion_tokens_details``; the
    model's own record rides along as ``radixnet``.
    """
    choices = []
    for c in reply.choices:
        message: dict[str, Any] = {"role": "assistant", "content": c.text if c.text or not c.tool_calls else None}
        if reply.thinking:
            message["reasoning_content"] = c.thinking
        if c.tool_calls:
            message["tool_calls"] = [_openai_tool_call(t) for t in c.tool_calls]
        choices.append({"index": c.index, "message": message, "logprobs": None, "finish_reason": FINISH_REASONS[c.stop_reason]})
    usage = reply.usage
    return {
        "id": "chatcmpl-" + reply.token, "object": "chat.completion", "created": reply.created, "model": reply.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage["input"], "completion_tokens": usage["output"], "total_tokens": usage["total"],
            "completion_tokens_details": {"reasoning_tokens": usage["thinking"]},
        },
        "radixnet": _record(reply),
    }


def _anthropic_content(c: Choice, thinking: bool) -> list[dict]:
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": c.thinking, "signature": ""})
    if c.text or not c.tool_calls:
        blocks.append({"type": "text", "text": c.text})
    for call in c.tool_calls:
        blocks.append({"type": "tool_use", "id": "toolu_" + call.token, "name": call.name, "input": call.input})
    return blocks


def to_anthropic(reply: Reply) -> dict:
    """The reply as a Messages ``message`` object: a ``thinking`` block, a ``text`` block, ``tool_use`` blocks.

    A Messages reply is one message, so the first choice is the answer; the
    model's own record (every choice) rides along as ``radixnet``.
    """
    c = reply.choices[0] if reply.choices else Choice()
    usage = reply.usage
    return {
        "id": "msg_" + reply.token, "type": "message", "role": "assistant", "model": reply.model,
        "content": _anthropic_content(c, reply.thinking),
        "stop_reason": c.stop_reason, "stop_sequence": c.stop_sequence,
        "usage": {"input_tokens": usage["input"], "output_tokens": c.output_units},
        "radixnet": _record(reply),
    }


def to_format(reply: Reply, dialect: str) -> dict:
    """:func:`to_openai` or :func:`to_anthropic`, by dialect."""
    if dialect == OPENAI:
        return to_openai(reply)
    if dialect == ANTHROPIC:
        return to_anthropic(reply)
    raise AskError(f"format must be one of {', '.join(FORMATS)} (got {dialect!r})", "format")


def sse(data: Any, event: str | None = None) -> str:
    """One server-sent event: ``event: name`` (when given), then ``data: <json>`` and a blank line."""
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {body}\n\n"


class OpenAIStream:
    """Renders :func:`respond`'s events as Chat Completions chunks (``chat.completion.chunk``).

    ``frames(event)`` returns the SSE frames an event becomes; the ``end``
    event becomes the usage chunk (with ``include_usage``) and ``[DONE]``.
    ``error(message)`` is the frame a failure mid-stream is reported with.
    """

    def __init__(self, token: str, model: str, created: int, *, include_usage: bool = False, thinking: bool = True) -> None:
        self.token, self.model, self.created = token, model, created
        self.include_usage, self.thinking = include_usage, thinking

    def _chunk(self, choices: list[dict], **extra: Any) -> str:
        doc = {"id": "chatcmpl-" + self.token, "object": "chat.completion.chunk", "created": self.created,
               "model": self.model, "choices": choices, **extra}
        return sse(doc)

    def frames(self, event: dict) -> list[str]:
        kind = event["type"]
        if kind == "start":
            return [self._chunk([{"index": event["index"], "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])]
        if kind == "thinking":
            if not self.thinking:
                return []
            return [self._chunk([{"index": event["index"], "delta": {"reasoning_content": event["text"]}, "finish_reason": None}])]
        if kind == "text":
            return [self._chunk([{"index": event["index"], "delta": {"content": event["text"]}, "finish_reason": None}])]
        if kind == "tool_use":
            call = {"index": 0, "id": "call_" + event["id"], "type": "function",
                    "function": {"name": event["name"], "arguments": json.dumps(event["input"], ensure_ascii=False, sort_keys=True)}}
            return [self._chunk([{"index": event["index"], "delta": {"tool_calls": [call]}, "finish_reason": None}])]
        if kind == "done":
            return [self._chunk([{"index": event["index"], "delta": {}, "finish_reason": FINISH_REASONS[event["stop_reason"]]}],
                                radixnet={"stop_reason": event["stop_reason"], "turn": event["turn"], "guard": event["guard"]})]
        if kind == "end":
            out = []
            if self.include_usage:
                usage = event["usage"]
                out.append(self._chunk([], usage={
                    "prompt_tokens": usage["input"], "completion_tokens": usage["output"], "total_tokens": usage["total"],
                    "completion_tokens_details": {"reasoning_tokens": usage["thinking"]},
                }))
            out.append(sse("[DONE]"))
            return out
        return []

    @staticmethod
    def error(message: str) -> str:
        return sse({"error": {"message": message, "type": "server_error", "param": None, "code": None}})


class AnthropicStream:
    """Renders :func:`respond`'s events as Messages events, one content block at a time.

    ``message_start`` opens the message; a ``thinking`` block collects the
    thinking deltas, a ``text`` block the text deltas, a ``tool_use`` block
    carries the call's input as one ``input_json_delta``; ``message_delta``
    carries the stop reason and the output units, ``message_stop`` closes.
    """

    def __init__(self, token: str, model: str, created: int, *, input_units: int = 0, thinking: bool = True) -> None:
        self.token, self.model, self.created = token, model, created
        self.input_units, self.thinking = input_units, thinking
        self.block = -1
        self.open: str | None = None
        self.answered = False  # whether a text or tool_use block has been opened

    def _open(self, kind: str, block: dict) -> list[str]:
        out = self._close()
        self.block += 1
        self.open = kind
        if kind != "thinking":
            self.answered = True
        out.append(sse({"type": "content_block_start", "index": self.block, "content_block": block}, "content_block_start"))
        return out

    def _close(self) -> list[str]:
        if self.open is None:
            return []
        out = []
        if self.open == "thinking":
            out.append(sse({"type": "content_block_delta", "index": self.block, "delta": {"type": "signature_delta", "signature": ""}},
                           "content_block_delta"))
        out.append(sse({"type": "content_block_stop", "index": self.block}, "content_block_stop"))
        self.open = None
        return out

    def frames(self, event: dict) -> list[str]:
        kind = event["type"]
        if kind == "end":
            return []  # message_stop went out with the first choice's done
        if kind == "start":
            if event["index"] != 0:
                return []
            message = {"id": "msg_" + self.token, "type": "message", "role": "assistant", "model": self.model, "content": [],
                       "stop_reason": None, "stop_sequence": None,
                       "usage": {"input_tokens": self.input_units, "output_tokens": 0}}
            return [sse({"type": "message_start", "message": message}, "message_start")]
        if event["index"] != 0:
            return []  # a Messages reply is one message: the first choice
        if kind == "thinking":
            if not self.thinking:
                return []
            out = self._open("thinking", {"type": "thinking", "thinking": "", "signature": ""}) if self.open != "thinking" else []
            out.append(sse({"type": "content_block_delta", "index": self.block, "delta": {"type": "thinking_delta", "thinking": event["text"]}},
                           "content_block_delta"))
            return out
        if kind == "text":
            out = self._open("text", {"type": "text", "text": ""}) if self.open != "text" else []
            out.append(sse({"type": "content_block_delta", "index": self.block, "delta": {"type": "text_delta", "text": event["text"]}},
                           "content_block_delta"))
            return out
        if kind == "tool_use":
            out = self._open("tool_use", {"type": "tool_use", "id": "toolu_" + event["id"], "name": event["name"], "input": {}})
            partial = json.dumps(event["input"], ensure_ascii=False, sort_keys=True)
            out.append(sse({"type": "content_block_delta", "index": self.block, "delta": {"type": "input_json_delta", "partial_json": partial}},
                           "content_block_delta"))
            out.extend(self._close())
            return out
        if kind == "done":
            out = []
            if not self.answered:
                # a reply with nothing to say is still one (empty) text block
                out.extend(self._open("text", {"type": "text", "text": ""}))
            out.extend(self._close())
            out.append(sse({"type": "message_delta", "delta": {"stop_reason": event["stop_reason"], "stop_sequence": event["stop_sequence"]},
                            "usage": {"output_tokens": event["output_units"]},
                            "radixnet": {"turn": event["turn"], "guard": event["guard"]}}, "message_delta"))
            out.append(sse({"type": "message_stop"}, "message_stop"))
            return out
        return []

    @staticmethod
    def error(message: str) -> str:
        return sse({"type": "error", "error": {"type": "api_error", "message": message}}, "error")
