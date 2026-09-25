"""External tools the network can call, and the text format it learns them in.

The network is a character-level graph model: it cannot "decide" to call a
function, it can only *emit characters*.  So a tool call is a piece of text
like any other training data::

    <tool>web_fetch {"url": "https://example.com"}</tool>

and what the tool answers comes back as another piece of text::

    <result>Example Domain. This domain is for use in ...</result>

A whole attempt at a task is therefore one training text (:func:`transcript_text`)
that the network can be rewarded or punished for as a unit — exactly what
:mod:`radixnet.agent` does with 2NRL.  Because the emission is only text, it is
routinely malformed; :func:`parse_call` is deliberately lenient (JSON arguments,
``key=value`` pairs, or a bare value for a single-argument tool) and whatever it
still cannot read is handed to Ollama to repair (:mod:`radixnet.agent`).

The tools themselves:

* :class:`ToolBox` — the registry: names, JSON schemas (Ollama's own
  ``tools`` format), argument validation, and :meth:`ToolBox.call`, which never
  raises: a failure is a :class:`ToolResult` with ``ok=False`` and a message the
  model (and the LLM) can read.
* :class:`WebClient` — browsing with the standard library: ``http`` / ``https``
  only, no redirects to private addresses, a byte cap, a timeout, and HTML
  reduced to readable text (:func:`html_to_text`).
* :func:`default_toolbox` — the built-in set: ``web_search``, ``web_fetch``,
  ``web_links``, ``calculator``, ``python`` (the :mod:`radixnet.codegen`
  sandbox) and ``read_file`` (uploaded files).

Safety: the web tools refuse private / loopback / link-local addresses unless
``allow_private`` is set (the tests point them at a local fake site), cap the
response size, and never send cookies or credentials.  ``python`` runs in the
codegen sandbox — isolation from accidents, not from a hostile program.
"""

from __future__ import annotations

import ast
import dataclasses
import html
import ipaddress
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from typing import Any

__all__ = [
    "ANSWER_CLOSE",
    "ANSWER_OPEN",
    "CALL_CLOSE",
    "CALL_OPEN",
    "DEFAULT_SEARCH_ENGINES",
    "DEFAULT_SEARCH_URL",
    "DEFAULT_USER_AGENT",
    "PARAM_TYPES",
    "RESULT_CLOSE",
    "RESULT_OPEN",
    "Param",
    "SearchResult",
    "Tool",
    "ToolBox",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "WebClient",
    "WebError",
    "answer_text",
    "call_text",
    "default_toolbox",
    "find_answer",
    "format_observation",
    "html_to_text",
    "parse_call",
    "parse_arguments",
    "result_text",
    "safe_eval",
    "task_header",
    "transcript_text",
]

# ---------------------------------------------------------------------------
# the text format
# ---------------------------------------------------------------------------

CALL_OPEN, CALL_CLOSE = "<tool>", "</tool>"
RESULT_OPEN, RESULT_CLOSE = "<result>", "</result>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

_CALL_RE = re.compile(re.escape(CALL_OPEN) + r"(.*?)(?:" + re.escape(CALL_CLOSE) + r"|$)", re.DOTALL)
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)(?:" + re.escape(ANSWER_CLOSE) + r"|$)", re.DOTALL)
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(\"[^\"]*\"|'[^']*'|[^,;]+)")

PARAM_TYPES = ("string", "number", "integer", "boolean")
DEFAULT_OBSERVATION_CHARS = 600
"""How much of a tool's output goes back into the transcript by default."""


def task_header(prompt: str) -> str:
    """The first line of every transcript — what the network continues from."""
    return f"TASK: {' '.join(str(prompt).split())}\n"


def call_text(name: str, arguments: dict | None = None) -> str:
    """``<tool>name {"arg": "value"}</tool>`` — one line, JSON arguments."""
    payload = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    return f"{CALL_OPEN}{name} {payload}{CALL_CLOSE}\n"


def result_text(output: str, limit: int | None = DEFAULT_OBSERVATION_CHARS) -> str:
    """``<result>...</result>`` — the observation, whitespace collapsed and clipped."""
    text = " ".join(str(output).split())
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return f"{RESULT_OPEN}{text}{RESULT_CLOSE}\n"


def answer_text(answer: str) -> str:
    """``<answer>...</answer>`` — the last line of a finished transcript."""
    return f"{ANSWER_OPEN}{' '.join(str(answer).split())}{ANSWER_CLOSE}\n"


def format_observation(result: "ToolResult", limit: int | None = DEFAULT_OBSERVATION_CHARS) -> str:
    """What a tool call adds to the transcript: the output, or the error prefixed with ``ERROR:``."""
    return result_text(result.output if result.ok else f"ERROR: {result.error}", limit)


def transcript_text(
    prompt: str, steps: Iterable[tuple[str, dict, "ToolResult"]], answer: str | None = None,
    limit: int | None = DEFAULT_OBSERVATION_CHARS,
) -> str:
    """One attempt as a single training text: the task, every call with its result, then the answer."""
    parts = [task_header(prompt)]
    for name, arguments, result in steps:
        parts.append(call_text(name, arguments))
        parts.append(format_observation(result, limit))
    if answer is not None:
        parts.append(answer_text(answer))
    return "".join(parts)


def _loads_truncated(text: str) -> Any:
    """JSON out of text a generator may have cut off mid-string or mid-object.

    A call is one line by construction, so the line is tried first, then the
    text up to its last ``}``, then the same completed with the closing quote
    and braces it is missing.  Control characters inside strings are tolerated
    (``strict=False``): the network writes newlines wherever it likes.
    """
    head = text.split("\n", 1)[0].strip()
    bases = [head, text] if head != text else [text]  # a call is one line: prefer it over the run-on text
    candidates: list[str] = []
    for base in bases:
        candidates.append(base)
        if "}" in base:
            candidates.append(base[: base.rfind("}") + 1])
        candidates += [base + tail for tail in ('"}', "}", '"}}', "}}", '"]}')]
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except ValueError:
            continue
    return None


def parse_arguments(raw: str, tool: "Tool | None" = None) -> dict:
    """Arguments of a call: JSON, ``key=value`` pairs, or a bare value for a single-argument tool.

    Raises :class:`ToolError` when nothing can be read out of ``raw``.
    """
    text = (raw or "").strip().strip(",;")
    if not text:
        return {}
    if text[0] in "{[":  # JSON, possibly truncated by the generator
        data = _loads_truncated(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and tool is not None:
            return dict(zip([p.name for p in tool.params], data, strict=False))
    pairs = _PAIR_RE.findall(text)
    if pairs:
        return {key: value.strip().strip("\"'") for key, value in pairs}
    if tool is not None:
        wanted = [p for p in tool.params if p.required] or list(tool.params)
        if len(wanted) == 1:
            return {wanted[0].name: text.strip("\"'")}
    raise ToolError(f"cannot read the arguments of the call: {raw.strip()[:120]!r}")


def parse_call(text: str, toolbox: "ToolBox | None" = None) -> "ToolCall | None":
    """The first ``<tool>`` call in ``text`` (``None`` when there is none).

    Lenient on purpose: an unterminated call is read to the end of the text, the
    tool name is whatever leading identifier is there, and the arguments go
    through :func:`parse_arguments`.  A call that cannot be understood is
    returned with ``error`` set rather than dropped, so the caller can hand it
    to the LLM mediator instead of guessing.
    """
    match = _CALL_RE.search(text or "")
    if match is None:
        return None
    body = match.group(1).strip()
    span = match.span()
    name_match = _NAME_RE.match(body)
    if name_match is None:
        return ToolCall("", {}, raw=body, span=span, error=f"no tool name in {body[:80]!r}")
    name = name_match.group(0).rstrip(".-")
    rest = body[name_match.end():].strip()
    tool = toolbox.get(name) if toolbox is not None and toolbox.has(name) else None
    if toolbox is not None and tool is None:
        return ToolCall(name, {}, raw=rest, span=span, error=f"unknown tool {name!r} (have: {', '.join(toolbox.names())})")
    try:
        arguments = parse_arguments(rest, tool)
    except ToolError as exc:
        return ToolCall(name, {}, raw=rest, span=span, error=str(exc))
    return ToolCall(name, arguments, raw=rest, span=span)


def find_answer(text: str) -> str | None:
    """The text of the first ``<answer>`` block, ``None`` when the attempt did not answer."""
    match = _ANSWER_RE.search(text or "")
    if match is None:
        return None
    return " ".join(match.group(1).split()) or None


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A call could not be made: unknown tool, bad arguments, or a refused request."""


@dataclasses.dataclass(slots=True)
class Param:
    """One argument of a tool, with the little bit of schema the LLM needs to fill it in."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(f"param {self.name!r}: type must be one of {', '.join(PARAM_TYPES)} (got {self.type!r})")

    def schema(self) -> dict:
        data: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            data["enum"] = list(self.enum)
        return data

    def coerce(self, value: Any) -> Any:
        """``value`` as this parameter's type; raises :class:`ToolError` when it cannot be."""
        try:
            if self.type == "string":
                out: Any = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            elif self.type == "boolean":
                out = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
            elif self.type == "integer":
                out = int(float(value)) if not isinstance(value, bool) else int(value)
            else:
                out = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"argument {self.name!r} must be a {self.type} (got {value!r})") from exc
        if self.enum and str(out) not in self.enum:
            raise ToolError(f"argument {self.name!r} must be one of {', '.join(self.enum)} (got {out!r})")
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.type, "description": self.description,
            "required": self.required, "default": self.default, "enum": list(self.enum) if self.enum else None,
        }


@dataclasses.dataclass(slots=True)
class ToolCall:
    """A call read out of generated text: the name, the arguments and why it is unusable (``error``)."""

    name: str
    arguments: dict
    raw: str = ""
    span: tuple[int, int] = (0, 0)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.name)

    def text(self) -> str:
        return call_text(self.name, self.arguments)

    def to_dict(self) -> dict:
        return {"tool": self.name, "arguments": self.arguments, "raw": self.raw, "error": self.error}


@dataclasses.dataclass(slots=True)
class ToolResult:
    """What a tool answered: ``output`` when ``ok``, otherwise ``error`` (both readable by the model and the LLM)."""

    tool: str
    arguments: dict
    ok: bool
    output: str = ""
    error: str | None = None
    seconds: float = 0.0
    meta: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool, "arguments": self.arguments, "ok": self.ok, "output": self.output,
            "error": self.error, "seconds": round(self.seconds, 4), "meta": self.meta,
        }


@dataclasses.dataclass(slots=True)
class Tool:
    """A named function the network can call by emitting text.

    ``handler`` takes the validated arguments as keywords and returns a string,
    or ``(string, meta)``.  Raising :class:`ToolError` inside it is the way to
    report a clean failure; any other exception is caught by
    :meth:`ToolBox.call` and reported the same way.
    """

    name: str
    description: str
    params: tuple[Param, ...] = ()
    handler: Callable[..., Any] | None = None
    network: bool = False

    def signature(self) -> str:
        """``name(arg: type, [optional]: type)`` — the one-liner the prompts and the CLI show."""
        parts = [f"{p.name}: {p.type}" if p.required else f"[{p.name}: {p.type}]" for p in self.params]
        return f"{self.name}({', '.join(parts)})"

    def schema(self) -> dict:
        """The tool in Ollama's / OpenAI's ``tools`` format (a JSON-schema function)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.schema() for p in self.params},
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }

    def coerce(self, arguments: dict) -> dict:
        """Validated arguments: types coerced, defaults filled in, unknown names dropped, missing ones refused."""
        if not isinstance(arguments, dict):
            raise ToolError(f"{self.name}: arguments must be an object (got {type(arguments).__name__})")
        known = {p.name: p for p in self.params}
        aliases = {p.name.lower(): p.name for p in self.params}
        out: dict[str, Any] = {}
        for key, value in arguments.items():
            name = str(key) if str(key) in known else aliases.get(str(key).strip().lower())
            if name is None or value is None:
                continue
            out[name] = known[name].coerce(value)
        missing = [p.name for p in self.params if p.required and p.name not in out]
        if missing:
            raise ToolError(f"{self.name}: missing argument(s) {', '.join(missing)} — {self.signature()}")
        for param in self.params:
            if param.name not in out and param.default is not None:
                out[param.name] = param.coerce(param.default)
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description, "signature": self.signature(),
            "network": self.network, "params": [p.to_dict() for p in self.params],
        }


class ToolBox:
    """The registry: what exists, what it looks like to the LLM, and how a call is run.

    :meth:`call` never raises — a failure becomes a :class:`ToolResult` with
    ``ok=False``, because a failed call is training data too (the transcript
    keeps it, and 2NRL punishes the attempt that contained it).
    """

    __slots__ = ("_tools", "calls")

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self.calls = 0
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError("register() takes a Tool")
        if not _NAME_RE.fullmatch(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r} (have: {', '.join(self.names()) or 'none'})") from None

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def describe(self) -> list[dict]:
        return [tool.to_dict() for tool in self._tools.values()]

    def schemas(self) -> list[dict]:
        """Every tool in Ollama's ``tools`` format (what :meth:`OllamaClient.chat` sends)."""
        return [tool.schema() for tool in self._tools.values()]

    def catalogue(self) -> str:
        """The tool list as prompt text: one ``name(args) — description`` line each."""
        return "\n".join(f"- {tool.signature()} — {tool.description}" for tool in self._tools.values())

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        """Run one tool; failures are reported, not raised."""
        t0 = time.perf_counter()
        self.calls += 1
        try:
            tool = self.get(name)
            values = tool.coerce(arguments or {})
            if tool.handler is None:
                raise ToolError(f"{name}: no handler is installed")
            output = tool.handler(**values)
        except ToolError as exc:
            return ToolResult(name, dict(arguments or {}), False, error=str(exc), seconds=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - any tool failure is an observation, never a crash
            return ToolResult(
                name, dict(arguments or {}), False, error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0
            )
        meta: dict = {}
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            output, meta = output
        return ToolResult(
            name, values, True, output="" if output is None else str(output), seconds=time.perf_counter() - t0, meta=meta
        )

    def run(self, call: ToolCall) -> ToolResult:
        """:meth:`call` for a parsed :class:`ToolCall` (a broken one becomes a failed result)."""
        if not call.ok:
            return ToolResult(call.name, call.arguments, False, error=call.error or "unusable tool call")
        return self.call(call.name, call.arguments)


# ---------------------------------------------------------------------------
# browsing: HTML to text
# ---------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    "p div br li tr h1 h2 h3 h4 h5 h6 section article header footer nav aside blockquote pre table "
    "main ul ol dl dt dd figure figcaption form fieldset details summary caption hr address".split()
)
"""Tags that start a new line of the text."""
_CELL_TAGS = frozenset(("td", "th"))
"""Table cells: set off from the cell before them by a space rather than glued to it."""
_DROP_TAGS = frozenset(
    "script style noscript template svg canvas head "
    "nav footer aside dialog button select textarea iframe object audio video datalist".split()
)
"""Tags whose content goes with them: what is never shown, and the page's own chrome and controls."""
_VOID_TAGS = frozenset("area base br col embed hr img input link meta param source track wbr".split())
"""Elements that have no content and no end tag, so one of them can never open a drop: a
``<meta charset="utf-8">`` once did, and every page that has one read as empty."""
_OPTIONAL_END_TAGS = frozenset(
    "html head body p li dt dd tr td th thead tbody tfoot caption colgroup option optgroup rb rt rtc rp".split()
)
"""Elements whose end tag may be left out: hiding one by its attributes could hide the rest of the page."""
_DROP_ROLES = frozenset(
    "navigation banner contentinfo complementary search menu menubar toolbar dialog alertdialog tooltip".split()
)
"""ARIA roles of page chrome, dropped like the elements they stand for."""


def _letters(text: str) -> int:
    """How many letters and digits ``text`` has: what the link density of a line is measured in."""
    return sum(1 for ch in text if ch.isalnum())


def _role(attrs: dict) -> str:
    """An element's ARIA role: the first of the roles it lists, lower-cased (``""`` for none)."""
    roles = (attrs.get("role") or "").lower().split()
    return roles[0] if roles else ""


class _Line:
    """One line of a page as it is read: its pieces, how many of its letters are link text, and where it lies."""

    __slots__ = ("parts", "linked", "main", "article")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.linked = 0
        self.main = self.article = False


class _TextExtractor(HTMLParser):
    """HTML -> readable text, plus the title and the links, with the standard library only.

    The text is the page as a reader meets it rather than as its markup lists it:

    * what is never shown (``<head>``, ``<script>``, an element that is
      ``hidden`` or ``display: none``) and the page's own chrome (``<nav>``,
      ``<footer>``, ``<aside>``, the page's ``<header>``, buttons, menus, the
      ARIA navigation / banner / search roles) go with everything inside
      them, their links included;
    * the lines follow the page's blocks: a paragraph written over several
      lines of markup is one line, and a ``<pre>`` keeps its own;
    * a run of lines made of nothing but link text (a menu, a list of
      languages or of categories) moves after the rest, and so does what lies
      outside the page's ``<main>`` (its ``<article>`` when it has no main).
      So the first few hundred characters are what the page is about, and
      nothing is lost: the rest is still there, further down.

    The links come in the same order, the ones in the running text first.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines = [_Line()]
        self.title = ""
        self.anchors: list[tuple[str, str, int]] = []  # (text, href, the line its text starts on)
        self._drop: str | None = None  # the element being dropped with everything inside it
        self._drop_depth = 0           # how many elements of its name are open, itself included
        self._main: str | None = None  # the element that is the page's main region
        self._main_depth = 0
        self._article = self._section = self._pre = 0
        self._in_title = False
        self._href: str | None = None
        self._anchor: list[str] = []
        self._anchor_line: int | None = None

    # -- lines ----------------------------------------------------------------

    def _break(self) -> None:
        if self.lines[-1].parts:
            self.lines.append(_Line())

    def _space(self) -> None:
        if self.lines[-1].parts:
            self.lines[-1].parts.append(" ")

    def _add(self, data: str) -> None:
        if not data:
            return
        line = self.lines[-1]
        if not line.parts:
            line.main, line.article = self._main is not None, self._article > 0
        line.parts.append(data)
        if self._href is not None:
            letters = _letters(data)
            line.linked += letters
            if letters and self._anchor_line is None:
                self._anchor_line = len(self.lines) - 1

    # -- tags -----------------------------------------------------------------

    def _drops(self, tag: str, attrs: dict) -> bool:
        """Whether an element (not a void one) goes, with everything inside it."""
        if tag in _DROP_TAGS:
            return True
        if tag == "header" and self._main is None and not self._article and not self._section:
            return True  # the page's banner, not the header of an article or a section
        if tag in _OPTIONAL_END_TAGS:
            return False
        if "hidden" in attrs and (attrs["hidden"] or "").strip().lower() != "until-found":
            return True
        if (attrs.get("aria-hidden") or "").strip().lower() == "true":
            return True
        if _role(attrs) in _DROP_ROLES:
            return True
        style = "".join((attrs.get("style") or "").lower().split())
        return "display:none" in style or "visibility:hidden" in style

    def handle_starttag(self, tag: str, attrs: list) -> None:
        values = dict(attrs)
        void = tag in _VOID_TAGS
        if self._main is not None:
            if tag == self._main and not void:
                self._main_depth += 1  # the region counts its own name everywhere, dropped or not
        elif self._drop is None and not void and (tag == "main" or _role(values) == "main"):
            self._break()
            self._main, self._main_depth = tag, 1
        if self._drop is not None:
            if tag == self._drop and not void:
                self._drop_depth += 1
            elif tag == "body" and self._drop == "head":  # the head ends where the body begins, </head> or not
                self._drop = None
            elif tag == "title" and self._drop == "head":  # the title lives inside <head>, which is otherwise dropped
                self._in_title = True
            return
        if not void and self._drops(tag, values):
            if tag in _BLOCK_TAGS:
                self._break()
            self._drop, self._drop_depth = tag, 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            self._href = values.get("href")
            self._anchor, self._anchor_line = [], None
        else:
            if tag == "article":
                self._article += 1
            elif tag == "section":
                self._section += 1
            elif tag == "pre":
                self._pre += 1
            if tag in _BLOCK_TAGS:
                self._break()
            elif tag in _CELL_TAGS:
                self._space()

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag == "br" and self._drop is None:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        void = tag in _VOID_TAGS
        if tag == "title":
            self._in_title = False
        elif tag == "a":
            self._close_anchor()
        if self._drop is not None:
            if tag == self._drop and not void:
                self._drop_depth -= 1
                if not self._drop_depth:
                    self._drop = None
                    if tag in _BLOCK_TAGS:
                        self._break()
        else:
            if tag == "article":
                self._article = max(0, self._article - 1)
            elif tag == "section":
                self._section = max(0, self._section - 1)
            elif tag == "pre":
                self._pre = max(0, self._pre - 1)
            if tag in _BLOCK_TAGS:
                self._break()
        if self._main is not None and tag == self._main and not void:
            self._main_depth -= 1
            if not self._main_depth:
                self._main = None
                self._break()

    def _close_anchor(self) -> None:
        text = " ".join("".join(self._anchor).split())
        if self._href and text:
            line = self._anchor_line if self._anchor_line is not None else len(self.lines) - 1
            self.anchors.append((text, self._href, line))
        self._href, self._anchor, self._anchor_line = None, [], None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._drop is not None:
            return
        if self._href is not None:
            self._anchor.append(data)
        if not self._pre:
            self._add(data)
            return
        for i, piece in enumerate(data.split("\n")):  # a <pre> keeps its lines
            if i:
                self._break()
            self._add(piece)

    # -- the page -------------------------------------------------------------

    def result(self, reading_order: bool = True) -> dict:
        """``{"title", "text", "links"}``: the main text first, then the rest, and the links in the same order.

        With ``reading_order`` off, the lines and the links stay as the page has them.
        """
        texts = [" ".join("".join(line.parts).split()) for line in self.lines]
        shown = [i for i, text in enumerate(texts) if text]
        order, anchors = shown, self.anchors
        if reading_order:
            if any(self.lines[i].main for i in shown):
                primary = [line.main for line in self.lines]
            elif any(self.lines[i].article for i in shown):
                primary = [line.article for line in self.lines]
            else:
                primary = [True] * len(self.lines)
            # a line of nothing but link text is a menu item when the line beside it is one too
            weak = [self.lines[i].linked >= _letters(texts[i]) for i in shown]
            menu = set()
            for k, i in enumerate(shown):
                if weak[k] and ((k > 0 and weak[k - 1]) or (k + 1 < len(shown) and weak[k + 1])):
                    menu.add(i)
            tier = [(2 if i in menu else 0) + (0 if primary[i] else 1) for i in range(len(self.lines))]
            order = sorted(shown, key=lambda i: (tier[i], i))
            anchors = sorted(self.anchors, key=lambda anchor: tier[anchor[2]])
        return {
            "title": " ".join(self.title.split()),
            "text": "\n".join(texts[i] for i in order),
            "links": [(text, href) for text, href, _line in anchors],
        }


def html_to_text(markup: str, *, reading_order: bool = True) -> dict:
    """``{"title", "text", "links": [(text, href)]}`` for a page (never raises on broken HTML).

    The text starts with what the page is about, not with its menus: see
    :class:`_TextExtractor` for what is dropped and what moves to the end.
    ``reading_order=False`` keeps the lines and the links in the order the page
    has them (a search engine's hits are ranked by it).
    """
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup still yields whatever was parsed
        pass
    return parser.result(reading_order)


# ---------------------------------------------------------------------------
# browsing: the HTTP client
# ---------------------------------------------------------------------------


class WebError(ToolError):
    """A page could not be fetched, or the address was refused."""


DEFAULT_USER_AGENT = os.environ.get("RADIXNET_USER_AGENT", "").strip() or "radixnet/0.1 (+https://curtistechsolutions.com)"
DEFAULT_SEARCH_ENGINES = (
    "https://html.duckduckgo.com/html/?q={query}",
    "https://lite.duckduckgo.com/lite/?q={query}",
    "https://en.wikipedia.org/w/api.php?action=query&format=json&formatversion=2&generator=search&gsrlimit=10"
    "&prop=info%7Cextracts&inprop=url&exintro=1&explaintext=1&exsentences=2&exlimit=10&gsrsearch={query}",
)
"""The search endpoints tried in turn by default: DuckDuckGo's HTML page, its lite page, and then
Wikipedia's own search API, which is made for programs and so still answers when a search engine
takes this client for a bot (each of its hits carries the first sentences of the article)."""
DEFAULT_SEARCH_URL = os.environ.get("RADIXNET_SEARCH_URL", "").strip() or " ".join(DEFAULT_SEARCH_ENGINES)
"""Search endpoints, separated by spaces and tried in turn until one has results; ``{query}`` is replaced
by the URL-encoded query. JSON answers are understood too."""


_ADDRESS_KINDS = (
    ("is_unspecified", "an unspecified address (0.0.0.0 usually means DNS is blocking the name)"),
    ("is_loopback", "a loopback address"),
    ("is_link_local", "a link-local address"),
    ("is_multicast", "a multicast address"),
    ("is_private", "a private address"),
    ("is_reserved", "a reserved address"),
)
"""Address kinds the web tools refuse, most diagnosable first (127.0.0.1 is loopback *and* private)."""


def _address_problem(host: str) -> str | None:
    """Why ``host`` must not be fetched, in words, or ``None`` when it is an ordinary public host.

    The three reasons a host is refused are kept apart on purpose: a name that
    does not resolve, a name that resolves to somewhere internal, and a name
    that resolves to something that is not an address at all are very different
    problems, and one message for all three leaves nobody any the wiser.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        return f"cannot be resolved here ({exc.strerror or exc})"
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            return f"resolves to {info[4][0]!r}, which is not an IP address"
        for flag, what in _ADDRESS_KINDS:
            if getattr(address, flag, False):
                return f"resolves to {address}, {what}"
    return None


def _proxy_for(url: str) -> str | None:
    """The proxy this URL would go through (``None`` when it goes out directly).

    When there is one, *it* resolves the host name — this process may have no
    DNS at all, which is the normal case inside Docker and behind a corporate
    or sandbox proxy.
    """
    parts = urllib.parse.urlsplit(url)
    proxy = urllib.request.getproxies().get(parts.scheme)
    if not proxy:
        return None
    try:
        if urllib.request.proxy_bypass(parts.netloc):
            return None
    except (OSError, ValueError, AttributeError):
        pass
    return proxy


class WebClient:
    """Fetches pages with the standard library, refusing everything that is not a plain public web page.

    ``http`` / ``https`` only, no credentials in the URL, at most ``max_bytes``
    read, ``timeout`` seconds per request, redirects followed by hand (at most
    ``max_redirects``) so every hop is checked again, and — unless
    ``allow_private`` — no address inside a private, loopback or link-local
    range.
    """

    __slots__ = ("timeout", "max_bytes", "max_redirects", "user_agent", "allow_private", "search_url", "fetched",
                 "browser")

    def __init__(
        self,
        timeout: float = 20.0,
        max_bytes: int = 2_000_000,
        max_redirects: int = 4,
        user_agent: str | None = None,
        allow_private: bool = False,
        search_url: str | None = None,
        browser: Any = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if max_bytes < 1024:
            raise ValueError("max_bytes must be >= 1024")
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)
        self.max_redirects = max(0, int(max_redirects))
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.allow_private = bool(allow_private)
        self.search_url = " ".join((search_url or "").split()) or DEFAULT_SEARCH_URL
        self.fetched = 0
        self.browser = browser
        """Optional :class:`~radixnet.browser.BrowserClient`: pages are then drawn by a real Chrome,
        so a site that renders itself with JavaScript is readable.  The address guards run first
        either way."""

    # -- transport -----------------------------------------------------------

    def check(self, url: str) -> str:
        """The normalised URL, or :class:`WebError` explaining why it is refused."""
        text = (url or "").strip().strip("<>\"'")
        if not text:
            raise WebError("the URL is empty")
        if "://" not in text:
            text = "https://" + text
        parts = urllib.parse.urlsplit(text)
        if parts.scheme not in ("http", "https"):
            raise WebError(f"only http and https are allowed (got {parts.scheme or 'no scheme'})")
        if parts.username or parts.password:
            raise WebError("URLs with credentials are refused")
        if not parts.hostname:
            raise WebError(f"no host in {text!r}")
        if not self.allow_private:
            problem = _address_problem(parts.hostname)
            # A host this process cannot resolve is still fetchable when a proxy does the resolving,
            # and refusing it there would make browsing impossible rather than safer.
            if problem and problem.startswith("cannot be resolved") and _proxy_for(text):
                problem = None
            if problem:
                raise WebError(
                    f"refusing {parts.hostname}: it {problem}"
                    + (" - pass allow_private (--allow-private) to fetch it anyway"
                       if not problem.startswith("cannot be resolved") else "")
                )
        return urllib.parse.urlunsplit(parts)

    def fetch(self, url: str) -> dict:
        """``{"url", "status", "content_type", "bytes", "truncated", "body"}`` for one page."""
        target = self.check(url)
        if self.browser is not None:
            return self._fetch_with_browser(target)
        seen = [target]
        for _hop in range(self.max_redirects + 1):
            request = urllib.request.Request(
                target, method="GET",
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1",
                    "Accept-Language": "en",
                },
            )
            opener = urllib.request.build_opener(_NoRedirect)
            try:
                response = opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                with exc:  # an HTTPError holds the (still open) response body
                    if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                        target = self.check(urllib.parse.urljoin(target, exc.headers["Location"]))
                        if target in seen:
                            raise WebError(f"redirect loop at {target}") from exc
                        seen.append(target)
                        continue
                    raise WebError(
                        f"{urllib.parse.urlsplit(target).hostname} answered HTTP {exc.code} ({exc.reason})"
                    ) from exc
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                reason = getattr(exc, "reason", None) or exc
                raise WebError(f"cannot fetch {target}: {reason}") from exc
            with response:
                raw = response.read(self.max_bytes + 1)
                status = response.status
                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                charset = response.headers.get_content_charset() or "utf-8"
                final = response.geturl() or target
            truncated = len(raw) > self.max_bytes
            try:
                body = raw[: self.max_bytes].decode(charset, errors="replace")
            except LookupError:  # a charset Python has never heard of: read it as UTF-8 rather than fail
                body = raw[: self.max_bytes].decode("utf-8", errors="replace")
            self.fetched += 1
            return {
                "url": final, "status": status, "content_type": content_type or "text/html",
                "bytes": len(raw[: self.max_bytes]), "truncated": truncated, "body": body,
            }
        raise WebError(f"too many redirects starting at {url}")

    def _fetch_with_browser(self, target: str) -> dict:
        """The same answer as :meth:`fetch`, drawn by Chrome (see :mod:`radixnet.browser`).

        WebDriver does not report the HTTP status, so a page that loaded is
        ``200``; a page that did not raises instead.
        """
        from .browser import BrowserError

        try:
            page = self.browser.get(target)
        except BrowserError as exc:
            raise WebError(f"cannot open {target} in the browser: {exc}") from exc
        body = page["html"]
        truncated = len(body) > self.max_bytes
        self.fetched += 1
        return {
            "url": page["url"], "status": page.get("status", 200), "content_type": "text/html",
            "bytes": len(body[: self.max_bytes].encode("utf-8", "replace")), "truncated": truncated,
            "body": body[: self.max_bytes],
        }

    # -- pages ---------------------------------------------------------------

    def page(self, url: str) -> dict:
        """A fetched page reduced to text: ``{"url", "title", "text", "links", "status", "content_type", "truncated"}``.

        The links are absolute ``http(s)`` addresses, each once, and none of them
        back into this same page (a table of contents, a "back to top").
        """
        response = self.fetch(url)
        kind = response["content_type"]
        if kind in ("text/html", "application/xhtml+xml", ""):
            page = html_to_text(response["body"])
            here = response["url"].split("#", 1)[0]
            links: list[tuple[str, str]] = []
            seen: set[str] = set()
            for text, href in page["links"]:
                href = urllib.parse.urljoin(response["url"], href)
                if urllib.parse.urlsplit(href).scheme not in ("http", "https") or href in seen:
                    continue
                if href.split("#", 1)[0] == here:
                    continue
                seen.add(href)
                links.append((text, href))
        elif kind == "application/json":
            page = {"title": "", "text": response["body"]}
            links = []
        else:
            page = {"title": "", "text": html.unescape(response["body"])}
            links = []
        return {
            "url": response["url"], "title": page["title"], "text": page["text"].strip(), "links": links,
            "status": response["status"], "content_type": kind, "truncated": response["truncated"],
        }

    # -- search --------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list["SearchResult"]:
        """Search results for ``query`` from the first endpoint of :attr:`search_url` that has any.

        :attr:`search_url` may list several endpoints separated by spaces, and
        the default does (:data:`DEFAULT_SEARCH_ENGINES`).  An endpoint that
        refuses rather than answers — an HTTP error, a ``202 Accepted``, a
        CAPTCHA where the results should be — is passed over for the
        next one, and when every one of them refused, the error says what each
        answered.  A JSON answer is read as such (SearxNG, MediaWiki,
        OpenSearch, ...), anything else as the engine's HTML.
        """
        text = (query or "").strip()
        if not text:
            raise WebError("the search query is empty")
        quoted = urllib.parse.quote_plus(text)
        endpoints = self.search_url.split()
        refusals: list[str] = []
        answered = False
        for endpoint in endpoints:
            url = endpoint.replace("{query}", quoted) if "{query}" in endpoint else (
                endpoint + ("&" if "?" in endpoint else "?") + "q=" + quoted
            )
            try:
                results = self._search_at(url)
            except WebError as exc:
                if len(endpoints) == 1:
                    raise
                refusals.append(str(exc))
                continue
            if results:
                return results[:limit]
            answered = True
        if answered or not refusals:
            return []
        raise WebError("no search engine answered: " + "; ".join(refusals))

    def _search_at(self, url: str) -> list["SearchResult"]:
        """The hits one search endpoint answers with; :class:`WebError` when it refuses instead."""
        response = self.fetch(url)
        host = urllib.parse.urlsplit(response["url"]).hostname or response["url"]
        if response["status"] == 202:  # "Accepted", and not answered: DuckDuckGo's way of saying no
            raise WebError(
                f"{host} answered HTTP {response['status']} instead of results"
                " - it may be turning automated searches away"
            )
        results = _search_results(response["body"], response["content_type"], response["url"])
        if not results and _is_challenge(response["body"]):
            raise WebError(f"{host} answered with a CAPTCHA instead of results - it takes this client for a bot")
        return results


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turns a redirect into an :class:`~urllib.error.HTTPError` so :meth:`WebClient.fetch` can check the new host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - base-class signature
        return None


@dataclasses.dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _search_results(body: str, content_type: str, base: str) -> list[SearchResult]:
    """The hits in one answer of a search endpoint: a JSON API's results, else the links out of the engine's page."""
    if content_type == "application/json" or body.lstrip()[:1] in ("{", "["):
        results = _search_from_json(body)
        if results:
            return results
    page = html_to_text(body, reading_order=False)  # the hits in the engine's own ranking
    results = _search_from_links(page["links"], base)
    if not results and page["text"][:1] in ("{", "["):  # a JSON answer that a browser drew as a page
        results = _search_from_json(page["text"])
    return results


def _search_from_json(body: str) -> list[SearchResult]:
    """Results out of a JSON search API: SearxNG, Brave, MediaWiki (``query.pages``), OpenSearch and the like."""
    try:
        data = json.loads(body)
    except ValueError:
        return []
    items: Any = None
    if isinstance(data, dict):
        for key in ("results", "items", "webPages", "web", "data", "query"):
            value = data.get(key)
            if isinstance(value, dict):
                value = next((value[k] for k in ("value", "results", "pages", "search") if k in value), None)
            if isinstance(value, list):
                items = value
                break
    elif isinstance(data, list):
        if len(data) >= 4 and isinstance(data[0], str) and all(isinstance(part, list) for part in data[1:4]):
            # OpenSearch suggestions: [query, [titles], [descriptions], [urls]]
            items = [{"title": t, "description": d, "url": u} for t, d, u in zip(data[1], data[2], data[3])]
        else:
            items = data
    if not isinstance(items, list):
        return []
    if items and all(isinstance(item, dict) and _is_number(item.get("index")) for item in items):
        items = sorted(items, key=lambda item: item["index"])  # MediaWiki lists the pages of a search out of rank
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = next(
            (item[k] for k in ("url", "link", "href", "fullurl", "canonicalurl") if isinstance(item.get(k), str)), None
        )
        if not url:
            continue
        title = next((item[k] for k in ("title", "name", "heading") if isinstance(item.get(k), str)), url)
        snippet = next(
            (item[k] for k in ("content", "snippet", "description", "body", "summary", "extract")
             if isinstance(item.get(k), str)), ""
        )
        results.append(SearchResult(" ".join(title.split()), url, " ".join(snippet.split())[:300]))
    return results


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_DDG_REDIRECT = re.compile(r"^(?:https?:)?//duckduckgo\.com/l/\?uddg=", re.I)


def _search_from_links(links: list[tuple[str, str]], base: str) -> list[SearchResult]:
    """Results out of a search engine's page: the links that leave the engine, each once.

    An engine links a hit more than once — its title, its address, a snippet of
    the page — so a later link to a hit already listed lends it its text as the
    snippet: the longest such text that has a space in it (an address has
    none), up to 300 characters.
    """
    engine = urllib.parse.urlsplit(base).hostname or ""
    results: list[SearchResult] = []
    listed: dict[str, SearchResult] = {}
    for title, href in links:
        if _DDG_REDIRECT.match(href):  # DuckDuckGo wraps every hit in a redirect
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
            href = (query.get("uddg") or [""])[0]
        href = urllib.parse.urljoin(base, href)
        parts = urllib.parse.urlsplit(href)
        host = parts.hostname or ""
        if parts.scheme not in ("http", "https") or not host:
            continue
        if host == engine or engine.endswith("." + host) or host.endswith("." + engine):
            continue
        if host == "duckduckgo.com" or host.endswith(".duckduckgo.com"):  # its ads (y.js), its feedback page
            continue
        hit = listed.get(href)
        if hit is not None:
            snippet = title[:300]
            if " " in snippet and title[:200] != hit.title and len(snippet) > len(hit.snippet):
                hit.snippet = snippet
            continue
        if len(title) < 3:
            continue
        listed[href] = hit = SearchResult(title[:200], href)
        results.append(hit)
    return results


_CHALLENGE_MARKERS = (
    "captcha", "anomaly-modal", "unusual traffic", "are you a robot", "not a robot", "bots use duckduckgo",
)
"""What a search engine's "are you a human?" page says where its results should be."""


def _is_challenge(body: str) -> bool:
    lower = body.lower()
    return any(marker in lower for marker in _CHALLENGE_MARKERS)


# ---------------------------------------------------------------------------
# the calculator
# ---------------------------------------------------------------------------

_MATH_FUNCTIONS: dict[str, Any] = {
    name: getattr(math, name)
    for name in ("sqrt", "floor", "ceil", "log", "log2", "log10", "exp", "sin", "cos", "tan", "atan", "hypot", "fabs", "pow")
}
_MATH_FUNCTIONS.update(abs=abs, round=round, min=min, max=max, sum=sum, int=int, float=float)
_MATH_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name, ast.Load, ast.Tuple, ast.List,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.Not,
)


def safe_eval(expression: str) -> float | int | bool:
    """Evaluate an arithmetic expression — numbers, operators and the :mod:`math` functions, nothing else."""
    text = (expression or "").strip()
    if not text:
        raise ToolError("the expression is empty")
    if len(text) > 500:
        raise ToolError("the expression is too long (500 characters max)")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not an arithmetic expression: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ToolError(f"{type(node).__name__} is not allowed in an expression")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCTIONS):
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ToolError(f"unknown function {name!r} (have: {', '.join(sorted(_MATH_FUNCTIONS))})")
        if isinstance(node, ast.Name) and node.id not in _MATH_FUNCTIONS and node.id not in _MATH_NAMES:
            raise ToolError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ToolError("only numbers are allowed in an expression")
    try:
        return eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, {**_MATH_FUNCTIONS, **_MATH_NAMES})
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"cannot evaluate the expression: {exc}") from exc


# ---------------------------------------------------------------------------
# the built-in tools
# ---------------------------------------------------------------------------


def _tool_web_search(web: WebClient) -> Tool:
    def handler(query: str, limit: int = 5) -> tuple[str, dict]:
        limit = max(1, min(int(limit), 10))
        results = web.search(query, limit)
        if not results:
            return f"no results for {query!r}", {"results": []}
        lines = [f"{i}. {r.title} — {r.url}" + (f" — {r.snippet}" if r.snippet else "") for i, r in enumerate(results, 1)]
        return "\n".join(lines), {"results": [r.to_dict() for r in results]}

    return Tool(
        "web_search", "Search the web and get back a numbered list of titles with their URLs.",
        (Param("query", "string", "what to search for"),
         Param("limit", "integer", "how many results (1-10)", required=False, default=5)),
        handler, network=True,
    )


def _tool_web_fetch(web: WebClient) -> Tool:
    def handler(url: str, max_chars: int = 2000) -> tuple[str, dict]:
        page = web.page(url)
        text = page["text"]
        limit = max(100, min(int(max_chars), 20000))
        body = text[:limit] + (" ..." if len(text) > limit else "")
        header = f"{page['title']} — {page['url']}\n" if page["title"] else f"{page['url']}\n"
        return header + body, {
            "url": page["url"], "title": page["title"], "status": page["status"],
            "chars": len(text), "truncated": page["truncated"] or len(text) > limit,
            "link_count": len(page["links"]),
            # the pages this one leads to: what an exploration follows next
            "links": [{"text": t, "url": h} for t, h in page["links"][:40]],
        }

    return Tool(
        "web_fetch", "Open a web page and read it as plain text (the title, then the body).",
        (Param("url", "string", "the address of the page"),
         Param("max_chars", "integer", "how much of the page to read", required=False, default=2000)),
        handler, network=True,
    )


def _tool_web_links(web: WebClient) -> Tool:
    def handler(url: str, limit: int = 20) -> tuple[str, dict]:
        page = web.page(url)
        links = page["links"][: max(1, min(int(limit), 100))]
        if not links:
            return f"no links on {page['url']}", {"links": []}
        return (
            "\n".join(f"{i}. {text} — {href}" for i, (text, href) in enumerate(links, 1)),
            {"links": [{"text": t, "url": h} for t, h in links], "url": page["url"]},
        )

    return Tool(
        "web_links", "List the links on a web page, so another page can be opened from it.",
        (Param("url", "string", "the address of the page"),
         Param("limit", "integer", "how many links (1-100)", required=False, default=20)),
        handler, network=True,
    )


def _tool_calculator() -> Tool:
    def handler(expression: str) -> str:
        return str(safe_eval(expression))

    return Tool(
        "calculator", "Work out an arithmetic expression, for example 2 * (3 + 4) or sqrt(841).",
        (Param("expression", "string", "the expression to evaluate"),), handler,
    )


def _tool_python(sandbox: Any) -> Tool:
    def handler(code: str) -> tuple[str, dict]:
        run = sandbox.run(code)
        if run.ok:
            return (run.stdout.strip() or "(the program printed nothing)"), {"seconds": run.seconds, "exit_code": run.exit_code}
        raise ToolError(f"the program failed: {run.error}\n{run.stderr[-400:]}".strip())

    return Tool(
        "python", "Run a short Python program in a sandbox and get back what it printed.",
        (Param("code", "string", "the program to run"),), handler,
    )


def _tool_read_file(upload_dir: str) -> Tool:
    def handler(name: str, max_chars: int = 2000) -> tuple[str, dict]:
        safe = os.path.basename(str(name).strip().replace("\\", "/"))
        if not safe or safe in (".", ".."):
            raise ToolError(f"{name!r} is not a file name")
        path = os.path.join(upload_dir, safe)
        if not os.path.isfile(path):
            have = sorted(f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f)))[:20]
            raise ToolError(f"no uploaded file named {safe!r} (have: {', '.join(have) or 'none'})")
        limit = max(100, min(int(max_chars), 100000))
        with open(path, "rb") as fh:
            blob = fh.read(limit + 1)
        text = blob.decode("utf-8", errors="replace").lstrip("﻿")
        return text[:limit] + (" ..." if len(text) > limit else ""), {"name": safe, "chars": len(text)}

    return Tool(
        "read_file", "Read one of the files uploaded to the server.",
        (Param("name", "string", "the file name"),
         Param("max_chars", "integer", "how much to read", required=False, default=2000)),
        handler,
    )


def default_toolbox(
    web: WebClient | None = None,
    *,
    sandbox: Any = None,
    upload_dir: str | None = None,
    offline: bool = False,
    browser: Any = None,
    extra: Iterable[Tool] = (),
) -> ToolBox:
    """The built-in tools: browsing (unless ``offline``), the calculator, and — when given — ``python`` / ``read_file``.

    ``web`` defaults to a fresh :class:`WebClient`; pass one to change the
    timeout, the byte cap, the search endpoint or ``allow_private``.
    ``browser`` is a :class:`~radixnet.browser.BrowserClient` the pages are
    drawn in, for sites that render themselves with JavaScript.
    """
    box = ToolBox()
    if not offline:
        client = web or WebClient()
        if browser is not None:
            client.browser = browser
        box.register(_tool_web_search(client))
        box.register(_tool_web_fetch(client))
        box.register(_tool_web_links(client))
    box.register(_tool_calculator())
    if sandbox is not None:
        box.register(_tool_python(sandbox))
    if upload_dir:
        box.register(_tool_read_file(upload_dir))
    for tool in extra:
        box.register(tool)
    return box
