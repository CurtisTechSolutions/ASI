"""An MCP server: the network's tools *and* the network itself, over the Model Context Protocol.

Two things are exposed to any MCP client (Claude Desktop, an editor, another
agent), and the second is the interesting one:

* the **tools** of :mod:`radixnet.tools` — ``web_search``, ``web_fetch``,
  ``web_links``, ``calculator``, and ``python`` / ``read_file`` when they are
  switched on.  The same registry the network calls by writing text, offered to
  whoever else wants it;
* the **model** — ``radixnet_predict``, ``radixnet_generate``,
  ``radixnet_score``, ``radixnet_stats``, ``radixnet_solve`` (one task through
  the whole agent loop: criteria, tool calls, judging) and ``radixnet_judge``
  (the negative network's verdict on a text: has this way of going wrong been
  seen before, and why).  A client can therefore ask *this* network what it
  thinks, not just borrow its browser.

MCP is JSON-RPC 2.0 over a stream, so this is the standard library and nothing
else: newline-delimited JSON on stdin and stdout (``radixnet mcp``), which is
how MCP's stdio transport works.  ``initialize``, ``tools/list`` and
``tools/call`` are implemented, plus ``ping`` and the ``notifications/*`` a
client sends and expects no answer to.

Anything written to stdout that is not a response would corrupt the stream, so
nothing here prints: the log goes to stderr.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from .tools import Tool, ToolBox, ToolError

__all__ = [
    "PROTOCOL_VERSION",
    "SERVER_NAME",
    "McpServer",
    "model_tools",
    "serve_stdio",
]

PROTOCOL_VERSION = "2024-11-05"
"""The MCP revision this server speaks (clients may ask for another; theirs is echoed when it is newer)."""
SERVER_NAME = "radixnet"

# JSON-RPC error codes (the spec's, plus MCP's use of them)
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _schema(tool: Tool) -> dict:
    """One tool in MCP's shape (``inputSchema`` rather than ``function.parameters``)."""
    function = tool.schema()["function"]
    return {"name": tool.name, "description": tool.description, "inputSchema": function["parameters"]}


# ---------------------------------------------------------------------------
# the model as tools
# ---------------------------------------------------------------------------


def model_tools(
    service: Any,
    *,
    toolbox: ToolBox | None = None,
    client: Any = None,
    negative: Any = None,
    prefix: str = "radixnet_",
) -> list[Tool]:
    """The network's own operations as :class:`~radixnet.tools.Tool` objects.

    ``service`` is anything with the model behind it: a
    :class:`~radixnet.api.ModelService` (whose lock is then respected) or a
    bare model.  ``toolbox`` and ``client`` enable ``solve``, ``negative``
    enables ``judge``; each is left out when what it needs is missing.
    """
    from .tools import Param

    def with_model(work: Callable[[Any], Any]) -> Any:
        session = getattr(service, "session", None)
        if session is None:
            return work(service)
        with session() as model:
            return work(model)

    def predict(prefix_text: str, length: int = 40, mode: str = "dijkstra", temperature: float = 1.0) -> str:
        result = with_model(lambda m: m.predict(prefix_text, length=length, mode=mode, temperature=temperature,
                                                max_length=max(length, 1)))
        return result.full_text

    def generate(count: int = 3, max_length: int = 80, temperature: float = 1.0) -> str:
        results = with_model(lambda m: m.generate(max_length=max_length, mode="sample",
                                                  temperature=temperature, count=count))
        return "\n".join(f"{i}. {r.text}" for i, r in enumerate(results, 1)) or "(the network generated nothing)"

    def score(text: str) -> str:
        data = with_model(lambda m: m.score(text))
        return (f"log probability {data['log_prob']:.3f}, {data['per_char']:.4f} per character over "
                f"{data['chars']} characters ({data['unknown_transitions']} unknown transitions)")

    def stats() -> str:
        return json.dumps(with_model(lambda m: m.stats()), indent=2, default=str)

    tools = [
        Tool(f"{prefix}predict", "Continue a prefix with the RadixCyclicNN network.",
             (Param("prefix_text", "string", "the text to continue"),
              Param("length", "integer", "characters to add", required=False, default=40),
              Param("mode", "string", "dijkstra (cheapest path), beam or sample", required=False, default="dijkstra",
                    enum=("dijkstra", "beam", "sample")),
              Param("temperature", "number", "sampling temperature", required=False, default=1.0)),
             predict),
        Tool(f"{prefix}generate", "Generate whole texts from the network.",
             (Param("count", "integer", "how many", required=False, default=3),
              Param("max_length", "integer", "characters per text", required=False, default=80),
              Param("temperature", "number", "sampling temperature", required=False, default=1.0)),
             generate),
        Tool(f"{prefix}score", "How likely the network thinks a text is (log probability per character).",
             (Param("text", "string", "the text to score"),), score),
        Tool(f"{prefix}stats", "The network's size, compression, training history and backend.", (), stats),
    ]

    if negative is not None:
        def judge(text: str) -> str:
            verdict = negative.judge(text)
            reasons = ", ".join(f"{r['reason']} ({r['blame']:.1f})" for r in verdict.get("reasons") or ()) or "none"
            spans = "; ".join(str(s.get("fragment")) for s in (verdict.get("spans") or ())[:3])
            return (f"{verdict['verdict']}: {verdict['why']}\nrisk {verdict['risk']:.2f}, coverage "
                    f"{verdict['coverage']:.2f}\nreasons: {reasons}" + (f"\nworst fragments: {spans}" if spans else ""))

        tools.append(Tool(
            f"{prefix}judge",
            "Ask the negative network whether a text looks like something that has gone wrong before, and why.",
            (Param("text", "string", "the text to judge"),), judge,
        ))

    if toolbox is not None and client is not None:
        def solve(task: str, max_steps: int = 4, source: str = "model") -> str:
            from .agent import AgentConfig, AgentTrainer, Task

            config = AgentConfig(max_steps=max_steps, model_attempts=1, teach_on_failure=False)
            problem = Task("mcp", task)
            trainer = AgentTrainer(
                getattr(service, "model", service), client, toolbox, config, negative=negative,
            )
            criteria = trainer.criteria_for(problem)
            attempt = (trainer.solve_with_teacher(problem, criteria, 0) if source == "teacher"
                       else trainer.solve_with_model(problem, 0, None, "model"))
            attempt.verdict = trainer.judge(problem, criteria, attempt)
            marks = "\n".join(f"  - {c}" for c in criteria)
            return (f"answer: {attempt.answer or '(none)'}\nverdict: "
                    f"{'correct' if attempt.verdict.correct else 'failed'}"
                    f"{'' if attempt.verdict.score is None else f' (score {attempt.verdict.score:g})'}\n"
                    f"criteria:\n{marks}\n\ntranscript:\n{attempt.text}")

        tools.append(Tool(
            f"{prefix}solve",
            "Put one task through the whole agent loop: acceptance criteria, tool calls and a judged answer.",
            (Param("task", "string", "the question to settle"),
             Param("max_steps", "integer", "tool calls allowed", required=False, default=4),
             Param("source", "string", "model (the network attempts it) or teacher (the LLM demonstrates)",
                   required=False, default="model", enum=("model", "teacher"))),
            solve, network=True,
        ))
    return tools


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------


class McpServer:
    """JSON-RPC 2.0 over a byte stream: MCP's stdio transport, with one toolbox behind it.

    :meth:`handle` turns one request object into one response object (or
    ``None`` for a notification, which must not be answered), so the protocol
    can be exercised without any streams at all — which is how it is tested.
    """

    def __init__(self, toolbox: ToolBox, name: str = SERVER_NAME, version: str = "", instructions: str = "") -> None:
        if not len(toolbox):
            raise ValueError("the toolbox is empty: an MCP server with no tools is of no use")
        self.toolbox = toolbox
        self.name = name
        if not version:
            from . import __version__

            version = __version__
        self.version = version
        self.instructions = instructions or (
            "The tools of a RadixCyclicNN instance: browsing and a calculator, plus the network's own "
            "predictions, generations, scores and judgements."
        )
        self.initialized = False
        self.calls = 0

    # -- one message ---------------------------------------------------------

    def handle(self, message: Any) -> dict | None:
        """One JSON-RPC request -> one response (``None`` for a notification)."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        method = message.get("method")
        ident = message.get("id")
        if not isinstance(method, str):
            return _error(ident, INVALID_REQUEST, "no method")
        params = message.get("params")
        if params is None or params == []:
            params = {}  # JSON-RPC allows positional params; MCP never uses them, and an empty list means none
        if not isinstance(params, dict):
            return _error(ident, INVALID_PARAMS, "params must be an object")
        if ident is None:  # a notification: acted on, never answered
            if method == "notifications/initialized":
                self.initialized = True
            return None
        try:
            handler = {
                "initialize": self._initialize,
                "ping": lambda _params: {},
                "tools/list": self._tools_list,
                "tools/call": self._tools_call,
            }.get(method)
            if handler is None:
                return _error(ident, METHOD_NOT_FOUND, f"unknown method {method!r}")
            return {"jsonrpc": "2.0", "id": ident, "result": handler(params)}
        except ToolError as exc:
            return _error(ident, INVALID_PARAMS, str(exc))
        except Exception as exc:  # noqa: BLE001 - a crash here must not end the session
            return _error(ident, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _initialize(self, params: dict) -> dict:
        self.initialized = True
        asked = params.get("protocolVersion")
        return {
            "protocolVersion": asked if isinstance(asked, str) and asked else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
            "instructions": self.instructions,
        }

    def _tools_list(self, params: dict) -> dict:
        return {"tools": [_schema(tool) for tool in self.toolbox.tools()]}

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ToolError("tools/call needs a tool 'name'")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ToolError("'arguments' must be an object")
        if not self.toolbox.has(name):
            raise ToolError(f"unknown tool {name!r} (have: {', '.join(self.toolbox.names())})")
        result = self.toolbox.call(name, arguments)
        self.calls += 1
        # a tool that failed is a result with isError, not a protocol error: the client shows it to its model
        text = result.output if result.ok else f"ERROR: {result.error}"
        return {"content": [{"type": "text", "text": text}], "isError": not result.ok}

    # -- the stream ----------------------------------------------------------

    def run(self, stream_in: Any = None, stream_out: Any = None, log: Any = None) -> int:
        """Read newline-delimited JSON requests and write the responses until the input ends."""
        stream_in = stream_in if stream_in is not None else sys.stdin
        stream_out = stream_out if stream_out is not None else sys.stdout
        for line in stream_in:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError as exc:
                self._write(stream_out, _error(None, PARSE_ERROR, f"invalid JSON: {exc}"))
                continue
            try:
                response = self.handle(message)
            except Exception:  # noqa: BLE001 - never let one message end the session
                if log is not None:
                    log.write(traceback.format_exc())
                response = _error(message.get("id") if isinstance(message, dict) else None,
                                  INTERNAL_ERROR, "the server failed to handle the message")
            if response is not None:
                self._write(stream_out, response)
        return 0

    @staticmethod
    def _write(stream: Any, payload: dict) -> None:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        stream.flush()


def _error(ident: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def serve_stdio(toolbox: ToolBox, extra: Iterable[Tool] = (), **options: Any) -> int:
    """Serve MCP on stdin / stdout until the client closes it."""
    for tool in extra:
        toolbox.register(tool)
    return McpServer(toolbox, **options).run(log=sys.stderr)
