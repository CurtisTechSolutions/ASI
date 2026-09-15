"""MCP servers as tools, in the same embedding layer as everything else.

The Model Context Protocol is how a capability that lives in another process
announces itself: a server exposes named tools with JSON Schemas, and a client
calls them. That is exactly the shape of `toolsmith.ToolSpec`, so MCP tools are
not given a parallel registry. They are embedded into the same memory, retrieved
by the same graded recall, and invoked through the same `Toolbox.invoke`.

The point of unifying them is retrieval. A goal like "read the open issues on
this repo" should surface an MCP server's `list_issues` and a locally forged
Python helper *in the same ranked list*, because at the moment of recall the
question is "what can act on this?" and the transport is an implementation
detail. Keeping two registries would mean every caller has to remember to search
both, and the one they forget is the one that had the answer.

**Transport.** stdio only: JSON-RPC 2.0 messages, one per line, over a
subprocess's stdin/stdout. That is the common local case and it needs nothing but
`subprocess` and `json`. HTTP/SSE servers are out of scope here and would be a
second `Transport` class rather than a change to this one.

**The handshake is not optional.** `initialize`, wait for the result, then send
the `notifications/initialized` notification, and only then `tools/list`. A server
that receives `tools/list` before the notification is entitled to ignore it, and
the resulting hang looks like a broken server rather than a protocol error.

**Everything here degrades instead of raising.** A server that will not start, a
tool that times out, a malformed frame -- each returns a result object saying so.
An agent whose memory layer crashes because one of a dozen configured servers is
down is worse than one that records the outage as a fact and carries on.

**Grading.** An MCP tool cannot be verified the way a forged Python tool can --
there are no contract tests to run, and the server's behaviour is not this
system's to check. So MCP tools enter memory **ungraded**, at the credibility
prior, and earn or lose standing through `Toolbox.record_use` like anything else.
Registering them as verified would be claiming evidence that does not exist.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .memory import Kind, Source

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "distil", "version": "1"}


@dataclass
class McpTool:
    server: str
    name: str
    description: str = ""
    schema: dict = field(default_factory=dict)

    @property
    def qualified(self) -> str:
        """`server.tool`. Namespaced because two servers may both expose `search`
        and a flat name space would make the second one unreachable."""
        return f"{self.server}.{self.name}"

    def arguments(self) -> list[str]:
        return sorted((self.schema.get("properties") or {}).keys())

    def required(self) -> list[str]:
        return list(self.schema.get("required") or [])

    def embed_text(self) -> str:
        """What the embedding layer indexes.

        Name, description, argument names and the server -- the same shape as a
        forged tool's text, so the two are comparable in one ranked list. The
        schema's *property names* are included because they carry most of the
        signal about what a tool operates on: `owner`, `repo`, `pullNumber` says
        more about `get_pull_request` than its one-line description usually does.
        """
        parts = [f"tool {self.qualified}: {self.description or self.name}",
                 f"arguments: {', '.join(self.arguments()) or 'none'}",
                 f"provided by the {self.server} mcp server"]
        if self.required():
            parts.append(f"required: {', '.join(self.required())}")
        return "\n".join(parts)

    def to_json(self) -> dict:
        return {"server": self.server, "name": self.name, "description": self.description,
                "schema": self.schema}


@dataclass
class McpResult:
    ok: bool
    content: str = ""
    raw: dict = field(default_factory=dict)
    error: str = ""

    def summary(self) -> str:
        return self.content[:400] if self.ok else f"error: {self.error}"


class McpServer:
    """One stdio MCP server subprocess.

    Not a context manager by accident -- `close()` is idempotent and `__exit__`
    calls it, because a server left running after an exception is a leaked
    process that holds a port or a file lock and is invisible until something
    else fails.
    """

    def __init__(self, name: str, command: list[str], env: dict | None = None,
                 cwd: str | None = None, timeout: float = 30.0) -> None:
        self.name = name
        self.command = command
        self.env = env or {}
        self.cwd = cwd
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self.server_info: dict = {}
        self.started = False
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> McpResult:
        if self.started:
            return McpResult(True, "already started")
        env = {**os.environ, **self.env}
        try:
            self.proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, env=env, cwd=self.cwd)
        except (OSError, ValueError) as exc:
            return McpResult(False, error=f"could not start {self.name}: {exc}")

        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

        init = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO})
        if not init.ok:
            self.close()
            return init
        self.server_info = init.raw.get("result", {})
        # The notification that completes the handshake. A server is entitled to
        # ignore tools/list until it arrives, and that hang reads as a broken
        # server rather than a missing message.
        self._notify("notifications/initialized")
        self.started = True
        return McpResult(True, f"{self.name} ready: "
                               f"{self.server_info.get('serverInfo', {}).get('name', 'unknown')}")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except (OSError, ValueError):
            pass
        finally:
            self.proc = None
            self.started = False
            self._reader = None
            self._lines = queue.Queue()

    def __enter__(self) -> "McpServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the wire -------------------------------------------------------------

    def _send(self, payload: dict) -> bool:
        if self.proc is None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None) -> McpResult:
        if self.proc is None:
            return McpResult(False, error="server is not running")
        self._id += 1
        want = self._id
        if not self._send({"jsonrpc": "2.0", "id": want, "method": method,
                           "params": params or {}}):
            return McpResult(False, error=f"could not write to {self.name}")
        return self._await(want, timeout if timeout is not None else self.timeout)

    def _await(self, want_id: int, timeout: float) -> McpResult:
        """Read until the response with our id arrives, or the budget runs out.

        Messages that are not our response are discarded rather than treated as
        an error: a server may interleave notifications (progress, logging) and
        its own requests with the reply, and a client that trips over them works
        against exactly one server and then breaks.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return McpResult(False, error=f"{self.name} timed out after {timeout:.0f}s")
            line = self._readline(remaining)
            if line is None:
                code = self.proc.poll() if self.proc else None
                if code is not None:
                    return McpResult(False, error=f"{self.name} exited ({code}): {self._stderr()}")
                return McpResult(False, error=f"{self.name} timed out after {timeout:.0f}s")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue          # servers do print stray text to stdout; skip it
            if msg.get("id") != want_id:
                continue          # a notification or another request; not ours
            if "error" in msg:
                err = msg["error"]
                return McpResult(False, raw=msg,
                                 error=f"{err.get('code', '?')}: {err.get('message', 'unknown')}")
            return McpResult(True, raw=msg)

    def _pump(self) -> None:
        """Blocking reads on a daemon thread, into a queue.

        This used to be `select()` on `proc.stdout` and it was subtly,
        intermittently wrong. `select` polls the *file descriptor*; `readline()`
        reads from Python's *text-mode buffer*. When one read pulls several
        lines' worth of bytes into that buffer -- which is exactly what happens
        when a server emits a log line, a notification and a response in quick
        succession -- the next `select` sees an idle fd and reports a timeout
        while a complete, unread response is already sitting in memory.
        `tools/list` then returned an empty list about a third of the time.

        A thread doing blocking `readline()` has no such split: one place reads,
        one buffer, and the timeout lives on `queue.get` where it belongs.
        """
        stdout = self.proc.stdout if self.proc else None
        if stdout is None:
            return
        try:
            for line in stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)          # EOF sentinel

    def _readline(self, timeout: float) -> str | None:
        """One line from stdout, or None on timeout/EOF."""
        try:
            return self._lines.get(timeout=max(timeout, 0.01))
        except queue.Empty:
            return None

    def _stderr(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return (self.proc.stderr.read() or "")[-300:]
        except (OSError, ValueError):
            return ""

    # -- the two calls that matter ---------------------------------------------

    def list_tools(self) -> list[McpTool]:
        if not self.started:
            started = self.start()
            if not started.ok:
                return []
        out = self._request("tools/list")
        if not out.ok:
            return []
        tools = out.raw.get("result", {}).get("tools", []) or []
        return [McpTool(server=self.name, name=t.get("name", ""),
                        description=t.get("description", "") or "",
                        schema=t.get("inputSchema", {}) or {})
                for t in tools if t.get("name")]

    def call(self, tool: str, arguments: dict | None = None,
             timeout: float | None = None) -> McpResult:
        out = self._request("tools/call", {"name": tool, "arguments": arguments or {}}, timeout)
        if not out.ok:
            return out
        result = out.raw.get("result", {})
        text = "\n".join(block.get("text", "") for block in (result.get("content") or [])
                         if block.get("type") == "text")
        # `isError` is the protocol's way of saying the tool ran and failed, which
        # is a different fact from the call failing. Both are `ok=False` to the
        # caller, but the diagnostic distinguishes them.
        if result.get("isError"):
            return McpResult(False, content=text, raw=result,
                             error=f"{tool} reported failure: {text[:200]}")
        return McpResult(True, content=text, raw=result)


class McpRegistry:
    """Every configured server, and their tools in the shared embedding layer."""

    def __init__(self, memory, workspace=None) -> None:
        self.memory = memory
        self.workspace = workspace
        self.servers: dict[str, McpServer] = {}

    def add(self, name: str, command: list[str], env: dict | None = None,
            cwd: str | None = None) -> McpServer:
        server = McpServer(name, command, env, cwd)
        self.servers[name] = server
        return server

    def load_config(self, path) -> list[str]:
        """Read an `mcpServers` config -- the same shape Claude Desktop and
        Claude Code use, so an existing file works unchanged:

            {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@mcp/fs", "/tmp"]}}}
        """
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return []
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError:
            return []
        added = []
        for name, spec in (config.get("mcpServers") or {}).items():
            command = [spec.get("command", "")] + list(spec.get("args") or [])
            if not command[0]:
                continue
            self.add(name, command, spec.get("env"), spec.get("cwd"))
            added.append(name)
        return added

    def discover(self, name: str | None = None) -> dict:
        """Start servers, list their tools, and embed each one into memory.

        Ungraded on purpose (see the module docstring): there is no contract test
        to run against someone else's server, and entering them as verified would
        be manufacturing evidence -- the same rule `explore.py` applies to
        experiments.
        """
        report = {"servers": {}, "tools": 0, "failed": []}
        targets = [self.servers[name]] if name else list(self.servers.values())
        for server in targets:
            started = server.start()
            if not started.ok:
                report["failed"].append({"server": server.name, "error": started.error})
                self.memory.remember(
                    Kind.FAILURE, f"mcp server {server.name!r} unavailable: {started.error}",
                    meta={"server": server.name, "transport": "mcp"},
                    grade=-0.5, source=Source.SELF)
                continue
            tools = server.list_tools()
            for tool in tools:
                self.memory.remember(
                    Kind.TOOL, tool.embed_text(),
                    meta={"tool": tool.qualified, "transport": "mcp", "server": server.name,
                          "mcp_name": tool.name, "schema": tool.schema,
                          "signature": f"{tool.qualified}({', '.join(tool.arguments())})",
                          "purpose": tool.description or tool.name},
                    identity=f"mcp:{tool.qualified}")
            report["servers"][server.name] = [t.name for t in tools]
            report["tools"] += len(tools)
        return report

    def call(self, qualified: str, arguments: dict | None = None) -> McpResult:
        server_name, _, tool_name = qualified.partition(".")
        server = self.servers.get(server_name)
        if server is None:
            return McpResult(False, error=f"no such mcp server: {server_name}")
        if not server.started:
            started = server.start()
            if not started.ok:
                return started
        return server.call(tool_name, arguments)

    def close(self) -> None:
        for server in self.servers.values():
            server.close()

    def names(self) -> list[str]:
        return sorted(self.servers)
