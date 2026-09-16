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

import collections
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
        #: True when `list_tools` could not read the catalogue to the end.
        self.catalogue_truncated = False
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        # stderr must be drained continuously. A server that logs to stderr --
        # most of them do -- fills the 64KB pipe buffer and then blocks forever
        # on its next write, which looks like a hung server and is actually a
        # client that never read. Bounded, because a chatty server should cost a
        # fixed amount of memory.
        self._errors: collections.deque = collections.deque(maxlen=200)
        self._errthread: threading.Thread | None = None
        # One request at a time. Two concurrent _request calls each read from
        # the same queue, so each consumed the other's reply, discarded it as
        # "not mine", and both timed out.
        self._lock = threading.Lock()
        # Separate from the request lock, and guarding the process itself.
        # Without it two threads calling start() each spawned a child: the
        # second won `self.proc`, the first was orphaned, survived close(), and
        # its EOF sentinel later aborted a live request on the queue.
        self._lifecycle = threading.RLock()

    # -- lifecycle ------------------------------------------------------------

    @property
    def alive(self) -> bool:
        proc = self.proc
        return proc is not None and proc.poll() is None

    def start(self) -> McpResult:
        with self._lifecycle:
            return self._start_locked()

    def _start_locked(self) -> McpResult:
        if self.started and self.alive:
            return McpResult(True, "already started")
        if self.proc is not None:
            # A previous child died. Tear it down before replacing it, or its
            # reader thread and its queued output outlive it -- including the
            # EOF sentinel, which the next session would read as a dead pipe.
            self._close_locked()
        # Fresh sinks for a fresh process. Reusing them meant a crashed
        # process's buffered stdout was still queued when the replacement
        # started, and the first reply read was the dead server's.
        self._lines = queue.Queue()
        self._errors = collections.deque(maxlen=200)
        env = {**os.environ, **self.env}
        try:
            self.proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, env=env, cwd=self.cwd)
        except (OSError, ValueError) as exc:
            return McpResult(False, error=f"could not start {self.name}: {exc}")

        # Each pump is handed its own sink. Looking up `self._lines` at write
        # time meant a reader left over from a previous process wrote into the
        # *next* process's queue, injecting stale replies into a fresh session.
        self._reader = threading.Thread(target=self._pump,
                                        args=(self.proc.stdout, self._lines), daemon=True)
        self._reader.start()
        self._errthread = threading.Thread(target=self._drain,
                                           args=(self.proc.stderr, self._errors), daemon=True)
        self._errthread.start()

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
        with self._lifecycle:
            self._close_locked()

    def _close_locked(self) -> None:
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)       # reap it: kill() alone leaves a zombie
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        finally:
            self.started = False
            # Let the pumps finish against the pipes they were given, then start
            # clean. Because each pump holds its own sink, replacing these is
            # safe even if a thread is still draining.
            for thread in (self._reader, self._errthread):
                if thread is not None:
                    thread.join(timeout=2)
            self._reader = self._errthread = None
            self._lines = queue.Queue()
            self._errors = collections.deque(maxlen=200)

    def __enter__(self) -> "McpServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the wire -------------------------------------------------------------

    def _send(self, payload: dict) -> bool:
        # Bound to a local. `close()` may run on another thread and set
        # `self.proc` to None between the guard and the write, which surfaced as
        # an AttributeError escaping call() rather than a clean failure.
        proc = self.proc
        if proc is None or proc.stdin is None:
            return False
        if proc.poll() is not None:
            # The child died while idle. Nothing clears `started` on this path,
            # so without this the server stayed "started" forever and every
            # later call failed against a corpse.
            self.started = False
            return False
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            self.started = False
            return False

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None) -> McpResult:
        if self.proc is None:
            return McpResult(False, error="server is not running")
        budget = timeout if timeout is not None else self.timeout
        deadline = time.monotonic() + budget
        # The lock is held across _await, so a queued caller would otherwise wait
        # its predecessor's timeout AND then its own -- the per-call timeout
        # bounded the wire, not the call. Waiting against the same deadline makes
        # the budget mean what it says.
        if not self._lock.acquire(timeout=budget):
            return McpResult(False, error=f"{self.name} busy; timed out after {budget:.0f}s")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return McpResult(False, error=f"{self.name} busy; timed out after {budget:.0f}s")
            self._id += 1
            want = self._id
            if not self._send({"jsonrpc": "2.0", "id": want, "method": method,
                               "params": params or {}}):
                if not self.alive:
                    return McpResult(False, error=f"{self.name} is not running: {self._stderr()}")
                return McpResult(False, error=f"could not write to {self.name}")
            return self._await(want, remaining)
        finally:
            self._lock.release()

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
                    # The child is gone. Clear `started` so the next call brings
                    # up a fresh process instead of writing into a dead pipe
                    # forever -- a crashed server used to stay "started" for the
                    # life of the registry and every later call failed.
                    self.started = False
                    return McpResult(False, error=f"{self.name} exited ({code}): {self._stderr()}")
                return McpResult(False, error=f"{self.name} timed out after {timeout:.0f}s")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue          # servers do print stray text to stdout; skip it
            if not isinstance(msg, dict):
                continue          # a bare JSON scalar or list is not a frame
            if msg.get("id") != want_id:
                continue          # a notification or another request; not ours
            if "error" in msg:
                err = msg["error"]
                if not isinstance(err, dict):
                    # `err.get` on a string raised AttributeError straight out of
                    # call() -- from an API whose entire contract is to degrade
                    # into an McpResult rather than raise.
                    return McpResult(False, raw=msg, error=f"malformed error frame: {err!r}"[:200])
                return McpResult(False, raw=msg,
                                 error=f"{err.get('code', '?')}: {err.get('message', 'unknown')}")
            return McpResult(True, raw=msg)

    def _pump(self, stdout, sink) -> None:
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
        if stdout is None:
            return
        try:
            for line in stdout:
                sink.put(line)
        except (OSError, ValueError):
            pass
        finally:
            sink.put(None)                 # EOF sentinel

    @staticmethod
    def _drain(stderr, sink) -> None:
        """Keep the stderr pipe empty so the child never blocks writing to it."""
        if stderr is None:
            return
        try:
            for line in stderr:
                sink.append(line)
        except (OSError, ValueError):
            pass

    def _readline(self, timeout: float) -> str | None:
        """One line from stdout, or None on timeout/EOF."""
        try:
            return self._lines.get(timeout=max(timeout, 0.01))
        except queue.Empty:
            return None

    def _stderr(self) -> str:
        """Whatever the drain thread has collected.

        This used to be `proc.stderr.read()`, an unbounded read with no timeout
        on a pipe whose writer may still be alive -- so the call that was meant
        to explain a failure could itself hang indefinitely.
        """
        return "".join(self._errors)[-300:]

    # -- the two calls that matter ---------------------------------------------

    def list_tools(self, page_limit: int = 50) -> list[McpTool]:
        """Every tool, following `nextCursor` to the end of the catalogue.

        The protocol paginates. Reading only the first page silently truncated
        any server exposing more tools than its page size -- and a missing tool
        is indistinguishable from a server that does not offer it, so the
        truncation would never have been noticed.
        """
        if not self.started:
            started = self.start()
            if not started.ok:
                return []
        found: list[McpTool] = []
        cursor, pages, seen_cursors = None, 0, set()
        truncated = False
        while pages < page_limit:
            out = self._request("tools/list", {"cursor": cursor} if cursor else {})
            if not out.ok:
                # Stopping quietly here returns a partial catalogue that is
                # indistinguishable from a complete one. Record that it is
                # partial so a caller can tell.
                truncated = pages > 0
                break
            result = out.raw.get("result", {}) or {}
            if not isinstance(result, dict):
                truncated = True
                break
            tools = result.get("tools")
            for t in (tools if isinstance(tools, list) else []):
                if not isinstance(t, dict):
                    continue
                name = t.get("name")
                # Dedupe by name while collecting. Cursor-repeat detection can
                # only fire after a page has been read, so without this a server
                # that repeats a cursor still contributes one duplicate page --
                # and a server may legitimately repeat a tool across pages.
                if name and name not in {f.name for f in found}:
                    found.append(McpTool(server=self.name, name=name,
                                         description=t.get("description", "") or "",
                                         schema=t.get("inputSchema", {}) or {}))
            cursor = result.get("nextCursor")
            pages += 1
            if not cursor:
                break
            if not isinstance(cursor, str) or cursor in seen_cursors:
                # A server that repeats a cursor would otherwise be paged until
                # `page_limit`, re-adding the same tools each time.
                truncated = True
                break
            seen_cursors.add(cursor)
        else:
            truncated = True
        self.catalogue_truncated = truncated
        return found

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
        if name is not None and name not in self.servers:
            # Degrade rather than KeyError: a caller naming a server that is not
            # configured has made a mistake worth reporting, not worth crashing
            # the whole discovery pass over.
            report["failed"].append({"server": name, "error": "no such server configured"})
            return report
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
                          "purpose": tool.description or tool.name,
                          # Seeded so `Toolbox.load` has something to read, but
                          # only once. `remember` merges meta on an identity hit,
                          # so re-writing [] on every discovery pass erased
                          # everything `record_use` had taught the store about
                          # this tool -- and discovery runs on every startup.
                          **({} if self._known(tool.qualified) else {"solved": []})},
                    identity=f"mcp:{tool.qualified}")
            report["servers"][server.name] = [t.name for t in tools]
            report["tools"] += len(tools)
        return report

    def _known(self, qualified: str) -> bool:
        """Is this tool already in the store, with a history worth keeping?"""
        from .memory import Kind
        return any(t.meta.get("tool") == qualified for t in self.memory.of_kind(Kind.TOOL))

    def call(self, qualified: str, arguments: dict | None = None,
             tool_name: str | None = None) -> McpResult:
        """`qualified` is "server.tool". `tool_name` overrides the split, which
        matters because a server name may itself contain a dot and `partition`
        would then hand the server half a tool it does not have."""
        # Longest configured server name that prefixes this qualified name.
        # `partition(".")` took everything before the FIRST dot, so a server
        # called "acme.tools" had every one of its tools permanently uncallable:
        # the lookup asked for a server named "acme".
        server_name = next((n for n in sorted(self.servers, key=len, reverse=True)
                            if qualified == n or qualified.startswith(n + ".")), None)
        if server_name is None:
            server_name, _, _rest = qualified.partition(".")
        tool_name = tool_name or qualified[len(server_name) + 1:]
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
