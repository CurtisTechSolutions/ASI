"""A minimal, real MCP server over stdio. Test fixture for `distil.mcp`.

Speaks enough of the protocol to exercise the client honestly: the initialize
handshake, `tools/list`, and `tools/call` with both a success and an `isError`
path. It also emits a stray log line and an unsolicited notification before
replying, because a client that only works against a perfectly behaved server
works against exactly one server.
"""
import json
import sys

#: Behaviour switches, selected by argv, so one fixture can exercise the failure
#: modes a well-behaved server never shows: a chatty stderr that fills the pipe,
#: a paginated tool catalogue, and a process that dies mid-session.
MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

TOOLS = [
    {"name": "echo", "description": "Return the text you were given",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "add", "description": "Add two numbers and return the sum",
     "inputSchema": {"type": "object",
                     "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                     "required": ["a", "b"]}},
    {"name": "explode", "description": "Always fails, to exercise the error path",
     "inputSchema": {"type": "object", "properties": {}}},
]


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def reply(rid, result):
    send({"jsonrpc": "2.0", "id": rid, "result": result})


def main():
    if MODE == "chatty":
        # ~200KB to stderr before doing anything. A client that does not drain
        # the pipe blocks the child here, forever.
        for i in range(4000):
            sys.stderr.write("log line %d: padding padding padding padding padding\n" % i)
        sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, rid = msg.get("method"), msg.get("id")
        if method == "initialize":
            reply(rid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                        "serverInfo": {"name": "echo-fixture", "version": "1"}})
        elif method == "notifications/initialized":
            continue                                  # a notification: no reply
        elif method == "tools/list" and MODE == "paged":
            cursor = (msg.get("params") or {}).get("cursor")
            if not cursor:
                reply(rid, {"tools": TOOLS[:1], "nextCursor": "page2"})
            elif cursor == "page2":
                reply(rid, {"tools": TOOLS[1:2], "nextCursor": "page3"})
            else:
                reply(rid, {"tools": TOOLS[2:]})
        elif method == "tools/call" and MODE == "crash":
            raise SystemExit(3)
        elif method == "tools/list":
            # Noise the client must survive: a non-JSON line, then a notification
            # with no id, then the actual response.
            sys.stdout.write("starting up, please ignore this line\n")
            sys.stdout.flush()
            send({"jsonrpc": "2.0", "method": "notifications/message",
                  "params": {"level": "info", "data": "listing tools"}})
            reply(rid, {"tools": TOOLS})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            if name == "echo":
                reply(rid, {"content": [{"type": "text", "text": args.get("text", "")}]})
            elif name == "add":
                total = (args.get("a") or 0) + (args.get("b") or 0)
                reply(rid, {"content": [{"type": "text", "text": str(total)}]})
            elif name == "explode":
                reply(rid, {"content": [{"type": "text", "text": "deliberate failure"}],
                            "isError": True})
            else:
                send({"jsonrpc": "2.0", "id": rid,
                      "error": {"code": -32602, "message": f"unknown tool: {name}"}})
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32601, "message": f"unknown method: {method}"}})


if __name__ == "__main__":
    main()
