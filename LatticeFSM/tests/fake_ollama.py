"""A stand-in for Ollama's HTTP API, for the tests (and for trying autotrain without a language model).

It answers ``GET /api/tags`` and ``POST /api/generate`` the way Ollama does, deterministically, and keeps every
prompt it was sent (``FakeOllama.prompts``) so a test can check what was asked:

* asked for sentences about a topic, it writes them - numbered, as models often do, to exercise the parsing;
* asked to continue a text, it continues it;
* asked to judge numbered texts, it approves the odd-numbered ones (JSON, the way it was asked).

Run on its own: ``python3 tests/fake_ollama.py --port 11500`` and point ``OLLAMA_HOST`` at it.  Its replies are
sent with chunked transfer encoding, as Go's HTTP server sends a large reply, so the clients' decoding is tested too.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOPIC = re.compile(r"English sentences about (.+?)\. Write one sentence")
COUNT = re.compile(r"Write (\d+) different")
TEXT = re.compile(r"Text: (.*)$", re.S)
NUMBERED = re.compile(r"^(\d+)\. ", re.M)


def reply_to(prompt: str) -> str:
    if "different, natural English sentences" in prompt:
        topic = TOPIC.search(prompt).group(1)
        n = int(COUNT.search(prompt).group(1))
        lines = [f"{i + 1}. The {topic} is part of a larger pattern, and pattern {i % 3} repeats." for i in range(n)]
        return "Here are the sentences:\n" + "\n".join(lines)
    if prompt.startswith("Continue the following text"):
        text = TEXT.search(prompt).group(1).strip()
        return f"\"which is how the {text.split()[-1] if text.split() else 'graph'} grows over time.\""
    if "decide whether it is fluent" in prompt:
        ids = [int(m.group(1)) for m in NUMBERED.finditer(prompt)]
        return "```json\n" + json.dumps([{"id": i, "ok": i % 2 == 1} for i in ids]) + "\n```"
    return "I am a stand-in."


class FakeOllama:
    """A threaded server on 127.0.0.1 (an ephemeral port unless one is given), started at once."""

    def __init__(self, port: int = 0, model: str = "fake-model") -> None:
        self.prompts: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # quiet
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i in range(0, len(data), 64):
                    part = data[i:i + 64]
                    self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")

            def do_GET(self) -> None:
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": f"{model}:latest"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if self.path != "/api/generate":
                    self._send(404, {"error": "not found"})
                    return
                prompt = body.get("prompt", "")
                outer.prompts.append(prompt)
                self._send(200, {"model": body.get("model"), "response": reply_to(prompt), "done": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=11500)
    args = p.parse_args()
    fake = FakeOllama(args.port)
    print(f"a stand-in Ollama on {fake.url} (Ctrl-C stops it)")
    try:
        fake.thread.join()
    except KeyboardInterrupt:
        fake.close()
