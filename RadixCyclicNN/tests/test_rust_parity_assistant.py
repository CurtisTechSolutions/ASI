"""The Rust port's today's-format area (``rust/src/assistant.rs``) against ``radixnet/assistant.py``.

The same model, the same request: both implementations must think the same
lines, say the same text, end for the same reason and count the same units -
from the CLI (``talk --request``, the dialect's own document) and from the
server (``/v1/chat/completions``, ``/v1/messages``, streamed and not).  Ids and
timestamps are the only fields allowed to differ.
"""

import json
import os
import shutil
import socket
import unittest

from tests.rust_harness import CORPUS, build, py, rust, serve, tmpdir

FLOATS = ("cost", "probability", "step_costs")
RUN_NAMES = ("backend", "device")
"""What the guard's report says about the run, not the reply: the negative network's ``stats`` name the port that ran it."""


def setUpModule():
    build()


def strip(doc):
    """A document without its id and its clock."""
    doc = json.loads(json.dumps(doc))
    doc.pop("id", None)
    doc.pop("created", None)
    return doc


def scrub_guard(report):
    """The guard's report without the negative network's run names (``backend`` / ``device``)."""
    if not report:
        return report
    report = dict(report)
    report["negative"] = {k: v for k, v in (report.get("negative") or {}).items() if k not in RUN_NAMES}
    return report


class TestRustTalkParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(tmpdir(), "assistant")
        os.makedirs(cls.root, exist_ok=True)
        cls.model = os.path.join(cls.root, "m.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)

    def copy(self, name):
        model = os.path.join(self.root, f"{name}.count.json")
        shutil.copy(self.model, model)
        return model

    def request(self, name, body):
        path = os.path.join(self.root, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        return path

    def assert_same_reply(self, a, b, dialect):
        """The same document, ids and timestamps aside, the turn's floats to 1e-9."""
        a, b = strip(a), strip(b)
        ra, rb = a.pop("radixnet"), b.pop("radixnet")
        self.assertEqual(a, b)
        self.assertEqual((ra["kind"], ra["units"], len(ra["choices"])), (rb["kind"], rb["units"], len(rb["choices"])))
        for x, y in zip(ra["choices"], rb["choices"]):
            self.assertEqual((x["index"], x["stop_reason"], scrub_guard(x["guard"])),
                             (y["index"], y["stop_reason"], scrub_guard(y["guard"])))
            tx, ty = x["turn"], y["turn"]
            if tx is None or ty is None:
                self.assertEqual(tx, ty)
                continue
            self.assertEqual({k: v for k, v in tx.items() if k not in FLOATS}, {k: v for k, v in ty.items() if k not in FLOATS})
            self.assertLessEqual(abs(tx["cost"] - ty["cost"]), 1e-9)
            self.assertLessEqual(abs(tx["probability"] - ty["probability"]), 1e-9)
            self.assertEqual(len(tx["step_costs"]), len(ty["step_costs"]))
            for u, v in zip(tx["step_costs"], ty["step_costs"]):
                self.assertLessEqual(abs(u - v), 1e-9)

    def test_the_same_thinking_and_the_same_text(self):
        for dialect, body in (
            ("openai", {"messages": [{"role": "user", "content": "tell me about the cat"}], "max_tokens": 40, "learn": False}),
            ("openai", {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "tell me about the cat"},
                                     {"role": "assistant", "content": "the cat night"}, {"role": "user", "content": "and the dog"}],
                        "n": 3, "max_tokens": 40, "learn": False}),
            ("openai", {"messages": [{"role": "user", "content": "the cat sat"}, {"role": "assistant", "content": "on the"}],
                        "stop": ["at"], "learn": False}),
            ("openai", {"messages": [{"role": "user", "content": "the dog runs"}], "mode": "sample", "seed": 7, "max_tokens": 25, "learn": False}),
            ("anthropic", {"system": "be brief", "messages": [{"role": "user", "content": "the dog runs in the park"}],
                           "max_tokens": 30, "stop_sequences": ["zzz"], "learn": False}),
            ("anthropic", {"messages": [{"role": "user", "content": "the cat sat"}], "thinking": {"type": "disabled"}, "learn": False}),
            ("openai", {"messages": [{"role": "user", "content": "TASK: add 2 and 2"}], "learn": False,
                        "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object",
                                   "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]}),
        ):
            request = self.request("req", body)
            a = py("talk", "--format", dialect, "--request", request, model=self.copy("tp"))
            b = rust("talk", "--format", dialect, "--request", request, model=self.copy("tr"))
            self.assert_same_reply(a, b, dialect)

    def test_a_conversation_from_the_flags(self):
        args = ["talk", "--message", "tell me about the cat", "--message", "and the dog", "--message", "the cat sat", "--no-learn"]
        a = py(*args, model=self.copy("cp"))
        b = rust(*args, model=self.copy("cr"))
        self.assertEqual((a["format"], len(a["exchanges"]), a["taught"]), (b["format"], len(b["exchanges"]), b["taught"]))
        for x, y in zip(a["exchanges"], b["exchanges"]):
            self.assert_same_reply(x, y, "openai")

    def test_what_a_rethink_teaches_the_graph_is_the_same(self):
        args = ["talk", "--message", "the cat sat on the mat", "--message", "the cat sat on the mat", "--message", "the cat sat on the mat",
                "--message", "the cat sat on the mat", "--k", 2, "--save"]
        pm, rm = self.copy("lp"), self.copy("lr")
        a = py(*args, model=pm)
        b = rust(*args, model=rm)
        self.assertEqual(a["taught"], b["taught"])
        for x, y in zip(a["exchanges"], b["exchanges"]):
            self.assert_same_reply(x, y, "openai")
        if not a["taught"]:
            self.skipTest("this conversation never made the voice back up")
        with open(pm, encoding="utf-8") as fh:
            pg = json.load(fh)["graph"]
        with open(rm, encoding="utf-8") as fh:
            rg = json.load(fh)["graph"]
        self.assertEqual(pg["nodes"]["labels"], rg["nodes"]["labels"])
        self.assertEqual(pg["edges"]["reward"], rg["edges"]["reward"])

    def test_the_guard_vetoes_the_same_replies(self):
        plain = py("talk", "--message", "tell me about the cat", "--no-learn", model=self.copy("gp0"))
        said = plain["choices"][0]["message"]["content"]
        negative = os.path.join(self.root, "m.count.negative.json")
        py("negative", "blame", "--reason", "nonsense", "--severity", 5, "--text", said, model=self.model)
        for name in ("gp", "gr"):
            shutil.copy(negative, os.path.join(self.root, f"{name}.count.negative.json"))
        args = ["talk", "--message", "tell me about the cat", "--no-learn"]
        a = py(*args, model=self.copy("gp"))
        b = rust(*args, model=self.copy("gr"))
        self.assert_same_reply(a, b, "openai")
        report = a["radixnet"]["choices"][0]["guard"]
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report["vetoed"], 1)
        self.assertIn("the negative network vetoed", a["choices"][0]["message"]["reasoning_content"])
        self.assertNotEqual(a["choices"][0]["message"]["content"], said)


class TestRustTalkRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(tmpdir(), "assistant-route")
        os.makedirs(root, exist_ok=True)
        cls.model = os.path.join(root, "model.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)
        request = os.path.join(root, "request.json")
        cls.body = {"messages": [{"role": "user", "content": "tell me about the cat"}], "max_tokens": 30, "learn": False}
        with open(request, "w", encoding="utf-8") as fh:
            json.dump(cls.body, fh)
        cls.expected = {dialect: py("talk", "--format", dialect, "--request", request, model=cls.model) for dialect in ("openai", "anthropic")}
        cls.server = serve(cls, cls.model)

    def raw(self, path, body):
        payload = json.dumps(body).encode("utf-8")
        head = (f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode("ascii")
        with socket.create_connection(("127.0.0.1", self.server.port), timeout=120) as sock:
            sock.sendall(head + payload)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        raw = b"".join(chunks)
        headers, _, body = raw.partition(b"\r\n\r\n")
        return headers.decode("latin-1"), body.decode("utf-8")

    def test_the_routes_answer_python_s_documents(self):
        status, models, _ = self.server.get("/v1/models")
        self.assertEqual(status, 200, models)
        self.assertEqual([m["id"] for m in models["data"]], ["radixnet-count"])
        self.assertEqual(set(models["data"][0]), {"id", "object", "created", "owned_by", "kind", "label", "encoding", "units", "active"})
        for dialect, path in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
            status, doc, _ = self.server.post(path, self.body)
            self.assertEqual(status, 200, doc)
            expected = strip(self.expected[dialect])
            got = strip(doc)
            expected.pop("radixnet")
            got.pop("radixnet")
            self.assertEqual(got, expected, dialect)
        status, counted, _ = self.server.post("/v1/messages/count_tokens", {"messages": self.body["messages"]})
        self.assertEqual((status, counted), (200, {"input_tokens": len("tell me about the cat")}))
        status, bad, _ = self.server.post("/v1/chat/completions", {"messages": "no"})
        self.assertEqual(status, 400)
        self.assertEqual(bad, {"error": {"message": "messages must be a list of {role, content} objects",
                                         "type": "invalid_request_error", "param": "messages", "code": None}})
        status, bad, _ = self.server.post("/v1/messages", {"model": "resonant", "messages": self.body["messages"]})
        self.assertEqual((status, bad["type"], bad["error"]["type"]), (404, "error", "not_found_error"))

    def test_the_streams_carry_the_same_text(self):
        said = self.expected["openai"]["choices"][0]["message"]["content"]
        thought = self.expected["openai"]["choices"][0]["message"]["reasoning_content"]
        headers, body = self.raw("/v1/chat/completions", {**self.body, "stream": True, "stream_options": {"include_usage": True}})
        self.assertIn("text/event-stream", headers)
        frames = [f for f in body.split("\n\n") if f.strip()]
        self.assertEqual(frames[-1], "data: [DONE]")
        chunks = [json.loads(f[len("data: "):]) for f in frames[:-1]]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]), said)
        self.assertEqual("".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"]), thought)
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], self.expected["openai"]["usage"]["prompt_tokens"])
        headers, body = self.raw("/v1/messages", {**self.body, "stream": True})
        names, texts = [], []
        for frame in body.split("\n\n"):
            if not frame.strip():
                continue
            lines = frame.split("\n")
            names.append(lines[0].replace("event: ", ""))
            data = json.loads(lines[1][len("data: "):])
            if data["type"] == "content_block_delta" and data["delta"]["type"] == "text_delta":
                texts.append(data["delta"]["text"])
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        self.assertEqual("".join(texts), said)


if __name__ == "__main__":
    unittest.main()
