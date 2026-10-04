"""agent.json files exercised through a fake endpoint and the real model server."""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parent / "RadixCyclicNN"
sys.path.insert(0, str(KIT))
sys.path.insert(1, str(ROOT))

from modelkit.agent_config import AgentConfig, load_agent_config  # noqa: E402

SAMPLE = json.loads((ROOT / "agent.json").read_text(encoding="utf-8"))


class ConfigFile(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="radixnet-agent-config-")
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "agent.json"

    def write(self, data, encoding="utf-8"):
        self.path.write_text(json.dumps(data), encoding=encoding)
        return load_agent_config(self.path)


class TestAgentConfig(ConfigFile):
    def test_sample_and_optional_defaults(self):
        expected = {"id": "radixcyclicnn-v1", "type": "chat", "base_url": "http://127.0.0.1:8000/v1",
                    "model": "radixcyclicnn", "timeout": 60, "max_tokens": 128, "temperature": 0, "structured": True}
        self.assertEqual(SAMPLE, expected)
        self.assertEqual(dataclasses.asdict(self.write(SAMPLE)), expected)
        self.assertEqual(dataclasses.asdict(self.write({k: SAMPLE[k] for k in ("id", "type", "base_url", "model")})), expected)
        self.assertEqual(self.write(SAMPLE, encoding="utf-8-sig").model, "radixcyclicnn")

    def test_url_prefixes_and_trailing_slashes(self):
        for url, expected in (
            ("http://localhost:8000", "http://localhost:8000/v1"),
            ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
            ("http://[::1]:8000/", "http://[::1]:8000/v1"),
            ("https://example.test/prefix/v1/", "https://example.test/prefix/v1"),
        ):
            with self.subTest(url=url):
                self.assertEqual(self.write({**SAMPLE, "base_url": url}).base_url, expected)

    def test_invalid_fields_are_refused_before_network_access(self):
        bad_fields = {
            "id": [None, "", " ", 3], "type": [None, "completion", True], "model": [None, "", []],
            "base_url": [None, "localhost:8000/v1", "file:///tmp/agent", "http:///v1", "http://a:bad",
                         "http://a:99999", "http://user:pass@localhost/v1", "http://a/v1?key=x", "http://a/#v1", "http://a b/v1"],
            "timeout": [0, -1, True, "60", None, float("inf"), float("nan")],
            "max_tokens": [0, -1, True, 1.5, "128", None],
            "temperature": [-1, True, "0", None, float("inf"), float("nan")],
            "structured": [None, 0, 1, "true"],
        }
        with mock.patch("urllib.request.build_opener") as network:
            for field, values in bad_fields.items():
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "cannot load agent config"):
                        self.write({**SAMPLE, field: value})
            network.assert_not_called()

    def test_bad_files_missing_and_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "cannot load agent config"):
            load_agent_config(self.path)
        for raw in ("{oops", "null", "[]", '"hello"', "{}"):
            self.path.write_text(raw, encoding="utf-8")
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "cannot load agent config"):
                load_agent_config(self.path)
        for name in ("id", "type", "base_url", "model"):
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "cannot load agent config"):
                self.write({k: v for k, v in SAMPLE.items() if k != name})
        with self.assertRaisesRegex(ValueError, "temperatur"):
            self.write({**SAMPLE, "temperatur": 1})

    def test_timeout_is_used_and_reported(self):
        agent = self.write({**SAMPLE, "timeout": 0.25})
        with mock.patch("urllib.request.build_opener") as factory:
            factory.return_value.open.side_effect = TimeoutError("timed out")
            with self.assertRaisesRegex(ValueError, "cannot reach agent.*timed out"):
                agent.request({"messages": [{"role": "user", "content": "hello"}]})
            self.assertEqual(factory.return_value.open.call_args.kwargs["timeout"], 0.25)


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body, dict(self.headers)))
        message = {"role": "assistant", "content": "reply " + str(len(self.server.requests)), "reasoning_content": "a trace"}
        doc = {"object": "chat.completion", "model": body["model"], "choices": [{"message": message, "finish_reason": "stop"}],
               "usage": {"total_tokens": 5}}
        raw = self.server.raw_response
        if raw is None:
            raw = json.dumps(doc).encode("utf-8")
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def serve(server, add_cleanup):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        thread.join(5)
        server.server_close()

    add_cleanup(stop)


class TestChatConnection(ConfigFile):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        self.server.requests = []
        self.server.status = 200
        self.server.raw_response = None
        serve(self.server, self.addCleanup)
        self.agent = self.write({**SAMPLE, "base_url": f"http://127.0.0.1:{self.server.server_port}/v1"})

    def run_cli(self, *args, input=None, expected=0, human=False):
        command = [sys.executable, "-m", "radixnet", "--backend", "python", "--model", str(self.path.parent / "absent-model.json")]
        if not human:
            command.append("--json")
        command += ["talk", "--agent-config", str(self.path), *args]
        env = dict(os.environ, OPENAI_API_KEY="must-not-be-sent", OPENAI_ORG_ID="must-not-be-sent",
                   HTTP_PROXY="http://127.0.0.1:1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        proc = subprocess.run(command, cwd=ROOT, env=env, input=input, capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, expected, proc.stderr)
        return proc

    def test_defaults_reach_the_wire_and_response_is_preserved(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-be-sent"}):
            doc = self.agent.request(body)
        path, sent, headers = self.server.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(sent, {**body, "model": "radixcyclicnn", "max_tokens": 128, "temperature": 0,
                                "response_format": {"type": "json_object"}, "stream": False})
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("max_tokens", body, "the caller's request must not be mutated")
        self.assertEqual(doc["usage"], {"total_tokens": 5})
        self.assertEqual(doc["choices"][0]["message"]["reasoning_content"], "a trace")

    def test_overrides_and_unstructured_requests(self):
        unstructured = dataclasses.replace(self.agent, structured=False)
        unstructured.request({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.75, "max_tokens": 7})
        sent = self.server.requests[-1][1]
        self.assertNotIn("response_format", sent)
        self.assertEqual((sent["max_tokens"], sent["temperature"]), (7, 0.75))
        self.agent.request({"messages": [{"role": "user", "content": "hi"}], "model": "radixnet-count",
                            "max_completion_tokens": 9, "response_format": {"type": "text"}})
        sent = self.server.requests[-1][1]
        self.assertNotIn("max_tokens", sent)
        self.assertEqual((sent["model"], sent["max_completion_tokens"], sent["response_format"]),
                         ("radixnet-count", 9, {"type": "text"}))

    def test_http_errors_and_bad_responses(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.server.status = 404
        self.server.raw_response = b'{"error": {"message": "unknown model"}}'
        with self.assertRaisesRegex(ValueError, "HTTP 404: unknown model"):
            self.agent.request(body)
        self.server.status = 200
        for raw in (b"not json", b"[]", b"{}", b'{"choices": []}', b'{"choices": [null]}', b'{"choices": [{"message": null}]}'):
            self.server.raw_response = raw
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.agent.request(body)
        with self.assertRaisesRegex(ValueError, "streaming"):
            self.agent.request({**body, "stream": True})

    def test_cli_conversation_without_a_local_model(self):
        doc = json.loads(self.run_cli("--message", "hello", "--message", "more", "--system", "be brief", "--no-learn").stdout)
        self.assertEqual(len(doc["exchanges"]), 2)
        first, second = [entry[1] for entry in self.server.requests]
        self.assertEqual((first["model"], first["max_tokens"], first["temperature"], first["learn"]),
                         ("radixcyclicnn", 128, 0, False))
        self.assertEqual(second["messages"], [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply 1"},
                         {"role": "user", "content": "more"}])
        for _, _, headers in self.server.requests:
            self.assertNotIn("Authorization", headers)
            self.assertNotIn("OpenAI-Organization", headers)

    def test_cli_flags_request_file_and_stdin(self):
        self.run_cli("--message", "hello", "--max-tokens", "11", "--temperature", "0.5")
        sent = self.server.requests[-1][1]
        self.assertEqual((sent["max_tokens"], sent["temperature"]), (11, 0.5))
        request = self.path.parent / "request.json"
        request.write_text(json.dumps({"messages": [{"role": "user", "content": "request"}], "max_tokens": 4,
                                       "temperature": 0, "response_format": {"type": "text"}}), encoding="utf-8")
        self.run_cli("--request", str(request))
        sent = self.server.requests[-1][1]
        self.assertEqual((sent["max_tokens"], sent["temperature"], sent["response_format"]), (4, 0, {"type": "text"}))
        output = self.run_cli(input="hello\nmore\n", human=True).stdout
        self.assertEqual(output.count("model: "), 2)
        self.assertIn("a trace", output)

    def test_cli_invalid_combinations_and_errors(self):
        for flags, error in ((("--format", "anthropic"), "openai chat format"), (("--save",), "local model"),
                             (("--out", "unused.json"), "local model")):
            with self.subTest(flags=flags):
                self.assertIn(error, self.run_cli(*flags, "--message", "hi", expected=1).stderr)
        self.assertFalse(self.server.requests)
        self.server.status = 503
        self.server.raw_response = b'{"error": {"message": "unavailable"}}'
        proc = self.run_cli("--message", "hi", expected=1)
        self.assertIn("HTTP 503: unavailable", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.path.write_text("{oops", encoding="utf-8")
        self.assertIn("cannot load agent config", self.run_cli("--message", "hi", expected=1).stderr)


class TestRealServer(ConfigFile):
    def test_sample_descriptor_against_radixcyclicnn(self):
        from modelkit.api import create_server

        server, service = create_server("127.0.0.1", 0, kind="count", backend="python", quiet=True)
        self.addCleanup(service.shutdown)
        serve(server, self.addCleanup)
        service.model.train(["the cat sat on the mat", "the cat ran away", "the dog runs home"], epochs=2)
        agent = self.write({**SAMPLE, "base_url": server.url.rstrip("/") + "/v1"})
        doc = agent.request({"messages": [{"role": "user", "content": "the cat"}], "learn": False})
        self.assertEqual((doc["object"], doc["model"]), ("chat.completion", "radixnet-count"))
        self.assertTrue(doc["choices"][0]["message"]["content"])
        self.assertIn("usage", doc)
        proc = subprocess.run([sys.executable, "-m", "radixnet", "--json", "talk", "--agent-config", str(self.path),
                               "--message", "the cat", "--no-learn"], cwd=ROOT,
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        again = json.loads(proc.stdout)
        self.assertEqual(again["choices"][0]["message"]["content"], doc["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
