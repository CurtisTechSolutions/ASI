"""Tests for ``radixnet.mcp``: the JSON-RPC protocol, the tools it offers and the stdio stream.

:meth:`radixnet.mcp.McpServer.handle` turns one request object into one
response object, so the protocol is exercised directly; :meth:`run` is driven
over real streams, and the CLI is driven as a client would drive it, through a
subprocess whose stdout must carry nothing but the protocol.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet  # noqa: E402
from radixnet.mcp import (  # noqa: E402
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    McpServer,
    model_tools,
)
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.tools import Tool, ToolBox, default_toolbox  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def request(ident, method, **params):
    return {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.box = default_toolbox(offline=True)
        self.server = McpServer(self.box)

    def test_initialize(self):
        answer = self.server.handle(request(1, "initialize", protocolVersion=PROTOCOL_VERSION, capabilities={}))
        result = answer["result"]
        self.assertEqual(answer["id"], 1)
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "radixnet")
        self.assertTrue(result["serverInfo"]["version"])
        self.assertIn("instructions", result)
        self.assertTrue(self.server.initialized)

    def test_a_clients_protocol_version_is_echoed(self):
        answer = self.server.handle(request(1, "initialize", protocolVersion="2025-06-18"))
        self.assertEqual(answer["result"]["protocolVersion"], "2025-06-18")

    def test_tools_list_is_mcp_shaped(self):
        tools = self.server.handle(request(2, "tools/list"))["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["calculator"])
        schema = tools[0]["inputSchema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("expression", schema["properties"])
        self.assertEqual(schema["required"], ["expression"])
        self.assertTrue(tools[0]["description"])

    def test_tools_call(self):
        answer = self.server.handle(request(3, "tools/call", name="calculator", arguments={"expression": "6*7"}))
        result = answer["result"]
        self.assertEqual(result["content"], [{"type": "text", "text": "42"}])
        self.assertFalse(result["isError"])
        self.assertEqual(self.server.calls, 1)

    def test_a_failing_tool_is_a_result_not_a_protocol_error(self):
        answer = self.server.handle(request(4, "tools/call", name="calculator", arguments={"expression": "1/0"}))
        self.assertNotIn("error", answer)
        self.assertTrue(answer["result"]["isError"])
        self.assertIn("division by zero", answer["result"]["content"][0]["text"])

    def test_unknown_tool_and_bad_arguments(self):
        answer = self.server.handle(request(5, "tools/call", name="nope"))
        self.assertEqual(answer["error"]["code"], INVALID_PARAMS)
        self.assertIn("unknown tool", answer["error"]["message"])
        self.assertEqual(self.server.handle(request(6, "tools/call"))["error"]["code"], INVALID_PARAMS)
        answer = self.server.handle(request(7, "tools/call", name="calculator", arguments="text"))
        self.assertEqual(answer["error"]["code"], INVALID_PARAMS)

    def test_a_missing_required_argument_comes_back_as_a_tool_error(self):
        answer = self.server.handle(request(8, "tools/call", name="calculator", arguments={}))
        self.assertTrue(answer["result"]["isError"])
        self.assertIn("missing argument", answer["result"]["content"][0]["text"])

    def test_ping_and_unknown_methods(self):
        self.assertEqual(self.server.handle(request(9, "ping"))["result"], {})
        self.assertEqual(self.server.handle(request(10, "nosuch"))["error"]["code"], METHOD_NOT_FOUND)

    def test_notifications_are_never_answered(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertTrue(self.server.initialized)
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}))

    def test_malformed_messages(self):
        self.assertEqual(self.server.handle("not an object")["error"]["code"], INVALID_REQUEST)
        self.assertEqual(self.server.handle({"id": 1, "method": "ping"})["error"]["code"], INVALID_REQUEST)
        self.assertEqual(self.server.handle({"jsonrpc": "2.0", "id": 1})["error"]["code"], INVALID_REQUEST)
        # JSON-RPC allows positional params; an empty list means "none", anything else is refused
        self.assertEqual(self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": []})["result"], {})
        self.assertEqual(
            self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1, 2]})["error"]["code"],
            INVALID_PARAMS,
        )

    def test_a_tool_that_raises_does_not_end_the_session(self):
        def boom():
            raise RuntimeError("kaboom")

        self.box.register(Tool("boom", "Fails.", (), boom))
        answer = self.server.handle(request(11, "tools/call", name="boom"))
        self.assertTrue(answer["result"]["isError"])
        self.assertEqual(self.server.handle(request(12, "ping"))["result"], {})

    def test_an_empty_toolbox_is_refused(self):
        with self.assertRaises(ValueError):
            McpServer(ToolBox())


class StreamTests(unittest.TestCase):
    def test_the_stream_answers_requests_and_skips_notifications(self):
        server = McpServer(default_toolbox(offline=True))
        lines = [
            json.dumps(request(1, "initialize")),
            "",  # blank lines are skipped
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps(request(2, "tools/call", name="calculator", arguments={"expression": "2+2"})),
        ]
        out = io.StringIO()
        self.assertEqual(server.run(io.StringIO("\n".join(lines) + "\n"), out), 0)
        answers = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([a["id"] for a in answers], [1, 2])  # the notification got no answer
        self.assertEqual(answers[1]["result"]["content"][0]["text"], "4")

    def test_invalid_json_is_reported_and_the_stream_carries_on(self):
        server = McpServer(default_toolbox(offline=True))
        out = io.StringIO()
        server.run(io.StringIO("{not json}\n" + json.dumps(request(1, "ping")) + "\n"), out)
        answers = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(answers[0]["error"]["code"], PARSE_ERROR)
        self.assertEqual(answers[1]["result"], {})


class ModelToolsTests(unittest.TestCase):
    def setUp(self):
        self.model = RadixNet(seed=0, backend="python")
        self.model.train(["the cat sat on the mat", "the dog runs in the park"], epochs=20, lr=0.5, batch_size=4)

    def test_the_network_is_offered_as_tools(self):
        names = [t.name for t in model_tools(self.model)]
        self.assertEqual(names, ["radixnet_predict", "radixnet_generate", "radixnet_score", "radixnet_stats"])

    def test_predict_generate_score_and_stats(self):
        box = ToolBox(model_tools(self.model))
        self.assertTrue(box.call("radixnet_predict", {"prefix_text": "the cat", "length": 10}).output.startswith("the cat"))
        self.assertTrue(box.call("radixnet_generate", {"count": 2, "max_length": 20}).ok)
        self.assertIn("per character", box.call("radixnet_score", {"text": "the cat sat on the mat"}).output)
        self.assertIn("nodes", json.loads(box.call("radixnet_stats", {}).output))

    def test_judge_is_offered_only_with_a_negative_network(self):
        negative = NegativeNet(seed=1)
        negative.blame(["the cat sat on the sky"], reason="false", severity=2.0)
        names = [t.name for t in model_tools(self.model, negative=negative)]
        self.assertIn("radixnet_judge", names)
        box = ToolBox(model_tools(self.model, negative=negative))
        answer = box.call("radixnet_judge", {"text": "the cat sat on the sky"})
        self.assertTrue(answer.ok)
        self.assertIn("false", answer.output)
        self.assertIn("risk", answer.output)

    def test_solve_is_offered_only_with_a_toolbox_and_a_client(self):
        self.assertNotIn("radixnet_solve", [t.name for t in model_tools(self.model, toolbox=default_toolbox(offline=True))])
        names = [t.name for t in model_tools(self.model, toolbox=default_toolbox(offline=True), client=object())]
        self.assertIn("radixnet_solve", names)

    def test_a_service_with_a_lock_is_used_through_it(self):
        entered = []

        class Service:  # a stand-in for ModelService: the model is reached through session()
            model = self.model

            def session(inner):
                import contextlib

                @contextlib.contextmanager
                def held():
                    entered.append(True)
                    yield inner.model

                return held()

        box = ToolBox(model_tools(Service()))
        self.assertTrue(box.call("radixnet_score", {"text": "the cat"}).ok)
        self.assertTrue(entered)

    def test_the_prefix_can_be_changed(self):
        self.assertTrue(all(t.name.startswith("net.") for t in model_tools(self.model, prefix="net.")))


class CliTests(unittest.TestCase):
    """The server as a client meets it: a subprocess whose stdout is the protocol and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-mcp-")
        cls.model = os.path.join(cls.tmp.name, "model.json")
        net = RadixNet(seed=0, backend="python")
        net.train(["the cat sat on the mat"], epochs=10, lr=0.5, batch_size=4)
        net.save(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_client_can_initialize_list_and_call(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python",
             "mcp", "--offline", "--no-solve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
        )
        try:
            def call(message):
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
                return json.loads(proc.stdout.readline())

            self.assertEqual(call(request(1, "initialize"))["result"]["serverInfo"]["name"], "radixnet")
            names = [t["name"] for t in call(request(2, "tools/list"))["result"]["tools"]]
            self.assertIn("calculator", names)
            self.assertIn("radixnet_predict", names)
            self.assertNotIn("radixnet_solve", names)  # --no-solve
            self.assertNotIn("web_fetch", names)  # --offline
            answer = call(request(3, "tools/call", name="radixnet_predict",
                                  arguments={"prefix_text": "the cat", "length": 8}))
            self.assertTrue(answer["result"]["content"][0]["text"].startswith("the cat"))
            proc.stdin.close()
            self.assertEqual(proc.wait(timeout=30), 0)
            # everything a person reads went to stderr; stdout carried only the protocol
            notes = proc.stderr.read()
            self.assertIn("MCP server", notes)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
