"""Tests for radixnet.assistant - the model in today's format: messages in, thinking, text and tool calls out, streamed.

The mapping is checked from both ends: what the two dialects' requests become
(``parse_openai`` / ``parse_anthropic``), what a reply is made of (the
thinking as the search's own trace, the text as the turn's, the chunks as the
walk's nodes), how it ends (the stop reasons), what it costs (usage in the
model's units), and how the two renderers and the two streams write it.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import assistant as A  # noqa: E402
from radixnet.dialogue import Turn, reply as dialogue_reply  # noqa: E402
from radixnet.duo import NegativeFilter  # noqa: E402
from radixnet.encoding import WORDS, Encoding  # noqa: E402
from radixnet.model import new_model  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as fh:
    CORPUS = [line for line in fh.read().splitlines() if line.strip()]

_MODELS = {}


def trained(kind="count", words=False):
    """A model trained once per kind and shared by the read-only tests (learn is switched off where it matters)."""
    key = (kind, words)
    if key not in _MODELS:
        encoding = Encoding(unit=WORDS, n=2, stride=1) if words else None
        model = new_model(kind, seed=1, encoding=encoding)
        model.train(CORPUS, epochs=2, **({"lr": 1.0, "batch_size": 1} if kind == "radix" else {}))
        _MODELS[key] = model
    return _MODELS[key]


def user(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


def events_of(model, ask, **options):
    seen = []
    reply = A.respond(model, ask, on_event=seen.append, **options)
    return reply, seen


# ---------------------------------------------------------------------------
# the requests
# ---------------------------------------------------------------------------


class TestParseOpenAI(unittest.TestCase):
    def test_a_plain_request(self):
        ask = A.parse_openai({"model": "radixnet-count", "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "developer", "content": "and kind"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": [{"type": "text", "text": "tell me"}, {"type": "text", "text": "more"}]},
        ], "max_tokens": 30, "temperature": 0.5, "stop": "xx", "n": 2, "stream": True,
            "stream_options": {"include_usage": True}, "top_p": 0.9, "user": "u", "logprobs": True})
        self.assertEqual(ask.model, "radixnet-count")
        self.assertEqual(ask.system, "be brief\nand kind")
        self.assertEqual([(m.role, m.text) for m in ask.messages], [("user", "hello"), ("assistant", "hi"), ("user", "tell me\nmore")])
        self.assertEqual((ask.max_tokens, ask.temperature, ask.stop, ask.n, ask.stream, ask.include_usage), (30, 0.5, ["xx"], 2, True, True))
        self.assertEqual(ask.previous, "tell me\nmore")
        self.assertFalse(ask.prefill)
        self.assertTrue(ask.thinking)
        # the defaults are the dialogue's
        self.assertEqual((ask.mode, ask.context, ask.k, ask.beam, ask.explore, ask.learn, ask.guard), ("beam", 12, 5, None, 3, True, True))
        self.assertEqual(ask.max_tokens, ask.dialogue_options()["max_length"])

    def test_the_dials_and_the_thinking_switch(self):
        ask = A.parse_openai(user("x", mode="sample", context=4, k=2, beam=8, step_penalty=0.5, explore=0, avoid_repeats=False,
                                  avoid_word_repeats=False, learn=False, guard=False, seed=7, reasoning_effort="none",
                                  max_completion_tokens=12))
        self.assertEqual((ask.mode, ask.context, ask.k, ask.beam, ask.step_penalty, ask.explore), ("sample", 4, 2, 8, 0.5, 0))
        self.assertEqual((ask.avoid_repeats, ask.avoid_word_repeats, ask.learn, ask.guard, ask.seed), (False, False, False, False, 7))
        self.assertFalse(ask.thinking)
        self.assertEqual(ask.max_tokens, 12)
        self.assertFalse(A.parse_openai(user("x", thinking=False)).thinking)
        self.assertEqual(A.parse_openai(user("x", mode="dijkstra")).mode, "beam")

    def test_tool_calls_and_results_become_the_transcript(self):
        ask = A.parse_openai({"messages": [
            {"role": "user", "content": "TASK: add"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": "{\"expression\": \"2+2\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ], "tools": [{"type": "function", "function": {"name": "calc", "description": "sums", "parameters": {"type": "object"}}}]})
        self.assertEqual(ask.messages[1].text, '<tool>calc {"expression": "2+2"}</tool>')
        self.assertEqual(ask.messages[2].text, "<result>4</result>")
        self.assertEqual(ask.messages[2].role, "tool")
        self.assertEqual(ask.tools, [{"name": "calc", "description": "sums", "parameters": {"type": "object"}}])
        self.assertEqual(A.parse_openai({**user("x"), "tools": ask.tools and [{"type": "function", "function": {"name": "calc"}}], "tool_choice": "none"}).tools, [])

    def test_what_is_refused(self):
        for body, param in (
            ({}, "messages"),
            ({"messages": "no"}, "messages"),
            ({"messages": []}, "messages"),
            ({"messages": [{"role": "system", "content": "only"}]}, "messages"),
            ({"messages": [{"role": "robot", "content": "x"}]}, "messages"),
            ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, "messages"),
            ({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}]}, "messages"),
            (user("x", max_tokens=0), "max_tokens"),
            (user("x", temperature=-1), "temperature"),
            (user("x", n=0), "n"),
            (user("x", n=99), "n"),
            (user("x", stop=[""]), "stop"),
            (user("x", mode="dijkstrah"), "mode"),
            (user("x", k=0), "k"),
            (user("x", tools=[{"type": "function", "function": {"name": "not a name"}}]), "tools"),
            (user("x", tools=[{"function": {"name": "a"}}, {"function": {"name": "a"}}]), "tools"),
            (user("x", stream_options="yes"), "stream_options"),
        ):
            with self.assertRaises(A.AskError, msg=body) as caught:
                A.parse_openai(body)
            self.assertEqual(caught.exception.param, param, body)
        with self.assertRaises(A.AskError):
            A.parse_openai("not an object")

    def test_empty_lines_are_not_heard(self):
        ask = A.parse_openai({"messages": [{"role": "user", "content": "  "}, {"role": "user", "content": "said"}]})
        self.assertEqual([m.text for m in ask.messages], ["said"])


class TestParseAnthropic(unittest.TestCase):
    def test_a_plain_request(self):
        ask = A.parse_anthropic({"model": "count", "system": [{"type": "text", "text": "be brief"}], "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": ""}, {"type": "text", "text": "hi"}]},
            {"role": "user", "content": [{"type": "text", "text": "more"}]},
        ], "max_tokens": 25, "stop_sequences": ["!"], "stream": True, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "metadata": {"user_id": "u"}})
        self.assertEqual(ask.system, "be brief")
        self.assertEqual([(m.role, m.text) for m in ask.messages], [("user", "hello"), ("assistant", "hi"), ("user", "more")])
        self.assertEqual((ask.max_tokens, ask.stop, ask.stream, ask.thinking, ask.n), (25, ["!"], True, True, 1))
        self.assertFalse(A.parse_anthropic(user("x", thinking={"type": "disabled"})).thinking)
        self.assertFalse(A.parse_anthropic(user("x", thinking=False)).thinking)
        self.assertEqual(A.parse_anthropic(user("x")).max_tokens, A.DEFAULT_MAX_TOKENS)

    def test_tool_blocks_become_the_transcript(self):
        ask = A.parse_anthropic({"messages": [
            {"role": "user", "content": "TASK: add"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"expression": "2+2"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "4"}]}]},
        ], "tools": [{"name": "calc", "description": "sums", "input_schema": {"type": "object"}}]})
        self.assertEqual(ask.messages[1].text, '<tool>calc {"expression": "2+2"}</tool>')
        self.assertEqual(ask.messages[2].text, "<result>4</result>")
        self.assertEqual(ask.tools[0], {"name": "calc", "description": "sums", "parameters": {"type": "object"}})
        self.assertTrue(ask.prefill is False)
        withdrawn = A.parse_anthropic({**user("x"), "tools": [{"name": "calc"}], "tool_choice": {"type": "none"}})
        self.assertEqual(withdrawn.tools, [])

    def test_what_is_refused(self):
        for body, param in (
            ({"messages": [{"role": "system", "content": "x"}]}, "messages"),
            ({"messages": [{"role": "tool", "content": "x"}]}, "messages"),
            ({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}, "messages"),
            (user("x", thinking={"type": "maybe"}), "thinking"),
            (user("x", stop_sequences="no"), "stop_sequences"),
            (user("x", max_tokens=-5), "max_tokens"),
        ):
            with self.assertRaises(A.AskError, msg=body) as caught:
                A.parse_anthropic(body)
            self.assertEqual(caught.exception.param, param, body)
        self.assertEqual(A.parse_request(user("x"), "anthropic").messages[0].text, "x")
        with self.assertRaises(A.AskError):
            A.parse_request(user("x"), "cohere")

    def test_a_prefill_is_the_assistant_s_own_line(self):
        ask = A.parse_anthropic({"messages": [{"role": "user", "content": "the cat"}, {"role": "assistant", "content": "sat on"}]})
        self.assertTrue(ask.prefill)
        self.assertEqual(ask.previous, "sat on")


class TestIds(unittest.TestCase):
    def test_a_model_is_named_by_its_kind(self):
        self.assertEqual(A.model_id(trained()), "radixnet-count")
        for name in ("", None, "radixnet", "default", "RadixNet"):
            self.assertEqual(A.kind_of_id(name), "")
        for name in ("radixnet-count", "count", "RadixNet-Count", " radixnet-count "):
            self.assertEqual(A.kind_of_id(name), "count")
        self.assertEqual(A.kind_of_id("radixnet-word"), "word")


# ---------------------------------------------------------------------------
# the reply
# ---------------------------------------------------------------------------


class TestRespond(unittest.TestCase):
    def test_the_reply_is_a_turn_and_the_thinking_is_its_trace(self):
        model = trained()
        ask = A.parse_openai(user("tell me about the cat", learn=False))
        reply, seen = events_of(model, ask)
        self.assertEqual((reply.model, reply.kind, reply.units, len(reply.choices)), ("radixnet-count", "count", "chars", 1))
        choice = reply.choices[0]
        turn = choice.turn
        self.assertIsNotNone(turn)
        self.assertEqual(choice.text, turn["text"])
        self.assertEqual(turn["speaker"], "assistant")
        self.assertEqual(turn["index"], 1)
        lines = choice.thinking.split("\n")
        self.assertEqual(lines[0], 'answering "tell me about the cat"')
        self.assertTrue(any(line.startswith('looking for "') and "in the graph" in line for line in lines), lines)
        self.assertTrue(lines[-1].startswith(f"saying {json.dumps(turn['text'])}: cost {turn['cost']:.4f}, probability"), lines[-1])
        self.assertIn("reached the end of a text" if turn["reached_end"] else "cut at 60 chars", lines[-1])
        self.assertEqual(choice.stop_reason, A.END_TURN if turn["reached_end"] else A.MAX_TOKENS)
        self.assertIsNone(choice.stop_sequence)
        self.assertIsNone(choice.guard)
        # the events join into the reply
        kinds = [e["type"] for e in seen]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-2:], ["done", "end"])
        self.assertEqual("".join(e["text"] for e in seen if e["type"] == "thinking"), choice.thinking)
        self.assertEqual("".join(e["text"] for e in seen if e["type"] == "text"), choice.text)
        self.assertLess(kinds.index("text"), kinds.index("done"))
        self.assertGreater(kinds.index("text"), max(i for i, k in enumerate(kinds) if k == "thinking"))
        start = seen[0]
        self.assertEqual((start["id"], start["model"], start["created"], start["input_units"]), (reply.token, reply.model, reply.created, reply.input_units))
        # usage counts units of the encoding: characters here
        self.assertEqual(reply.input_units, len("tell me about the cat"))
        self.assertEqual(choice.thinking_units, len(choice.thinking))
        self.assertEqual(choice.output_units, len(choice.text) + len(choice.thinking))
        self.assertEqual(reply.usage, {"input": 21, "output": choice.output_units, "thinking": choice.thinking_units, "total": 21 + choice.output_units})
        done = seen[-2]
        self.assertEqual((done["stop_reason"], done["output_units"], done["turn"]), (choice.stop_reason, choice.output_units, turn))

    def test_the_text_streams_one_node_at_a_time(self):
        model = trained()
        reply, seen = events_of(model, A.parse_openai(user("the cat sat on the", learn=False)))
        turn = reply.choices[0].turn
        pieces = [e["text"] for e in seen if e["type"] == "text"]
        self.assertEqual("".join(pieces), turn["text"])
        real = [label for node, label in zip(turn["node_ids"], turn["labels"]) if node >= 3]
        if len(real) >= 2:
            self.assertEqual(len(pieces), len(real), (pieces, real))
            # every piece after the first is what its node adds beyond the overlap
            self.assertEqual(pieces[1:], [label[model.encoding.overlap:] for label in real[1:]])

    def test_deltas(self):
        enc = Encoding()
        self.assertEqual(A.deltas(enc, ["<s>", "the", "he ", "e c", " ca", "cat", "</s>"], [0, 3, 4, 5, 6, 7, 1], "the cat"), ["the", " ", "c", "a", "t"])
        self.assertEqual(A.deltas(enc, ["the", "he ", "e cat"], [3, 4, 5], "xthe cat"), ["xthe", " ", "cat"])  # a merged node adds its whole tail
        self.assertEqual(A.deltas(enc, ["the", "he ", "e cat"], [3, 4, 5], "the ca"), ["the ca"])  # cut short: one piece
        self.assertEqual(A.deltas(enc, ["the"], [3], "the"), ["the"])
        self.assertEqual(A.deltas(enc, [], [], ""), [])
        words = Encoding(unit=WORDS, n=2, stride=1)
        self.assertEqual(A.deltas(words, ["<s>", "the cat", "cat sat", "sat on"], [0, 3, 4, 5], "the cat sat on"), ["the cat", " sat", " on"])
        self.assertEqual(A.deltas(words, ["the cat", "cat sat"], [3, 4], "sat"), ["sat"])

    def test_a_word_model_streams_words(self):
        model = trained(words=True)
        reply, seen = events_of(model, A.parse_openai(user("the cat sat on the", learn=False, max_tokens=8)))
        choice = reply.choices[0]
        self.assertEqual(reply.units, "words")
        pieces = [e["text"] for e in seen if e["type"] == "text"]
        self.assertEqual("".join(pieces), choice.text)
        self.assertTrue(all(p.startswith(" ") for p in pieces[1:]), pieces)
        self.assertEqual(reply.input_units, 5)
        self.assertEqual(choice.output_units, len(choice.text.split()) + len(choice.thinking.split()))

    def test_a_system_prompt_is_acknowledged_and_not_read(self):
        model = trained()
        body = {"messages": [{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "the cat sat"}], "learn": False}
        reply = A.respond(model, A.parse_openai(body))
        lines = reply.choices[0].thinking.split("\n")
        self.assertEqual(lines[1], "a system prompt was given; the network continues text and cannot follow instructions, so it is not read")
        self.assertEqual(reply.input_units, len("Answer in French.") + len("the cat sat"))
        without = A.respond(model, A.parse_openai(user("the cat sat", learn=False)))
        self.assertEqual(without.choices[0].text, reply.choices[0].text)

    def test_earlier_lines_are_heard_and_counted(self):
        model = trained()
        first = A.respond(model, A.parse_openai(user("the cat sat", learn=False)))
        said = first.choices[0].text
        body = {"messages": [{"role": "user", "content": "the cat sat"}, {"role": "assistant", "content": said},
                             {"role": "user", "content": "the cat sat"}], "learn": False}
        second = A.respond(model, A.parse_openai(body))
        self.assertIn("(2 earlier line(s) heard)", second.choices[0].thinking.split("\n")[0])
        self.assertNotEqual(second.choices[0].text, said)  # what was said is not said again

    def test_a_prefill_returns_only_what_it_adds(self):
        model = trained()
        body = {"messages": [{"role": "user", "content": "the cat"}, {"role": "assistant", "content": "the cat sat on the"}], "learn": False}
        reply, seen = events_of(model, A.parse_anthropic(body))
        choice = reply.choices[0]
        turn = choice.turn
        self.assertTrue(choice.thinking.startswith('continuing its own last line "the cat sat on the"'))
        if turn["context"]:
            self.assertEqual(choice.text, turn["reply"])
            self.assertEqual("".join(e["text"] for e in seen if e["type"] == "text"), turn["reply"])
            self.assertTrue(turn["text"].endswith(choice.text))

    def test_alternatives_differ_because_each_hears_the_last(self):
        model = trained()
        reply = A.respond(model, A.parse_openai(user("the cat sat", n=3, learn=False)))
        texts = [c.text for c in reply.choices]
        self.assertEqual([c.index for c in reply.choices], [0, 1, 2])
        self.assertEqual(len(set(t.casefold() for t in texts)), 3, texts)
        self.assertEqual(reply.usage["output"], sum(c.output_units for c in reply.choices))

    def test_a_stop_sequence_cuts_the_reply(self):
        model = trained()
        plain = A.respond(model, A.parse_openai(user("the cat sat", learn=False))).choices[0].text
        self.assertGreater(len(plain), 4)
        seq = plain[3:5]
        cut = A.respond(model, A.parse_openai(user("the cat sat", stop=[seq, "zzzz"], learn=False))).choices[0]
        self.assertEqual(cut.text, plain[:3])
        self.assertEqual((cut.stop_reason, cut.stop_sequence), (A.STOP_SEQUENCE, seq))
        self.assertIn(f"stopped at the stop sequence {json.dumps(seq)}", cut.thinking)
        self.assertEqual(cut.output_units, len(cut.text) + len(cut.thinking))

    def test_sampling_is_reproducible_under_a_seed(self):
        model = trained()
        a = A.respond(model, A.parse_openai(user("the dog", mode="sample", seed=3, learn=False, max_tokens=20)))
        b = A.respond(model, A.parse_openai(user("the dog", mode="sample", seed=3, learn=False, max_tokens=20)))
        self.assertEqual(a.choices[0].text, b.choices[0].text)
        self.assertIn("walk(s) drawn", a.choices[0].thinking)

    def test_nothing_to_say(self):
        empty = new_model("count", seed=1)
        reply, seen = events_of(empty, A.parse_openai(user("anything at all")))
        choice = reply.choices[0]
        self.assertEqual((choice.text, choice.stop_reason, choice.turn), ("", A.END_TURN, None))
        self.assertEqual(choice.thinking.split("\n")[-1], "nothing to say: the graph has no way on from here")
        self.assertFalse([e for e in seen if e["type"] == "text"])

    def test_learning_is_a_setting(self):
        model = new_model("count", seed=1)
        model.train(CORPUS, epochs=2)
        before = json.dumps(model.to_dict(), sort_keys=True)
        A.respond(model, A.parse_openai(user("the cat sat", n=4, learn=False)))
        self.assertEqual(json.dumps(model.to_dict(), sort_keys=True), before)


class _RejectsAll:
    """A pair that vetoes everything: what the guard looks like when every candidate is a known failure."""

    class _Negative:
        def __init__(self):
            self.encoding = Encoding()
            self.blamed = []

        def stats(self):
            return {"blame": 1.0}

        def blame(self, texts, **options):
            self.blamed.extend(texts)

    class _Config:
        learn = True
        reason = "filtered"

    def __init__(self):
        self.negative = self._Negative()
        self.config = self._Config()

    def judge(self, text):
        return {"text": text, "decision": "reject", "rule": "blame", "why": "it is all known failure; rejected", "reasons": [], "spans": []}

    def describe(self):
        return {"config": {"threshold": 0.5}}


class TestGuard(unittest.TestCase):
    def test_the_negative_network_vetoes_with_its_reason(self):
        model = trained()
        plain = A.respond(model, A.parse_openai(user("the cat sat", learn=False))).choices[0].text
        negative = NegativeNet(seed=2)
        negative.blame([plain], reason="nonsense", source="review", note="a test", severity=5)
        pair = NegativeFilter(model, negative)
        self.assertTrue(pair.ready)
        reply = A.respond(model, A.parse_openai(user("the cat sat", learn=False)), pair=pair)
        choice = reply.choices[0]
        self.assertIsNotNone(choice.guard)
        self.assertGreaterEqual(choice.guard["vetoed"], 1)
        self.assertEqual(choice.guard["rejected"][0]["text"], plain)
        self.assertIn(f"the negative network vetoed {json.dumps(plain)}: ", choice.thinking)
        self.assertNotEqual(choice.text, plain)
        self.assertEqual(set(choice.guard), {"on", "vetoed", "rejected", "verdicts", "negative", "config"})
        # the guard is the caller's to leave out
        unguarded = A.respond(model, A.parse_openai(user("the cat sat", learn=False)))
        self.assertEqual(unguarded.choices[0].text, plain)

    def test_everything_vetoed_is_a_refusal(self):
        model = trained()
        pair = _RejectsAll()
        reply = A.respond(model, A.parse_openai(user("the cat sat", learn=False)), pair=pair)
        choice = reply.choices[0]
        self.assertEqual((choice.text, choice.stop_reason, choice.turn), ("", A.REFUSAL, None))
        self.assertEqual(choice.thinking.split("\n")[-1], "nothing to say: the guard vetoed everything it could say")
        self.assertTrue(pair.negative.blamed, "with learn on, the filter blames what it refused")
        self.assertEqual(A.to_openai(reply)["choices"][0]["finish_reason"], "content_filter")
        self.assertEqual(A.to_anthropic(reply)["stop_reason"], "refusal")


def _turn(text, context=""):
    return Turn(index=1, speaker="assistant", text=text, context=context, reply=text[len(context):], cost=1.5,
                probability=0.2, reached_end=True)


class TestToolCalls(unittest.TestCase):
    """The model writes a call as text; the request decides whether that text is a tool call."""

    def test_a_call_to_an_offered_tool_comes_back_as_one(self):
        model = trained()
        written = 'TASK: add\n<tool>calc {"expression": "2+2"}</tool>\n'
        with mock.patch.object(A, "dialogue_reply", return_value=_turn(written)):
            ask = A.parse_openai({**user("TASK: add"), "learn": False,
                                  "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}]})
            reply, seen = events_of(model, ask)
        choice = reply.choices[0]
        self.assertEqual(choice.text, "TASK: add\n")
        self.assertEqual([(c.name, c.input) for c in choice.tool_calls], [("calc", {"expression": "2+2"})])
        self.assertEqual((choice.stop_reason, choice.stop_sequence), (A.TOOL_USE, None))
        self.assertIn('wrote a tool call: <tool>calc {"expression": "2+2"}</tool>', choice.thinking)
        self.assertIn("1 tool(s) offered (calc); a reply that writes one comes back as a tool call", choice.thinking)
        self.assertEqual([e["name"] for e in seen if e["type"] == "tool_use"], ["calc"])
        self.assertEqual("".join(e["text"] for e in seen if e["type"] == "text"), "TASK: add\n")
        doc = A.to_openai(reply)
        call = doc["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((call["type"], call["function"]["name"], json.loads(call["function"]["arguments"])), ("function", "calc", {"expression": "2+2"}))
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(doc["choices"][0]["finish_reason"], "tool_calls")
        blocks = A.to_anthropic(reply)["content"]
        self.assertEqual([b["type"] for b in blocks], ["thinking", "text", "tool_use"])
        self.assertEqual((blocks[2]["name"], blocks[2]["input"]), ("calc", {"expression": "2+2"}))
        self.assertTrue(blocks[2]["id"].startswith("toolu_"))

    def test_a_call_that_was_not_offered_stays_text(self):
        model = trained()
        written = '<tool>calc {"expression": "2+2"}</tool>\n'
        with mock.patch.object(A, "dialogue_reply", return_value=_turn(written)):
            plain = A.respond(model, A.parse_openai(user("TASK: add", learn=False))).choices[0]
            other = A.respond(model, A.parse_openai({**user("TASK: add"), "learn": False,
                                                     "tools": [{"type": "function", "function": {"name": "web_fetch"}}]})).choices[0]
        for choice, note in ((plain, "but no tools were offered"), (other, "which was not offered")):
            self.assertEqual(choice.text, written)
            self.assertEqual(choice.tool_calls, [])
            self.assertEqual(choice.stop_reason, A.END_TURN)
            self.assertIn(note, choice.thinking)

    def test_a_call_with_no_text_before_it(self):
        model = trained()
        schema = {"type": "object", "properties": {"expression": {"type": "string", "description": "what to work out"}},
                  "required": ["expression"]}
        with mock.patch.object(A, "dialogue_reply", return_value=_turn("<tool>calc 2+2</tool>")):
            ask = A.parse_anthropic({**user("add"), "learn": False, "tools": [{"name": "calc", "input_schema": schema}]})
            reply = A.respond(model, ask)
        choice = reply.choices[0]
        self.assertEqual(choice.text, "")
        # a bare value is the one argument of a one-argument tool, as the agent reads it
        self.assertEqual((choice.tool_calls[0].name, choice.tool_calls[0].input), ("calc", {"expression": "2+2"}))
        self.assertEqual(A.to_openai(reply)["choices"][0]["message"]["content"], None)
        self.assertEqual([b["type"] for b in A.to_anthropic(reply)["content"]], ["thinking", "tool_use"])


# ---------------------------------------------------------------------------
# the renderers and the streams
# ---------------------------------------------------------------------------


class TestRenderers(unittest.TestCase):
    def test_chat_completion(self):
        model = trained()
        reply = A.respond(model, A.parse_openai(user("the cat sat", learn=False)))
        doc = A.to_openai(reply)
        self.assertEqual(set(doc), {"id", "object", "created", "model", "choices", "usage", "radixnet"})
        self.assertTrue(doc["id"].startswith("chatcmpl-"))
        self.assertEqual((doc["object"], doc["model"]), ("chat.completion", "radixnet-count"))
        choice = doc["choices"][0]
        self.assertEqual(set(choice), {"index", "message", "logprobs", "finish_reason"})
        self.assertEqual(choice["message"], {"role": "assistant", "content": reply.choices[0].text, "reasoning_content": reply.choices[0].thinking})
        usage = doc["usage"]
        self.assertEqual(usage, {"prompt_tokens": 11, "completion_tokens": reply.usage["output"], "total_tokens": reply.usage["total"],
                                 "completion_tokens_details": {"reasoning_tokens": reply.usage["thinking"]}})
        self.assertEqual(doc["radixnet"]["choices"][0]["turn"], reply.choices[0].turn)
        quiet = A.respond(model, A.parse_openai(user("the cat sat", learn=False, thinking=False)))
        self.assertNotIn("reasoning_content", A.to_openai(quiet)["choices"][0]["message"])
        self.assertEqual(A.to_format(reply, "openai"), doc)

    def test_message(self):
        model = trained()
        reply = A.respond(model, A.parse_anthropic(user("the cat sat", learn=False)))
        doc = A.to_anthropic(reply)
        self.assertEqual(set(doc), {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage", "radixnet"})
        self.assertTrue(doc["id"].startswith("msg_"))
        self.assertEqual((doc["type"], doc["role"], doc["model"]), ("message", "assistant", "radixnet-count"))
        self.assertEqual(doc["content"], [{"type": "thinking", "thinking": reply.choices[0].thinking, "signature": ""},
                                          {"type": "text", "text": reply.choices[0].text}])
        self.assertEqual(doc["usage"], {"input_tokens": 11, "output_tokens": reply.choices[0].output_units})
        self.assertIn(doc["stop_reason"], A.STOP_REASONS)
        quiet = A.respond(model, A.parse_anthropic(user("the cat sat", learn=False, thinking={"type": "disabled"})))
        self.assertEqual([b["type"] for b in A.to_anthropic(quiet)["content"]], ["text"])
        with self.assertRaises(A.AskError):
            A.to_format(reply, "gemini")

    def test_sse_frames(self):
        self.assertEqual(A.sse({"a": 1}), 'data: {"a":1}\n\n')
        self.assertEqual(A.sse("[DONE]"), "data: [DONE]\n\n")
        self.assertEqual(A.sse({"type": "ping"}, "ping"), 'event: ping\ndata: {"type":"ping"}\n\n')


def _frames_of(renderer, events):
    out = []
    for event in events:
        out.extend(renderer.frames(event))
    return out


def _data_of(frame):
    lines = [line for line in frame.strip().split("\n") if line.startswith("data: ")]
    return json.loads(lines[0][6:]) if lines and lines[0] != "data: [DONE]" else lines[0][6:]


class TestStreams(unittest.TestCase):
    def test_chat_completion_chunks(self):
        model = trained()
        reply, seen = events_of(model, A.parse_openai(user("the cat sat", learn=False)))
        chunks = [_data_of(f) for f in _frames_of(A.OpenAIStream(reply.token, reply.model, reply.created, include_usage=True), seen)]
        self.assertEqual(chunks[-1], "[DONE]")
        self.assertEqual(chunks[-2]["choices"], [])
        self.assertEqual(chunks[-2]["usage"]["prompt_tokens"], 11)
        self.assertEqual(chunks[-2]["usage"]["completion_tokens_details"]["reasoning_tokens"], reply.usage["thinking"])
        body = chunks[:-2]
        self.assertTrue(all(c["object"] == "chat.completion.chunk" and c["id"] == "chatcmpl-" + reply.token for c in body))
        self.assertEqual(body[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        thinking = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in body)
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in body)
        self.assertEqual((thinking, text), (reply.choices[0].thinking, reply.choices[0].text))
        self.assertEqual(body[-1]["choices"][0], {"index": 0, "delta": {}, "finish_reason": A.FINISH_REASONS[reply.choices[0].stop_reason]})
        # without include_usage there is no usage chunk; without thinking there is no reasoning_content
        plain = [_data_of(f) for f in _frames_of(A.OpenAIStream(reply.token, reply.model, reply.created, thinking=False), seen)]
        self.assertEqual(plain[-1], "[DONE]")
        self.assertTrue("finish_reason" in plain[-2]["choices"][0])
        self.assertFalse(any("reasoning_content" in c["choices"][0]["delta"] for c in plain[:-1]))
        self.assertIn('"type":"server_error"', A.OpenAIStream.error("boom"))

    def test_message_events(self):
        model = trained()
        reply, seen = events_of(model, A.parse_anthropic(user("the cat sat", learn=False)))
        frames = _frames_of(A.AnthropicStream(reply.token, reply.model, reply.created, input_units=reply.input_units), seen)
        names = [f.split("\n", 1)[0].replace("event: ", "") for f in frames]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        docs = [_data_of(f) for f in frames]
        self.assertEqual(docs[0]["message"]["id"], "msg_" + reply.token)
        self.assertEqual(docs[0]["message"]["usage"], {"input_tokens": 11, "output_tokens": 0})
        starts = [d for d in docs if d["type"] == "content_block_start"]
        self.assertEqual([s["content_block"]["type"] for s in starts], ["thinking", "text"])
        self.assertEqual([s["index"] for s in starts], [0, 1])
        thinking = "".join(d["delta"]["thinking"] for d in docs if d["type"] == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
        text = "".join(d["delta"]["text"] for d in docs if d["type"] == "content_block_delta" and d["delta"]["type"] == "text_delta")
        self.assertEqual((thinking, text), (reply.choices[0].thinking, reply.choices[0].text))
        self.assertEqual([d["index"] for d in docs if d["type"] == "content_block_stop"], [0, 1])
        self.assertEqual(docs[-2]["delta"], {"stop_reason": reply.choices[0].stop_reason, "stop_sequence": None})
        self.assertEqual(docs[-2]["usage"], {"output_tokens": reply.choices[0].output_units})
        # thinking off: one text block; nothing to say: still one (empty) text block
        quiet = _frames_of(A.AnthropicStream("t", "m", 1, thinking=False), seen)
        self.assertEqual([_data_of(f)["content_block"]["type"] for f in quiet if "content_block_start" in f], ["text"])
        empty = new_model("count", seed=1)
        _reply, silent = events_of(empty, A.parse_anthropic(user("anything")))
        docs = [_data_of(f) for f in _frames_of(A.AnthropicStream("t", "m", 1), silent)]
        self.assertEqual([d["content_block"]["type"] for d in docs if d["type"] == "content_block_start"], ["thinking", "text"])
        self.assertIn('"type":"api_error"', A.AnthropicStream.error("boom"))

    def test_stream_yields_the_same_events(self):
        model = trained()
        ask = A.parse_openai(user("the cat sat", learn=False))
        streamed = list(A.stream(model, ask))
        _reply, direct = events_of(model, ask)
        strip = lambda events: [{k: v for k, v in e.items() if k not in ("id", "created")} for e in events]  # noqa: E731
        self.assertEqual(strip(streamed), strip(direct))
        self.assertEqual(streamed[-1]["type"], "end")
        with self.assertRaises(A.AskError):
            list(A.stream(model, A.Ask()))


class TestTrace(unittest.TestCase):
    def test_reply_says_what_it_does(self):
        model = trained()
        heard = []
        turn = dialogue_reply(model, "the cat sat on the mat", trace=heard.append, learn=False)
        self.assertIsNotNone(turn)
        kinds = [e["kind"] for e in heard]
        self.assertEqual(kinds[0], "context")
        self.assertIn("candidates", kinds)
        self.assertIn("pick", kinds)
        self.assertTrue(set(kinds) <= {"context", "candidates", "pick", "rethink", "fresh"})
        for event in heard:
            if event["kind"] == "candidates":
                self.assertEqual(set(event), {"kind", "context", "offered", "mode"})
            if event["kind"] == "pick":
                self.assertEqual(set(event), {"kind", "skipped", "vetoed", "repeat", "spoken", "caught"})
        # without a trace nothing changes
        again = dialogue_reply(model, "the cat sat on the mat", learn=False)
        self.assertEqual(again.text, turn.text)


if __name__ == "__main__":
    unittest.main()
