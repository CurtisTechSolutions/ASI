"""Talking with the model by voice (``radixnet/voicechat.py``, D-089): heard, trained, answered, spoken, taught.

One turn of the Voice tab, on the module alone (the model's parts as callables, Ollama faked), over the API
(whole and streamed) and on the command line (``speech talk``).  The voice needs the phonetic tokenizer beside
this checkout; the tests that speak skip without it.
"""

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(os.path.dirname(ROOT), "PhoneticTokenizer"))

from radixnet.dialogue import Heard, Turn  # noqa: E402
from radixnet.encoding import ACOUSTIC, PHONES, Encoding  # noqa: E402
from radixnet.ollama import OllamaClient  # noqa: E402
from radixnet.voicechat import (  # noqa: E402
    ANSWERS, CHUNK_SAMPLES, SPEAKERS, VoiceOptions, audio_events, describe, hear, history_pairs, speech_encoding, turn,
)
from tests.test_ollama import closed_port_url, start_fake  # noqa: E402
from tests.test_speech import tone, wav  # noqa: E402

try:
    import phonetok  # noqa: F401

    from radixnet.voice import say
except ImportError:  # pragma: no cover
    phonetok = None

CORPUS = ["the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "a bird in the hand"]


def recording(seconds: float = 0.2) -> bytes:
    return wav(tone(seconds=seconds))


def a_turn(text: str, fresh: bool = False) -> Turn:
    return Turn(index=1, speaker=SPEAKERS[1], text=text, context="" if fresh else text.split()[0], reply=text,
                cost=1.0, probability=0.5, reached_end=True, fresh=fresh)


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestOneTurn(unittest.TestCase):
    """The turn on its own: the model's parts are callables, and every step is an event."""

    def test_heard_trained_answered_and_spoken(self):
        events: list[dict] = []
        learned: list[list[str]] = []
        asked: dict = {}

        def learn(texts):
            learned.append(list(texts))
            return {"texts": len(texts), "epochs": 1}

        def model_reply(line, heard):
            asked.update(line=line, heard=heard)
            return a_turn("sat on the mat")

        out = turn(
            recording(), encoding=Encoding(), options=VoiceOptions(transcript="the cat sat", answer="model", epochs=1),
            history=[["You", "hello"], {"speaker": "Model", "text": "hi there"}], learn=learn, model_reply=model_reply,
            write=events.append,
        )
        doc = out.document
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[:3], ["heard", "trained", "reply"])
        self.assertEqual(kinds[-1], "spoken")
        self.assertTrue(all(k == "audio" for k in kinds[3:-1]) and len(kinds) > 4)
        # heard: the transcript and the waveform behind one token, both learned before the reply
        self.assertEqual(doc["transcript"], "the cat sat")
        self.assertTrue(doc["token"].startswith("<speech:"))
        self.assertEqual(len(doc["texts"]), 2)
        self.assertTrue(doc["texts"][0].endswith(" the cat sat") and "aud:mu:" in doc["texts"][1])
        self.assertEqual(learned, [doc["texts"]])
        self.assertEqual(doc["trained"]["texts"], 2)
        self.assertEqual(doc["audio"]["codec"], "mu")
        # answered: the line, with the conversation so far heard
        self.assertEqual(asked["line"], "the cat sat")
        self.assertIsInstance(asked["heard"], Heard)
        self.assertTrue(asked["heard"].match("hi there"))
        self.assertTrue(asked["heard"].match("the cat sat"))
        self.assertEqual((doc["by"], doc["reply"]["text"], doc["reply"]["spelled"]), ("model", "sat on the mat", "sat on the mat"))
        self.assertEqual(doc["reply"]["turn"]["text"], "sat on the mat")
        self.assertIsNone(doc["reply"]["ollama_error"])
        self.assertEqual(doc["history"], [["You", "hello"], ["Model", "hi there"], ["You", "the cat sat"], ["Model", "sat on the mat"]])
        # spoken: the audio events are the reply's speech, chunk by chunk, exactly what the decoder says
        chunks = [e for e in events if e["event"] == "audio"]
        pcm = b"".join(base64.b64decode(e["pcm_base64"]) for e in chunks)
        self.assertEqual(pcm, out.pcm)
        self.assertEqual(pcm, say(Encoding(), ["sat on the mat"]).pcm)
        self.assertTrue(all(e["samples"] <= CHUNK_SAMPLES and e["rate"] == 16000 for e in chunks))
        self.assertEqual(sum(e["samples"] for e in chunks), doc["spoken"]["samples"])
        self.assertEqual((doc["spoken"]["decoder"], out.rate), ("voice", 16000))
        self.assertTrue(out.wav().startswith(b"RIFF"))
        self.assertEqual(len(out.wav()), 44 + len(pcm))
        self.assertIsNone(doc["taught"])
        self.assertEqual(doc["encoding"], "char:3:1")

    def test_typed_text_nothing_to_say_and_nobody_asked(self):
        events: list[dict] = []
        calls = {"learn": 0, "reply": 0}

        def learn(texts):
            calls["learn"] += 1
            return {"texts": len(texts)}

        def model_reply(line, heard):
            calls["reply"] += 1
            return None

        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="hello there"), learn=learn,
                   model_reply=model_reply, write=events.append)
        doc = out.document
        self.assertEqual(len(doc["texts"]), 1)  # the transcript alone, behind its token
        self.assertIsNone(doc["audio"])
        self.assertEqual((doc["by"], doc["reply"]["text"], doc["spoken"], out.pcm), ("none", "", None, b""))
        self.assertEqual([e["event"] for e in events], ["heard", "trained", "reply"])
        self.assertEqual(doc["history"], [["You", "hello there"]])
        self.assertEqual(calls, {"learn": 1, "reply": 1})
        # nobody asked: learned, not answered
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="hello", answer="none"), learn=learn,
                   model_reply=model_reply)
        self.assertEqual((out.document["by"], calls["reply"]), ("none", 1))
        # not learned when told not to, or when there is nothing to learn from
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="hello", train=False), learn=learn,
                   model_reply=model_reply)
        self.assertIsNone(out.document["trained"])
        self.assertEqual(calls["learn"], 2)
        with self.assertRaises(ValueError):
            turn(None, encoding=Encoding(), options=VoiceOptions())  # nothing said at all
        with self.assertRaises(ValueError):
            turn(None, encoding=Encoding(), options=VoiceOptions(transcript="hi", answer="nobody"))
        with self.assertRaises(ValueError):
            turn(None, encoding=Encoding(), options=VoiceOptions(transcript="hi"), history=["not a pair"])

    def test_ollama_answers_for_the_model_and_the_model_is_taught(self):
        fake = start_fake(self.addCleanup)
        client = OllamaClient(fake.url, "fake:latest")
        learned: list[list[str]] = []

        def learn(texts):
            learned.append(list(texts))
            return {"texts": len(texts)}

        events: list[dict] = []
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="how are you", answer="ollama", persona="a cat"),
                   history=[["You", "hello"], ["Model", "hi"]], learn=learn, model_reply=lambda line, heard: a_turn("never"),
                   ollama=client, write=events.append)
        doc = out.document
        self.assertEqual(doc["by"], "ollama")
        self.assertTrue(doc["reply"]["text"])
        self.assertEqual(doc["reply"]["spelled"], doc["reply"]["text"])
        self.assertIsNone(doc["reply"]["turn"])
        self.assertEqual(learned, [doc["texts"], [doc["reply"]["text"]]])  # what was said, then what Ollama said
        self.assertEqual(doc["taught"]["texts"], 1)
        self.assertEqual([e["event"] for e in events if e["event"] not in ("audio",)], ["heard", "trained", "reply", "spoken", "taught"])
        self.assertGreater(len(out.pcm), 1000)
        method, path, body = fake.requests[-1]
        self.assertEqual((method, path, body["model"]), ("POST", "/api/generate", "fake:latest"))
        self.assertIn("voice of a very small language model", body["system"])
        self.assertIn("You are a cat", body["system"])
        self.assertIn("You: how are you", body["prompt"])
        self.assertIn("Model: hi", body["prompt"])
        # auto: the model first; Ollama when it has nothing to say, or only a fresh text that picked up none of the line
        asked_before = len(fake.requests)
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="auto"), learn=learn,
                   model_reply=lambda line, heard: a_turn("the cat sat"), ollama=client)
        self.assertEqual((out.document["by"], len(fake.requests)), ("model", asked_before))
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="auto"), learn=learn,
                   model_reply=lambda line, heard: None, ollama=client)
        self.assertEqual((out.document["by"], len(fake.requests)), ("ollama", asked_before + 1))
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="auto"), learn=learn,
                   model_reply=lambda line, heard: a_turn("a bird", fresh=True), ollama=client)
        self.assertEqual((out.document["by"], len(fake.requests)), ("ollama", asked_before + 2))
        # not taught when told not to
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="ollama", learn_reply=False),
                   learn=learn, ollama=client)
        self.assertEqual((out.document["by"], out.document["taught"]), ("ollama", None))

    def test_an_unreachable_ollama_is_recorded_not_raised(self):
        down = OllamaClient(closed_port_url(), "fake:latest", timeout=1)
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="ollama"), ollama=down)
        self.assertEqual((out.document["by"], out.document["reply"]["text"]), ("none", ""))
        self.assertIn("cannot reach Ollama", out.document["reply"]["ollama_error"])
        # in auto the model's fresh text stands when Ollama is down, and the error is on record
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="auto"),
                   model_reply=lambda line, heard: a_turn("a bird", fresh=True), ollama=down)
        self.assertEqual((out.document["by"], out.document["reply"]["text"]), ("model", "a bird"))
        self.assertIn("cannot reach Ollama", out.document["reply"]["ollama_error"])
        out = turn(None, encoding=Encoding(), options=VoiceOptions(transcript="the cat", answer="auto"), ollama=None)
        self.assertEqual(out.document["reply"]["ollama_error"], "no Ollama to ask")

    def test_a_model_of_sounds_and_one_of_acoustic_units(self):
        # sounds: the model's reply is phones, spoken as they are and spelled for the record
        out = turn(None, encoding=Encoding(unit=PHONES), options=VoiceOptions(transcript="the cat", answer="model"),
                   model_reply=lambda line, heard: a_turn("DH AH0 # D AO1 G"))
        self.assertEqual((out.document["reply"]["spelled"], out.document["spoken"]["decoder"]), ("the dog", "voice"))
        self.assertEqual(out.pcm, say(Encoding(unit=PHONES), ["DH AH0 # D AO1 G"]).pcm)
        # acoustic units: the recording is heard as units, the one text of sound the model can learn
        enc = Encoding(unit=ACOUSTIC)
        heard = hear(recording(0.3), enc, VoiceOptions(transcript="the cat"))
        self.assertEqual(heard["transcript"], "the cat")
        self.assertIsNone(heard["token"])
        self.assertEqual(len(heard["texts"]), 1)
        self.assertTrue(all(u.startswith("q") for u in heard["texts"][0].split()))
        self.assertEqual(heard["units"], heard["texts"][0])
        learned: list[list[str]] = []
        out = turn(recording(0.3), encoding=enc, options=VoiceOptions(transcript="the cat", answer="model"),
                   learn=lambda texts: learned.append(list(texts)) or {"texts": len(texts)},
                   model_reply=lambda line, heard: a_turn("q2 q28 q55 q5"))
        self.assertEqual(learned, [out.document["texts"]])
        self.assertEqual((out.document["by"], out.document["spoken"]["decoder"]), ("model", "vocoder"))
        self.assertEqual(out.rate, 16000)
        # a reply Ollama wrote is words: spoken through the voice, and not taught to a model of units
        fake = start_fake(self.addCleanup)
        out = turn(None, encoding=enc, options=VoiceOptions(transcript="the cat", answer="ollama"),
                   learn=lambda texts: learned.append(list(texts)) or {"texts": len(texts)},
                   ollama=OllamaClient(fake.url, "fake:latest"))
        self.assertEqual((out.document["by"], out.document["spoken"]["decoder"], out.document["taught"]), ("ollama", "voice", None))
        self.assertEqual(out.document["texts"], [])  # nothing of sound to learn from a typed line
        self.assertEqual(len(learned), 1)
        # without a recording an acoustic model learns nothing, and without a transcript nobody can answer
        out = turn(None, encoding=enc, options=VoiceOptions(transcript="hi", answer="none"))
        self.assertEqual((out.document["texts"], out.document["trained"]), ([], None))

    def test_the_small_parts(self):
        self.assertEqual(history_pairs([("A", " x  y "), {"speaker": "", "text": "z"}, ["B", "  "]]), [("A", "x y"), ("You", "z")])
        for bad in (["x"], [("A", 1)], [1]):
            with self.assertRaises(ValueError):
                history_pairs(bad)
        self.assertEqual(speech_encoding(Encoding(unit=ACOUSTIC), "ollama"), Encoding())
        self.assertEqual(speech_encoding(Encoding(unit=ACOUSTIC), "model"), Encoding(unit=ACOUSTIC))
        self.assertEqual(speech_encoding(Encoding(unit=PHONES), "ollama"), Encoding(unit=PHONES))
        chunks = list(audio_events(bytes(2 * 20000), 16000, chunk=8000))
        self.assertEqual([c["samples"] for c in chunks], [8000, 8000, 4000])
        self.assertEqual(len(base64.b64decode(chunks[0]["pcm_base64"])), 16000)
        for bad in (dict(answer="x"), dict(mode="dijkstra"), dict(epochs=-1), dict(pitch=0), dict(k=0), dict(chunk=0)):
            with self.assertRaises(ValueError):
                VoiceOptions(**bad).validate()
        VoiceOptions().validate()
        info = describe(Encoding(unit=PHONES), OllamaClient("http://127.0.0.1:1", "m"))
        self.assertEqual((info["speakers"], info["answers"], info["decoder"], info["acoustic"]), (list(SPEAKERS), list(ANSWERS), "voice", False))
        self.assertEqual(info["ollama"], {"url": "http://127.0.0.1:1", "model": "m"})
        self.assertIn("backends", info["transcription"])


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestVoiceApi(unittest.TestCase):
    """``POST /api/voice/turn``, whole and streamed, and ``GET /api/voice``."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup)
        cls.service.model.train(CORPUS, epochs=2, batch_size=8)

    def test_a_turn_whole(self):
        body = {"name": "u.wav", "content_base64": base64.b64encode(recording()).decode("ascii"),
                "transcript": "the cat sat on the mat", "answer": "model", "epochs": 1, "seed": 1}
        before = self.service.model.stats()["trained_texts"]
        status, doc, _ = self.client.post("/api/voice/turn", body)
        self.assertEqual(status, 200, doc)
        self.assertEqual((doc["transcript"], doc["by"], doc["encoding"], doc["rate"]), ("the cat sat on the mat", "model", "char:3:1", 16000))
        self.assertEqual((doc["trained"]["texts"], doc["trained"]["epochs"]), (2, 1))
        self.assertEqual(self.service.model.stats()["trained_texts"], before + 2)
        self.assertTrue(doc["reply"]["text"])
        self.assertEqual(doc["reply"]["turn"]["speaker"], "Model")
        self.assertTrue(base64.b64decode(doc["wav_base64"]).startswith(b"RIFF"))
        self.assertEqual(len(base64.b64decode(doc["wav_base64"])), 44 + 2 * doc["spoken"]["samples"])
        self.assertEqual(doc["history"][-2:], [["You", "the cat sat on the mat"], ["Model", doc["reply"]["text"]]])
        # a recording alone, with no words heard in it and no transcription backend to hear them: the sound is
        # learned, nobody can answer
        status, doc, _ = self.client.post("/api/voice/turn", {"name": "u.wav", "content_base64": base64.b64encode(recording()).decode("ascii"), "answer": "model", "backend": "given"})
        self.assertEqual((status, doc["transcript"], doc["by"], len(doc["texts"])), (200, "", "none", 1), doc)
        self.assertIn("aud:mu:", doc["texts"][0])
        self.assertEqual((doc["reply"]["text"], doc["spoken"], doc["wav_base64"]), ("", None, None))
        # typed, unheard of a recording, not spoken, nobody answering
        status, doc, _ = self.client.post("/api/voice/turn", {"transcript": "a bird in the hand", "answer": "none", "speak": False})
        self.assertEqual((status, doc["by"], doc["wav_base64"], doc["spoken"], len(doc["texts"])), (200, "none", None, None, 1))

    def test_a_turn_streamed(self):
        body = {"transcript": "the cat sat", "answer": "model", "history": [["You", "hello"], ["Model", "hi there"]],
                "epochs": 1}
        status, raw, headers = self.client.post("/api/voice/turn/stream", body)
        self.assertEqual(status, 200, raw)
        self.assertTrue(headers.get("Content-Type", "").startswith("application/x-ndjson"), headers)
        events = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[:3], ["heard", "trained", "reply"])
        self.assertEqual(kinds[-2:], ["spoken", "done"])
        self.assertIn("audio", kinds)
        done = events[-1]
        self.assertNotIn("wav_base64", done)
        self.assertEqual(done["history"][:2], [["You", "hello"], ["Model", "hi there"]])
        pcm = b"".join(base64.b64decode(e["pcm_base64"]) for e in events if e["event"] == "audio")
        self.assertEqual(len(pcm) // 2, done["spoken"]["samples"])
        # a request refused before anything was streamed is an ordinary 400
        status, err, headers = self.client.post("/api/voice/turn/stream", {"answer": "model"})
        self.assertEqual(status, 400, err)
        self.assertIn("nothing was said", err["error"])

    def test_what_the_tab_has_to_work_with_and_the_refusals(self):
        status, info, _ = self.client.get("/api/voice")
        self.assertEqual(status, 200, info)
        self.assertEqual((info["speakers"], info["answers"], info["encoding"], info["decoder"]), (["You", "Model"], list(ANSWERS), "char:3:1", "voice"))
        self.assertTrue(info["ollama"]["url"].startswith("http"))
        for body in ({}, {"transcript": "hi", "answer": "nobody"}, {"transcript": "hi", "epochs": "many"},
                     {"transcript": "hi", "history": "not a list"}, {"transcript": "hi", "pitch": 0},
                     {"name": "u.wav", "content_base64": "bm90IGF1ZGlv", "answer": "model"}):
            status, err, _ = self.client.post("/api/voice/turn", body)
            self.assertEqual(status, 400, (body, err))
        # Ollama that cannot be reached: the turn still happens, the error is on record
        status, doc, _ = self.client.post("/api/voice/turn", {"transcript": "the cat sat", "answer": "ollama", "url": closed_port_url(), "timeout": 1, "speak": False})
        self.assertEqual((status, doc["by"]), (200, "none"), doc)
        self.assertIn("cannot reach Ollama", doc["reply"]["ollama_error"])
        self.assertEqual(doc["trained"]["texts"], 1)


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestVoiceCli(unittest.TestCase):
    def test_speech_talk(self):
        from radixnet.cli import main

        def run(argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = main(argv)
            return code, out.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            corpus = os.path.join(tmp, "c.txt")
            clip = os.path.join(tmp, "clip.wav")
            reply = os.path.join(tmp, "reply.wav")
            with open(corpus, "w", encoding="utf-8") as fh:
                fh.write("\n".join(CORPUS) + "\n")
            with open(clip, "wb") as fh:
                fh.write(recording())
            code, out = run(["--model", model, "--kind", "count", "--encoding", "char:3:1", "--seed", "1", "train", "--data", corpus, "--epochs", "1"])
            self.assertEqual(code, 0, out)
            code, out = run(["--model", model, "--json", "speech", "talk", clip, "--text", "the cat sat on the mat", "--answer", "model", "--epochs", "1", "--out", reply])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual((doc["by"], doc["source"], doc["sink"], doc["trained"]["texts"]), ("model", clip, reply, 2))
            self.assertEqual(doc["saved"]["path"], model)
            self.assertTrue(doc["reply"]["text"])
            with wave.open(reply) as w:
                self.assertEqual(w.getnframes(), doc["spoken"]["samples"])
            # typed, with a conversation so far, nobody answering, nothing learned, nothing saved
            history = os.path.join(tmp, "so-far.txt")
            with open(history, "w", encoding="utf-8") as fh:
                fh.write("You: hello\nModel: hi there\nthe dog sat\n")
            code, out = run(["--model", model, "--json", "speech", "talk", "--text", "the cat", "--answer", "none", "--no-train", "--history", history, "--out", reply])
            self.assertEqual(code, 0, out)
            doc = json.loads(out)
            self.assertEqual(doc["history"], [["You", "hello"], ["Model", "hi there"], ["You", "the dog sat"], ["You", "the cat"]])
            self.assertEqual((doc["by"], doc["saved"], doc["trained"], doc["sink"]), ("none", None, None, None))
            # nothing said
            self.assertNotEqual(run(["--model", model, "speech", "talk", "--answer", "model"])[0], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
