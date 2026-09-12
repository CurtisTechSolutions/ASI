"""Tests for radixnet.speech - teaching the model by talking to it.

Everything here runs without a transcription backend, ffmpeg or a microphone:
WAV files are built by hand (every sample format the reader accepts), the
"given" transcriber covers the speech-to-text path the browser uses, and the
optional backends are exercised with fakes that have the real interface.
"""

import array
import base64
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import speech  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tone(seconds=0.25, rate=16000, freq=220.0, amplitude=0.5, channels=1):
    """A sine wave as floats in [-1, 1], interleaved over ``channels``."""
    frames = int(seconds * rate)
    values = array.array("f")
    for i in range(frames):
        sample = amplitude * math.sin(2.0 * math.pi * freq * i / rate)
        for channel in range(channels):
            values.append(sample if channel == 0 else -sample)
    return values


def wav(samples, rate=16000, channels=1, bits=16, fmt=speech._WAVE_PCM, extra=b""):
    """Pack samples into a RIFF/WAVE file of the requested sample format."""
    if fmt == speech._WAVE_FLOAT and bits == 32:
        body = array.array("f", samples).tobytes()
    elif fmt == speech._WAVE_FLOAT and bits == 64:
        body = array.array("d", samples).tobytes()
    elif bits == 8:
        body = bytes(max(0, min(255, int(round(v * 127.0)) + 128)) for v in samples)
    elif bits == 24:
        body = b"".join(
            (max(-8388608, min(8388607, int(round(v * 8388607.0)))) & 0xFFFFFF).to_bytes(3, "little") for v in samples
        )
    elif bits == 32:
        body = array.array("i", (max(-2147483648, min(2147483647, int(round(v * 2147483647.0)))) for v in samples)).tobytes()
    else:
        body = array.array("h", (max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples)).tobytes()
    if sys.byteorder == "big":
        raise unittest.SkipTest("the fixtures assume a little-endian host")
    block = channels * (bits // 8)
    fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, rate * block, block, bits) + extra
    return (
        struct.pack("<4sI4s", b"RIFF", 36 + len(fmt_chunk) - 16 + len(body), b"WAVE")
        + struct.pack("<4sI", b"fmt ", len(fmt_chunk)) + fmt_chunk
        + struct.pack("<4sI", b"data", len(body)) + body
    )


SPOKEN = "the cat sat on the mat"


class TestTextFormat(unittest.TestCase):
    def test_pack_and_parse(self):
        text = speech.pack_text("mu", 8000, 1, bytes(range(64)))
        self.assertTrue(text.startswith("aud:mu:8000x1:"))
        self.assertEqual(speech.parse_text(text), ("mu", 8000, 1, bytes(range(64)), False))

    def test_the_header_is_found_after_a_token_and_before_a_transcript(self):
        text = speech.pack_text("mu", 8000, 1, bytes(range(48)))
        token = speech.utterance_token(text)
        self.assertEqual(speech.parse_text(f"{token} {text}")[3], bytes(range(48)))
        # a paired text carries the transcript behind the waveform: the payload stops at the space
        self.assertEqual(speech.parse_text(f"{token} {text} {SPOKEN}")[3], bytes(range(48)))

    def test_parse_repairs_junk_and_truncation(self):
        text = speech.pack_text("mu", 8000, 1, bytes(range(256)))
        codec, rate, channels, payload, repaired = speech.parse_text(text[:-30])
        self.assertEqual((codec, rate, channels, repaired), ("mu", 8000, 1, True))
        self.assertLess(len(payload), 256)
        self.assertEqual(payload, bytes(range(256))[: len(payload)])
        self.assertEqual(speech.parse_text(text[:40] + "!*" + text[40:])[3], bytes(range(256)))
        for bad in ("hello", "aud:mu:8000", "aud:mu:0x1:AAAA", ""):
            with self.assertRaises(speech.SpeechError):
                speech.parse_text(bad)

    def test_tokens_are_unique_per_utterance_and_shared_by_both_texts(self):
        one, two = speech.utterance_token(b"first"), speech.utterance_token(b"second")
        self.assertNotEqual(one, two)
        self.assertEqual(one, speech.utterance_token(b"first"))  # deterministic: the same sound, the same token
        self.assertTrue(one.startswith("<speech:") and one.endswith(">"))
        self.assertEqual(speech.utterance_token(b"first", unique=False), speech.SPEECH_TOKEN)
        self.assertNotIn("<s>", one)  # never collides with the graph's START / END labels

    def test_speech_texts(self):
        token = "<speech:abcd1234>"
        audio_text = "aud:mu:8000x1:AAAA"
        self.assertEqual(
            speech.speech_texts(token, SPOKEN, audio_text),
            [f"{token} {SPOKEN}", f"{token} {audio_text}"],
        )
        paired = speech.speech_texts(token, SPOKEN, audio_text, pair=True)
        self.assertEqual(paired[2], f"{token} {audio_text} {SPOKEN}")
        self.assertEqual(speech.speech_texts(token, "  the   cat  ", ""), [f"{token} the cat"])
        self.assertEqual(speech.speech_texts(token, "", ""), [])
        for bad in ("", "   ", "two words"):
            with self.assertRaises(speech.SpeechError):
                speech.speech_texts(bad, SPOKEN, audio_text)


class TestCodecs(unittest.TestCase):
    def test_mu_law_beats_linear_on_quiet_audio(self):
        for amplitude in (0.5, 0.05, 0.005):
            samples = tone(seconds=0.02, amplitude=amplitude)
            mu = speech.mulaw_decode(speech.mulaw_encode(samples))
            pcm = speech.pcm8_decode(speech.pcm8_encode(samples))
            mu_error = max(abs(a - b) for a, b in zip(samples, mu)) / amplitude
            pcm_error = max(abs(a - b) for a, b in zip(samples, pcm)) / amplitude
            self.assertLess(mu_error, 0.05, f"mu-law error at amplitude {amplitude}")
            if amplitude < 0.01:  # linear 8-bit has no resolution left down here, mu-law still has
                self.assertGreater(pcm_error, 10 * mu_error)

    def test_codecs_are_stable_and_cover_the_byte_range(self):
        self.assertEqual(speech.mulaw_encode(speech.mulaw_decode(bytes(range(256)))), bytes(range(256)))
        extremes = speech.mulaw_encode(array.array("f", [-2.0, -1.0, 0.0, 1.0, 2.0]))
        self.assertEqual((extremes[0], extremes[1], extremes[3], extremes[4]), (0, 0, 255, 255))
        self.assertEqual(list(speech.pcm8_encode(array.array("f", [-2.0, -1.0, 0.0, 1.0, 2.0]))), [129, 129, 0, 127, 127])
        self.assertEqual(list(speech.pcm8_decode(bytes([128]))), [-1.0])  # only a predicted text holds -128
        self.assertEqual(speech.get_codec("auto")[0], speech.DEFAULT_CODEC)
        with self.assertRaises(speech.SpeechError):
            speech.get_codec("flac")

    def test_resample_and_downmix(self):
        stereo = array.array("f", [0.0, 1.0, 0.5, -0.5, 1.0, 0.0])
        self.assertEqual(list(speech.to_mono(stereo, 2)), [0.5, 0.0, 0.5])
        up = speech.resample(array.array("f", [0.0, 1.0]), 1000, 2000)
        self.assertEqual(len(up), 4)
        self.assertAlmostEqual(up[1], 0.5, places=5)
        down = speech.resample(array.array("f", [0.0, 0.25, 0.5, 0.75]), 4000, 2000)
        self.assertEqual(len(down), 2)
        self.assertEqual(speech.resample(stereo, 8000, 8000), stereo)

    def test_normalise(self):
        quiet = array.array("f", [0.01, -0.005, 0.0])
        loud = speech.normalise_samples(quiet)
        self.assertAlmostEqual(max(abs(v) for v in loud), 0.99, places=5)
        silence = array.array("f", [0.0, 0.0])
        self.assertEqual(list(speech.normalise_samples(silence)), [0.0, 0.0])


class TestReadingAudio(unittest.TestCase):
    def test_every_wav_sample_format(self):
        samples = tone(seconds=0.05)
        cases = [
            (wav(samples, bits=16), 16, 0.01),
            (wav(samples, bits=8), 8, 0.02),
            (wav(samples, bits=24), 24, 0.001),
            (wav(samples, bits=32), 32, 0.001),
            (wav(samples, bits=32, fmt=speech._WAVE_FLOAT), 32, 0.001),
            (wav(samples, bits=64, fmt=speech._WAVE_FLOAT), 64, 0.001),
        ]
        for data, bits, tolerance in cases:
            with self.subTest(bits=bits):
                audio = speech.load_audio(data)
                self.assertEqual((audio.rate, audio.channels, audio.frames), (16000, 1, len(samples)))
                self.assertLess(max(abs(a - b) for a, b in zip(samples, audio.samples)), tolerance)

    def test_extensible_alaw_and_mulaw(self):
        samples = tone(seconds=0.02)
        extensible = wav(samples, bits=16, fmt=speech._WAVE_EXTENSIBLE,
                         extra=struct.pack("<HIH", 22, 16, 0xFFFF) + struct.pack("<H", speech._WAVE_PCM) + bytes(14))
        audio = speech.load_audio(extensible)
        self.assertEqual(audio.frames, len(samples))
        self.assertLess(max(abs(a - b) for a, b in zip(samples, audio.samples)), 0.01)
        # G.711 payloads are expanded by the reader: each law has its own code for silence and for a peak
        for fmt, silence, peak in ((speech._WAVE_MULAW, 0xFF, 0x80), (speech._WAVE_ALAW, 0xD5, 0x2A)):
            payload = bytes([silence] * 4 + [peak] * 4)
            head = struct.pack("<HHIIHH", fmt, 1, 8000, 8000, 1, 8)
            data = (struct.pack("<4sI4s", b"RIFF", 36 + len(payload), b"WAVE")
                    + struct.pack("<4sI", b"fmt ", 16) + head + struct.pack("<4sI", b"data", len(payload)) + payload)
            audio = speech.load_audio(data)
            self.assertEqual((audio.rate, audio.frames), (8000, 8))
            self.assertTrue(all(abs(v) < 0.001 for v in audio.samples[:4]), list(audio.samples[:4]))
            self.assertTrue(all(v > 0.5 for v in audio.samples[4:]), list(audio.samples[4:]))

    def test_stereo_is_mixed_down(self):
        data = wav(tone(seconds=0.05, channels=2), channels=2)
        audio = speech.load_audio(data)
        self.assertEqual(audio.channels, 2)
        mono = speech.to_mono(audio.samples, audio.channels)
        self.assertEqual(len(mono), audio.frames)
        self.assertTrue(all(abs(v) < 1e-3 for v in mono))  # the fixture's channels cancel out

    def test_broken_audio(self):
        for bad, message in (
            (b"", "empty"),
            (b"RIFFxxxxWAVE", "fmt"),
            (b"RIFF" + struct.pack("<I", 4) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
             + struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8), "no audio data"),
        ):
            with self.assertRaises(speech.SpeechError) as caught:
                speech.load_audio(bad)
            self.assertIn(message, str(caught.exception))

    def test_other_formats_need_ffmpeg(self):
        with mock.patch.object(speech, "ffmpeg_path", lambda: None):
            with self.assertRaises(speech.SpeechError) as caught:
                speech.load_audio(b"OggS" + bytes(64))
            self.assertIn("Ogg", str(caught.exception))
            self.assertIn("ffmpeg", str(caught.exception))
            with self.assertRaises(speech.SpeechError) as caught:
                speech.load_audio(b"ID3" + bytes(64))
            self.assertIn("MP3", str(caught.exception))

    def test_ffmpeg_is_used_for_other_formats(self):
        converted = wav(tone(seconds=0.02))
        calls = []

        class Result:
            returncode, stdout, stderr = 0, converted, b""

        def fake_run(command, **kwargs):
            calls.append((command, kwargs.get("input")))
            return Result()

        with mock.patch.object(speech, "ffmpeg_path", lambda: "/usr/bin/ffmpeg"), \
             mock.patch.object(speech.subprocess, "run", fake_run):
            audio = speech.load_audio(b"OggS" + bytes(64))
        self.assertEqual(audio.rate, 16000)
        self.assertEqual(calls[0][0][0], "/usr/bin/ffmpeg")
        self.assertEqual(calls[0][1][:4], b"OggS")


class TestWaveformAsText(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        data = wav(tone(seconds=0.2, rate=16000))
        result = speech.encode_audio(data, rate=8000)
        self.assertEqual((result["codec"], result["rate"], result["channels"]), ("mu", 8000, 1))
        self.assertEqual((result["samples"], result["bytes"]), (1600, 1600))
        self.assertAlmostEqual(result["seconds"], 0.2, places=3)
        self.assertEqual(result["source"], {"rate": 16000, "channels": 1, "frames": 3200, "seconds": 0.2})
        self.assertTrue(result["text"].startswith("aud:mu:8000x1:"))
        self.assertEqual(result["chars"], len(result["text"]))

        decoded = speech.decode_text(result["text"])
        self.assertEqual((decoded["codec"], decoded["rate"], decoded["samples"], decoded["repaired"]), ("mu", 8000, 1600, False))
        self.assertTrue(decoded["wav"].startswith(b"RIFF"))
        again = speech.load_audio(decoded["wav"])
        self.assertEqual((again.rate, again.frames), (8000, 1600))
        # the waveform survives quantisation: it still sounds like the tone it was
        original = speech.resample(speech.load_audio(data).samples, 16000, 8000)
        self.assertLess(max(abs(a - b) for a, b in zip(original, again.samples)), 0.03)

    def test_a_cut_off_prediction_still_decodes(self):
        text = speech.encode_audio(wav(tone(seconds=0.1)), rate=4000)["text"]
        decoded = speech.decode_text(text[:-33])
        self.assertTrue(decoded["repaired"])
        self.assertLess(decoded["samples"], 400)
        self.assertGreater(decoded["samples"], 300)

    def test_options_and_errors(self):
        data = wav(tone(seconds=0.05, amplitude=0.02))
        plain = speech.encode_audio(data, rate=4000, codec="pcm8")
        louder = speech.encode_audio(data, rate=4000, codec="pcm8", normalise=True)
        self.assertTrue(louder["normalised"])
        self.assertGreater(max(abs(v) for v in speech.pcm8_decode(base64.b64decode(louder["text"].split(":")[-1]))),
                           max(abs(v) for v in speech.pcm8_decode(base64.b64decode(plain["text"].split(":")[-1]))))
        with self.assertRaises(speech.SpeechError):
            speech.encode_audio(data, rate=10)
        with self.assertRaises(speech.SpeechError):
            speech.decode_text("aud:opus:8000x1:AAAA")

    def test_wav_bytes_header(self):
        data = speech.wav_bytes(array.array("f", [0.0, 0.5, -0.5]), 8000)
        self.assertEqual(len(data), 44 + 6)
        audio = speech.load_audio(data)
        self.assertEqual((audio.rate, audio.channels, audio.frames), (8000, 1, 3))


class FakeWhisper:
    """A stand-in with the openai-whisper interface."""

    def __init__(self, text=SPOKEN):
        self.text, self.calls = text, []

    def transcribe(self, path, language=None, fp16=False):
        self.calls.append((path, language, os.path.getsize(path)))
        return {"text": " " + self.text + " ", "language": language or "en"}


class TestTranscription(unittest.TestCase):
    def test_given_transcript_wins_and_needs_nothing(self):
        result = speech.transcribe(b"", text=f"  {SPOKEN}  ")
        self.assertEqual((result["transcript"], result["backend"], result["model"]), (SPOKEN, "given", None))
        self.assertGreaterEqual(result["seconds"], 0.0)

    def test_without_a_backend_the_error_says_what_to_do(self):
        with mock.patch.object(speech.FasterWhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech.WhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech, "DEFAULT_ASR_URL", ""):
            with self.assertRaises(speech.SpeechError) as caught:
                speech.transcribe(wav(tone(seconds=0.02)))
        message = str(caught.exception)
        self.assertIn("faster-whisper", message)
        self.assertIn("RADIXNET_ASR_URL", message)

    def test_whisper_backend(self):
        fake = FakeWhisper()
        transcriber = speech.WhisperTranscriber("tiny")
        with mock.patch.object(transcriber, "load", lambda: fake), \
             mock.patch.object(speech, "get_transcriber", lambda *a, **k: transcriber):
            result = speech.transcribe(wav(tone(seconds=0.02)), backend="whisper", language="en")
        self.assertEqual((result["transcript"], result["backend"], result["model"], result["language"]),
                         (SPOKEN, "whisper", "tiny", "en"))
        self.assertEqual(len(fake.calls), 1)
        path, language, size = fake.calls[0]
        self.assertTrue(path.endswith(".wav"))
        self.assertFalse(os.path.exists(path))  # the temporary file is removed again
        self.assertGreater(size, 44)

    def test_a_failing_backend_is_reported(self):
        transcriber = speech.WhisperTranscriber("tiny")
        with mock.patch.object(transcriber, "load", mock.Mock(side_effect=RuntimeError("no weights"))), \
             mock.patch.object(speech, "get_transcriber", lambda *a, **k: transcriber):
            with self.assertRaises(speech.SpeechError) as caught:
                speech.transcribe(wav(tone(seconds=0.02)), backend="whisper")
        self.assertIn("no weights", str(caught.exception))
        self.assertIn("no weights", transcriber.error)

    def test_server_backend_posts_multipart(self):
        sent = {}

        class Response:
            def read(self):
                return json.dumps({"text": SPOKEN}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["type"] = request.headers.get("Content-type")
            sent["body"] = request.data
            return Response()

        transcriber = speech.ServerTranscriber("http://asr.local/v1/audio/transcriptions", "whisper-1")
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            result = transcriber.transcribe(wav(tone(seconds=0.02)), language="en")
        self.assertEqual(result["transcript"], SPOKEN)
        self.assertEqual(sent["url"], "http://asr.local/v1/audio/transcriptions")
        self.assertIn("multipart/form-data; boundary=", sent["type"])
        self.assertIn(b'name="model"', sent["body"])
        self.assertIn(b'name="language"', sent["body"])
        self.assertIn(b'filename="speech.wav"', sent["body"])
        self.assertIn(b"RIFF", sent["body"])

    def test_auto_order_and_unknown_backend(self):
        with mock.patch.object(speech.FasterWhisperTranscriber, "importable", staticmethod(lambda: True)):
            self.assertEqual(speech.get_transcriber("auto").name, "faster-whisper")
            self.assertEqual(speech.get_transcriber("auto", text=SPOKEN).name, "given")  # a given text always wins
        with mock.patch.object(speech.FasterWhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech.WhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech, "DEFAULT_ASR_URL", "http://asr.local/v1/audio/transcriptions"):
            self.assertEqual(speech.get_transcriber("auto").name, "server")
        with self.assertRaises(speech.SpeechError):
            speech.get_transcriber("wav2vec")


class TestTeach(unittest.TestCase):
    def setUp(self):
        self.data = wav(tone(seconds=0.1))

    def test_both_texts_share_the_unique_token(self):
        result = speech.teach(self.data, transcript=SPOKEN, rate=4000)
        token = result["token"]
        self.assertTrue(token.startswith("<speech:"))
        self.assertEqual(result["transcript"], SPOKEN)
        self.assertEqual(result["asr"]["backend"], "given")
        self.assertIsNone(result["asr"]["error"])
        self.assertEqual(len(result["texts"]), 2)
        self.assertEqual(result["texts"][0], f"{token} {SPOKEN}")
        self.assertTrue(result["texts"][1].startswith(f"{token} aud:mu:4000x1:"))
        self.assertEqual(result["chars"], sum(len(t) for t in result["texts"]))
        self.assertEqual(result["audio"]["samples"], 400)
        self.assertFalse(result["pair"])
        # the same recording always gives the same token, a different one does not
        self.assertEqual(speech.teach(self.data, transcript="other words", rate=4000)["token"], token)
        self.assertNotEqual(speech.teach(wav(tone(seconds=0.1, freq=440.0)), transcript=SPOKEN, rate=4000)["token"], token)

    def test_pair_shared_token_and_transcript_only(self):
        paired = speech.teach(self.data, transcript=SPOKEN, rate=4000, pair=True)
        self.assertTrue(paired["pair"])
        self.assertEqual(len(paired["texts"]), 3)
        self.assertTrue(paired["texts"][2].endswith(" " + SPOKEN))
        self.assertIn("aud:mu:4000x1:", paired["texts"][2])

        shared = speech.teach(self.data, transcript=SPOKEN, rate=4000, unique=False)
        self.assertEqual(shared["token"], speech.SPEECH_TOKEN)
        named = speech.teach(self.data, transcript=SPOKEN, rate=4000, token="<voice>")
        self.assertTrue(all(t.startswith("<voice> ") for t in named["texts"]))

        words_only = speech.teach(self.data, transcript=SPOKEN, waveform=False)
        self.assertIsNone(words_only["audio"])
        self.assertEqual(words_only["texts"], [f"{words_only['token']} {SPOKEN}"])

    def test_a_failing_transcription_still_teaches_the_waveform(self):
        with mock.patch.object(speech.FasterWhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech.WhisperTranscriber, "importable", staticmethod(lambda: False)), \
             mock.patch.object(speech, "DEFAULT_ASR_URL", ""):
            result = speech.teach(self.data, rate=4000)
        self.assertEqual(result["transcript"], "")
        self.assertIn("no transcription backend", result["asr"]["error"])
        self.assertEqual(len(result["texts"]), 1)
        self.assertIn("aud:mu:4000x1:", result["texts"][0])

    def test_nothing_to_learn(self):
        with self.assertRaises(speech.SpeechError):
            speech.teach(b"", transcript="   ")

    def test_the_model_learns_and_predicts_what_was_spoken(self):
        from radixnet import RadixNet

        result = speech.teach(self.data, transcript=SPOKEN, rate=2000)
        net = RadixNet(seed=0, backend="python")
        net.train(result["texts"], epochs=3, lr=0.5, batch_size=8)
        token = result["token"]
        # the token is the entry node of both views: continuing it reaches the words and the sound
        self.assertGreater(net.score(result["texts"][0])["log_prob"], net.score("zzz qqq xxx")["log_prob"])
        prediction = net.predict(token + " ", length=12)
        self.assertTrue(prediction.full_text.startswith(token))
        waveform = net.predict(result["texts"][1][:60], length=40)
        self.assertGreater(len(waveform.text), 0)
        decoded = speech.decode_text(waveform.full_text)
        self.assertEqual(decoded["rate"], 2000)


class TestRecording(unittest.TestCase):
    def test_no_recorder_installed(self):
        with mock.patch.object(speech.shutil, "which", lambda name: None), \
             mock.patch.object(speech, "ffmpeg_path", lambda: None):
            self.assertEqual(speech.recorders(), [])
            with self.assertRaises(speech.SpeechError) as caught:
                speech.record(1.0)
        self.assertIn("no microphone recorder found", str(caught.exception))

    def test_arecord_is_called_and_its_file_is_read(self):
        data = wav(tone(seconds=0.02))
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            with open(command[-1], "wb") as fh:
                fh.write(data)

            class Result:
                returncode, stdout, stderr = 0, b"", b""

            return Result()

        with mock.patch.object(speech.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "arecord" else None), \
             mock.patch.object(speech, "ffmpeg_path", lambda: None), \
             mock.patch.object(speech.subprocess, "run", fake_run):
            self.assertEqual(speech.recorders(), ["arecord"])
            recorded = speech.record(seconds=1.0, rate=16000)
        self.assertEqual(recorded, data)
        self.assertEqual(calls[0][0], "/usr/bin/arecord")
        self.assertIn("16000", calls[0])

    def test_bad_arguments(self):
        with self.assertRaises(speech.SpeechError):
            speech.record(seconds=0)
        with self.assertRaises(speech.SpeechError):
            speech.record(seconds=1, recorder="microphone")


class TestDescribe(unittest.TestCase):
    def test_describe(self):
        info = speech.describe()
        self.assertEqual(info["backends"], list(speech.ASR_BACKENDS))
        self.assertEqual(info["codecs"], list(speech.CODECS))
        self.assertEqual(info["token"], "<speech>")
        self.assertTrue(info["given_always"])
        self.assertEqual(info["default_rate"], speech.DEFAULT_RATE)
        self.assertIn("aud:<codec>", info["text_format"])
        self.assertIsInstance(info["recorders"], list)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-speech-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, upload_dir=os.path.join(cls.tmp.name, "uploads"))
        cls.wav = wav(tone(seconds=0.1))

    def test_describe_endpoint(self):
        status, data, _ = self.client.get("/api/speech")
        self.assertEqual(status, 200)
        self.assertEqual(data["token"], "<speech>")
        self.assertIn("text_format", data)

    def test_transcribe_with_a_browser_transcript(self):
        status, data, _ = self.client.request(
            "POST", f"/api/speech/transcribe?transcript={SPOKEN.replace(' ', '%20')}",
            raw=self.wav, headers={"Content-Type": "audio/wav"},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["transcript"], data["backend"], data["name"]), (SPOKEN, "given", "speech"))

    def test_teach_raw_multipart_and_json(self):
        status, data, _ = self.client.request(
            "POST", f"/api/speech/teach?rate=4000&transcript={SPOKEN.replace(' ', '%20')}",
            raw=self.wav, headers={"Content-Type": "audio/wav"},
        )
        self.assertEqual(status, 200, data)
        token = data["token"]
        self.assertTrue(token.startswith("<speech:"))
        self.assertEqual(data["transcript"], SPOKEN)
        self.assertEqual(data["texts"][0], f"{token} {SPOKEN}")
        self.assertTrue(data["texts"][1].startswith(f"{token} aud:mu:4000x1:"))
        self.assertEqual(data["audio"]["samples"], 400)
        self.assertIsNone(data["job"])
        texts = data["texts"]

        boundary = "----radixnet-speech"
        body = (
            f"--{boundary}\r\n".encode("ascii")
            + b'Content-Disposition: form-data; name="file"; filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            + self.wav + f"\r\n--{boundary}--\r\n".encode("ascii")
        )
        status, data, _ = self.client.request(
            "POST", f"/api/speech/teach?rate=4000&transcript={SPOKEN.replace(' ', '%20')}", raw=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["name"], data["texts"]), ("utterance.wav", texts))

        status, data, _ = self.client.post("/api/speech/teach", {
            "name": "b64.wav", "content_base64": base64.b64encode(self.wav).decode("ascii"),
            "rate": 4000, "transcript": SPOKEN, "pair": True,
        })
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["texts"]), 3)
        self.assertTrue(data["pair"])

    def test_teach_can_save_and_train(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post("/api/speech/teach", {
            "name": "spoken.wav", "content_base64": base64.b64encode(self.wav).decode("ascii"),
            "rate": 2000, "transcript": SPOKEN, "train": True, "save_as": "spoken.txt",
            "epochs": 2, "batch_size": 8, "lr": 0.5,
        })
        self.assertEqual(status, 202, data)
        self.assertEqual(data["upload"]["name"], "spoken.txt")
        self.assertEqual(data["job"]["type"], "train")
        job = wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(job["history"]), 2)
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 2)
        status, prediction, _ = self.client.post("/api/predict", {"prefix": data["token"] + " ", "length": 8})
        self.assertEqual(status, 200)
        self.assertTrue(prediction["full_text"].startswith(data["token"]))

    def test_decode_endpoint(self):
        status, data, _ = self.client.post("/api/speech/teach", {
            "name": "x.wav", "content_base64": base64.b64encode(self.wav).decode("ascii"), "rate": 4000,
            "transcript": SPOKEN,
        })
        self.assertEqual(status, 200, data)
        status, decoded, _ = self.client.post("/api/speech/decode", {"text": data["texts"][1]})
        self.assertEqual(status, 200, decoded)
        self.assertEqual((decoded["codec"], decoded["rate"], decoded["samples"], decoded["repaired"]), ("mu", 4000, 400, False))
        self.assertTrue(base64.b64decode(decoded["wav_base64"]).startswith(b"RIFF"))
        status, decoded, _ = self.client.post("/api/speech/decode", {"text": data["texts"][1][:-25]})
        self.assertEqual(status, 200)
        self.assertTrue(decoded["repaired"])

    def test_errors(self):
        status, data, _ = self.client.request(
            "POST", "/api/speech/teach", raw=b"not audio at all", headers={"Content-Type": "audio/wav"}
        )
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.request(
            "POST", "/api/speech/teach?rate=2", raw=self.wav, headers={"Content-Type": "audio/wav"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("sample rate", data["error"])
        status, data, _ = self.client.post("/api/speech/decode", {"text": "hello"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/speech/teach", {"name": "x.wav", "content": "text, not bytes"})
        self.assertEqual(status, 400, data)
        self.assertIn("bytes", data["error"])


class TestCli(unittest.TestCase):
    def test_info_transcribe_teach_and_decode(self):
        from tests.test_cli import run_cli, run_json

        with tempfile.TemporaryDirectory(prefix="radixnet-speech-cli-") as tmp:
            source = os.path.join(tmp, "utterance.wav")
            with open(source, "wb") as fh:
                fh.write(wav(tone(seconds=0.1)))

            info = run_json("speech", "info")
            self.assertEqual(info["token"], "<speech>")
            self.assertIn("mu", info["codecs"])

            doc = run_json("speech", "transcribe", source, "--text", SPOKEN)
            self.assertEqual((doc["transcript"], doc["backend"]), (SPOKEN, "given"))

            texts = os.path.join(tmp, "texts.txt")
            doc = run_json("speech", "teach", source, "--text", SPOKEN, "--rate", 2000, "--out", texts)
            token = doc["token"]
            self.assertEqual(doc["transcript"], SPOKEN)
            self.assertEqual(len(doc["texts"]), 2)
            self.assertEqual(doc["audio"]["rate"], 2000)
            with open(texts, encoding="utf-8") as fh:
                self.assertEqual(fh.read().splitlines(), doc["texts"])

            out = os.path.join(tmp, "back.wav")
            decoded = run_json("speech", "decode", "--data", texts, "--out", out)
            self.assertEqual((decoded["codec"], decoded["rate"], decoded["out"]), ("mu", 2000, out))
            self.assertTrue(os.path.getsize(out) > 44)

            model = os.path.join(tmp, "m.json")
            trained = run_json("speech", "teach", source, "--text", SPOKEN, "--rate", 2000, "--pair",
                               "--train", "--epochs", 2, "--batch-size", 8, model=model)
            self.assertEqual(trained["trained"]["epochs"], 2)
            self.assertEqual(trained["trained"]["texts"], 3)
            self.assertTrue(os.path.exists(model))
            self.assertEqual(trained["token"], token)

            prediction = run_json("predict", "--prefix", token + " ", "--length", 8, model=model)
            self.assertTrue(prediction["full_text"].startswith(token))

            proc = run_cli("speech", "teach", os.path.join(tmp, "missing.wav"), expect=1)
            self.assertIn("not found", proc.stderr)
            proc = run_cli("speech", "decode", "--text", "hello", "--out", out, expect=1)
            self.assertIn("not an encoded waveform", proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
