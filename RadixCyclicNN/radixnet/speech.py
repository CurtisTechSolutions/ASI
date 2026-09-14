"""Speech as text: what was said *and* the waveform that said it, both behind one unique token.

Teaching the network by talking to it produces two texts from one utterance,
and both start with the **same unique token**::

    <speech:9f2a1c7d> the cat sat on the mat
    <speech:9f2a1c7d> aud:mu:8000x1:f39ta3t7e4R0...

The token is ``<speech>`` plus a short digest of the waveform, so it is unique
to that utterance and identical in both texts: in the cyclic graph the two
views of one utterance leave the *same* node, which is what ties the words to
the sound.  ``teach()`` builds both (and, with ``pair=True``, a third text that
runs the waveform straight into its transcript), :meth:`GraphModel.train`
learns them like any other text.

Speech to text
--------------
:func:`transcribe` turns audio into words with the first backend that is
available (:data:`ASR_BACKENDS`):

* ``"given"`` - the transcript comes from the caller: the browser's Web Speech
  API (the Speech tab dictates while it records), ``--text`` on the command
  line, or ``transcript`` in the API body.  Always available, and the reason
  the feature works with nothing installed.
* ``"faster-whisper"`` - ``pip install faster-whisper`` (CTranslate2 Whisper,
  fast on the CPU).
* ``"whisper"`` - ``pip install openai-whisper``.
* ``"server"`` - any OpenAI-compatible ``/v1/audio/transcriptions`` endpoint
  (whisper.cpp's server, Speaches, ...) over the standard library; used by
  ``auto`` only when ``$RADIXNET_ASR_URL`` names one.

``$RADIXNET_WHISPER_MODEL`` (default ``base``) picks the local Whisper size,
``$RADIXNET_ASR_URL`` / ``$RADIXNET_ASR_MODEL`` the remote endpoint.

The waveform as text
--------------------
:func:`encode_audio` decodes the audio (WAV natively - PCM, IEEE float, A-law
and mu-law; anything else through ``ffmpeg`` when it is installed), mixes it
down to mono, resamples it to :data:`DEFAULT_RATE` and quantises every sample
to one byte, which is base64-encoded into

    ``aud:<codec>:<rate>x<channels>:<base64 of one byte per sample>``

* ``"mu"`` (the default) - mu-law companding, ``sign(x) * ln(1 + 255|x|) / ln(256)``
  quantised to 256 levels: the classic 8-bit speech quantisation (G.711,
  WaveNet), which spends its levels where speech lives instead of on the peaks.
* ``"pcm8"`` - plain linear 8-bit, one signed byte per sample.

:func:`decode_text` turns an encoded - or *predicted* - text back into a WAV
file, so what the network says can be listened to; a cut-off or garbled base64
tail is repaired like the image encoder's.
"""

from __future__ import annotations

import array
import base64
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

from .encoding import repair_base64

__all__ = [
    "ASR_BACKENDS",
    "Audio",
    "CODECS",
    "DEFAULT_ASR_MODEL",
    "DEFAULT_ASR_URL",
    "DEFAULT_RATE",
    "DEFAULT_WHISPER_MODEL",
    "SPEECH_TOKEN",
    "SpeechError",
    "decode_text",
    "describe",
    "encode_audio",
    "load_audio",
    "pack_text",
    "parse_text",
    "record",
    "speech_texts",
    "teach",
    "transcribe",
    "utterance_token",
    "wav_bytes",
]

HEADER = "aud"
SPEECH_TOKEN = "<speech>"
"""Marker every spoken text starts with; :func:`utterance_token` makes it unique per utterance."""
CODECS = ("auto", "mu", "pcm8")
DEFAULT_CODEC = "mu"
DEFAULT_RATE = 8000
"""Samples per second the waveform is resampled to before it is quantised (8 kHz keeps speech intelligible)."""
MIN_RATE, MAX_RATE = 1000, 48000
MU = 255.0
"""Compression parameter of the mu-law codec (256 quantisation levels)."""

ASR_BACKENDS = ("auto", "given", "faster-whisper", "whisper", "server")
DEFAULT_WHISPER_MODEL = os.environ.get("RADIXNET_WHISPER_MODEL", "base")
DEFAULT_ASR_URL = os.environ.get("RADIXNET_ASR_URL", "")
DEFAULT_ASR_MODEL = os.environ.get("RADIXNET_ASR_MODEL", "whisper-1")
ASR_TIMEOUT = float(os.environ.get("RADIXNET_ASR_TIMEOUT", "300") or 300)

_TEXT_RE = re.compile(r"aud:([a-z0-9_]+):(\d+)x(\d+):")
_TOKEN_DIGITS = 8

# WAVE format tags understood by the reader
_WAVE_PCM, _WAVE_FLOAT, _WAVE_ALAW, _WAVE_MULAW, _WAVE_EXTENSIBLE = 0x0001, 0x0003, 0x0006, 0x0007, 0xFFFE

_MAGIC = (
    (b"ID3", "MP3"), (b"\xff\xfb", "MP3"), (b"\xff\xf3", "MP3"), (b"\xff\xf2", "MP3"),
    (b"OggS", "Ogg / Opus"), (b"fLaC", "FLAC"), (b"\x1a\x45\xdf\xa3", "WebM / Matroska"),
    (b"FORM", "AIFF"),
)
"""First bytes of the audio formats the error messages name (MP4 / M4A is recognised by its ``ftyp`` box instead)."""


class SpeechError(ValueError):
    """Unreadable audio, a missing transcription backend, an unknown codec or a text that is not an encoded waveform."""


# ---------------------------------------------------------------------------
# the text format
# ---------------------------------------------------------------------------


def pack_text(codec: str, rate: int, channels: int, payload: bytes) -> str:
    """``aud:<codec>:<rate>x<channels>:<base64 of one byte per sample>``."""
    return f"{HEADER}:{codec}:{int(rate)}x{int(channels)}:" + base64.b64encode(payload).decode("ascii")


def parse_text(text: str) -> tuple[str, int, int, bytes, bool]:
    """``(codec, rate, channels, payload, repaired)`` of an encoded-waveform text.

    The header is looked for *anywhere* in the text, because a spoken text
    carries its token first (``<speech:9f2a1c7d> aud:mu:8000x1:...``) and a
    paired one its transcript last; the payload therefore runs from the header
    to the next whitespace.  Inside it the base64 is repaired when it is not
    clean - stray characters, a cut-off tail: a *predicted* waveform rarely
    ends on a byte boundary.  ``repaired`` says whether anything had to be
    dropped or padded.
    """
    text = str(text or "")
    match = _TEXT_RE.search(text)
    if not match:
        raise SpeechError("not an encoded waveform: expected 'aud:<codec>:<rate>x<channels>:<base64>'")
    codec, rate, channels = match.group(1), int(match.group(2)), int(match.group(3))
    body = text[match.end() :].split(None, 1)[0] if text[match.end() :].strip() else ""
    try:
        payload, repaired = repair_base64(body)
    except ValueError as exc:  # pragma: no cover - the junk filter makes this rare
        raise SpeechError(str(exc)) from exc
    if rate <= 0 or channels <= 0:
        raise SpeechError(f"invalid waveform header {rate}x{channels}")
    return codec, rate, channels, payload, repaired


def utterance_token(payload: bytes | str, unique: bool = True) -> str:
    """The token both texts of one utterance start with.

    ``unique`` (the default) appends a short digest of the waveform -
    ``<speech:9f2a1c7d>`` - so every utterance has its own token and the
    transcript and the waveform share it; ``False`` returns the plain
    :data:`SPEECH_TOKEN`, which merges all spoken texts into one entry node.
    """
    if not unique:
        return SPEECH_TOKEN
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    digest = hashlib.blake2b(data, digest_size=(_TOKEN_DIGITS + 1) // 2).hexdigest()[:_TOKEN_DIGITS]
    return f"{SPEECH_TOKEN[:-1]}:{digest}>"


def check_token(token: str) -> str:
    """Validate a token: ``<speech>``, ``<speech:hex>`` or any non-empty label without whitespace."""
    token = str(token or "").strip()
    if not token:
        raise SpeechError("the token must not be empty")
    if any(ch.isspace() for ch in token):
        raise SpeechError(f"the token must not contain whitespace: {token!r}")
    return token


def speech_texts(token: str, transcript: str = "", audio_text: str = "", pair: bool = False) -> list[str]:
    """The texts one utterance trains on, each starting with ``token``.

    1. the transcript (what was said), when there is one;
    2. the waveform text (how it sounded), when there is one;
    3. with ``pair``, the waveform followed by its transcript, so the search
       can run from the sound straight into the words.
    """
    token = check_token(token)
    transcript = " ".join(str(transcript or "").split())
    audio_text = str(audio_text or "").strip()
    texts: list[str] = []
    if transcript:
        texts.append(f"{token} {transcript}")
    if audio_text:
        texts.append(f"{token} {audio_text}")
        if pair and transcript:
            texts.append(f"{token} {audio_text} {transcript}")
    return texts


# ---------------------------------------------------------------------------
# reading audio
# ---------------------------------------------------------------------------


class Audio:
    """Decoded audio: interleaved samples in ``[-1, 1]``, the sample rate and the channel count."""

    __slots__ = ("samples", "rate", "channels")

    def __init__(self, samples: array.array, rate: int, channels: int) -> None:
        self.samples = samples
        self.rate = int(rate)
        self.channels = int(channels)

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels if self.channels else 0

    @property
    def seconds(self) -> float:
        return self.frames / self.rate if self.rate else 0.0

    def to_dict(self) -> dict:
        return {"rate": self.rate, "channels": self.channels, "frames": self.frames, "seconds": round(self.seconds, 4)}

    def __repr__(self) -> str:
        return f"Audio(rate={self.rate}, channels={self.channels}, frames={self.frames})"


def _swap(values: array.array) -> array.array:
    if sys.byteorder == "big":  # every WAVE payload is little-endian
        values.byteswap()
    return values


def _ulaw_to_linear(byte: int) -> float:
    """One G.711 mu-law byte -> a sample in ``[-1, 1]`` (the telephony encoding, not our codec)."""
    byte = ~byte & 0xFF
    magnitude = (((byte & 0x0F) << 3) + 0x84) << ((byte >> 4) & 0x07)
    value = magnitude - 0x84
    return (-value if byte & 0x80 else value) / 32768.0


def _alaw_to_linear(byte: int) -> float:
    """One G.711 A-law byte -> a sample in ``[-1, 1]``."""
    byte ^= 0x55
    exponent, mantissa = (byte >> 4) & 0x07, byte & 0x0F
    magnitude = ((mantissa << 4) + 0x108) << (exponent - 1) if exponent else (mantissa << 4) + 8
    return (-magnitude if byte & 0x80 else magnitude) / 32768.0


def _samples_from_pcm(data: bytes, fmt: int, bits: int) -> array.array:
    """A WAVE data chunk -> floats in ``[-1, 1]``."""
    if fmt == _WAVE_MULAW:
        return array.array("f", [_ulaw_to_linear(b) for b in data])
    if fmt == _WAVE_ALAW:
        return array.array("f", [_alaw_to_linear(b) for b in data])
    if fmt == _WAVE_FLOAT:
        if bits == 32:
            values = _swap(array.array("f", data[: len(data) - len(data) % 4]))
            return array.array("f", [max(-1.0, min(1.0, v)) for v in values])
        if bits == 64:
            values = _swap(array.array("d", data[: len(data) - len(data) % 8]))
            return array.array("f", [max(-1.0, min(1.0, v)) for v in values])
        raise SpeechError(f"unsupported float WAV sample size: {bits} bits")
    if fmt != _WAVE_PCM:
        raise SpeechError(f"unsupported WAV format tag 0x{fmt:04x} (PCM, IEEE float, A-law and mu-law are read)")
    if bits == 8:  # 8-bit PCM is unsigned, everything wider is signed
        return array.array("f", [(b - 128) / 128.0 for b in data])
    if bits == 16:
        return array.array("f", [v / 32768.0 for v in _swap(array.array("h", data[: len(data) - len(data) % 2]))])
    if bits == 24:
        usable = len(data) - len(data) % 3
        out = array.array("f")
        for i in range(0, usable, 3):
            value = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            out.append(value / 8388608.0)
        return out
    if bits == 32:
        return array.array("f", [v / 2147483648.0 for v in _swap(array.array("i", data[: len(data) - len(data) % 4]))])
    raise SpeechError(f"unsupported PCM sample size: {bits} bits")


def parse_wav(data: bytes) -> Audio:
    """Decode a RIFF/WAVE file (PCM 8/16/24/32, IEEE float 32/64, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SpeechError("not a WAV file (no RIFF/WAVE header)")
    fmt = channels = rate = bits = 0
    payload = b""
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            fmt, channels, rate, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", body, 0)
            if fmt == _WAVE_EXTENSIBLE and len(body) >= 26:
                fmt = struct.unpack_from("<H", body, 24)[0]  # the SubFormat GUID starts with the real tag
        elif chunk_id == b"data":
            if not size:  # a streamed WAV may declare size 0: the rest of the file is audio
                payload = data[offset + 8 :]
                break
            payload = body
        offset += 8 + size + (size & 1)  # chunks are padded to an even length
    if not channels or not rate:
        raise SpeechError("the WAV file has no usable 'fmt ' chunk")
    if not payload:
        raise SpeechError("the WAV file has no audio data")
    samples = _samples_from_pcm(payload, fmt, bits)
    usable = len(samples) - len(samples) % channels
    return Audio(samples[:usable] if usable != len(samples) else samples, rate, channels)


def _looks_like(data: bytes) -> str:
    """The audio format the first bytes suggest - only used to make an error message helpful."""
    if data[4:8] == b"ftyp":
        return "MP4 / M4A"
    for magic, name in _MAGIC:
        if data.startswith(magic):
            return name
    return "an unknown format"


def ffmpeg_path() -> str | None:
    """``ffmpeg`` on ``PATH`` (``$RADIXNET_FFMPEG`` overrides it), or ``None``."""
    configured = os.environ.get("RADIXNET_FFMPEG")
    if configured:
        return configured if os.path.isfile(configured) else shutil.which(configured)
    return shutil.which("ffmpeg")


def convert_with_ffmpeg(data: bytes, rate: int | None = None) -> bytes:
    """Anything ffmpeg reads (MP3, Opus, WebM, M4A, FLAC, ...) -> a 16-bit PCM WAV."""
    binary = ffmpeg_path()
    if not binary:
        raise SpeechError(
            f"the audio is {_looks_like(data)}; install ffmpeg to read it (or send a WAV file)"
        )
    command = [binary, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn", "-map", "a:0", "-f", "wav",
               "-acodec", "pcm_s16le"]
    if rate:
        command += ["-ar", str(int(rate))]
    command.append("pipe:1")
    try:
        proc = subprocess.run(command, input=data, capture_output=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpeechError(f"ffmpeg could not be run: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SpeechError(f"ffmpeg could not decode the audio: {detail[-1] if detail else 'no output'}")
    return proc.stdout


def load_audio(data: bytes) -> Audio:
    """Audio bytes -> :class:`Audio`; WAV is read directly, everything else goes through ffmpeg."""
    if not data:
        raise SpeechError("the audio is empty")
    data = bytes(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return parse_wav(data)
    return parse_wav(convert_with_ffmpeg(data))


# ---------------------------------------------------------------------------
# the waveform as text
# ---------------------------------------------------------------------------


def to_mono(samples: array.array, channels: int) -> array.array:
    """Average the channels of an interleaved buffer."""
    if channels <= 1:
        return samples
    frames = len(samples) // channels
    out = array.array("f", bytes(4 * frames))
    for frame in range(frames):
        base = frame * channels
        total = 0.0
        for offset in range(channels):
            total += samples[base + offset]
        out[frame] = total / channels
    return out


def resample(samples: array.array, src_rate: int, dst_rate: int) -> array.array:
    """Linear interpolation to ``dst_rate`` (the waveform is quantised to a byte afterwards anyway)."""
    if src_rate == dst_rate or not samples:
        return samples
    if src_rate <= 0:
        raise SpeechError(f"invalid source sample rate {src_rate}")
    count = max(1, int(round(len(samples) * dst_rate / src_rate)))
    step = src_rate / dst_rate
    last = len(samples) - 1
    out = array.array("f", bytes(4 * count))
    for i in range(count):
        position = i * step
        left = int(position)
        if left >= last:
            out[i] = samples[last]
            continue
        fraction = position - left
        out[i] = samples[left] * (1.0 - fraction) + samples[left + 1] * fraction
    return out


def normalise_samples(samples: array.array, headroom: float = 0.99) -> array.array:
    """Scale the waveform so its loudest sample sits at ``headroom`` (silence is left alone)."""
    peak = 0.0
    for value in samples:
        magnitude = -value if value < 0.0 else value
        if magnitude > peak:
            peak = magnitude
    if peak <= 1e-9:
        return samples
    gain = headroom / peak
    return array.array("f", [value * gain for value in samples])


def mulaw_encode(samples) -> bytes:
    """Samples in ``[-1, 1]`` -> one mu-law byte each (256 levels, fine where speech lives)."""
    scale = math.log1p(MU)
    out = bytearray(len(samples))
    for i, value in enumerate(samples):
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        magnitude = math.log1p(MU * (value if value >= 0.0 else -value)) / scale
        companded = magnitude if value >= 0.0 else -magnitude
        out[i] = min(255, max(0, int(round((companded + 1.0) * 127.5))))
    return bytes(out)


def mulaw_decode(payload: bytes) -> array.array:
    """The inverse of :func:`mulaw_encode`."""
    out = array.array("f", bytes(4 * len(payload)))
    for i, byte in enumerate(payload):
        companded = byte / 127.5 - 1.0
        magnitude = (math.pow(1.0 + MU, companded if companded >= 0.0 else -companded) - 1.0) / MU
        out[i] = magnitude if companded >= 0.0 else -magnitude
    return out


def pcm8_encode(samples) -> bytes:
    """Samples in ``[-1, 1]`` -> one signed byte each (plain linear quantisation)."""
    return bytes((max(-127, min(127, int(round(max(-1.0, min(1.0, value)) * 127.0)))) & 0xFF) for value in samples)


def pcm8_decode(payload: bytes) -> array.array:
    """The inverse of :func:`pcm8_encode`; -128 (only a *predicted* text holds it) clamps to -1."""
    return array.array("f", [max(-1.0, ((byte - 256) if byte > 127 else byte) / 127.0) for byte in payload])


_CODECS = {
    "mu": (mulaw_encode, mulaw_decode),
    "pcm8": (pcm8_encode, pcm8_decode),
}


def get_codec(name: str = "auto") -> tuple[str, Any, Any]:
    """``(name, encode, decode)`` of a waveform codec (``auto`` -> :data:`DEFAULT_CODEC`)."""
    key = (name or "auto").strip().lower()
    if key == "auto":
        key = DEFAULT_CODEC
    if key not in _CODECS:
        raise SpeechError(f"unknown waveform codec {name!r}; expected one of: {', '.join(CODECS)}")
    encode, decode = _CODECS[key]
    return key, encode, decode


def check_rate(rate: int) -> int:
    rate = int(rate)
    if rate < MIN_RATE or rate > MAX_RATE:
        raise SpeechError(f"the sample rate must be between {MIN_RATE} and {MAX_RATE} Hz, got {rate}")
    return rate


def wav_bytes(samples, rate: int, channels: int = 1) -> bytes:
    """Samples in ``[-1, 1]`` -> a 16-bit PCM WAV file."""
    pcm = array.array("h", (max(-32768, min(32767, int(round(value * 32767.0)))) for value in samples))
    _swap(pcm)
    body = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(body), b"WAVE", b"fmt ", 16, _WAVE_PCM, channels, rate,
        rate * channels * 2, channels * 2, 16, b"data", len(body),
    )
    return header + body


def encode_audio(
    data: bytes, rate: int = DEFAULT_RATE, codec: str = "auto", normalise: bool = False
) -> dict:
    """Audio bytes -> ``{"text", "codec", "rate", "channels", "samples", "seconds", "bytes", "chars", "source"}``.

    The audio is mixed down to mono, resampled to ``rate`` and quantised to one
    byte per sample; ``normalise`` scales a quiet recording up first.
    """
    rate = check_rate(rate)
    name, encode, _decode = get_codec(codec)
    audio = load_audio(data)
    samples = resample(to_mono(audio.samples, audio.channels), audio.rate, rate)
    if normalise:
        samples = normalise_samples(samples)
    payload = encode(samples)
    text = pack_text(name, rate, 1, payload)
    return {
        "text": text,
        "codec": name,
        "rate": rate,
        "channels": 1,
        "samples": len(payload),
        "seconds": round(len(payload) / rate, 4),
        "bytes": len(payload),
        "chars": len(text),
        "normalised": bool(normalise),
        "source": audio.to_dict(),
    }


def decode_text(text: str, codec: str | None = None) -> dict:
    """Encoded (or predicted) text -> ``{"wav", "codec", "rate", "channels", "samples", "seconds", "bytes", "repaired"}``.

    ``codec`` overrides the one named in the text.  A predicted waveform whose
    base64 tail is cut off or garbled is repaired first (:func:`parse_text`).
    """
    header_codec, rate, channels, payload, repaired = parse_text(text)
    name, _encode, decode = get_codec(codec or header_codec)
    rate = check_rate(rate)
    samples = decode(payload)
    frames = len(samples) // channels if channels else 0
    return {
        "wav": wav_bytes(samples, rate, channels),
        "codec": name,
        "rate": rate,
        "channels": channels,
        "samples": len(samples),
        "seconds": round(frames / rate, 4) if rate else 0.0,
        "bytes": len(payload),
        "repaired": repaired,
    }


# ---------------------------------------------------------------------------
# speech to text
# ---------------------------------------------------------------------------


class GivenTranscriber:
    """The transcript was dictated elsewhere: the browser's Web Speech API, ``--text``, or the API body."""

    name = "given"

    def __init__(self, text: str = "") -> None:
        self.text = str(text or "")

    def available(self) -> bool:
        return bool(self.text.strip())

    def transcribe(self, wav: bytes, language: str | None = None) -> dict:
        if not self.available():
            raise SpeechError("no transcript was given (pass the text, or install a transcription backend)")
        return {"transcript": self.text.strip(), "model": None, "language": language}


class _WhisperFamily:
    """Shared plumbing of the two local Whisper backends: the model loads once per process."""

    module = ""
    install = ""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_WHISPER_MODEL
        self.error: str | None = None
        self._loaded: Any = None
        self._loaded_name: str | None = None

    @classmethod
    def importable(cls) -> bool:
        try:
            __import__(cls.module)
        except ImportError:
            return False
        return True

    def available(self) -> bool:
        return self.importable()

    def load(self) -> Any:
        raise NotImplementedError

    def _get(self) -> Any:
        if self._loaded is None or self._loaded_name != self.model:
            try:
                self._loaded = self.load()
            except ImportError as exc:
                self.error = f"{self.name} needs {self.install}"
                raise SpeechError(self.error) from exc
            except Exception as exc:  # noqa: BLE001 - a missing download, a bad model name, no disk space
                self.error = f"cannot load the {self.name} model {self.model!r}: {exc}"
                raise SpeechError(self.error) from exc
            self._loaded_name = self.model
            self.error = None
        return self._loaded


class WhisperTranscriber(_WhisperFamily):
    """``openai-whisper`` (``pip install openai-whisper``)."""

    name = "whisper"
    module = "whisper"
    install = "pip install openai-whisper"

    def load(self) -> Any:
        import whisper

        return whisper.load_model(self.model)

    def transcribe(self, wav: bytes, language: str | None = None) -> dict:
        model = self._get()
        with _temp_wav(wav) as path:
            try:
                result = model.transcribe(path, language=language or None, fp16=False)
            except Exception as exc:  # noqa: BLE001 - whisper raises many kinds
                raise SpeechError(f"whisper failed to transcribe the audio: {exc}") from exc
        return {
            "transcript": str(result.get("text", "")).strip(),
            "model": self.model,
            "language": result.get("language") or language,
        }


class FasterWhisperTranscriber(_WhisperFamily):
    """``faster-whisper`` (CTranslate2; ``pip install faster-whisper``)."""

    name = "faster-whisper"
    module = "faster_whisper"
    install = "pip install faster-whisper"

    def load(self) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(self.model, device="auto", compute_type="default")

    def transcribe(self, wav: bytes, language: str | None = None) -> dict:
        model = self._get()
        with _temp_wav(wav) as path:
            try:
                segments, info = model.transcribe(path, language=language or None)
                text = " ".join(segment.text.strip() for segment in segments)
            except Exception as exc:  # noqa: BLE001 - faster-whisper raises many kinds
                raise SpeechError(f"faster-whisper failed to transcribe the audio: {exc}") from exc
        return {
            "transcript": " ".join(text.split()),
            "model": self.model,
            "language": getattr(info, "language", None) or language,
        }


class ServerTranscriber:
    """An OpenAI-compatible ``/v1/audio/transcriptions`` endpoint, over the standard library.

    ``$RADIXNET_ASR_URL`` (whisper.cpp's server, Speaches, ...) and
    ``$RADIXNET_ASR_MODEL`` configure it; ``auto`` uses it when a URL is set.
    """

    name = "server"

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = ASR_TIMEOUT) -> None:
        self.url = str(url if url is not None else DEFAULT_ASR_URL or "").strip()
        self.model = model or DEFAULT_ASR_MODEL
        self.timeout = float(timeout)
        self.error: str | None = None

    def available(self) -> bool:
        return bool(self.url)

    def transcribe(self, wav: bytes, language: str | None = None) -> dict:
        import json
        import urllib.error
        import urllib.request

        if not self.url:
            raise SpeechError("no transcription server configured (set $RADIXNET_ASR_URL or pass --asr-url)")
        fields = {"model": self.model}
        if language:
            fields["language"] = language
        body, content_type = _multipart(fields, "file", "speech.wav", wav, "audio/wav")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": content_type}, method="POST")
        key = os.environ.get("RADIXNET_ASR_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            self.error = f"the transcription server answered {exc.code}: {detail[:200]}"
            raise SpeechError(self.error) from exc
        except (urllib.error.URLError, OSError) as exc:
            self.error = f"cannot reach the transcription server at {self.url}: {exc}"
            raise SpeechError(self.error) from exc
        try:
            document = json.loads(raw)
        except ValueError:
            document = {"text": raw}
        transcript = document.get("text") if isinstance(document, dict) else None
        if not isinstance(transcript, str):
            raise SpeechError("the transcription server did not answer with a 'text' field")
        self.error = None
        return {"transcript": transcript.strip(), "model": self.model, "language": language}


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    """A ``multipart/form-data`` body (no dependencies)."""
    boundary = "----radixnet-" + hashlib.blake2b(data[:4096], digest_size=8).hexdigest()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class _TempWav:
    __slots__ = ("data", "path")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.path = ""

    def __enter__(self) -> str:
        handle, self.path = tempfile.mkstemp(prefix="radixnet-speech-", suffix=".wav")
        with os.fdopen(handle, "wb") as fh:
            fh.write(self.data)
        return self.path

    def __exit__(self, *exc_info) -> None:
        try:
            os.unlink(self.path)
        except OSError:  # pragma: no cover - the file was removed by the backend
            pass


def _temp_wav(data: bytes) -> _TempWav:
    """The transcription libraries want a path, so the WAV is written to a temporary file."""
    return _TempWav(data)


_INSTANCES: dict[str, Any] = {}


def get_transcriber(backend: str = "auto", text: str = "", model: str | None = None, url: str | None = None) -> Any:
    """The transcription backend ``backend``; ``auto`` picks the first usable one.

    Order: a given transcript, faster-whisper, whisper, then a configured
    transcription server.  A named backend is returned even when it has to
    raise later, so the caller sees *why* it cannot transcribe.
    """
    key = (backend or "auto").strip().lower().replace("_", "-")
    if key not in ASR_BACKENDS:
        raise SpeechError(f"unknown transcription backend {backend!r}; expected one of: {', '.join(ASR_BACKENDS)}")
    if key == "given":
        return GivenTranscriber(text)
    if key == "server":
        return ServerTranscriber(url, model)
    if key in ("whisper", "faster-whisper"):
        cls = WhisperTranscriber if key == "whisper" else FasterWhisperTranscriber
        instance = _INSTANCES.get(key)
        if instance is None or not isinstance(instance, cls):
            instance = _INSTANCES[key] = cls(model)
        elif model and instance.model != model:
            instance.model = model
        return instance
    given = GivenTranscriber(text)
    if given.available():
        return given
    for name in ("faster-whisper", "whisper"):
        candidate = get_transcriber(name, model=model)
        if candidate.available():
            return candidate
    server = ServerTranscriber(url, model)
    if server.available():
        return server
    raise SpeechError(
        "no transcription backend: pass the text you spoke, install one "
        "(pip install faster-whisper, or openai-whisper), or set $RADIXNET_ASR_URL to an "
        "OpenAI-compatible /v1/audio/transcriptions endpoint"
    )


def transcribe(
    data: bytes | None,
    backend: str = "auto",
    text: str = "",
    language: str | None = None,
    model: str | None = None,
    url: str | None = None,
) -> dict:
    """Audio bytes -> ``{"transcript", "backend", "model", "language", "seconds"}``.

    ``text`` is a transcript the caller already has (the browser dictated it,
    or the user typed it); it wins over every other backend under ``auto``.
    Audio that is not a WAV is converted with ffmpeg first, because that is
    what every backend reads.
    """
    transcriber = get_transcriber(backend, text=text, model=model, url=url)
    started = time.monotonic()
    if isinstance(transcriber, GivenTranscriber):
        result = transcriber.transcribe(b"", language=language)
    else:
        if not data:
            raise SpeechError("no audio to transcribe")
        wav = bytes(data) if bytes(data)[:4] == b"RIFF" else convert_with_ffmpeg(bytes(data), rate=16000)
        result = transcriber.transcribe(wav, language=language)
    result["backend"] = transcriber.name
    result["seconds"] = round(time.monotonic() - started, 3)
    result.setdefault("model", None)
    result.setdefault("language", language)
    return result


# ---------------------------------------------------------------------------
# recording from a microphone
# ---------------------------------------------------------------------------

RECORDERS = ("arecord", "rec", "sox", "ffmpeg")


def _recorder_command(binary: str, name: str, seconds: float, rate: int, path: str) -> list[str] | None:
    if name == "arecord":
        return [binary, "-q", "-f", "S16_LE", "-c", "1", "-r", str(rate), "-d", str(int(math.ceil(seconds))), path]
    if name in ("rec", "sox"):
        head = [binary, "-q", "-d"] if name == "sox" else [binary, "-q"]
        return head + ["-c", "1", "-r", str(rate), "-b", "16", path, "trim", "0", str(seconds)]
    if name == "ffmpeg":
        device = {"linux": ("alsa", "default"), "darwin": ("avfoundation", ":0")}.get(sys.platform)
        if device is None:  # Windows needs a dshow device name we cannot guess
            return None
        return [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", device[0], "-i", device[1],
                "-t", str(seconds), "-ac", "1", "-ar", str(rate), path]
    return None  # pragma: no cover - RECORDERS is closed


def recorders() -> list[str]:
    """The microphone recorders found on ``PATH`` (``arecord``, ``rec`` / ``sox``, ``ffmpeg``)."""
    found = []
    for name in RECORDERS:
        binary = ffmpeg_path() if name == "ffmpeg" else shutil.which(name)
        if binary and _recorder_command(binary, name, 1.0, 16000, "out.wav"):
            found.append(name)
    return found


def record(seconds: float = 5.0, rate: int = 16000, recorder: str | None = None) -> bytes:
    """Record ``seconds`` of mono audio from the default input device; returns WAV bytes.

    Uses whichever of ``arecord``, ``rec`` / ``sox`` and ``ffmpeg`` is
    installed (the browser's Speech tab records without any of them).
    """
    if seconds <= 0:
        raise SpeechError(f"the recording length must be positive, got {seconds}")
    names = [recorder] if recorder else list(RECORDERS)
    if recorder and recorder not in RECORDERS:
        raise SpeechError(f"unknown recorder {recorder!r}; expected one of: {', '.join(RECORDERS)}")
    handle, path = tempfile.mkstemp(prefix="radixnet-record-", suffix=".wav")
    os.close(handle)
    try:
        for name in names:
            binary = ffmpeg_path() if name == "ffmpeg" else shutil.which(name)
            command = _recorder_command(binary, name, seconds, rate, path) if binary else None
            if command is None:
                continue
            try:
                proc = subprocess.run(command, capture_output=True, timeout=seconds + 60, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                raise SpeechError(f"{name} could not be run: {exc}") from exc
            if proc.returncode != 0 or not os.path.getsize(path):
                detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
                raise SpeechError(f"{name} could not record: {detail[-1] if detail else 'no audio was written'}")
            with open(path, "rb") as fh:
                return fh.read()
        raise SpeechError(
            "no microphone recorder found: install alsa-utils (arecord), sox (rec) or ffmpeg, "
            "or record in the browser's Speech tab"
        )
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# the whole flow
# ---------------------------------------------------------------------------


def teach(
    data: bytes | None = None,
    transcript: str = "",
    *,
    backend: str = "auto",
    language: str | None = None,
    asr_model: str | None = None,
    asr_url: str | None = None,
    rate: int = DEFAULT_RATE,
    codec: str = "auto",
    normalise: bool = False,
    waveform: bool = True,
    pair: bool = False,
    token: str | None = None,
    unique: bool = True,
) -> dict:
    """One utterance -> the texts the network learns, both behind the same unique token.

    The audio is transcribed (unless ``transcript`` already holds the words)
    and encoded as a waveform text; the token is derived from the waveform, so
    the transcript and the sound share it.  A failing transcription is
    *recorded*, not raised: the waveform alone is still worth training on
    (``asr.error`` says what went wrong).

    Returns ``{"token", "transcript", "asr", "audio", "texts", "chars", "pair"}``.
    """
    if not data and not str(transcript or "").strip():
        raise SpeechError("nothing to learn: pass audio, a transcript, or both")
    audio: dict | None = None
    if data and waveform:
        audio = encode_audio(data, rate=rate, codec=codec, normalise=normalise)
    asr: dict = {"backend": None, "model": None, "language": language, "seconds": 0.0, "error": None}
    words = " ".join(str(transcript or "").split())
    if not words and data:
        try:
            result = transcribe(data, backend=backend, text="", language=language, model=asr_model, url=asr_url)
            words = " ".join(str(result.get("transcript") or "").split())
            asr.update(backend=result.get("backend"), model=result.get("model"),
                       language=result.get("language"), seconds=result.get("seconds", 0.0))
        except SpeechError as exc:
            asr["error"] = str(exc)
    elif words:
        asr["backend"] = "given"
    seed = audio["text"] if audio else words
    chosen = check_token(token) if token else utterance_token(seed, unique=unique)
    texts = speech_texts(chosen, words, audio["text"] if audio else "", pair=pair)
    return {
        "token": chosen,
        "transcript": words,
        "asr": asr,
        "audio": audio,
        "texts": texts,
        "chars": sum(len(t) for t in texts),
        "pair": bool(pair and words and audio),
    }


def describe() -> dict:
    """What is available: the transcription backends, ffmpeg, the recorders and the text format."""
    faster = FasterWhisperTranscriber.importable()
    whisper = WhisperTranscriber.importable()
    server = bool(DEFAULT_ASR_URL)
    auto = "faster-whisper" if faster else ("whisper" if whisper else ("server" if server else None))
    return {
        "backends": list(ASR_BACKENDS),
        "faster_whisper": faster,
        "whisper": whisper,
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "server_url": DEFAULT_ASR_URL or None,
        "server_model": DEFAULT_ASR_MODEL,
        "auto": auto,
        "given_always": True,
        "ffmpeg": bool(ffmpeg_path()),
        "recorders": recorders(),
        "codecs": list(CODECS),
        "default_codec": DEFAULT_CODEC,
        "default_rate": DEFAULT_RATE,
        "token": SPEECH_TOKEN,
        "token_example": utterance_token(b"example"),
        "text_format": f"{HEADER}:<codec>:<rate>x<channels>:<base64 of one byte per sample>",
        "formats": "WAV natively (PCM, IEEE float, A-law, mu-law); other formats need ffmpeg",
    }
