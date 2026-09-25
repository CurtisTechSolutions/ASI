"""A voice for the sounds: a formant synthesizer that speaks phones as they arrive.

This is the *decoder* of the tokenizer - sounds in, speech out - built the way
speech synthesizers were built before there were models to train: as a
**source-filter vocoder**.  The source is what the vocal folds make (a glottal
pulse at the pitch, or nothing) and what turbulence makes (noise); the filter
is the vocal tract, a cascade of resonators tuned to the *formants* of each
sound.  Every phone in ``data/voice.tsv`` says where its formants are, how long
it lasts, whether it is voiced and what noise it carries; the synthesizer
glides the formants from one sound to the next (coarticulation), gives stressed
syllables a higher pitch and a longer stay, lets the pitch fall over an
utterance and at a full stop and rise at a question, and writes 16-bit PCM.

Nothing is learned and nothing is downloaded; it is arithmetic on a table, in
every port alike, so it runs anywhere the tokenizer runs.  It does not sound
like a person - it sounds like the machines that first spoke - but every sound
is where it should be, and that is what a model whose symbols are sounds needs
to be heard.

**Streaming.**  :class:`Synthesizer` is fed one token at a time
(:meth:`Synthesizer.feed`) and hands back the samples it can already commit
to: a word's sounds are rendered as soon as the token after the word arrives
(a boundary, a pause or the end), which is what phrase-final lengthening and the
final pitch movement need to know.  :meth:`Synthesizer.end` is the **final
sentinel**: the end of the utterance, ``</s>``, which flushes what is pending
with the closing intonation and the closing silence.  So a model that walks
its graph sound by sound can be heard as it walks, and the utterance is
finished the moment the walk reaches its END.
"""

from __future__ import annotations

import io
import math
import os
import struct
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .phones import BOUNDARY, BOS, EOS, PAUSES, PAUSE_FULL, PAUSE_QUESTION, PAUSE_SHORT, base, is_vowel, stress_of
from .tokenizer import parse_token

VOICE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voice.tsv")

RATE = 16000
"""The default sample rate.  8000 works too (the speech encoder of RadixCyclicNN's rate), a little duller."""


@dataclass(frozen=True)
class Sound:
    """One row of the voice table: what a phone is made of."""

    phone: str
    kind: str
    f: tuple[float, float, float]
    bw: tuple[float, float, float]
    ms: float
    voiced: bool
    noise_f: float
    noise_bw: float
    noise_amp: float
    amp: float
    f2: tuple[float, float, float] | None  # a diphthong's second target


def load_voice(path: str = VOICE_PATH) -> dict[str, Sound]:
    """Read ``voice.tsv``: phone -> :class:`Sound`."""
    table: dict[str, Sound] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split()
            phone, kind = cols[0], cols[1]
            nums = [float(x) for x in cols[2:]]
            f1, f2, f3, b1, b2, b3, ms, voiced, nf, nbw, namp, amp, f1b, f2b, f3b = nums
            table[phone] = Sound(
                phone, kind, (f1, f2, f3), (b1, b2, b3), ms, voiced >= 1, nf, nbw, namp, amp,
                (f1b, f2b, f3b) if kind == "diph" else None,
            )
    return table


_VOICE: dict[str, Sound] | None = None


def voice() -> dict[str, Sound]:
    global _VOICE
    if _VOICE is None:
        _VOICE = load_voice()
    return _VOICE


@dataclass
class Voice:
    """How the voice is set: its pitch, its pace, its loudness, the pauses it takes."""

    pitch: float = 120.0
    """The base F0 in Hz: about 120 for a low voice, 210 for a high one."""
    tempo: float = 1.0
    """1 is the table's pace; 1.5 is half again as fast."""
    gain: float = 0.5
    """Peak level, as a share of full scale."""
    short_pause: float = 0.15
    """Seconds of silence at a comma."""
    full_pause: float = 0.40
    """Seconds of silence at a full stop or a question mark."""
    lead: float = 0.05
    """Seconds of silence before the first sound."""


class _Noise:
    """A deterministic noise source (xorshift32), the same numbers in every port."""

    __slots__ = ("state",)

    def __init__(self, seed: int = 0x9E3779B9) -> None:
        self.state = seed & 0xFFFFFFFF or 1

    def next(self) -> float:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x / 2147483648.0 - 1.0  # [-1, 1)


def _resonator(f: float, bw: float, rate: float) -> tuple[float, float, float]:
    """The coefficients of a two-pole resonator with unit gain at DC: ``y = a x + b y1 + c y2``."""
    c = -math.exp(-2.0 * math.pi * bw / rate)
    b = 2.0 * math.exp(-math.pi * bw / rate) * math.cos(2.0 * math.pi * f / rate)
    a = 1.0 - b - c
    return a, b, c


@dataclass
class _Segment:
    """One stretch of sound to render: its targets at start and end, and its source."""

    f_from: tuple[float, float, float]
    f_to: tuple[float, float, float]
    bw: tuple[float, float, float]
    seconds: float
    voiced: bool
    amp: float
    noise_f: float
    noise_bw: float
    noise_amp: float
    pitch_from: float
    pitch_to: float
    aspiration: bool = False  # noise shaped by the formants (HH, the release of P T K)


class Synthesizer:
    """Speak phones as they arrive.  See the module docstring.

    Feed it the tokens of the phonetic tokenizer at any level - phones, syllables,
    constituents, ``#``, the pauses, ``<s>`` and ``</s>`` - and take the PCM it
    returns; :meth:`end` (or an ``</s>``) closes the utterance.
    """

    FRAME_MS = 5.0

    def __init__(self, rate: int = RATE, voice_settings: Voice | None = None) -> None:
        self.rate = rate
        self.voice = voice_settings or Voice()
        self.table = voice()
        self._noise = _Noise()
        self._pending: list[str] = []  # the phones of the word being read
        self._last_f: tuple[float, float, float] = (500.0, 1500.0, 2500.0)
        self._y = [[0.0, 0.0] for _ in range(4)]  # the resonators' memories (F1-F3 and the noise one)
        self._phase = 1.0  # position in the glottal period, 1 = a new pulse is due
        self._period = rate / self.voice.pitch
        self._src_prev = 0.0
        self._started = False
        self._elapsed = 0.0  # seconds of speech so far in the utterance (for the declination)
        self._words = 0
        self.total_samples = 0

    # -- the stream -----------------------------------------------------------------

    def feed(self, token: str) -> bytes:
        """One token in; the PCM that can be committed comes back (often ``b""``)."""
        if token == EOS:
            return self.end()
        if token == BOS:
            return b""
        parsed = parse_token(token)
        if parsed is None:
            return b""
        kind, phones = parsed
        if kind == "boundary":
            return self._flush(final=None)
        if kind == "pause":
            return self._flush(final=token) + self._silence(
                self.voice.short_pause if token == PAUSE_SHORT else self.voice.full_pause
            )
        if kind == "special":
            return b""
        self._pending.extend(phones)
        return b""

    def end(self) -> bytes:
        """The final sentinel: what is pending is spoken with the closing intonation, then the utterance ends."""
        out = self._flush(final=PAUSE_FULL)
        if self._started:
            out += self._silence(0.12)
        self._reset_utterance()
        return out

    def speak(self, tokens: Iterable[str]) -> bytes:
        """The whole of a token stream, ended."""
        chunks = [self.feed(t) for t in tokens]
        chunks.append(self.end())
        return b"".join(chunks)

    def stream(self, tokens: Iterable[str]) -> Iterator[bytes]:
        """The PCM of a token stream as it comes: a chunk whenever a word can be committed, and the end."""
        for t in tokens:
            chunk = self.feed(t)
            if chunk:
                yield chunk
        chunk = self.end()
        if chunk:
            yield chunk

    def _reset_utterance(self) -> None:
        self._started = False
        self._elapsed = 0.0
        self._words = 0
        self._last_f = (500.0, 1500.0, 2500.0)

    # -- prosody and segments ---------------------------------------------------------

    def _flush(self, final: str | None) -> bytes:
        """Render the pending word.  ``final`` is the pause that follows it, if any (``None``: a boundary)."""
        phones = self._pending
        self._pending = []
        if not phones:
            return b""
        out = bytearray()
        if not self._started:
            out += self._silence(self.voice.lead)
            self._started = True
        segments = self._word_segments(phones, final)
        for seg in segments:
            out += self._render(seg)
        self._words += 1
        return bytes(out)

    def _pitch_at(self, seconds: float, stress: int | None, final: str | None, last_syllable: bool) -> float:
        """The F0 at a moment: the base, the declination, the stress, the final movement."""
        f0 = self.voice.pitch * (1.0 - 0.12 * min(seconds / 3.0, 1.0))
        if stress == 1:
            f0 *= 1.18
        elif stress == 2:
            f0 *= 1.08
        if last_syllable and final == PAUSE_QUESTION:
            f0 *= 1.35
        elif last_syllable and final == PAUSE_FULL:
            f0 *= 0.82
        elif last_syllable and final == PAUSE_SHORT:
            f0 *= 1.06
        return f0

    def _word_segments(self, phones: list[str], final: str | None) -> list[_Segment]:
        table = self.table
        vowels = [i for i, p in enumerate(phones) if is_vowel(p)]
        last_vowel = vowels[-1] if vowels else len(phones) - 1
        segs: list[_Segment] = []
        t = self._elapsed
        tempo = self.voice.tempo
        for i, phone in enumerate(phones):
            b = base(phone)
            sound = table.get(b)
            if sound is None:
                continue
            stress = stress_of(phone)
            last_syllable = i >= last_vowel
            seconds = sound.ms / 1000.0 / tempo
            if is_vowel(b):
                if stress == 1:
                    seconds *= 1.25
                elif stress == 0:
                    seconds *= 0.75
            if last_syllable and final is not None:
                seconds *= 1.35 if is_vowel(b) else 1.15
            amp = sound.amp
            if is_vowel(b) and stress == 0:
                amp *= 0.7
            f_to = sound.f
            pitch_from = self._pitch_at(t, stress, final, last_syllable)
            pitch_to = self._pitch_at(t + seconds, stress, final, last_syllable)
            if sound.kind in ("stop", "affr"):
                # a closure (silence, or a faint voice bar), a burst of noise, and for a voiceless stop aspiration
                closure = seconds * (0.55 if sound.kind == "stop" else 0.4)
                segs.append(_Segment(self._last_f, sound.f, sound.bw, closure, sound.voiced, sound.amp * 0.4,
                                     0.0, 0.0, 0.0, pitch_from, pitch_from))
                burst = 0.012 if sound.kind == "stop" else seconds - closure
                segs.append(_Segment(sound.f, sound.f, sound.bw, burst, False, 0.0, sound.noise_f, sound.noise_bw,
                                     sound.noise_amp, pitch_from, pitch_from))
                if not sound.voiced and sound.kind == "stop":
                    segs.append(_Segment(sound.f, sound.f, sound.bw, 0.035, False, 0.0, 0.0, 0.0, 0.18,
                                         pitch_from, pitch_from, aspiration=True))
                self._last_f = sound.f
            elif sound.kind == "asp":
                segs.append(_Segment(self._last_f, sound.f, sound.bw, seconds, False, 0.0, 0.0, 0.0, sound.noise_amp,
                                     pitch_from, pitch_to, aspiration=True))
                self._last_f = sound.f
            else:
                if sound.kind == "diph" and sound.f2 is not None:
                    half = seconds / 2.0
                    segs.append(_Segment(self._last_f, sound.f, sound.bw, half, True, amp, 0.0, 0.0, 0.0,
                                         pitch_from, (pitch_from + pitch_to) / 2.0))
                    segs.append(_Segment(sound.f, sound.f2, sound.bw, half, True, amp, 0.0, 0.0, 0.0,
                                         (pitch_from + pitch_to) / 2.0, pitch_to))
                    self._last_f = sound.f2
                else:
                    segs.append(_Segment(self._last_f, f_to, sound.bw, seconds, sound.voiced, amp, sound.noise_f,
                                         sound.noise_bw, sound.noise_amp, pitch_from, pitch_to))
                    self._last_f = f_to
            t += seconds
        self._elapsed = t
        return segs

    # -- the vocoder --------------------------------------------------------------------

    def _silence(self, seconds: float) -> bytes:
        n = int(round(seconds * self.rate))
        self.total_samples += n
        self._elapsed += seconds
        return b"\0\0" * n

    def _render(self, seg: _Segment) -> bytes:
        """The samples of one segment: source through the cascade, in 5 ms frames of fixed coefficients."""
        rate = self.rate
        total = max(1, int(round(seg.seconds * rate)))
        frame = max(1, int(rate * self.FRAME_MS / 1000.0))
        out = bytearray()
        y = self._y
        noise = self._noise
        gain = self.voice.gain * 32767.0
        nyquist = rate * 0.45
        done = 0
        while done < total:
            n = min(frame, total - done)
            mid = (done + n / 2.0) / total  # where this frame sits in the segment, for the glides
            # the transition into a sound takes its first 40 %, then the formants hold
            g = min(mid / 0.4, 1.0)
            f = [seg.f_from[k] + (seg.f_to[k] - seg.f_from[k]) * g for k in range(3)]
            coefs = [_resonator(min(f[k], nyquist), seg.bw[k], rate) for k in range(3)]
            noise_coefs = _resonator(min(seg.noise_f, nyquist), max(seg.noise_bw, 50.0), rate) if seg.noise_amp > 0 and not seg.aspiration else None
            pitch = seg.pitch_from + (seg.pitch_to - seg.pitch_from) * mid
            period = rate / max(pitch, 40.0)
            # a short fade at the edges of a voiced segment keeps the joins clean
            for i in range(n):
                # -- the sources --
                src = 0.0
                if seg.voiced and seg.amp > 0:
                    self._phase += 1.0 / period
                    if self._phase >= 1.0:
                        self._phase -= 1.0
                    ph = self._phase
                    # a glottal pulse (Rosenberg's): open for 60 % of the period, rising then falling
                    if ph < 0.4:
                        flow = 0.5 * (1.0 - math.cos(math.pi * ph / 0.4))
                    elif ph < 0.6:
                        flow = math.cos(math.pi * (ph - 0.4) / 0.4)
                    else:
                        flow = 0.0
                    src = (flow - self._src_prev) * 4.0 * seg.amp  # the derivative: the lips radiate
                    self._src_prev = flow
                if seg.aspiration and seg.noise_amp > 0:
                    src += noise.next() * seg.noise_amp * 0.6
                # -- the tract: F1, F2, F3 in cascade --
                x = src
                for k in range(3):
                    a, b, c = coefs[k]
                    m = y[k]
                    v = a * x + b * m[0] + c * m[1]
                    m[1] = m[0]
                    m[0] = v
                    x = v
                # -- frication: noise through its own resonator, added at the lips --
                if noise_coefs is not None:
                    a, b, c = noise_coefs
                    m = y[3]
                    v = a * noise.next() * seg.noise_amp + b * m[0] + c * m[1]
                    m[1] = m[0]
                    m[0] = v
                    x += v * 3.0
                s = x * gain * 0.09  # the cascade peaks near 4 x full scale at a gain of 1; this keeps 0.5 at half scale
                if s > 32767.0:
                    s = 32767.0
                elif s < -32767.0:
                    s = -32767.0
                out += struct.pack("<h", int(s))
            done += n
        self.total_samples += total
        return bytes(out)


# -- files and players ---------------------------------------------------------------------

def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    """A WAV file (16-bit mono) holding ``pcm``."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav(path: str, pcm: bytes, rate: int = RATE) -> None:
    with open(path, "wb") as fh:
        fh.write(wav_bytes(pcm, rate))


def wav_header(rate: int = RATE, samples: int | None = None) -> bytes:
    """The 44-byte header of a 16-bit mono WAV; ``samples`` unknown gives the streaming form (sizes of 0xFFFFFFFF)."""
    size = 0xFFFFFFFF if samples is None else samples * 2
    riff = 0xFFFFFFFF if samples is None else 36 + size
    return b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) \
        + b"data" + struct.pack("<I", size)


def find_player() -> list[str] | None:
    """A command that plays a WAV from stdin, if one is installed: aplay, paplay, ffplay, play or afplay."""
    import shutil

    for cmd, args in (("aplay", ["-q", "-"]), ("paplay", []), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", "-"]),
                      ("play", ["-q", "-t", "wav", "-"]), ("afplay", [])):
        path = shutil.which(cmd)
        if path:
            return [path, *args]
    return None


def duration(pcm: bytes, rate: int = RATE) -> float:
    return len(pcm) / 2.0 / rate
