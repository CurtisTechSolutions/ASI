"""Acoustic units: speech as the sounds a codebook learned from it, with nothing written down.

The rest of the package reads *text* as sounds.  This module reads *audio* as
sounds, and the sounds are its own: a codebook learned from recordings by
k-means, with no transcript, no dictionary and no rule.  Every 10 ms of audio
becomes a frame of 40 log-mel energies; every frame goes to its nearest
codebook entry; a run of one entry is one unit.  A unit is a token like
``q17`` - ``q`` for quantised, then the codebook index - and a text of them
(``"q3 q17 q4"``) is what a model over sounds can be trained on straight from
a microphone, with the same machinery that trains it on phones.

    >>> from phonetok.acoustic import AcousticTokenizer
    >>> tok = AcousticTokenizer()                          # the bundled codebook
    >>> units = tok.hear(open("speech.wav", "rb").read())  # ['q3', 'q17', 'q4', ...]
    >>> pcm = tok.synthesize(units)                        # and back to 16-bit audio

**Analysis.** The audio is brought to 16 kHz mono, pre-emphasised, cut into
25 ms frames every 10 ms, Hann-windowed, transformed (a 512-point FFT), and
its power spectrum is summed under 40 triangular filters spaced evenly on the
mel scale between 20 Hz and 8 kHz; the log of each sum is the frame, and the
mean over the utterance of every band is subtracted, which removes the
recording's level and channel.  Those settings live in the codebook file, so
a codebook always carries the analysis that made it.

**Learning** (:func:`learn`) is unsupervised: the frames of any number of
recordings, k-means++ seeding drawn from the Mersenne Twister the other ports
implement, then Lloyd's iterations until the assignments settle.  The result
is a :class:`Codebook`: the centroids, how many frames each one claimed, how
long a run of it typically lasted (decoding holds a unit that long), and the
mean log-mel of the training audio (decoding adds it back).

**Decoding** (:class:`Vocoder`, :func:`synthesize`) is the reverse path: a
unit is its centroid held for its typical run, the mel frame is spread back
over the linear spectrum, and a frame-by-frame vocoder with continuous phase
turns the magnitudes into a waveform as the units come - so a model walking
its graph is heard as it walks - and Griffin-Lim iterations polish the whole
utterance when it is known.  What comes back is speech-like and recognisably
the recording's sounds, not the recording.

Everything here is plain arithmetic, in the same order in the Go and Rust
ports, so a unit is the same unit in every port.  Standard library only.
"""

from __future__ import annotations

import math
import os
import random
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .synth import _Noise

RATE = 16000
"""The sample rate the analysis runs at; audio at another rate is resampled to it."""

UNIT_PREFIX = "q"
"""A unit token is ``q`` (for quantised) and the codebook index: ``q0``, ``q1``, ..."""

DEFAULT_CODEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "acoustic.tsv")
"""The bundled codebook, learned from the formant synthesizer's speech (``tests/make_codebook.py``)."""

LOG_FLOOR = 1e-10
"""Added to every band's energy before the log, so silence has a floor instead of minus infinity."""

TWO_PI = 2.0 * math.pi


# -- audio files ----------------------------------------------------------------------------

def read_wav(data: bytes) -> tuple[list[float], int]:
    """The samples of a WAV file, in ``[-1, 1]`` and mono (channels averaged), and its rate.

    PCM of 8, 16, 24 and 32 bits and IEEE float of 32 and 64 bits are read,
    plain or in the extensible header; other chunks are skipped.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file (no RIFF/WAVE header)")
    fmt: tuple[int, int, int, int] | None = None
    body: bytes | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        payload = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt " and len(payload) >= 16:
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", payload, 0)
            if tag == 0xFFFE and len(payload) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real tag is in the sub-format
                tag = struct.unpack_from("<H", payload, 24)[0]
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            body = payload
        pos += 8 + size + (size & 1)
    if fmt is None or body is None:
        raise ValueError("not a WAV file (no fmt or data chunk)")
    tag, channels, rate, bits = fmt
    if channels < 1 or rate < 1:
        raise ValueError(f"WAV with {channels} channels at {rate} Hz")
    if tag == 1:
        if bits == 8:
            values = [(b - 128) / 128.0 for b in body]
        elif bits == 16:
            values = [v / 32768.0 for v in struct.unpack(f"<{len(body) // 2}h", body[:len(body) // 2 * 2])]
        elif bits == 24:
            values = []
            for i in range(0, len(body) - 2, 3):
                v = body[i] | (body[i + 1] << 8) | (body[i + 2] << 16)
                if v & 0x800000:
                    v -= 0x1000000
                values.append(v / 8388608.0)
        elif bits == 32:
            values = [v / 2147483648.0 for v in struct.unpack(f"<{len(body) // 4}i", body[:len(body) // 4 * 4])]
        else:
            raise ValueError(f"{bits}-bit PCM WAV is not supported")
    elif tag == 3:
        if bits == 32:
            values = list(struct.unpack(f"<{len(body) // 4}f", body[:len(body) // 4 * 4]))
        elif bits == 64:
            values = list(struct.unpack(f"<{len(body) // 8}d", body[:len(body) // 8 * 8]))
        else:
            raise ValueError(f"{bits}-bit float WAV is not supported")
    else:
        raise ValueError(f"WAV format tag {tag} is not supported (PCM and IEEE float are)")
    if channels > 1:
        frames = len(values) // channels
        values = [sum(values[i * channels:(i + 1) * channels]) / channels for i in range(frames)]
    return values, rate


def load_wav(path: str) -> tuple[list[float], int]:
    with open(path, "rb") as fh:
        return read_wav(fh.read())


def pcm_samples(pcm: bytes) -> list[float]:
    """16-bit mono PCM (what the synthesizer makes) as samples in ``[-1, 1]``."""
    return [v / 32768.0 for v in struct.unpack(f"<{len(pcm) // 2}h", pcm[:len(pcm) // 2 * 2])]


def resample(samples: Sequence[float], src: int, dst: int, taps: int = 16) -> list[float]:
    """``samples`` at ``src`` Hz brought to ``dst`` Hz by a windowed-sinc interpolation."""
    if src == dst:
        return list(samples)
    if src < 1 or dst < 1:
        raise ValueError(f"cannot resample {src} Hz to {dst} Hz")
    n = len(samples)
    count = n * dst // src
    ratio = src / dst
    cutoff = min(1.0, dst / src) * 0.95
    out: list[float] = []
    for j in range(count):
        center = j * ratio
        base = int(math.floor(center))
        acc = 0.0
        norm = 0.0
        for i in range(base - taps + 1, base + taps + 1):
            if i < 0 or i >= n:
                continue
            x = i - center
            if x == 0.0:
                w = 1.0
            else:
                w = math.sin(math.pi * cutoff * x) / (math.pi * cutoff * x) * (0.5 + 0.5 * math.cos(math.pi * x / taps))
            acc += samples[i] * w
            norm += w
        out.append(acc / norm if norm != 0.0 else 0.0)
    return out


# -- the analysis ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    """How audio becomes frames: the settings a codebook carries."""

    rate: int = RATE
    frame: int = 400
    """Samples per frame: 25 ms at 16 kHz."""
    hop: int = 160
    """Samples between frames: 10 ms at 16 kHz."""
    fft: int = 512
    """Transform length, a power of two of at least ``frame``."""
    bands: int = 40
    """Mel bands per frame: the frame's dimension."""
    fmin: float = 20.0
    fmax: float = 8000.0
    preemphasis: float = 0.97
    normalize: bool = True
    """Subtract the mean of every band over the utterance (the level and the channel go with it)."""

    def validate(self) -> None:
        if self.rate < 1 or self.frame < 2 or self.hop < 1 or self.bands < 1:
            raise ValueError(f"impossible analysis: {self}")
        if self.fft < self.frame or self.fft & (self.fft - 1):
            raise ValueError(f"fft must be a power of two of at least the frame ({self.frame}), got {self.fft}")
        if not 0.0 <= self.fmin < self.fmax <= self.rate / 2.0:
            raise ValueError(f"the bands must lie in 0 .. {self.rate / 2:g} Hz, got {self.fmin:g} .. {self.fmax:g}")

    @property
    def seconds_per_frame(self) -> float:
        return self.hop / self.rate


DEFAULT_ANALYSIS = Analysis()


def hann(n: int) -> list[float]:
    """The periodic Hann window of ``n`` samples."""
    return [0.5 - 0.5 * math.cos(TWO_PI * i / n) for i in range(n)]


def fft(re: list[float], im: list[float], inverse: bool = False) -> None:
    """In place: the discrete Fourier transform of ``re + i im``, whose length is a power of two.

    An iterative radix-2 transform with the twiddle factor advanced by
    multiplication, exactly as the Go and Rust ports do it, so every port's
    spectrum is the same spectrum.  ``inverse`` transforms the other way and
    divides by the length.
    """
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        ang = sign * TWO_PI / length
        wr = math.cos(ang)
        wi = math.sin(ang)
        half = length >> 1
        for start in range(0, n, length):
            cr = 1.0
            ci = 0.0
            for k in range(start, start + half):
                m = k + half
                tr = re[m] * cr - im[m] * ci
                ti = re[m] * ci + im[m] * cr
                re[m] = re[k] - tr
                im[m] = im[k] - ti
                re[k] = re[k] + tr
                im[k] = im[k] + ti
                ncr = cr * wr - ci * wi
                nci = cr * wi + ci * wr
                cr = ncr
                ci = nci
        length <<= 1
    if inverse:
        for i in range(n):
            re[i] = re[i] / n
            im[i] = im[i] / n


def mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


_FILTERS: dict[Analysis, list[tuple[int, list[float]]]] = {}


def mel_filters(a: Analysis) -> list[tuple[int, list[float]]]:
    """The triangular filters: per band, the first bin it touches and its weight on each bin from there on."""
    cached = _FILTERS.get(a)
    if cached is not None:
        return cached
    lo, hi = mel(a.fmin), mel(a.fmax)
    edges = [mel_to_hz(lo + (hi - lo) * i / (a.bands + 1)) for i in range(a.bands + 2)]
    half = a.fft // 2
    filters: list[tuple[int, list[float]]] = []
    for b in range(a.bands):
        left, center, right = edges[b], edges[b + 1], edges[b + 2]
        first = -1
        weights: list[float] = []
        for k in range(half + 1):
            f = k * a.rate / a.fft
            if f <= left or f >= right:
                w = 0.0
            elif f <= center:
                w = (f - left) / (center - left)
            else:
                w = (right - f) / (right - center)
            if w > 0.0:
                if first < 0:
                    first = k
                weights.append(w)
            elif first >= 0:
                break
        if first < 0:  # a band narrower than a bin: give it the bin nearest its centre
            first = int(center * a.fft / a.rate + 0.5)
            weights = [1.0]
        filters.append((first, weights))
    _FILTERS[a] = filters
    return filters


def preemphasize(samples: Sequence[float], p: float) -> list[float]:
    out = list(samples)
    for i in range(len(out) - 1, 0, -1):
        out[i] = out[i] - p * out[i - 1]
    return out


def frames_of(samples: Sequence[float], a: Analysis = DEFAULT_ANALYSIS, normalize: bool | None = None) -> list[list[float]]:
    """The log-mel frames of ``samples`` (already at ``a.rate``); at least one frame, zero-padded if need be.

    ``normalize`` overrides the analysis' own setting (the learner reads the
    raw frames once to find their mean, then normalises).
    """
    a.validate()
    y = preemphasize(samples, a.preemphasis) if a.preemphasis else list(samples)
    if len(y) < a.frame:
        y.extend([0.0] * (a.frame - len(y)))
    count = (len(y) - a.frame) // a.hop + 1
    window = hann(a.frame)
    filters = mel_filters(a)
    half = a.fft // 2
    pad = [0.0] * (a.fft - a.frame)
    out: list[list[float]] = []
    for t in range(count):
        start = t * a.hop
        re = [y[start + i] * window[i] for i in range(a.frame)] + pad
        im = [0.0] * a.fft
        fft(re, im)
        power = [re[k] * re[k] + im[k] * im[k] for k in range(half + 1)]
        frame: list[float] = []
        for first, weights in filters:
            e = 0.0
            for i, w in enumerate(weights):
                e += w * power[first + i]
            frame.append(math.log(e + LOG_FLOOR))
        out.append(frame)
    if (a.normalize if normalize is None else normalize) and out:
        means = band_means(out)
        for frame in out:
            for b in range(a.bands):
                frame[b] = frame[b] - means[b]
    return out


def band_means(frames: Sequence[Sequence[float]]) -> list[float]:
    if not frames:
        return []
    dims = len(frames[0])
    sums = [0.0] * dims
    for frame in frames:
        for b in range(dims):
            sums[b] += frame[b]
    return [s / len(frames) for s in sums]


def dist2(x: Sequence[float], y: Sequence[float]) -> float:
    """The squared Euclidean distance, summed in order."""
    d = 0.0
    for a, b in zip(x, y):
        diff = a - b
        d += diff * diff
    return d


# -- the codebook ---------------------------------------------------------------------------

@dataclass
class Codebook:
    """What was learned: the centroids and what the training frames said about them."""

    analysis: Analysis
    centroids: list[list[float]]
    runs: list[float]
    """Mean length in frames of a run of each unit in the training audio (decoding holds a unit that long)."""
    counts: list[int]
    """Frames each unit claimed in the training audio."""
    mean: list[float]
    """The mean log-mel of the training frames before normalisation: decoding adds it back."""
    seed: int = 0
    note: str = ""
    inertia: float = 0.0
    """The summed squared distance of every training frame to its centroid, at the end."""

    @property
    def k(self) -> int:
        return len(self.centroids)

    def __len__(self) -> int:
        return len(self.centroids)

    def validate(self) -> None:
        self.analysis.validate()
        if not self.centroids:
            raise ValueError("an empty codebook")
        for c in self.centroids:
            if len(c) != self.analysis.bands:
                raise ValueError(f"a centroid of {len(c)} values in a codebook of {self.analysis.bands} bands")
        if len(self.runs) != self.k or len(self.counts) != self.k or len(self.mean) != self.analysis.bands:
            raise ValueError("a codebook whose tables disagree about its size")

    # -- units ------------------------------------------------------------------------------

    def name(self, index: int) -> str:
        return f"{UNIT_PREFIX}{index}"

    def names(self) -> list[str]:
        return [self.name(i) for i in range(self.k)]

    def index(self, token: str) -> int | None:
        """The codebook index of a unit token, or ``None`` when the token is not one of this codebook's units."""
        if len(token) < 2 or not token.startswith(UNIT_PREFIX) or not token[1:].isdigit():
            return None
        i = int(token[1:])
        if i >= self.k or token != self.name(i):  # no leading zeros, no aliases
            return None
        return i

    def nearest(self, frame: Sequence[float]) -> int:
        """The index of the centroid nearest the frame (the first of equals)."""
        best = 0
        best_d = dist2(frame, self.centroids[0])
        for i in range(1, self.k):
            d = dist2(frame, self.centroids[i])
            if d < best_d:
                best, best_d = i, d
        return best

    def codes(self, frames: Iterable[Sequence[float]]) -> list[int]:
        return [self.nearest(f) for f in frames]

    def units(self, frames: Iterable[Sequence[float]], collapse: bool = True) -> list[str]:
        codes = self.codes(frames)
        if collapse:
            codes = collapse_runs(codes)
        return [self.name(c) for c in codes]

    def hold(self, index: int) -> int:
        """Frames a unit is held for when decoded: its typical run, at least one."""
        return max(1, int(self.runs[index] + 0.5))

    # -- the file -----------------------------------------------------------------------------

    def dumps(self) -> str:
        a = self.analysis
        lines = [f"# phonetok acoustic codebook: {self.k} units learned by k-means over log-mel frames"]
        for line in self.note.splitlines():
            lines.append(f"# {line}")
        lines += [
            f"rate\t{a.rate}", f"frame\t{a.frame}", f"hop\t{a.hop}", f"fft\t{a.fft}", f"bands\t{a.bands}",
            f"fmin\t{a.fmin!r}", f"fmax\t{a.fmax!r}", f"preemphasis\t{a.preemphasis!r}",
            f"normalize\t{1 if a.normalize else 0}", f"seed\t{self.seed}", f"inertia\t{self.inertia!r}",
            f"units\t{self.k}",
            "mean\t" + "\t".join(repr(v) for v in self.mean),
        ]
        for i, c in enumerate(self.centroids):
            lines.append(f"{self.name(i)}\t{self.counts[i]}\t{self.runs[i]!r}\t" + "\t".join(repr(v) for v in c))
        return "\n".join(lines) + "\n"

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())

    @classmethod
    def loads(cls, text: str) -> Codebook:
        settings: dict[str, str] = {}
        note: list[str] = []
        mean: list[float] = []
        rows: list[tuple[str, int, float, list[float]]] = []
        for raw in text.splitlines():
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if not line.startswith("# phonetok acoustic codebook"):
                    note.append(line[2:] if line.startswith("# ") else line[1:])
                continue
            parts = line.split("\t")
            key = parts[0]
            if key == "mean":
                mean = [float(v) for v in parts[1:]]
            elif key.startswith(UNIT_PREFIX) and key[1:].isdigit():
                if len(parts) < 4:
                    raise ValueError(f"codebook row {key!r} is incomplete")
                rows.append((key, int(parts[1]), float(parts[2]), [float(v) for v in parts[3:]]))
            elif len(parts) == 2:
                settings[key] = parts[1]
            else:
                raise ValueError(f"unreadable codebook line: {line!r}")
        try:
            analysis = Analysis(
                rate=int(settings.get("rate", RATE)), frame=int(settings.get("frame", 400)),
                hop=int(settings.get("hop", 160)), fft=int(settings.get("fft", 512)),
                bands=int(settings.get("bands", 40)), fmin=float(settings.get("fmin", 20.0)),
                fmax=float(settings.get("fmax", 8000.0)), preemphasis=float(settings.get("preemphasis", 0.97)),
                normalize=settings.get("normalize", "1") not in ("0", "false", "no"),
            )
        except ValueError as exc:
            raise ValueError(f"unreadable codebook settings: {exc}") from None
        book = cls(
            analysis=analysis, centroids=[r[3] for r in rows], runs=[r[2] for r in rows], counts=[r[1] for r in rows],
            mean=mean, seed=int(settings.get("seed", 0)), note="\n".join(note), inertia=float(settings.get("inertia", 0.0)),
        )
        for i, r in enumerate(rows):
            if r[0] != book.name(i):
                raise ValueError(f"codebook rows out of order: {r[0]!r} where {book.name(i)!r} was expected")
        book.validate()
        return book

    @classmethod
    def load(cls, path: str = DEFAULT_CODEBOOK) -> Codebook:
        with open(path, encoding="utf-8") as fh:
            return cls.loads(fh.read())


_DEFAULT: list[Codebook] = []


def default_codebook() -> Codebook:
    """The bundled codebook, read once."""
    if not _DEFAULT:
        _DEFAULT.append(Codebook.load(DEFAULT_CODEBOOK))
    return _DEFAULT[0]


def collapse_runs(codes: Sequence[int]) -> list[int]:
    out: list[int] = []
    for c in codes:
        if not out or out[-1] != c:
            out.append(c)
    return out


def run_lengths(codes: Sequence[int], k: int) -> tuple[list[int], list[int]]:
    """Per code: frames it claimed and runs it made (their ratio is the mean run)."""
    frames = [0] * k
    runs = [0] * k
    prev = -1
    for c in codes:
        frames[c] += 1
        if c != prev:
            runs[c] += 1
        prev = c
    return frames, runs


# -- learning -------------------------------------------------------------------------------

@dataclass
class Learned:
    """What :func:`kmeans` found."""

    centroids: list[list[float]]
    assignments: list[int]
    inertia: float
    iterations: int
    converged: bool


def kmeans(points: Sequence[Sequence[float]], k: int, rng: random.Random, iterations: int = 50) -> Learned:
    """k-means++ seeding, then Lloyd's iterations until nothing moves (or ``iterations`` are up).

    An empty cluster takes over the point farthest from its centroid.  Every
    draw comes from ``rng`` (CPython's Mersenne Twister, which the Go and Rust
    ports reproduce), so the same seed learns the same codebook in every port.
    """
    n = len(points)
    if k < 1:
        raise ValueError("k must be at least 1")
    if n < k:
        raise ValueError(f"{n} frames cannot make {k} units")
    # k-means++: the first centre at random, every next one with probability proportional to its squared distance
    first = rng.randrange(n)
    centroids = [list(points[first])]
    d2 = [dist2(p, centroids[0]) for p in points]
    for _ in range(1, k):
        total = 0.0
        for d in d2:
            total += d
        chosen = n - 1
        if total > 0.0:
            r = rng.random() * total
            acc = 0.0
            for i, d in enumerate(d2):
                acc += d
                if acc >= r:
                    chosen = i
                    break
        else:  # every point sits on a centre already: any point will do, deterministically
            chosen = rng.randrange(n)
        centre = list(points[chosen])
        centroids.append(centre)
        for i, p in enumerate(points):
            d = dist2(p, centre)
            if d < d2[i]:
                d2[i] = d
    dims = len(points[0]) if n else 0
    assignments = [-1] * n
    inertia = 0.0
    converged = False
    done = 0
    for it in range(iterations):
        done = it + 1
        # assign
        changed = 0
        inertia = 0.0
        for i, p in enumerate(points):
            best = 0
            best_d = dist2(p, centroids[0])
            for c in range(1, k):
                d = dist2(p, centroids[c])
                if d < best_d:
                    best, best_d = c, d
            if best != assignments[i]:
                changed += 1
                assignments[i] = best
            d2[i] = best_d
            inertia += best_d
        if changed == 0:
            converged = True
            break
        # update
        sums = [[0.0] * dims for _ in range(k)]
        counts = [0] * k
        for i, p in enumerate(points):
            c = assignments[i]
            s = sums[c]
            for b in range(dims):
                s[b] += p[b]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = [v / counts[c] for v in sums[c]]
            else:  # an empty cluster: the point farthest from where it was assigned starts it afresh
                far = 0
                far_d = -1.0
                for i in range(n):
                    if d2[i] > far_d:
                        far, far_d = i, d2[i]
                centroids[c] = list(points[far])
                assignments[far] = c
                d2[far] = 0.0
    return Learned(centroids=centroids, assignments=assignments, inertia=inertia, iterations=done, converged=converged)


def learn(
    recordings: Iterable[Sequence[float]],
    k: int = 64,
    seed: int = 1,
    iterations: int = 50,
    analysis: Analysis = DEFAULT_ANALYSIS,
    note: str = "",
) -> Codebook:
    """A codebook of ``k`` units learned from ``recordings`` (each a sequence of samples at ``analysis.rate``).

    Nothing but the audio is looked at.  The frames are analysed raw, their
    mean over all the recordings is kept for decoding, every recording is
    normalised on its own (as it will be when it is heard), and k-means
    clusters the lot; each unit's count and typical run come from the final
    assignments read back recording by recording.
    """
    analysis.validate()
    utterances: list[list[list[float]]] = []
    sums = [0.0] * analysis.bands
    total = 0
    for samples in recordings:
        frames = frames_of(samples, analysis, normalize=False)
        for f in frames:
            for b in range(analysis.bands):
                sums[b] += f[b]
        total += len(frames)
        if analysis.normalize:
            means = band_means(frames)
            for f in frames:
                for b in range(analysis.bands):
                    f[b] = f[b] - means[b]
        utterances.append(frames)
    if total == 0:
        raise ValueError("no audio to learn from")
    mean = [s / total for s in sums]
    points = [f for frames in utterances for f in frames]
    found = kmeans(points, k, random.Random(seed), iterations)
    counts = [0] * k
    runs = [0] * k
    pos = 0
    for frames in utterances:
        codes = found.assignments[pos:pos + len(frames)]
        pos += len(frames)
        fr, ru = run_lengths(codes, k)
        for c in range(k):
            counts[c] += fr[c]
            runs[c] += ru[c]
    mean_runs = [counts[c] / runs[c] if runs[c] else 1.0 for c in range(k)]
    return Codebook(
        analysis=analysis, centroids=found.centroids, runs=mean_runs, counts=counts, mean=mean, seed=seed,
        note=note, inertia=found.inertia,
    )


# -- decoding -------------------------------------------------------------------------------

_AREAS: dict[Analysis, list[float]] = {}


def band_areas(a: Analysis) -> list[float]:
    """Per band, the summed weight of its filter: the width in bins its energy was summed over."""
    areas = _AREAS.get(a)
    if areas is None:
        areas = []
        for _, weights in mel_filters(a):
            total = 0.0
            for w in weights:
                total += w
            areas.append(total)
        _AREAS[a] = areas
    return areas


def magnitudes_of(logmel: Sequence[float], a: Analysis) -> list[float]:
    """The linear magnitude spectrum (``fft/2 + 1`` bins) a log-mel frame stands for.

    A band's energy was summed over its bins, so it is spread back as that
    energy per unit of filter weight through the same triangles; where two
    filters overlap their weights sum to one, so a bin under both takes their
    mean, and at the ends of the range the weights taper off on their own.
    """
    half = a.fft // 2
    power = [0.0] * (half + 1)
    for b, ((first, ws), area) in enumerate(zip(mel_filters(a), band_areas(a))):
        e = math.exp(logmel[b]) / area
        for i, w in enumerate(ws):
            power[first + i] += w * e
    return [math.sqrt(p) for p in power]


def _pack(samples: Iterable[float], gain: float) -> bytes:
    out = bytearray()
    scale = gain * 32767.0
    for x in samples:
        s = x * scale
        if s > 32767.0:
            s = 32767.0
        elif s < -32767.0:
            s = -32767.0
        out += struct.pack("<h", int(s))
    return bytes(out)


PITCH = 120.0
"""The pitch of the vocoder's pulse train, in Hz: the voice the units are given back in."""


def band_centres(a: Analysis) -> list[float]:
    """The centre frequency of every band, in Hz."""
    lo, hi = mel(a.fmin), mel(a.fmax)
    return [mel_to_hz(lo + (hi - lo) * (b + 1) / (a.bands + 1)) for b in range(a.bands)]


_VOICING_BANDS: dict[Analysis, tuple[list[int], list[int]]] = {}


def voicing_of(logmel: Sequence[float], a: Analysis) -> float:
    """How voiced a frame sounds, 0 to 1, read off its tilt: the low bands (under 1 kHz) over the high (over 3 kHz).

    A vowel's energy sits low, a fricative's high; the tilt between them
    decides how much pulse and how much noise the vocoder excites the frame
    with.  Fully voiced from a tilt of 4 nats, fully unvoiced from -2.
    """
    bands = _VOICING_BANDS.get(a)
    if bands is None:
        centres = band_centres(a)
        bands = ([b for b, c in enumerate(centres) if c < 1000.0], [b for b, c in enumerate(centres) if c > 3000.0])
        _VOICING_BANDS[a] = bands
    low, high = bands
    if not low or not high:
        return 1.0
    tilt = sum(logmel[b] for b in low) / len(low) - sum(logmel[b] for b in high) / len(high)
    v = (tilt + 2.0) / 6.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class _Excitation:
    """What the vocoder takes its phases from: a pulse train at the voice's pitch, and noise.

    Both run on continuously (the pulse phase and the noise state carry
    across frames), so consecutive frames agree about the phase of every bin
    and the overlap-add is coherent; a frame mixes the two by its voicing.
    Only the phases are used - the magnitudes are the frame's own.
    """

    def __init__(self, a: Analysis, pitch: float) -> None:
        if pitch <= 0.0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        self.a = a
        self.step = pitch / a.rate
        self.phase = 0.0
        self.noise = _Noise()
        self.pulses: list[float] = []
        self.noises: list[float] = []
        self.offset = 0
        self.window = hann(a.frame)

    def _ensure(self, upto: int) -> None:
        while self.offset + len(self.pulses) < upto:
            self.phase += self.step
            if self.phase >= 1.0:
                self.phase -= 1.0
                self.pulses.append(1.0)
            else:
                self.pulses.append(0.0)
            self.noises.append(self.noise.next())

    def phases(self, start: int, voicing: float) -> tuple[list[float], list[float]]:
        """The unit phasors (cos, sin per bin, ``fft/2 + 1`` of them) of the excitation frame at ``start``."""
        a = self.a
        self._ensure(start + a.frame)
        if start - self.offset > 4 * a.frame:  # what no frame will look at again
            drop = start - self.offset - a.frame
            del self.pulses[:drop]
            del self.noises[:drop]
            self.offset += drop
        noise_gain = 0.25 * (1.0 - voicing) + 0.02
        base = start - self.offset
        re = [(voicing * self.pulses[base + i] + noise_gain * self.noises[base + i]) * self.window[i] for i in range(a.frame)]
        re.extend([0.0] * (a.fft - a.frame))
        im = [0.0] * a.fft
        fft(re, im)
        half = a.fft // 2
        cos = [1.0] * (half + 1)
        sin = [0.0] * (half + 1)
        for k in range(half + 1):
            m = math.sqrt(re[k] * re[k] + im[k] * im[k])
            if m > 0.0:
                cos[k] = re[k] / m
                sin[k] = im[k] / m
        return cos, sin


def _spectrum(mags: Sequence[float], cos: Sequence[float], sin: Sequence[float], n: int) -> tuple[list[float], list[float]]:
    """The full (Hermitian) spectrum of a frame from its magnitudes and unit phasors."""
    half = n // 2
    re = [0.0] * n
    im = [0.0] * n
    for k in range(half + 1):
        re[k] = mags[k] * cos[k]
        im[k] = mags[k] * sin[k]
    im[0] = 0.0
    im[half] = 0.0
    for k in range(1, half):
        re[n - k] = re[k]
        im[n - k] = -im[k]
    return re, im


class Vocoder:
    """Units to a waveform, as they come.

    Every unit is its centroid (plus the codebook's mean) held for its typical
    run of frames.  Every frame is spread back over the linear spectrum; its
    phases are taken from a pulse train at ``pitch`` mixed with noise by how
    voiced the frame looks (:func:`voicing_of`), both running on continuously
    so consecutive frames agree; the inverse transform of the frame is
    windowed and overlap-added, and the samples no later frame can touch are
    handed back at once.  :meth:`end` flushes the tail.  A frame at a time is
    what lets a model be heard while it walks.
    """

    def __init__(self, codebook: Codebook, gain: float = 1.0, pitch: float = PITCH) -> None:
        codebook.validate()
        self.codebook = codebook
        self.gain = gain
        self.pitch = pitch
        a = codebook.analysis
        self._a = a
        self._window = hann(a.frame)
        self._excitation = _Excitation(a, pitch)
        self._acc: list[float] = []
        self._norm: list[float] = []
        self._frames = 0
        self._emitted = 0
        self.total_samples = 0

    def feed(self, unit: str) -> bytes:
        """One unit; the PCM that no later unit can change comes back."""
        index = self.codebook.index(unit)
        if index is None:
            raise ValueError(f"not a unit of this codebook: {unit!r}")
        frame = [c + m for c, m in zip(self.codebook.centroids[index], self.codebook.mean)]
        out = bytearray()
        for _ in range(self.codebook.hold(index)):
            out += self.feed_frame(frame)
        return bytes(out)

    def feed_frame(self, logmel: Sequence[float]) -> bytes:
        """One absolute log-mel frame (the codebook's mean already added); the PCM that is ready comes back."""
        a = self._a
        start = self._frames * a.hop
        mags = magnitudes_of(logmel, a)
        cos, sin = self._excitation.phases(start, voicing_of(logmel, a))
        re, im = _spectrum(mags, cos, sin, a.fft)
        fft(re, im, inverse=True)
        end = start + a.frame
        if len(self._acc) < end:
            grow = end - len(self._acc)
            self._acc.extend([0.0] * grow)
            self._norm.extend([0.0] * grow)
        for i in range(a.frame):
            w = self._window[i]
            self._acc[start + i] += re[i] * w
            self._norm[start + i] += w * w
        self._frames += 1
        return self._emit(min(self._frames * a.hop, len(self._acc)))  # every sample before the next frame is complete

    def _emit(self, upto: int) -> bytes:
        if upto <= self._emitted:
            return b""
        chunk = []
        for i in range(self._emitted, upto):
            norm = self._norm[i]
            chunk.append(self._acc[i] / norm if norm > 1e-3 else self._acc[i] / 1e-3)
        self._emitted = upto
        self.total_samples += len(chunk)
        return _pack(chunk, self.gain)

    def end(self) -> bytes:
        """The tail: what the last frames left pending.  The vocoder is ready for the next utterance."""
        out = self._emit(len(self._acc))
        self._excitation = _Excitation(self._a, self.pitch)
        self._acc = []
        self._norm = []
        self._frames = 0
        self._emitted = 0
        return out


def _stft(samples: Sequence[float], a: Analysis, count: int) -> list[tuple[list[float], list[float]]]:
    window = hann(a.frame)
    pad = [0.0] * (a.fft - a.frame)
    spectra = []
    for t in range(count):
        start = t * a.hop
        re = [(samples[start + i] if start + i < len(samples) else 0.0) * window[i] for i in range(a.frame)] + pad
        im = [0.0] * a.fft
        fft(re, im)
        spectra.append((re, im))
    return spectra


def _istft(spectra: Sequence[tuple[list[float], list[float]]], a: Analysis) -> list[float]:
    window = hann(a.frame)
    length = (len(spectra) - 1) * a.hop + a.frame if spectra else 0
    acc = [0.0] * length
    norm = [0.0] * length
    for t, (re, im) in enumerate(spectra):
        r = list(re)
        i = list(im)
        fft(r, i, inverse=True)
        start = t * a.hop
        for j in range(a.frame):
            w = window[j]
            acc[start + j] += r[j] * w
            norm[start + j] += w * w
    return [acc[i] / norm[i] if norm[i] > 1e-3 else acc[i] / 1e-3 for i in range(length)]


def griffin_lim(
    frames: Sequence[Sequence[float]], a: Analysis, iterations: int = 32, pitch: float = PITCH,
) -> list[float]:
    """Samples for a run of absolute log-mel frames, polished.

    The first pass is exactly the :class:`Vocoder`'s (the frames' magnitudes
    with the excitation's phases); each of ``iterations`` then transforms the
    samples back, keeps the phases that came out and restores the magnitudes -
    Griffin and Lim's projection - so the frames agree with one another better
    each time.
    """
    n = a.fft
    half = n // 2
    count = len(frames)
    if count == 0:
        return []
    magnitudes = [magnitudes_of(f, a) for f in frames]
    excitation = _Excitation(a, pitch)
    spectra: list[tuple[list[float], list[float]]] = []
    for t, f in enumerate(frames):
        cos, sin = excitation.phases(t * a.hop, voicing_of(f, a))
        spectra.append(_spectrum(magnitudes[t], cos, sin, n))
    for _ in range(iterations):
        samples = _istft(spectra, a)
        fresh = _stft(samples, a, count)
        for t, (re, im) in enumerate(fresh):
            mags = magnitudes[t]
            for k in range(half + 1):
                m = math.sqrt(re[k] * re[k] + im[k] * im[k])
                if m > 0.0:
                    re[k] = re[k] / m * mags[k]
                    im[k] = im[k] / m * mags[k]
                else:
                    re[k] = mags[k]
                    im[k] = 0.0
            im[0] = 0.0
            im[half] = 0.0
            for k in range(1, half):
                re[n - k] = re[k]
                im[n - k] = -im[k]
        spectra = fresh
    return _istft(spectra, a)


def unit_frames(units: Iterable[str], codebook: Codebook) -> list[list[float]]:
    """The absolute log-mel frames a run of units stands for: each centroid plus the mean, held for its run."""
    frames: list[list[float]] = []
    for unit in units:
        index = codebook.index(unit)
        if index is None:
            raise ValueError(f"not a unit of this codebook: {unit!r}")
        frame = [c + m for c, m in zip(codebook.centroids[index], codebook.mean)]
        for _ in range(codebook.hold(index)):
            frames.append(frame)
    return frames


def synthesize(
    units: Iterable[str], codebook: Codebook, polish: int = 0, gain: float = 1.0, pitch: float = PITCH,
) -> bytes:
    """16-bit PCM at the codebook's rate for a run of units.

    ``polish`` is the number of Griffin-Lim iterations run over the whole
    utterance once it is known; ``0`` is exactly what the streaming
    :class:`Vocoder` makes.
    """
    units = list(units)
    if polish <= 0:
        voc = Vocoder(codebook, gain, pitch)
        out = bytearray()
        for u in units:
            out += voc.feed(u)
        out += voc.end()
        return bytes(out)
    return _pack(griffin_lim(unit_frames(units, codebook), codebook.analysis, polish, pitch), gain)


# -- the facade -----------------------------------------------------------------------------

@dataclass
class Heard:
    """What an utterance was heard as."""

    units: list[str]
    codes: list[int]
    """Every frame's code, before runs collapse."""
    seconds: float
    frames: int

    @property
    def text(self) -> str:
        return " ".join(self.units)


class AcousticTokenizer:
    """Audio to units and back, over one codebook.

    ``codebook`` is a :class:`Codebook`, a path, or ``None`` for the bundled
    one.  ``collapse`` folds a run of one unit into one token (the default);
    off, every frame is a token.
    """

    def __init__(self, codebook: Codebook | str | None = None, collapse: bool = True) -> None:
        if codebook is None:
            self.codebook = default_codebook()
        elif isinstance(codebook, str):
            self.codebook = Codebook.load(codebook)
        else:
            self.codebook = codebook
        self.codebook.validate()
        self.collapse = collapse

    @property
    def analysis(self) -> Analysis:
        return self.codebook.analysis

    @property
    def vocab(self) -> list[str]:
        return self.codebook.names()

    def samples(self, audio: bytes | Sequence[float], rate: int | None = None) -> list[float]:
        """``audio`` as samples at the codebook's rate: a WAV file's bytes, or samples at ``rate``."""
        if isinstance(audio, (bytes, bytearray)):
            values, src = read_wav(bytes(audio))
        else:
            values, src = list(audio), (rate or self.analysis.rate)
        return resample(values, src, self.analysis.rate)

    def frames(self, audio: bytes | Sequence[float], rate: int | None = None) -> list[list[float]]:
        return frames_of(self.samples(audio, rate), self.analysis)

    def listen(self, audio: bytes | Sequence[float], rate: int | None = None) -> Heard:
        """Everything the utterance was heard as: its units, its codes frame by frame, its length."""
        samples = self.samples(audio, rate)
        frames = frames_of(samples, self.analysis)
        codes = self.codebook.codes(frames)
        units = [self.codebook.name(c) for c in (collapse_runs(codes) if self.collapse else codes)]
        return Heard(units=units, codes=codes, seconds=len(samples) / self.analysis.rate, frames=len(frames))

    def hear(self, audio: bytes | Sequence[float], rate: int | None = None) -> list[str]:
        """The units of an utterance."""
        return self.listen(audio, rate).units

    def hear_file(self, path: str) -> list[str]:
        with open(path, "rb") as fh:
            return self.hear(fh.read())

    def is_unit(self, token: str) -> bool:
        return self.codebook.index(token) is not None

    def units_of(self, text: str) -> list[str]:
        """The unit tokens of a text of units, checked against the codebook (runs collapsed when the tokenizer does)."""
        units = []
        for token in text.split():
            if self.codebook.index(token) is None:
                raise ValueError(f"not a unit of this codebook: {token!r}")
            if self.collapse and units and units[-1] == token:
                continue
            units.append(token)
        return units

    def text(self, units_or_text: str | Iterable[str]) -> str:
        """The text form of units: one line, space-separated; idempotent on a text of units."""
        if isinstance(units_or_text, str):
            return " ".join(self.units_of(units_or_text))
        return " ".join(self.units_of(" ".join(units_or_text)))

    def synthesize(self, units: str | Iterable[str], polish: int = 0, gain: float = 1.0, pitch: float = PITCH) -> bytes:
        """16-bit PCM at the codebook's rate: the units spoken back."""
        return synthesize(
            self.units_of(units) if isinstance(units, str) else list(units), self.codebook, polish, gain, pitch,
        )

    def vocoder(self, gain: float = 1.0, pitch: float = PITCH) -> Vocoder:
        return Vocoder(self.codebook, gain, pitch)

    def stream(self, units: Iterable[str], gain: float = 1.0, pitch: float = PITCH) -> Iterator[bytes]:
        """The PCM of the units as they come, a chunk per unit that completed samples, and the tail."""
        voc = Vocoder(self.codebook, gain, pitch)
        for u in units:
            chunk = voc.feed(u)
            if chunk:
                yield chunk
        tail = voc.end()
        if tail:
            yield tail


__all__ = [
    "RATE", "UNIT_PREFIX", "DEFAULT_CODEBOOK", "Analysis", "DEFAULT_ANALYSIS", "Codebook", "Learned", "Heard",
    "AcousticTokenizer", "Vocoder", "read_wav", "load_wav", "pcm_samples", "resample", "frames_of", "band_means",
    "mel_filters", "mel", "mel_to_hz", "hann", "fft", "dist2", "kmeans", "learn", "collapse_runs", "run_lengths",
    "default_codebook", "magnitudes_of", "unit_frames", "griffin_lim", "synthesize", "PITCH", "voicing_of",
    "band_centres", "band_areas",
]
