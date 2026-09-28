"""A neural vocoder for the acoustic units: the units heard back through a network learned from recordings.

The streaming :class:`~phonetok.acoustic.Vocoder` gives every unit its
centroid - the average of thousands of frames - and takes the phases of a
pulse train, and what comes out is speech-like and blurred.  This module keeps
the second half and learns the first: the same excitation - a pulse train at
the voice's pitch, each pulse spread into a short chirp, and the same
deterministic noise - runs on under the utterance, and a small network that
reads the run of unit codes with its
context writes, for every frame and every bin of the transform, how loud the
pulse train and how loud the noise should be there.  A source and a filter,
as the formant synthesizer is; the filter is learned, and it is the units'
own.  Nothing about the units changes: a codebook's units are decoded by
whichever of the two vocoders is there.

* **The model** (:class:`UnitVocoder`): an embedding of the codes, a stack
  of residual blocks of dilated convolutions at the frame rate (so a frame
  hears the units before and after it), and a head that writes two log gains
  per bin - the pulse train's and the noise's.  Each frame's spectrum is the
  excitation's frames times those gains, and the frames are overlap-added
  exactly as the vocoder overlap-adds its own, so the Go and Rust ports run
  the same weights through the same excitation and the same inverse transform
  and hear the same samples.  ``pitch`` is the excitation's, as it is the
  vocoder's: the units carry none.
* **The weights** are one JSON file beside the codebook (``<codebook stem>.vocoder.json``,
  or ``$PHONETOK_VOCODER``): the architecture, the codebook it belongs to,
  and the tensors as base64 little-endian float32.  Inference needs nothing
  but the standard library; ``numpy`` makes it fast when it is there.
* **Training** (:func:`train`) needs ``numpy`` and recordings, nothing else:
  the recordings become codes under the codebook, their pitch is tracked so
  the pulse train can follow it, and the network is fitted to the waveforms
  by gradient descent on a multi-resolution spectral loss (spectral
  convergence and log-magnitude distance at three transform sizes) plus the
  distance between the log-mel frames the analysis hears in the original and
  in the output.  The gradients are written out by hand here: every layer is
  a few lines, and the whole thing runs on a laptop's CPU in minutes for a
  few minutes of audio.

A codebook learned from a person's recordings and a vocoder trained on the
same recordings give that person's voice back; the bundled vocoder is trained
on the synthesizer's speech, like the bundled codebook, and is the bootstrap.
"""

from __future__ import annotations

import base64
import json
import math
import os
import random
import struct
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from operator import mul
from typing import Any

from .acoustic import (
    DEFAULT_CODEBOOK, LOG_FLOOR, PITCH, Analysis, Codebook, _istft, _pack, _stft, frames_of, hann, mel_filters,
)
from .synth import _Noise

VOCODER_SUFFIX = ".vocoder.json"
DEFAULT_VOCODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "acoustic.vocoder.json")
"""The vocoder trained for the bundled codebook."""

SLOPE = 0.1
"""The slope of the leaky rectifier below zero."""
CLIP = (-24.0, 8.0)
"""The range a predicted log gain is held to: the floor is silence to a 16-bit ear, the ceiling forty decibels of
boost on the excitation."""
EPS_INIT = 1e-9
"""What is added to a magnitude before its log when the templates are read off the recordings: low enough that
digital silence gets a template of silence."""
EPS_LOG = 1e-4
"""What is added to a magnitude before its log in the loss: a floor a hundred decibels down."""
F0_MIN, F0_MAX = 50.0, 400.0
"""The pitch range the tracker searches, in Hz."""
RESOLUTIONS = ((256, 64, 256), (512, 128, 512), (1024, 256, 1024))
"""The (transform, hop, window) sizes of the multi-resolution spectral loss."""
PEAK = 0.5
"""The level every training recording is brought to: the synthesizer's own default, so a vocoder at gain 1 plays
about as loud as the voice does."""
HEADROOM = 0.7
"""What :meth:`UnitVocoder.synthesize` scales the waveform by at gain 1: the dispersed excitation still peaks
somewhat higher than a recording of the same loudness, and three decibels keep the peaks off the rails."""


def vocoder_path_for(codebook_path: str) -> str:
    """Where a codebook's vocoder lives: ``codebook.tsv`` -> ``codebook.vocoder.json``."""
    stem, ext = os.path.splitext(codebook_path)
    return (stem if ext.lower() == ".tsv" else codebook_path) + VOCODER_SUFFIX


def codebook_checksum(codebook: Codebook) -> float:
    """A cheap fingerprint of a codebook, kept in its vocoder so the two are not mixed up."""
    total = 0.0
    for c in codebook.centroids:
        for v in c:
            total += v
    for v in codebook.mean:
        total += v
    return round(total + codebook.k, 6)


def _f32(values: Sequence[float]) -> str:
    data = struct.pack("<%df" % len(values), *values)
    return base64.b64encode(data).decode("ascii")


def _from_f32(text: str, count: int) -> list[float]:
    data = base64.b64decode(text)
    if len(data) != 4 * count:
        raise ValueError(f"a tensor of {count} values packed as {len(data)} bytes")
    return list(struct.unpack("<%df" % count, data))


def _shape_size(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _reshape(flat: Sequence[float], shape: Sequence[int]) -> Any:
    """A flat list as nested lists of the shape (row-major)."""
    if len(shape) == 1:
        return list(flat)
    step = _shape_size(shape[1:])
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def _flatten(nested: Any) -> list[float]:
    if isinstance(nested, (int, float)):
        return [float(nested)]
    out: list[float] = []
    for item in nested:
        out.extend(_flatten(item))
    return out


def _numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


# -- the excitation: the source every port makes the same ------------------------------------------------------

CHIRP = 64
"""How many samples each pulse is spread over: a Hann-windowed linear chirp of unit energy sweeping from zero to
the Nyquist frequency.  An impulse's harmonics are all in phase at the instant of the pulse, and a filter that
only scales them rings symmetrically around it; the chirp disperses the energy over four milliseconds, as a
glottal pulse's is, so the waveform peaks where the recordings' do instead of three times higher."""


def chirp_kernel(length: int = CHIRP) -> list[float]:
    """The chirp every pulse becomes, of unit energy."""
    window = hann(length)
    kernel = [math.cos(math.pi * t * t / (2.0 * length)) * window[t] for t in range(length)]
    norm = math.sqrt(sum(v * v for v in kernel))
    return [v / norm for v in kernel]


_KERNEL: list[float] = []


def disperse(pulses: Sequence[float]) -> list[float]:
    """The pulse train with every pulse spread into the chirp (pulses taken in order, so every port sums alike)."""
    if not _KERNEL:
        _KERNEL.extend(chirp_kernel())
    kernel = _KERNEL
    out = [0.0] * len(pulses)
    for p, v in enumerate(pulses):
        if v == 0.0:
            continue
        for t, k in enumerate(kernel):
            if p + t >= len(out):
                break
            out[p + t] += v * k
    return out


def pulse_train(length: int, pitch: float, rate: int) -> list[float]:
    """Unit impulses at ``pitch`` from phase zero: exactly the vocoder's pulse train (see :func:`disperse`)."""
    if pitch <= 0.0:
        raise ValueError(f"pitch must be positive, got {pitch}")
    step = pitch / rate
    phase = 0.0
    out = [0.0] * length
    for n in range(length):
        phase += step
        if phase >= 1.0:
            phase -= 1.0
            out[n] = 1.0
    return out


_NOISE_STATE = _Noise()
_NOISE: list[float] = []


def noise_train(length: int) -> list[float]:
    """The first ``length`` values of the vocoder's noise (xorshift32 from its fixed seed), in [-1, 1)."""
    while len(_NOISE) < length:
        _NOISE.append(_NOISE_STATE.next())
    return _NOISE[:length]


def excitation_length(count: int, a: Analysis) -> int:
    return (count - 1) * a.hop + a.frame if count > 0 else 0


HalfSpectra = list[tuple[list[float], list[float]]]


def excitation_spectra(count: int, a: Analysis, pitch: float = PITCH) -> tuple[HalfSpectra, HalfSpectra]:
    """The half spectra (``fft/2 + 1`` bins) of the dispersed pulse train's frames and of the noise's, ``count`` of
    each."""
    length = excitation_length(count, a)
    half = a.fft // 2 + 1
    pulses = _stft(disperse(pulse_train(length, pitch, a.rate)), a, count)
    noises = _stft(noise_train(length), a, count)
    return [(re[:half], im[:half]) for re, im in pulses], [(re[:half], im[:half]) for re, im in noises]


# -- the model -------------------------------------------------------------------------------------------------

@dataclass
class Spec:
    """The architecture: what the file says, what the ports build."""

    units: int
    bins: int
    channels: int = 64
    kernel: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)

    def validate(self) -> None:
        if self.units < 1 or self.bins < 2 or self.channels < 1:
            raise ValueError(f"impossible vocoder: {self}")
        if self.kernel < 1 or self.kernel % 2 == 0:
            raise ValueError(f"the kernel must be odd, got {self.kernel}")
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError(f"the dilations must be at least 1, got {self.dilations}")

    @property
    def context(self) -> int:
        """Frames on either side that can reach a frame's output: the receptive field's radius."""
        return sum((self.kernel - 1) // 2 * d for d in self.dilations)

    @property
    def parameters(self) -> int:
        c, k = self.channels, self.kernel
        return (self.units * c + len(self.dilations) * (c * c * k + c + c * c + c) + 2 * self.bins * c + 2 * self.bins
                + self.units * 2 * self.bins)


@dataclass
class Weights:
    """The tensors, as nested lists (row-major): the portable form."""

    embed: list[list[float]]
    """``[units][channels]``."""
    blocks: list[dict[str, Any]]
    """Per block ``w1 [channels][channels][kernel]`` (out, in, tap), ``b1 [channels]``, ``w2 [channels][channels]``,
    ``b2 [channels]``."""
    head: dict[str, Any]
    """``w [2 * bins][channels]``, ``b [2 * bins]``: the pulse train's log gains, then the noise's."""
    template: list[list[float]]
    """``[units][2 * bins]``: every unit's own log gains, which the network's output is added to - the unit's
    typical spectrum, read off the recordings before training starts and refined by it."""


class UnitVocoder:
    """Codes to samples through the learned filter over the vocoder's excitation.

    ``synthesize`` takes the codes of an utterance - a code per frame, as
    :func:`codes_of` expands a run of units - and returns 16-bit PCM at the
    analysis' rate; ``samples`` the floats behind it.  The whole utterance is
    read before anything is written, since every frame hears its neighbours.
    """

    def __init__(self, analysis: Analysis, spec: Spec, weights: Weights, note: str = "", checksum: float = 0.0,
                 trained: dict[str, Any] | None = None) -> None:
        spec.validate()
        analysis.validate()
        if spec.bins != analysis.fft // 2 + 1:
            raise ValueError(f"a vocoder of {spec.bins} bins for a transform of {analysis.fft} points")
        self.analysis = analysis
        self.spec = spec
        self.weights = weights
        self.note = note
        self.checksum = checksum
        self.trained = dict(trained or {})
        self._np_cache: dict[str, Any] | None = None

    # -- making one ------------------------------------------------------------------------------------------

    @classmethod
    def new(cls, codebook: Codebook, channels: int = 64, kernel: int = 5, dilations: Sequence[int] = (1, 2, 4, 8, 1, 2),
            seed: int = 1, template: Sequence[Sequence[float]] | None = None) -> UnitVocoder:
        """A fresh, untrained vocoder for a codebook: small random weights, the blocks nearly the identity, and
        every unit's template the log gains ``template`` gives it (``[units][2 * bins]``, as :func:`templates`
        reads them off recordings), or a flat guess."""
        a = codebook.analysis
        spec = Spec(units=codebook.k, bins=a.fft // 2 + 1, channels=channels, kernel=kernel, dilations=tuple(dilations))
        spec.validate()
        rng = random.Random(seed)
        c, k = spec.channels, spec.kernel

        def normal(count: int, scale: float) -> list[float]:
            return [rng.gauss(0.0, scale) for _ in range(count)]

        embed = _reshape(normal(spec.units * c, 0.5), (spec.units, c))
        blocks = []
        for _ in spec.dilations:
            blocks.append({
                "w1": _reshape(normal(c * c * k, math.sqrt(2.0 / (c * k))), (c, c, k)),
                "b1": [0.0] * c,
                "w2": _reshape(normal(c * c, 0.05 / math.sqrt(c)), (c, c)),
                "b2": [0.0] * c,
            })
        bins = spec.bins
        head_w = _reshape(normal(2 * bins * c, 0.1 / math.sqrt(c)), (2 * bins, c))
        head_b = [0.0] * (2 * bins)
        lo, hi = CLIP
        if template is None:
            rows = [[0.0] * bins + [-2.0] * bins for _ in range(spec.units)]
        else:
            if len(template) != spec.units or any(len(row) != 2 * bins for row in template):
                raise ValueError(f"a template of {len(template)} units for a codebook of {spec.units}")
            rows = [[min(hi, max(lo, float(v))) for v in row] for row in template]
        return cls(a, spec, Weights(embed, blocks, {"w": head_w, "b": head_b}, rows), checksum=codebook_checksum(codebook))

    # -- the file --------------------------------------------------------------------------------------------

    def dumps(self) -> str:
        a = self.analysis
        s = self.spec

        def tensor(nested: Any, shape: Sequence[int]) -> dict[str, Any]:
            flat = _flatten(nested)
            if len(flat) != _shape_size(shape):
                raise ValueError(f"a tensor of {len(flat)} values for a shape of {list(shape)}")
            return {"shape": list(shape), "f32": _f32(flat)}

        c, k = s.channels, s.kernel
        doc = {
            "phonetok": "acoustic vocoder",
            "version": 1,
            "model": "source-filter",
            "note": self.note,
            "units": s.units,
            "checksum": self.checksum,
            "analysis": {"rate": a.rate, "frame": a.frame, "hop": a.hop, "fft": a.fft},
            "channels": c,
            "kernel": k,
            "dilations": list(s.dilations),
            "slope": SLOPE,
            "clip": list(CLIP),
            "trained": self.trained,
            "weights": {
                "embed": tensor(self.weights.embed, (s.units, c)),
                "blocks": [
                    {
                        "w1": tensor(b["w1"], (c, c, k)),
                        "b1": tensor(b["b1"], (c,)),
                        "w2": tensor(b["w2"], (c, c)),
                        "b2": tensor(b["b2"], (c,)),
                    }
                    for b in self.weights.blocks
                ],
                "head": {"w": tensor(self.weights.head["w"], (2 * s.bins, c)),
                         "b": tensor(self.weights.head["b"], (2 * s.bins,))},
                "template": tensor(self.weights.template, (s.units, 2 * s.bins)),
            },
        }
        return json.dumps(doc, separators=(",", ":")) + "\n"

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())

    @classmethod
    def loads(cls, text: str) -> UnitVocoder:
        doc = json.loads(text)
        if not isinstance(doc, dict) or doc.get("phonetok") != "acoustic vocoder":
            raise ValueError("not a phonetok acoustic vocoder file")
        if doc.get("model", "source-filter") != "source-filter" or int(doc.get("version", 1)) != 1:
            raise ValueError(f"a vocoder of a kind this version does not know: {doc.get('model')} v{doc.get('version')}")
        an = doc["analysis"]
        analysis = Analysis(rate=int(an["rate"]), frame=int(an["frame"]), hop=int(an["hop"]), fft=int(an["fft"]))
        spec = Spec(units=int(doc["units"]), bins=int(an["fft"]) // 2 + 1, channels=int(doc["channels"]),
                    kernel=int(doc["kernel"]), dilations=tuple(int(d) for d in doc["dilations"]))
        spec.validate()

        def tensor(t: dict[str, Any], shape: Sequence[int]) -> Any:
            if [int(v) for v in t["shape"]] != list(shape):
                raise ValueError(f"a tensor of shape {t['shape']} where {list(shape)} was expected")
            return _reshape(_from_f32(t["f32"], _shape_size(shape)), shape)

        c, k = spec.channels, spec.kernel
        w = doc["weights"]
        if len(w["blocks"]) != len(spec.dilations):
            raise ValueError(f"{len(w['blocks'])} blocks of weights for {len(spec.dilations)} dilations")
        weights = Weights(
            embed=tensor(w["embed"], (spec.units, c)),
            blocks=[
                {
                    "w1": tensor(b["w1"], (c, c, k)),
                    "b1": tensor(b["b1"], (c,)),
                    "w2": tensor(b["w2"], (c, c)),
                    "b2": tensor(b["b2"], (c,)),
                }
                for b in w["blocks"]
            ],
            head={"w": tensor(w["head"]["w"], (2 * spec.bins, c)), "b": tensor(w["head"]["b"], (2 * spec.bins,))},
            template=tensor(w["template"], (spec.units, 2 * spec.bins)),
        )
        return cls(analysis, spec, weights, note=str(doc.get("note", "")), checksum=float(doc.get("checksum", 0.0)),
                   trained=doc.get("trained") or {})

    @classmethod
    def load(cls, path: str = DEFAULT_VOCODER) -> UnitVocoder:
        with open(path, encoding="utf-8") as fh:
            return cls.loads(fh.read())

    def matches(self, codebook: Codebook) -> bool:
        """Whether this vocoder was trained for ``codebook``: the same units, analysis and fingerprint."""
        a, b = self.analysis, codebook.analysis
        return (self.spec.units == codebook.k and (a.rate, a.frame, a.hop, a.fft) == (b.rate, b.frame, b.hop, b.fft)
                and abs(self.checksum - codebook_checksum(codebook)) < 1e-3)

    def describe(self) -> dict[str, Any]:
        s = self.spec
        a = self.analysis
        return {
            "units": s.units, "channels": s.channels, "kernel": s.kernel, "dilations": list(s.dilations), "bins": s.bins,
            "parameters": s.parameters, "context_frames": s.context, "context_seconds": s.context * a.hop / a.rate,
            "analysis": {"rate": a.rate, "frame": a.frame, "hop": a.hop, "fft": a.fft},
            "checksum": self.checksum, "note": self.note, "trained": self.trained,
        }

    # -- inference -------------------------------------------------------------------------------------------

    def filters(self, codes: Sequence[int]) -> tuple[list[list[float]], list[list[float]]]:
        """The log gains of every frame: the pulse train's per bin, and the noise's (each ``[frames][bins]``)."""
        if not codes:
            return [], []
        for c in codes:
            if not 0 <= c < self.spec.units:
                raise ValueError(f"code {c} is not one of this vocoder's {self.spec.units} units")
        np = _numpy()
        if np is not None:
            out = forward_numpy(np, self._numpy_weights(np), self.spec, np.asarray(codes, dtype=np.int64))
            return out["lp"].tolist(), out["ln"].tolist()
        return self._filters_python(codes)

    def spectra(self, codes: Sequence[int], pitch: float = PITCH) -> list[tuple[list[float], list[float]]]:
        """The full (Hermitian) spectrum of every frame, as the acoustic module's inverse transform takes it."""
        lp, ln = self.filters(codes)
        pulses, noises = excitation_spectra(len(codes), self.analysis, pitch)
        n = self.analysis.fft
        half = n // 2
        out = []
        for t in range(len(codes)):
            gp = [math.exp(v) for v in lp[t]]
            gn = [math.exp(v) for v in ln[t]]
            (pre, pim), (nre, nim) = pulses[t], noises[t]
            re = [0.0] * n
            im = [0.0] * n
            for k in range(half + 1):
                re[k] = pre[k] * gp[k] + nre[k] * gn[k]
                im[k] = pim[k] * gp[k] + nim[k] * gn[k]
            im[0] = 0.0
            im[half] = 0.0
            for k in range(1, half):
                re[n - k] = re[k]
                im[n - k] = -im[k]
            out.append((re, im))
        return out

    def samples(self, codes: Sequence[int], pitch: float = PITCH) -> list[float]:
        """The waveform of a run of codes, in [-1, 1]."""
        np = _numpy()
        if np is None or not codes:
            return _istft(self.spectra(codes, pitch), self.analysis)
        lp, ln = self.filters(codes)
        ep, en = excitation_numpy(np, self.analysis, len(codes), pitch=pitch)
        out, _ = istft_numpy(np, combine_numpy(np, ep, en, np.asarray(lp), np.asarray(ln)), self.analysis)
        return out.tolist()

    def synthesize(self, codes: Sequence[int], gain: float = 1.0, pitch: float = PITCH) -> bytes:
        """16-bit PCM at the analysis' rate: the waveform at ``gain`` times :data:`HEADROOM`."""
        return _pack(self.samples(codes, pitch), gain * HEADROOM)

    def _numpy_weights(self, np):
        if self._np_cache is None:
            w = self.weights
            self._np_cache = {
                "embed": np.asarray(w.embed, dtype=np.float64),
                "blocks": [
                    {
                        "w1": np.asarray(b["w1"], dtype=np.float64),
                        "b1": np.asarray(b["b1"], dtype=np.float64),
                        "w2": np.asarray(b["w2"], dtype=np.float64),
                        "b2": np.asarray(b["b2"], dtype=np.float64),
                    }
                    for b in w.blocks
                ],
                "head_w": np.asarray(w.head["w"], dtype=np.float64),
                "head_b": np.asarray(w.head["b"], dtype=np.float64),
                "template": np.asarray(w.template, dtype=np.float64),
            }
        return self._np_cache

    def _filters_python(self, codes: Sequence[int]) -> tuple[list[list[float]], list[list[float]]]:
        s = self.spec
        w = self.weights
        c, k = s.channels, s.kernel
        centre = (k - 1) // 2
        count = len(codes)
        x = [list(w.embed[code]) for code in codes]
        for block, d in zip(w.blocks, s.dilations):
            w1, b1, w2, b2 = block["w1"], block["b1"], block["w2"], block["b2"]
            taps = [[w1[o][i] for i in range(c)] for o in range(c)]  # taps[o][i][j]
            per_tap = [[[taps[o][i][j] for i in range(c)] for o in range(c)] for j in range(k)]  # [j][o][i]
            pre = [list(b1) for _ in range(count)]
            for j in range(k):
                shift = (j - centre) * d
                tap = per_tap[j]
                for t in range(max(0, -shift), min(count, count - shift)):
                    xs = x[t + shift]
                    acc = pre[t]
                    for o in range(c):
                        acc[o] += sum(map(mul, tap[o], xs))
            for t in range(count):
                h = [v if v > 0.0 else SLOPE * v for v in pre[t]]
                xt = x[t]
                for o in range(c):
                    xt[o] += b2[o] + sum(map(mul, w2[o], h))
        hw, hb = w.head["w"], w.head["b"]
        bins = s.bins
        lo, hi = CLIP
        lp = []
        ln = []
        for t, xt in enumerate(x):
            row = w.template[codes[t]]
            z = [min(hi, max(lo, row[o] + hb[o] + sum(map(mul, hw[o], xt)))) for o in range(2 * bins)]
            lp.append(z[:bins])
            ln.append(z[bins:])
        return lp, ln


def codes_of(units: Iterable[str], codebook: Codebook) -> list[int]:
    """The code of every frame a run of units stands for: each unit held for its typical run."""
    codes: list[int] = []
    for unit in units:
        index = codebook.index(unit)
        if index is None:
            raise ValueError(f"not a unit of this codebook: {unit!r}")
        codes.extend([index] * codebook.hold(index))
    return codes


def find_vocoder(codebook: Codebook, source: str | None = None, path: str | None = None) -> UnitVocoder | None:
    """The vocoder that belongs to a codebook, when there is one.

    ``path`` names the file outright; else ``$PHONETOK_VOCODER``; else the
    file beside the codebook's own (``source``, the codebook's path), or the
    bundled vocoder for the bundled codebook (and for a codebook with no
    ``source``, when it matches).  A file that claims to belong to the
    codebook but does not match it is a :class:`ValueError`; none at all is
    ``None``.
    """
    named = path or os.environ.get("PHONETOK_VOCODER") or ""
    if named:
        if not os.path.isfile(named):
            raise ValueError(f"vocoder file not found: {named}")
        candidates = [(named, True)]
    else:
        candidates = []
        if source:
            candidates.append((vocoder_path_for(source), True))
        if not source or os.path.abspath(source) == os.path.abspath(DEFAULT_CODEBOOK):
            # a codebook that came from nowhere may be the bundled one: the bundled vocoder is tried, and
            # merely not the codebook's when it does not match; a file that claims to belong must match
            candidates.append((DEFAULT_VOCODER, bool(source)))
    for candidate, claimed in candidates:
        if not os.path.isfile(candidate):
            continue
        vocoder = UnitVocoder.load(candidate)
        if vocoder.matches(codebook):
            return vocoder
        if claimed:
            raise ValueError(f"{candidate} was not trained for this codebook")
    return None


# -- the network in numpy (inference, and the training's own) --------------------------------------------------

def _conv_np(np, x, w1, b1, d: int):
    """``x [T][C_in]`` through a dilated convolution ``w1 [C_out][C_in][k]`` (zero padding): ``[T][C_out]``."""
    count, _ = x.shape
    k = w1.shape[2]
    centre = (k - 1) // 2
    out = np.tile(b1, (count, 1))
    for j in range(k):
        shift = (j - centre) * d
        lo = max(0, -shift)
        hi = min(count, count - shift)
        if hi <= lo:
            continue
        out[lo:hi] += x[lo + shift:hi + shift] @ np.ascontiguousarray(w1[:, :, j].T)
    return out


def forward_numpy(np, w: dict[str, Any], spec: Spec, codes) -> dict[str, Any]:
    """The network on a run of codes: the log gains ``lp`` and ``ln`` (``[T][bins]``, clipped), and every
    intermediate :func:`backward_numpy` needs."""
    x = w["embed"][codes]
    blocks = []
    for block, d in zip(w["blocks"], spec.dilations):
        pre = _conv_np(np, x, block["w1"], block["b1"], d)
        h = np.where(pre > 0.0, pre, SLOPE * pre)
        y = h @ block["w2"].T + block["b2"]
        blocks.append({"x": x, "pre": pre, "h": h})
        x = x + y
    z = x @ w["head_w"].T + w["head_b"] + w["template"][codes]
    bins = spec.bins
    clipped = np.clip(z, CLIP[0], CLIP[1])
    return {"codes": codes, "blocks": blocks, "x": x, "z": z, "lp": clipped[:, :bins], "ln": clipped[:, bins:]}


def combine_numpy(np, pulses, noises, lp, ln):
    """The half spectra ``[T][bins]`` (complex) of the excitation's frames under the log gains."""
    return pulses * np.exp(lp) + noises * np.exp(ln)


def backward_numpy(np, w: dict[str, Any], spec: Spec, fwd: dict[str, Any], g_lp, g_ln) -> dict[str, Any]:
    """The gradients of every weight from the gradients of the (clipped) log gains."""
    z = fwd["z"]
    g_z = np.concatenate([g_lp, g_ln], axis=1)
    g_z = np.where((z > CLIP[0]) & (z < CLIP[1]), g_z, 0.0)
    grads: dict[str, Any] = {"head_w": g_z.T @ fwd["x"], "head_b": g_z.sum(axis=0), "blocks": []}
    g_template = np.zeros_like(w["template"])
    np.add.at(g_template, fwd["codes"], g_z)
    grads["template"] = g_template
    g_x = g_z @ w["head_w"]
    for block, d, kept in zip(reversed(w["blocks"]), reversed(spec.dilations), reversed(fwd["blocks"])):
        x_in, pre, h = kept["x"], kept["pre"], kept["h"]
        g_y = g_x  # the residual: the same gradient goes on to the block's input and its branch
        g_w2 = g_y.T @ h
        g_b2 = g_y.sum(axis=0)
        g_h = g_y @ block["w2"]
        g_pre = np.where(pre > 0.0, g_h, SLOPE * g_h)
        w1 = block["w1"]
        k = w1.shape[2]
        centre = (k - 1) // 2
        count = x_in.shape[0]
        g_w1 = np.zeros_like(w1)
        g_in = g_x.copy()
        for j in range(k):
            shift = (j - centre) * d
            lo = max(0, -shift)
            hi = min(count, count - shift)
            if hi <= lo:
                continue
            g_w1[:, :, j] = g_pre[lo:hi].T @ x_in[lo + shift:hi + shift]
            g_in[lo + shift:hi + shift] += g_pre[lo:hi] @ np.ascontiguousarray(w1[:, :, j])
        grads["blocks"].insert(0, {"w1": g_w1, "b1": g_pre.sum(axis=0), "w2": g_w2, "b2": g_b2})
        g_x = g_in
    g_embed = np.zeros_like(w["embed"])
    np.add.at(g_embed, fwd["codes"], g_x)
    grads["embed"] = g_embed
    return grads


# -- the transforms, the excitation and the loss, in numpy -----------------------------------------------------

def stft_numpy(np, y, n: int, hop: int, win: int):
    """Frames of ``y`` (Hann of ``win``, hopped by ``hop``, transformed at ``n``): the complex spectra
    ``[frames][n/2+1]``, the frames' sample indices and the window."""
    if len(y) < win:
        y = np.concatenate([y, np.zeros(win - len(y))])
    count = (len(y) - win) // hop + 1
    window = np.asarray(hann(win))
    idx = np.arange(win)[None, :] + hop * np.arange(count)[:, None]
    frames = y[idx] * window
    return np.fft.rfft(frames, n=n), idx, window


def stft_numpy_backward(np, g_spec, idx, window, n: int, length: int):
    """The gradient of ``y`` from the gradient of its spectra (``g_spec`` complex: d/d re + i d/d im)."""
    scale = np.full(n // 2 + 1, 0.5)
    scale[0] = 1.0
    scale[-1] = 1.0
    g_frames = np.fft.irfft(g_spec * scale, n=n) * n
    win = idx.shape[1]
    g_y = np.zeros(max(length, int(idx.max()) + 1))
    np.add.at(g_y, idx, g_frames[:, :win] * window)
    return g_y[:length]


def istft_numpy(np, spectra, a: Analysis):
    """The acoustic module's inverse transform of half spectra ``[T][bins]`` (complex): samples, and the normaliser."""
    count = spectra.shape[0]
    window = np.asarray(hann(a.frame))
    frames = np.fft.irfft(spectra, n=a.fft)[:, :a.frame] * window
    length = excitation_length(count, a)
    acc = np.zeros(length)
    norm = np.zeros(length)
    for t in range(count):
        acc[t * a.hop:t * a.hop + a.frame] += frames[t]
        norm[t * a.hop:t * a.hop + a.frame] += window * window
    norm = np.maximum(norm, 1e-3)
    return acc / norm, norm


def istft_numpy_backward(np, g_samples, norm, count: int, a: Analysis):
    """The gradient of the half spectra (complex) from the gradient of the samples: the transform's adjoint."""
    window = np.asarray(hann(a.frame))
    g_acc = g_samples / norm
    g_frames = np.zeros((count, a.fft))
    for t in range(count):
        g_frames[t, :a.frame] = g_acc[t * a.hop:t * a.hop + a.frame] * window
    spec = np.fft.rfft(g_frames, n=a.fft)
    scale = np.full(a.fft // 2 + 1, 2.0 / a.fft)
    scale[0] = 1.0 / a.fft
    scale[-1] = 1.0 / a.fft
    return spec * scale


def pulse_train_numpy(np, f0, count: int, a: Analysis):
    """The pulse train that follows a pitch track (``f0`` per frame, Hz) over ``count`` frames' samples, dispersed:
    every sample takes the pitch of the frame centred nearest it, and the phase runs exactly as
    :func:`pulse_train`'s does, so a flat track gives the very pulses the ports make."""
    length = excitation_length(count, a)
    frame = np.clip(np.rint((np.arange(length) - a.frame / 2.0) / a.hop).astype(np.int64), 0, max(0, count - 1))
    steps = (np.asarray(f0, dtype=np.float64)[frame] / a.rate).tolist()
    out = [0.0] * length
    phase = 0.0
    for n, step in enumerate(steps):
        phase += step
        if phase >= 1.0:
            phase -= 1.0
            out[n] = 1.0
    return np.asarray(disperse(out))


def excitation_numpy(np, a: Analysis, count: int, pitch: float | None = PITCH, f0=None, pulses=None):
    """The excitation's half spectra (complex ``[T][bins]``): the dispersed pulse train at ``pitch`` (or following
    the track ``f0``, or the dispersed ``pulses`` given) and the noise."""
    length = excitation_length(count, a)
    if pulses is None:
        pulses = (pulse_train_numpy(np, f0, count, a) if f0 is not None
                  else np.asarray(disperse(pulse_train(length, pitch, a.rate))))
    noise = np.asarray(noise_train(length))
    ep, _, _ = stft_numpy(np, pulses, a.fft, a.hop, a.frame)
    en, _, _ = stft_numpy(np, noise, a.fft, a.hop, a.frame)
    return ep, en


def track_pitch(np, samples, a: Analysis, count: int) -> tuple[list[float], list[bool]]:
    """The pitch of every analysis frame, in Hz, by normalised autocorrelation over a 40 ms window around the
    frame's centre: voiced frames get their own (the first clear peak, refined between samples, then a median
    over five frames), the rest the interpolation of their voiced neighbours, so the pulse train has somewhere
    to be at every moment.  Returns the track and which frames were voiced."""
    y = np.asarray(samples, dtype=np.float64)
    width = 4 * a.hop
    half_width = width // 2
    lag_lo = max(2, int(round(a.rate / F0_MAX)))
    lag_hi = min(width - 2, int(round(a.rate / F0_MIN)))
    padded = np.concatenate([np.zeros(half_width), y, np.zeros(half_width + width)])
    centres = np.arange(count) * a.hop + a.frame // 2 + half_width
    idx = centres[:, None] + np.arange(-half_width, half_width)[None, :]
    seg = padded[np.clip(idx, 0, len(padded) - 1)]
    spec = np.fft.rfft(seg, n=2 * width)
    ac = np.fft.irfft(spec * np.conj(spec), n=2 * width)[:, :width]
    e = seg * seg
    cum = np.concatenate([np.zeros((count, 1)), np.cumsum(e, axis=1)], axis=1)
    lags = np.arange(width)
    e0 = cum[:, width - lags]  # energy of the first width - lag samples
    e1 = cum[:, width][:, None] - cum[:, lags]  # energy of the last width - lag samples
    r = ac / np.sqrt(e0 * e1 + 1e-12)
    energy = cum[:, width]
    loud = energy >= 1e-4 * max(float(energy.max()), 1e-12)
    f0 = np.full(count, float("nan"))
    voiced = np.zeros(count, dtype=bool)
    previous = 0.0  # the last voiced frame's lag: a candidate near it is preferred (no octave jumps for nothing)
    for t in range(count):
        if not loud[t]:
            continue
        row = r[t]
        best = float(row[lag_lo:lag_hi + 1].max())
        if best < 0.6:
            continue
        floor = 0.85 * best
        chosen = 0.0
        score = -math.inf
        for lag in range(lag_lo, lag_hi + 1):
            v = row[lag]
            if v < floor or v < row[lag - 1] or v < row[lag + 1]:
                continue
            denom = row[lag - 1] - 2.0 * v + row[lag + 1]
            shift = 0.5 * (row[lag - 1] - row[lag + 1]) / denom if denom < 0.0 else 0.0
            fine = lag + max(-0.5, min(0.5, shift))
            s = v - (0.15 * abs(math.log2(fine / previous)) if previous else 0.01 * math.log2(fine / lag_lo))
            if s > score:
                score, chosen = s, fine
        if chosen <= 0.0:  # the best lag sits at the edge of the range: no clear period
            continue
        f0[t] = a.rate / chosen
        voiced[t] = True
        previous = chosen
    if voiced.any():
        smooth = f0.copy()
        for t in np.flatnonzero(voiced):
            lo, hi = max(0, t - 2), min(count, t + 3)
            window = f0[lo:hi]
            smooth[t] = float(np.median(window[~np.isnan(window)]))
        where = np.flatnonzero(voiced)
        track = np.interp(np.arange(count), where, smooth[where])
    else:
        track = np.full(count, PITCH)
    return [float(v) for v in track], [bool(v) for v in voiced]


def _mel_matrix(np, a: Analysis):
    """``[bands][bins]``: the analysis' triangles as a matrix."""
    m = np.zeros((a.bands, a.fft // 2 + 1))
    for b, (first, weights) in enumerate(mel_filters(a)):
        for i, wgt in enumerate(weights):
            m[b, first + i] = wgt
    return m


class SpectralLoss:
    """The distance between two waveforms as the ear hears it: spectral convergence and log-magnitude distance at
    three resolutions, plus the distance between their log-mel frames under the analysis (after its pre-emphasis)."""

    def __init__(self, np, a: Analysis, mel_weight: float = 1.0) -> None:
        self.np = np
        self.a = a
        self.mel_weight = mel_weight
        self.mel = _mel_matrix(np, a) if mel_weight > 0.0 else None

    def __call__(self, target, output, region: slice | None = None) -> tuple[float, Any, dict[str, float]]:
        """The loss, its gradient with respect to ``output``, and the terms; ``region`` restricts the comparison."""
        np = self.np
        eps = EPS_LOG
        if region is not None:
            o_full = output
            target, output = target[region], output[region]
        total = 0.0
        g_out = np.zeros_like(output)
        terms: dict[str, float] = {}
        for n, hop, win in RESOLUTIONS:
            spec_t, _, _ = stft_numpy(np, target, n, hop, win)
            spec_o, idx, window = stft_numpy(np, output, n, hop, win)
            mag_t = np.abs(spec_t)
            mag_o = np.abs(spec_o)
            diff = mag_o - mag_t
            frob_diff = float(np.sqrt((diff * diff).sum()) + 1e-9)
            frob_t = float(np.sqrt((mag_t * mag_t).sum()) + 1e-9)
            sc = frob_diff / frob_t
            log_diff = np.log(mag_o + eps) - np.log(mag_t + eps)
            lm = float(np.abs(log_diff).mean())
            terms[f"sc{n}"] = sc
            terms[f"lm{n}"] = lm
            total += sc + lm
            g_mag = diff / (frob_diff * frob_t) + np.sign(log_diff) / (mag_o + eps) / log_diff.size
            g_spec = g_mag * spec_o / np.maximum(mag_o, 1e-12)
            g_out += stft_numpy_backward(np, g_spec, idx, window, n, len(output))
        if self.mel is not None:
            a = self.a
            p = a.preemphasis
            pre_t = target.copy()
            pre_o = output.copy()
            if p:
                pre_t[1:] -= p * target[:-1]
                pre_o[1:] -= p * output[:-1]
            spec_t, _, _ = stft_numpy(np, pre_t, a.fft, a.hop, a.frame)
            spec_o, idx, window = stft_numpy(np, pre_o, a.fft, a.hop, a.frame)
            pow_t = np.abs(spec_t) ** 2
            pow_o = np.abs(spec_o) ** 2
            e_t = pow_t @ self.mel.T + LOG_FLOOR
            e_o = pow_o @ self.mel.T + LOG_FLOOR
            log_diff = np.log(e_o) - np.log(e_t)
            mel_l1 = float(np.abs(log_diff).mean())
            terms["mel"] = mel_l1
            total += self.mel_weight * mel_l1
            g_e = self.mel_weight * np.sign(log_diff) / e_o / log_diff.size
            g_pow = g_e @ self.mel
            g_spec = 2.0 * g_pow * spec_o  # d|S|^2/dS = 2 S (as re, im)
            g_pre = stft_numpy_backward(np, g_spec, idx, window, a.fft, len(output))
            g_o = g_pre.copy()
            if p:
                g_o[:-1] -= p * g_pre[1:]
            g_out += g_o
        if region is not None:
            g_full = np.zeros_like(o_full)
            g_full[region] = g_out
            g_out = g_full
        return total, g_out, terms


# -- training ----------------------------------------------------------------------------------------------------

@dataclass
class Clip:
    """One recording as the trainer holds it: its codes under the codebook, its samples at the analysis' rate
    (peak-normalised), the pitch track and the pulse train that follows it."""

    codes: list[int]
    samples: Any
    f0: list[float]
    voiced: list[bool]
    pulses: Any

    @property
    def frames(self) -> int:
        return len(self.codes)

    @property
    def pitch(self) -> float:
        """The median pitch of the voiced frames (the default pitch when none were)."""
        voiced = [f for f, v in zip(self.f0, self.voiced) if v]
        return float(sorted(voiced)[len(voiced) // 2]) if voiced else PITCH


def prepare(np, codebook: Codebook, recordings: Iterable[Sequence[float]]) -> list[Clip]:
    """Recordings (samples at the analysis' rate) as clips: codes, samples peak-normalised, pitch, pulses."""
    a = codebook.analysis
    clips = []
    for samples in recordings:
        frames = frames_of(samples, a)
        codes = codebook.codes(frames)
        if not codes:
            continue
        y = np.asarray(samples, dtype=np.float64)
        length = excitation_length(len(codes), a)
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])
        y = y[:length]
        peak = float(np.abs(y).max())
        if peak > 0.0:
            y = y * (PEAK / peak)
        f0, voiced = track_pitch(np, y, a, len(codes))
        clips.append(Clip(codes, y, f0, voiced, pulse_train_numpy(np, f0, len(codes), a)))
    return clips


def templates(np, clips: Sequence[Clip], units: int, a: Analysis) -> list[list[float]]:
    """Every unit's starting template, read off the clips: per bin, the mean log magnitude of the unit's frames
    over the pulse train's there, and over the noise's a little lower, so the first output has every unit about
    as loud as it was recorded (digital silence included) and mostly voiced.  A unit no clip holds gets the
    mean of them all."""
    bins = a.fft // 2 + 1
    target = np.zeros((units, bins))
    pulse = np.zeros((units, bins))
    noise = np.zeros((units, bins))
    count = np.zeros(units)
    for clip in clips:
        spec, _, _ = stft_numpy(np, clip.samples, a.fft, a.hop, a.frame)
        ep, en = excitation_numpy(np, a, clip.frames, pulses=clip.pulses)
        codes = np.asarray(clip.codes)
        np.add.at(target, codes, np.log(np.abs(spec) + EPS_INIT))
        np.add.at(pulse, codes, np.log(np.abs(ep) + EPS_INIT))
        np.add.at(noise, codes, np.log(np.abs(en) + EPS_INIT))
        np.add.at(count, codes, 1.0)
    seen = count > 0
    if not seen.any():
        return [[0.0] * bins + [-2.0] * bins for _ in range(units)]
    lp = np.zeros((units, bins))
    ln = np.zeros((units, bins))
    lp[seen] = (target[seen] - pulse[seen]) / count[seen, None]
    ln[seen] = (target[seen] - noise[seen]) / count[seen, None] - 1.0
    lp[~seen] = lp[seen].mean(axis=0)
    ln[~seen] = ln[seen].mean(axis=0)
    lo, hi = CLIP
    both = np.clip(np.concatenate([lp, ln], axis=1), lo, hi)
    return [[float(v) for v in row] for row in both]


def _flatten_grads(grads: dict[str, Any]) -> list[Any]:
    out = [grads["embed"]]
    for b in grads["blocks"]:
        out += [b["w1"], b["b1"], b["w2"], b["b2"]]
    out += [grads["head_w"], grads["head_b"], grads["template"]]
    return out


def _flatten_weights(w: dict[str, Any]) -> list[Any]:
    out = [w["embed"]]
    for b in w["blocks"]:
        out += [b["w1"], b["b1"], b["w2"], b["b2"]]
    out += [w["head_w"], w["head_b"], w["template"]]
    return out


class Adam:
    """Adam over a list of arrays, with the global gradient norm clipped."""

    def __init__(self, np, params: list[Any], lr: float, betas: tuple[float, float] = (0.9, 0.99), clip: float = 5.0) -> None:
        self.np = np
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.clip = clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[Any], lr: float | None = None) -> float:
        np = self.np
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = self.clip / norm if norm > self.clip else 1.0
        self.t += 1
        rate = self.lr if lr is None else lr
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= rate * (m / c1) / (np.sqrt(v / c2) + 1e-8)
        return norm


def _numpy_or_raise():
    np = _numpy()
    if np is None:
        raise ValueError("training a vocoder needs numpy (pip install numpy); the vocoder itself does not")
    return np


def _example_loss(np, w, spec: Spec, a: Analysis, loss_fn: SpectralLoss, clip: Clip, start: int, length: int):
    """One training example: the segment's loss, its terms, and the gradients of the weights."""
    context = spec.context
    in0 = max(0, start - context)
    in1 = min(clip.frames, start + length + context)
    count = in1 - in0
    codes = np.asarray(clip.codes[in0:in1], dtype=np.int64)
    first_sample = in0 * a.hop
    span = excitation_length(count, a)
    ep, en = excitation_numpy(np, a, count, pulses=clip.pulses[first_sample:first_sample + span])
    fwd = forward_numpy(np, w, spec, codes)
    gp = np.exp(fwd["lp"])
    gn = np.exp(fwd["ln"])
    output, norm = istft_numpy(np, ep * gp + en * gn, a)
    target = clip.samples[first_sample:first_sample + len(output)]
    first = (start - in0) * a.hop
    region = slice(first, first + excitation_length(length, a))
    loss, g_out, terms = loss_fn(target, output, region)
    g_spec = istft_numpy_backward(np, g_out, norm, count, a)
    g_lp = (g_spec.real * ep.real + g_spec.imag * ep.imag) * gp
    g_ln = (g_spec.real * en.real + g_spec.imag * en.imag) * gn
    return loss, terms, _flatten_grads(backward_numpy(np, w, spec, fwd, g_lp, g_ln))


def train(
    codebook: Codebook,
    recordings: Iterable[Sequence[float]],
    *,
    steps: int = 2000,
    segment: int = 200,
    batch: int = 8,
    lr: float = 2e-3,
    seed: int = 1,
    channels: int = 64,
    kernel: int = 5,
    dilations: Sequence[int] = (1, 2, 4, 8, 1, 2),
    hold_out: int = 0,
    mel_weight: float = 1.0,
    note: str = "",
    log: Callable[[str], None] | None = None,
    vocoder: UnitVocoder | None = None,
    polish: int = 32,
) -> tuple[UnitVocoder, dict[str, Any]]:
    """Fit a vocoder to recordings (samples at the codebook's rate) under the codebook.

    ``segment`` frames are drawn at random from a random recording for each of
    the ``batch`` examples of a step, with the network's context on either side
    so the frames at a segment's edges hear their neighbours as they would in a
    whole utterance; the loss is taken over the segment alone.  Adam, the
    learning rate cosine-decayed from ``lr`` to a tenth of it, the gradient
    clipped.  The last ``hold_out`` recordings are kept back and measured
    (:func:`evaluate`, against the codebook's own vocoder polished ``polish``
    times), never trained on.  ``vocoder`` continues training one instead of
    starting fresh.  Returns the vocoder and a report.
    """
    np = _numpy_or_raise()
    started = time.perf_counter()
    say = log or (lambda line: None)
    clips = prepare(np, codebook, recordings)
    if len(clips) <= hold_out:
        raise ValueError(f"{len(clips)} recording(s) with {hold_out} held out leaves nothing to train on")
    held = clips[len(clips) - hold_out:] if hold_out else []
    clips = clips[:len(clips) - hold_out] if hold_out else clips
    a = codebook.analysis
    if vocoder is None:
        vocoder = UnitVocoder.new(codebook, channels=channels, kernel=kernel, dilations=dilations, seed=seed,
                                  template=templates(np, clips, codebook.k, a))
    elif not vocoder.matches(codebook):
        raise ValueError("the vocoder to continue was not trained for this codebook")
    spec = vocoder.spec
    w = vocoder._numpy_weights(np)
    params = _flatten_weights(w)
    optimiser = Adam(np, params, lr)
    loss_fn = SpectralLoss(np, a, mel_weight)
    rng = random.Random(seed)
    weights = [max(1, c.frames) for c in clips]
    history: list[dict[str, float]] = []
    seconds = sum(len(c.samples) for c in clips) / a.rate
    pitches = sorted(c.pitch for c in clips)
    say(f"training on {len(clips)} recording(s), {seconds:.1f} s, {sum(c.frames for c in clips)} frames, pitch "
        f"{pitches[0]:.0f}-{pitches[-1]:.0f} Hz; {spec.parameters} parameters, {steps} steps of {batch} x {segment} "
        f"frames; {time.perf_counter() - started:.0f} s to prepare")
    for step in range(1, steps + 1):
        rate = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, steps))))
        total_grads: list[Any] | None = None
        total_loss = 0.0
        term_sums: dict[str, float] = {}
        for _ in range(batch):
            clip = rng.choices(clips, weights=weights, k=1)[0]
            length = min(segment, clip.frames)
            start = rng.randint(0, clip.frames - length)
            loss, terms, grads = _example_loss(np, w, spec, a, loss_fn, clip, start, length)
            total_loss += loss
            for key, value in terms.items():
                term_sums[key] = term_sums.get(key, 0.0) + value
            if total_grads is None:
                total_grads = grads
            else:
                for acc, g in zip(total_grads, grads):
                    acc += g
        assert total_grads is not None
        grad_norm = optimiser.step([g / batch for g in total_grads], rate)
        record = {"step": step, "loss": total_loss / batch, "lr": rate, "grad_norm": grad_norm}
        record.update({k: v / batch for k, v in term_sums.items()})
        history.append(record)
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            say(f"step {step}/{steps}  loss {record['loss']:.4f}  mel {record.get('mel', 0.0):.4f}  "
                f"lr {rate:.2e}  |g| {record['grad_norm']:.3f}  {time.perf_counter() - started:.0f} s")
    vocoder._np_cache = None
    _write_back(vocoder, w)
    report: dict[str, Any] = {
        "steps": steps, "batch": batch, "segment": segment, "lr": lr, "seed": seed, "mel_weight": mel_weight,
        "recordings": len(clips), "seconds": round(seconds, 2), "frames": sum(c.frames for c in clips),
        "pitch": [round(pitches[0], 1), round(pitches[-1], 1)] if pitches else [],
        "loss": history[-1]["loss"] if history else None, "history": history,
        "elapsed": round(time.perf_counter() - started, 1),
    }
    if held:
        report["held_out"] = evaluate(vocoder, codebook, [], prepared=held, polish=polish)
        say(f"held out: {json.dumps(report['held_out'])}")
    vocoder.trained = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "history"}
    vocoder.note = note or vocoder.note
    return vocoder, report


def _write_back(vocoder: UnitVocoder, w: dict[str, Any]) -> None:
    """The numpy weights back into the portable lists."""
    vocoder.weights.embed = w["embed"].tolist()
    vocoder.weights.blocks = [
        {"w1": b["w1"].tolist(), "b1": b["b1"].tolist(), "w2": b["w2"].tolist(), "b2": b["b2"].tolist()}
        for b in w["blocks"]
    ]
    vocoder.weights.head = {"w": w["head_w"].tolist(), "b": w["head_b"].tolist()}
    vocoder.weights.template = w["template"].tolist()


def logmel_distance(a: Analysis, reference: Sequence[float], output: Sequence[float]) -> float:
    """The mean absolute distance between the log-mel frames the analysis hears in two waveforms."""
    count = min(len(reference), len(output))
    x = frames_of([float(v) for v in reference[:count]], a, normalize=False)
    y = frames_of([float(v) for v in output[:count]], a, normalize=False)
    total = 0.0
    n = 0
    for fx, fy in zip(x, y):
        for p, q in zip(fx, fy):
            total += abs(p - q)
            n += 1
    return total / max(1, n)


def evaluate(vocoder: UnitVocoder, codebook: Codebook, recordings: Iterable[Sequence[float]],
             prepared: Sequence[Clip] | None = None, polish: int = 32) -> dict[str, Any]:
    """How close the vocoder comes to recordings, against the codebook's own vocoder.

    For each recording: its codes, the neural vocoder's samples for them and
    the codebook's vocoder's (``polish`` Griffin-Lim iterations), both at the
    recording's own median pitch - and for each, the log-mel distance to the
    original over every frame and over the speech alone (the frames within
    six nats of the loudest), and how many of the original's codes the output
    is heard as again (the round trip).  Lower distances and higher round
    trips are better; the numbers are averaged over the recordings.  The
    round trip favours the codebook's own vocoder, whose frames *are* the
    centroids, and turns on the silences: the codebook hears every frame
    against the utterance's mean, so a vocoder whose silence is a little
    louder than the recording's moves every unit.
    """
    from .acoustic import griffin_lim

    np = _numpy_or_raise()
    a = codebook.analysis
    clips = list(prepared) if prepared is not None else prepare(np, codebook, recordings)
    sums = {"neural_logmel": 0.0, "griffin_logmel": 0.0, "neural_logmel_speech": 0.0, "griffin_logmel_speech": 0.0,
            "neural_round_trip": 0.0, "griffin_round_trip": 0.0}
    for clip in clips:
        codes = clip.codes
        pitch = clip.pitch
        neural = vocoder.samples(codes, pitch)
        # the original's own runs, not the typical ones unit_frames would hold each unit for
        frames = [[c + m for c, m in zip(codebook.centroids[code], codebook.mean)] for code in codes]
        griffin = griffin_lim(frames, a, polish, pitch)
        original = frames_of([float(v) for v in clip.samples], a, normalize=False)
        levels = [sum(f) / len(f) for f in original]
        loud = max(levels)
        speech = [level >= loud - 6.0 for level in levels]  # within six nats of the loudest frame
        for name, output in (("neural", neural), ("griffin", griffin)):
            heard_frames = frames_of([float(v) for v in output], a, normalize=False)
            distances = [sum(abs(p - q) for p, q in zip(fx, fy)) / len(fx) for fx, fy in zip(original, heard_frames)]
            sums[f"{name}_logmel"] += sum(distances) / max(1, len(distances))
            spoken = [d for d, s in zip(distances, speech) if s]
            sums[f"{name}_logmel_speech"] += sum(spoken) / max(1, len(spoken))
            heard = codebook.codes(frames_of([float(v) for v in output], a))
            agree = sum(1 for p, q in zip(heard, codes) if p == q)
            sums[f"{name}_round_trip"] += agree / max(1, len(codes))
    n = max(1, len(clips))
    return {k: round(v / n, 4) for k, v in sums.items()} | {"recordings": len(clips), "polish": polish}


def gradient_check(codebook: Codebook, np=None, seed: int = 3, channels: int = 4, dilations: Sequence[int] = (1, 2),
                   frames: int = 9, mel_weight: float = 1.0) -> float:
    """The largest relative disagreement between the hand-written gradients and finite differences, on a tiny
    vocoder over a random target: the trainer's own proof.  Small (below 1e-4) when every adjoint is right."""
    np = np or _numpy_or_raise()
    a = codebook.analysis
    vocoder = UnitVocoder.new(codebook, channels=channels, kernel=3, dilations=dilations, seed=seed)
    w = vocoder._numpy_weights(np)
    spec = vocoder.spec
    rng = np.random.default_rng(seed)
    codes = [int(c) for c in rng.integers(0, codebook.k, size=frames)]
    target = rng.standard_normal(excitation_length(frames, a)) * 0.1
    f0 = [float(v) for v in 100.0 + 20.0 * rng.random(frames)]
    clip = Clip(codes, target, f0, [True] * frames, pulse_train_numpy(np, f0, frames, a))
    loss_fn = SpectralLoss(np, a, mel_weight)

    def loss_of() -> float:
        return _example_loss(np, w, spec, a, loss_fn, clip, 0, frames)[0]

    grads = _example_loss(np, w, spec, a, loss_fn, clip, 0, frames)[2]
    params = _flatten_weights(w)
    worst = 0.0
    for p, g in zip(params, grads):
        flat = p.reshape(-1)
        gflat = g.reshape(-1)
        for _ in range(min(6, flat.size)):
            i = int(rng.integers(0, flat.size))
            keep = flat[i]
            best = math.inf
            for h in (1e-5, 1e-6):  # the loss has kinks (the L1 terms) and strong curvature (exp): two step sizes
                flat[i] = keep + h
                up = loss_of()
                flat[i] = keep - h
                down = loss_of()
                flat[i] = keep
                numeric = (up - down) / (2.0 * h)
                scale = max(abs(numeric), abs(gflat[i]), 1e-6)
                best = min(best, abs(numeric - gflat[i]) / scale)
            worst = max(worst, best)
    return worst


__all__ = [
    "VOCODER_SUFFIX", "DEFAULT_VOCODER", "Spec", "Weights", "UnitVocoder", "codes_of", "find_vocoder",
    "vocoder_path_for", "codebook_checksum", "pulse_train", "disperse", "chirp_kernel", "CHIRP", "noise_train",
    "excitation_spectra", "train", "evaluate",
    "prepare", "templates", "track_pitch", "gradient_check", "SpectralLoss", "RESOLUTIONS", "PEAK", "HEADROOM", "CLIP",
    "SLOPE",
]
