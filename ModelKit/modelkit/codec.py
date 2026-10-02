"""The diffusion codec: Stable Diffusion run backwards as an image compressor - the encoder and the decoder.

:mod:`modelkit.vision` runs one part of Stable Diffusion backwards: the VAE,
image -> latent.  This module runs the *whole* generator backwards.

Stable Diffusion makes a picture out of nothing in steps.  It starts from
noise and at every step the model looks at what it has and predicts the clean
picture; the noise is reduced, the prediction improves, and after the last
step the latent is decoded into pixels.  All the information in the picture
comes from the model's own predictions - nothing is sent in.

The codec keeps those steps and adds one thing: at every step, the encoder
compares what the generator predicts with what is *true* (the real picture's
latent), and sends only the difference, quantised.  The decoder runs the same
generator from the same noise, and at every step adds the correction it is
sent.  The bitstream is the sequence of corrections and nothing else::

    noise ─ step 1 ─ step 2 ─ ... ─ step n ─ latent ─ VAE ─ picture
             +c1      +c2            +cn                      (the decoder)
    picture ─ VAE ─ latent ─┐
                   c_k = quantise(latent - what the generator predicts at step k)   (the encoder)

What the generator can guess is never sent, so the better it knows pictures,
the fewer bits are left to send.  Early steps are quantised coarsely - only
the gross shape is corrected, at the precision the next step's noise would
drown anyway - and the last step finely, so the stream is **progressive**: cut
it off after any step and the decoder runs the rest of the generator with no
corrections at all, and the diffusion fills in what was not sent.  The
quantisation step at the last step is the quality knob.

Processes
---------
* ``"sd"`` - :class:`DiffusionProcess`: Stable Diffusion's UNet and VAE run by
  DDIM (needs ``pillow``, ``torch``, ``diffusers``; ``transformers`` for the
  prompt).  The weights are ``$RADIXNET_SD_MODEL`` (default
  ``stable-diffusion-v1-5/stable-diffusion-v1-5``) and the VAE of
  :mod:`modelkit.vision`.  A *prompt* can ride in the header as side
  information: a few bytes of caption the generator conditions on.
* ``"pyramid"`` - :class:`PyramidProcess`: the stand-in, Pillow only.  The same
  algorithm, with resolution for the noise schedule and bicubic upsampling for
  the model: the generator's "prediction" at each level is the previous level
  enlarged, and the corrections are what the enlargement misses.  This is a
  cold diffusion (blur instead of noise) with a fixed denoiser, and it is a
  real codec: it compresses, it is progressive, and it works anywhere.
* ``"auto"`` (default) - ``sd`` when it loads, else ``pyramid``.

Both are deterministic: the same picture and settings give the same bytes.

The corrections are entropy-coded with an adaptive binary arithmetic coder
(standard library, below): a zero flag, a sign and a unary magnitude, each with
a context from the neighbouring corrections.  Each step is flushed on its own
and prefixed with its length, so a cut-off stream still decodes every complete
step.

File format (``rdc``)::

    "RDC\\x01"  u16 header length  header (JSON: process, width, height, step, ...)
    then one block per step: varint length + the arithmetic-coded corrections
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
from dataclasses import dataclass
from typing import Any, Iterator

from . import vision
from .vision import VisionError, load_image

__all__ = [
    "CHROMA_STEP",
    "DEADZONE",
    "DEFAULT_SD_MODEL",
    "DEFAULT_STEP",
    "DEFAULT_STEPS",
    "DEFAULT_SEED",
    "MAGIC",
    "PROCESSES",
    "TEXT_NAME",
    "Block",
    "CodecError",
    "DiffusionProcess",
    "PyramidProcess",
    "compare_images",
    "decode",
    "decode_text",
    "describe",
    "encode",
    "encode_text",
    "get_process",
    "iter_blocks",
    "psnr",
    "read_header",
    "truncate",
]

MAGIC = b"RDC\x01"
TEXT_NAME = "rdc"
"""The encoder name in the ``img:rdc:<w>x<h>:<base64>`` text form (:func:`encode_text`)."""
PROCESSES = ("auto", "sd", "pyramid")
DEFAULT_STEP = 4.0
"""The quantisation step of the last (finest) block: pixel units for ``pyramid``, latent units (x 1/16) for ``sd``."""
DEFAULT_STEPS = 10
"""DDIM steps of the ``sd`` process."""
DEFAULT_SEED = 0
"""The seed of the noise the ``sd`` process starts from (both sides draw the same)."""
DEADZONE = 0.3
"""Residuals within ``(0.5 + DEADZONE)`` quantisation steps of zero round to zero: fewer, cheaper symbols."""
CHROMA_STEP = 2.0
"""The ``pyramid`` process quantises the two chroma channels this much coarser than luma."""
LEVEL_GAIN = 0.5
"""The ``pyramid`` process's quantisation step shrinks by this factor per level below the finest (1 = the same step everywhere)."""
DEFAULT_SD_MODEL = os.environ.get("RADIXNET_SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
SD_LATENT_UNIT = 1.0 / 16.0
"""``sd`` steps are given in the same units as ``pyramid`` ones (``DEFAULT_STEP`` = 4): one unit is 1/16 of a latent."""
MAX_SIZE = 4096


class CodecError(VisionError):
    """A missing dependency, an unreadable image, an unknown process or bytes that are not an ``rdc`` stream."""


# ---------------------------------------------------------------------------
# the arithmetic coder
# ---------------------------------------------------------------------------

_CODE_BITS = 32
_TOP = 1 << _CODE_BITS
_HALF = _TOP >> 1
_QUARTER = _TOP >> 2
_THREE_QUARTERS = _HALF + _QUARTER
_COUNT_LIMIT = 1 << 12
"""An adaptive bit model halves its counts past this total, so it keeps adapting."""
_NEIGHBOUR_CLASSES = 4
_MAG_BINS = 8
_MAG_CAP = 16
"""Magnitudes beyond this many unary bins escape to Exp-Golomb raw bits (tiny steps, wild residuals)."""


def _new_model() -> list[int]:
    return [1, 1]  # counts of 0 and 1, both seen once


class _BitWriter:
    __slots__ = ("out", "acc", "nbits")

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def bit(self, b: int) -> None:
        self.acc = (self.acc << 1) | b
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def finish(self) -> bytes:
        if self.nbits:
            self.out.append(self.acc << (8 - self.nbits))
            self.acc = 0
            self.nbits = 0
        return bytes(self.out)


class _BitReader:
    __slots__ = ("data", "pos", "acc", "nbits")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def bit(self) -> int:
        if self.nbits == 0:
            if self.pos < len(self.data):
                self.acc = self.data[self.pos]
                self.pos += 1
            else:
                self.acc = 0  # past the end: zeros (the coder's flush makes them harmless)
            self.nbits = 8
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1


class ArithmeticEncoder:
    """A binary arithmetic encoder (Witten-Neal-Cleary, 32-bit registers) over adaptive two-count models."""

    def __init__(self) -> None:
        self.low = 0
        self.high = _TOP - 1
        self.pending = 0
        self._bits = _BitWriter()

    def _emit(self, b: int) -> None:
        bits = self._bits
        bits.bit(b)
        while self.pending:
            bits.bit(1 - b)
            self.pending -= 1

    def encode(self, bit: int, model: list[int]) -> None:
        n0, n1 = model
        total = n0 + n1
        low, high = self.low, self.high
        split = low + (high - low + 1) * n0 // total
        if bit:
            low = split
            model[1] = n1 + 1
        else:
            high = split - 1
            model[0] = n0 + 1
        if total + 1 >= _COUNT_LIMIT:
            model[0] = (model[0] + 1) >> 1
            model[1] = (model[1] + 1) >> 1
        while True:
            if high < _HALF:
                self._emit(0)
            elif low >= _HALF:
                self._emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                self.pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        self.low, self.high = low, high

    def encode_raw(self, bit: int) -> None:
        """An equiprobable bit (no model, no adaptation)."""
        self.encode(bit, [1, 1])

    def finish(self) -> bytes:
        self.pending += 1
        if self.low < _QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self._bits.finish()


class ArithmeticDecoder:
    """The decoder matching :class:`ArithmeticEncoder`; reads past the end as zero bits (an empty block is all zeros)."""

    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = _TOP - 1
        self._bits = _BitReader(data)
        value = 0
        for _ in range(_CODE_BITS):
            value = (value << 1) | self._bits.bit()
        self.value = value

    def decode(self, model: list[int]) -> int:
        n0, n1 = model
        total = n0 + n1
        low, high, value = self.low, self.high, self.value
        split = low + (high - low + 1) * n0 // total
        if value >= split:
            bit = 1
            low = split
            model[1] = n1 + 1
        else:
            bit = 0
            high = split - 1
            model[0] = n0 + 1
        if total + 1 >= _COUNT_LIMIT:
            model[0] = (model[0] + 1) >> 1
            model[1] = (model[1] + 1) >> 1
        read = self._bits.bit
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | read()
        self.low, self.high, self.value = low, high, value
        return bit

    def decode_raw(self) -> int:
        return self.decode([1, 1])


class _Contexts:
    """The adaptive models of one block: a zero flag, a sign and unary magnitude bins, each per neighbour class."""

    __slots__ = ("zero", "sign", "mag")

    def __init__(self) -> None:
        self.zero = [_new_model() for _ in range(_NEIGHBOUR_CLASSES)]
        self.sign = [_new_model() for _ in range(_NEIGHBOUR_CLASSES)]
        self.mag = [[_new_model() for _ in range(_MAG_BINS)] for _ in range(_NEIGHBOUR_CLASSES)]


def _neighbour_class(left: int, above: int) -> int:
    m = max(left if left >= 0 else -left, above if above >= 0 else -above)
    return m if m < _NEIGHBOUR_CLASSES - 1 else _NEIGHBOUR_CLASSES - 1


def _write_symbols(symbols: list[int], width: int) -> bytes:
    """Entropy-code one block of quantised residuals laid out in rows of ``width``."""
    enc = ArithmeticEncoder()
    ctx = _Contexts()
    encode = enc.encode
    for i, q in enumerate(symbols):
        left = symbols[i - 1] if i % width else 0
        above = symbols[i - width] if i >= width else 0
        nb = _neighbour_class(left, above)
        if q == 0:
            encode(0, ctx.zero[nb])
            continue
        encode(1, ctx.zero[nb])
        encode(1 if q < 0 else 0, ctx.sign[nb])
        m = (q if q > 0 else -q) - 1
        bins = ctx.mag[nb]
        k = 0
        while k < _MAG_CAP and m > k:
            encode(1, bins[k if k < _MAG_BINS else _MAG_BINS - 1])
            k += 1
        if k < _MAG_CAP:
            encode(0, bins[k if k < _MAG_BINS else _MAG_BINS - 1])
        else:  # escape: Exp-Golomb of the excess, equiprobable bits
            excess = m - _MAG_CAP + 1
            nbits = excess.bit_length()
            for _ in range(nbits - 1):
                enc.encode_raw(1)
            enc.encode_raw(0)
            for shift in range(nbits - 2, -1, -1):
                enc.encode_raw((excess >> shift) & 1)
    return enc.finish()


def _read_symbols(data: bytes, count: int, width: int) -> list[int]:
    """The inverse of :func:`_write_symbols`."""
    dec = ArithmeticDecoder(data)
    ctx = _Contexts()
    decode = dec.decode
    symbols: list[int] = []
    append = symbols.append
    for i in range(count):
        left = symbols[i - 1] if i % width else 0
        above = symbols[i - width] if i >= width else 0
        nb = _neighbour_class(left, above)
        if not decode(ctx.zero[nb]):
            append(0)
            continue
        negative = decode(ctx.sign[nb])
        bins = ctx.mag[nb]
        m = 0
        while m < _MAG_CAP and decode(bins[m if m < _MAG_BINS else _MAG_BINS - 1]):
            m += 1
        if m >= _MAG_CAP:
            nbits = 1
            while dec.decode_raw():
                nbits += 1
            excess = 1
            for _ in range(nbits - 1):
                excess = (excess << 1) | dec.decode_raw()
            m = _MAG_CAP - 1 + excess
        q = m + 1
        append(-q if negative else q)
    return symbols


# ---------------------------------------------------------------------------
# the varint block framing
# ---------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        if pos >= len(data):
            raise CodecError("truncated block length")
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, pos
        shift += 7
        if shift > 35:
            raise CodecError("bad block length")


# ---------------------------------------------------------------------------
# the processes: what the generator predicts at each step, and what is true
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One step of the generator, as the codec sees it: ``count`` numbers in rows of ``width``, quantised at ``max(floor, step * scale)``."""

    index: int
    label: str
    count: int
    width: int
    scale: float
    floor: float = 0.0


def _delta(step: float, block: Block) -> float:
    """The quantisation step of a block."""
    return max(block.floor, step * block.scale)


class PyramidProcess:
    """The stand-in: resolution for the noise schedule, bicubic enlargement for the model, Pillow only.

    The levels are the picture halved until it is a few pixels across; the
    coarsest level is predicted as mid-grey, every other level as the previous
    one enlarged.  Luma and the two chroma channels (YCbCr) are separate
    blocks, chroma quantised :data:`CHROMA_STEP` times coarser.
    """

    name = "pyramid"
    channels = ("Y", "Cb", "Cr")

    def __init__(self) -> None:
        self.error: str | None = None
        self.run_lock = threading.Lock()

    @staticmethod
    def sizes(width: int, height: int) -> list[tuple[int, int]]:
        """Level sizes, coarsest first."""
        levels = [(width, height)]
        while max(levels[-1]) > 2:
            w, h = levels[-1]
            levels.append((max(1, (w + 1) // 2), max(1, (h + 1) // 2)))
        levels.reverse()
        return levels

    def coded_size(self, width: int, height: int, size: int | None) -> tuple[int, int]:
        if size and max(width, height) > size:
            ratio = size / max(width, height)
            return max(1, round(width * ratio)), max(1, round(height * ratio))
        return width, height

    def header(self) -> dict:
        return {}

    def configure(self, header: dict) -> None:
        pass

    def blocks(self, width: int, height: int) -> list[Block]:
        out: list[Block] = []
        sizes = self.sizes(width, height)
        for level, (w, h) in enumerate(sizes):
            gain = LEVEL_GAIN ** (len(sizes) - 1 - level)
            for c, name in enumerate(self.channels):
                # pixels are integers: a step below one would spend bits on nothing
                out.append(Block(len(out), f"{w}x{h}/{name}", w * h, w, gain * (1.0 if c == 0 else CHROMA_STEP), floor=1.0))
        return out

    def truths(self, image, width: int, height: int) -> list[list[float]]:
        Image = vision._pil()
        channels = image.convert("YCbCr").split()
        out: list[list[float]] = []
        for size in self.sizes(width, height):
            for ch in channels:
                out.append([float(v) for v in ch.resize(size, Image.LANCZOS).tobytes()])
        return out

    def start(self, width: int, height: int) -> dict:
        return {"sizes": self.sizes(width, height), "level": None, "next": [None, None, None]}

    def predict(self, state: dict, block: Block) -> list[float]:
        Image = vision._pil()
        level, channel = divmod(block.index, 3)
        if state["level"] is None:
            return [128.0] * block.count
        size = state["sizes"][level]
        return [float(v) for v in state["level"][channel].resize(size, Image.BICUBIC).tobytes()]

    def advance(self, state: dict, block: Block, corrected: list[float]) -> dict:
        Image = vision._pil()
        level, channel = divmod(block.index, 3)
        size = state["sizes"][level]
        pixels = bytes(0 if v < 0 else (255 if v > 255 else int(v + 0.5)) for v in corrected)
        state["next"][channel] = Image.frombytes("L", size, pixels)
        if channel == 2:
            state = {"sizes": state["sizes"], "level": state["next"], "next": [None, None, None]}
        return state

    def finish(self, state: dict, width: int, height: int):
        Image = vision._pil()
        return Image.merge("YCbCr", state["level"]).convert("RGB")


class DiffusionProcess:
    """Stable Diffusion run backwards: the UNet's DDIM steps with the true latent corrected in at each one.

    ``model`` is a Hugging Face model id or a local ``diffusers`` directory
    with ``unet``, ``scheduler`` and (for the prompt) ``tokenizer`` /
    ``text_encoder`` subfolders; the VAE is :class:`modelkit.vision.SDVaeEncoder`'s.
    ``steps`` DDIM steps from a Gaussian latent drawn with ``seed``; the
    ``prompt`` (default empty) conditions the UNet and rides in the header.

    Step ``k`` corrects the predicted clean latent at precision
    ``max(step, noise-to-signal ratio of step k+1)``: differences the next
    step's noise would drown are not worth sending yet.  The last step's
    precision is ``step`` itself.
    """

    name = "sd"

    def __init__(self, model: str | None = None, device: str | None = None, steps: int = DEFAULT_STEPS,
                 prompt: str = "", seed: int = DEFAULT_SEED) -> None:
        self.model = model or DEFAULT_SD_MODEL
        self.device = device
        self.steps = int(steps)
        self.prompt = prompt or ""
        self.seed = int(seed)
        self.unet: Any = None
        self.scheduler: Any = None
        self.vae: Any = None
        self.tokenizer: Any = None
        self.text_encoder: Any = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self.run_lock = threading.Lock()
        """One encode or decode at a time: the settings of a run live on the instance."""
        self._embedding: Any = None
        self._embedding_prompt: str | None = None

    @staticmethod
    def importable() -> bool:
        return vision.SDVaeEncoder.importable()

    def header(self) -> dict:
        return {"steps": self.steps, "seed": self.seed, "prompt": self.prompt, "model": self.model}

    def configure(self, header: dict) -> None:
        """Take a stream's settings; a stream made with other weights than the loaded ones is refused, not drifted."""
        self.steps = int(header.get("steps", self.steps))
        self.seed = int(header.get("seed", self.seed))
        self.prompt = str(header.get("prompt", self.prompt) or "")
        wanted = str(header.get("model") or "")
        if wanted and wanted != self.model:
            if self.unet is not None:
                raise CodecError(f"the stream was made with {wanted!r} but {self.model!r} is loaded: set RADIXNET_SD_MODEL")
            self.model = wanted

    def coded_size(self, width: int, height: int, size: int | None) -> tuple[int, int]:
        if size and max(width, height) > size:
            ratio = size / max(width, height)
            width, height = max(8, round(width * ratio)), max(8, round(height * ratio))
        return max(8, width // 8 * 8), max(8, height // 8 * 8)

    def load(self) -> None:
        """Load the UNet, the scheduler, the VAE and (when transformers is there) the text encoder; once per process."""
        if self.unet is not None:
            return
        with self._lock:
            if self.unet is not None:
                return
            try:
                import torch
                from diffusers import DDIMScheduler, UNet2DConditionModel
            except ImportError as exc:
                self.error = "the sd process needs torch and diffusers: pip install torch diffusers"
                raise CodecError(self.error) from exc
            vae = vision.SDVaeEncoder(device=self.device)
            try:
                vae.load()
            except VisionError as exc:
                self.error = str(exc)
                raise CodecError(self.error) from exc
            self.device = vae.device
            try:
                unet = UNet2DConditionModel.from_pretrained(self.model, subfolder="unet", torch_dtype=torch.float32)
                scheduler = DDIMScheduler.from_pretrained(self.model, subfolder="scheduler")
            except Exception as exc:  # noqa: BLE001 - network, missing files, bad model ids
                self.error = f"cannot load the diffusion model {self.model!r}: {exc}"
                raise CodecError(self.error) from exc
            try:
                from transformers import CLIPTextModel, CLIPTokenizer

                self.tokenizer = CLIPTokenizer.from_pretrained(self.model, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(self.model, subfolder="text_encoder").to(self.device).eval()
            except Exception:  # noqa: BLE001 - no transformers, or no text encoder in the directory: unconditioned
                self.tokenizer = self.text_encoder = None
            self.vae = vae
            self.scheduler = scheduler
            self.unet = unet.to(self.device).eval()
            self.error = None

    # -- the schedule ------------------------------------------------------

    def _timesteps(self) -> list[int]:
        self.scheduler.set_timesteps(self.steps)
        return [int(t) for t in self.scheduler.timesteps]

    def _alpha(self, t: int) -> float:
        if t < 0:
            return float(getattr(self.scheduler, "final_alpha_cumprod", 1.0))
        return float(self.scheduler.alphas_cumprod[t])

    @staticmethod
    def _ntsr(alpha: float) -> float:
        """The noise-to-signal ratio of a DDIM state with this cumulative alpha, in clean-latent units."""
        return math.sqrt(max(0.0, 1.0 - alpha) / max(alpha, 1e-12))

    def blocks(self, width: int, height: int) -> list[Block]:
        self.load()
        timesteps = self._timesteps()
        w8, h8 = width // 8, height // 8
        count = 4 * w8 * h8
        out: list[Block] = []
        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            ntsr = self._ntsr(self._alpha(t_next)) / SD_LATENT_UNIT  # in the codec's step units
            out.append(Block(i, f"t={t}", count, w8, max(1.0, ntsr / self._step_hint)))
        return out

    _step_hint = DEFAULT_STEP
    """The ``step`` the blocks are scaled against (set by :func:`encode` / :func:`decode` before ``blocks``)."""

    # -- the generator -----------------------------------------------------

    def _embedding_for(self, prompt: str):
        import torch

        if self._embedding is not None and self._embedding_prompt == prompt:
            return self._embedding
        if self.text_encoder is None:
            dim = int(self.unet.config.cross_attention_dim)
            emb = torch.zeros((1, 77, dim), dtype=torch.float32, device=self.device)
        else:
            ids = self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                emb = self.text_encoder(ids)[0].to(torch.float32)
        self._embedding, self._embedding_prompt = emb, prompt
        return emb

    def _predict_x0(self, x, t: int):
        """The UNet's estimate of the clean latent from the DDIM state ``x`` at timestep ``t``; also its noise estimate."""
        import torch

        alpha = self._alpha(t)
        sa, sb = math.sqrt(alpha), math.sqrt(1.0 - alpha)
        with torch.no_grad():
            out = self.unet(x, t, encoder_hidden_states=self._embedding_for(self.prompt)).sample
        kind = str(getattr(self.scheduler.config, "prediction_type", "epsilon"))
        if kind == "epsilon":
            eps = out
            x0 = (x - sb * eps) / sa
        elif kind == "v_prediction":
            x0 = sa * x - sb * out
        elif kind == "sample":
            x0 = out
        else:
            raise CodecError(f"unsupported prediction type {kind!r}")
        return x0

    def _tensor(self, image):
        import torch

        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = raw.to(torch.float32).view(image.height, image.width, 3).permute(2, 0, 1).unsqueeze(0)
        return (x / 127.5 - 1.0).to(self.device)

    def truths(self, image, width: int, height: int) -> list[list[float]]:
        import torch

        self.load()
        with torch.no_grad():
            posterior = self.vae.vae.encode(self._tensor(image))
            latent = posterior.latent_dist.mean * self.vae.scaling
        values = [v / SD_LATENT_UNIT for v in latent.flatten().tolist()]
        return [values] * self.steps  # the truth is the same clean latent at every step

    def start(self, width: int, height: int) -> dict:
        import torch

        self.load()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn((1, 4, height // 8, width // 8), generator=gen, dtype=torch.float32).to(self.device)
        return {"x": noise, "timesteps": self._timesteps(), "x0": None}

    def predict(self, state: dict, block: Block) -> list[float]:
        x0 = self._predict_x0(state["x"], state["timesteps"][block.index])
        state["x0"] = x0
        return [v / SD_LATENT_UNIT for v in x0.flatten().tolist()]

    def advance(self, state: dict, block: Block, corrected: list[float]) -> dict:
        import torch

        timesteps = state["timesteps"]
        t = timesteps[block.index]
        t_next = timesteps[block.index + 1] if block.index + 1 < len(timesteps) else -1
        x = state["x"]
        x0 = torch.tensor([v * SD_LATENT_UNIT for v in corrected], dtype=torch.float32, device=x.device).view(x.shape)
        alpha, alpha_next = self._alpha(t), self._alpha(t_next)
        sa, sb = math.sqrt(alpha), math.sqrt(max(0.0, 1.0 - alpha))
        eps = (x - sa * x0) / sb if sb > 0 else torch.zeros_like(x)  # the noise consistent with the corrected x0
        x_next = math.sqrt(alpha_next) * x0 + math.sqrt(max(0.0, 1.0 - alpha_next)) * eps
        return {"x": x_next, "timesteps": timesteps, "x0": x0}

    def finish(self, state: dict, width: int, height: int):
        import torch

        Image = vision._pil()
        z = state["x0"] if state["x0"] is not None else state["x"]
        with torch.no_grad():
            sample = self.vae.vae.decode(z / self.vae.scaling).sample
            pixels = ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        data = bytes(pixels[0].permute(1, 2, 0).contiguous().flatten().tolist())
        return Image.frombytes("RGB", (int(pixels.shape[3]), int(pixels.shape[2])), data)


_PROCESSES: dict[str, Any] = {}
_PROCESSES_LOCK = threading.Lock()


def _instance(name: str) -> Any:
    with _PROCESSES_LOCK:
        proc = _PROCESSES.get(name)
        if proc is None:
            proc = _PROCESSES[name] = DiffusionProcess() if name == "sd" else PyramidProcess()
        return proc


def get_process(name: str = "auto") -> Any:
    """The process ``name`` (``sd`` / ``pyramid``), or with ``auto`` the diffusion model when it loads, else the stand-in."""
    key = (name or "auto").strip().lower()
    if key not in PROCESSES:
        raise CodecError(f"unknown process {name!r}; expected one of: {', '.join(PROCESSES)}")
    try:
        vision._pil()
    except VisionError as exc:
        raise CodecError(str(exc)) from exc
    if key == "pyramid":
        return _instance("pyramid")
    if key == "sd":
        proc = _instance("sd")
        proc.load()
        return proc
    if DiffusionProcess.importable():
        proc = _instance("sd")
        if proc.unet is not None:
            return proc
        if proc.error is None:  # try once per process; a failed download is not retried on every image
            try:
                proc.load()
                return proc
            except CodecError:
                pass
    return _instance("pyramid")


# ---------------------------------------------------------------------------
# the codec
# ---------------------------------------------------------------------------


def _quantise(x: float) -> int:
    ax = x if x >= 0 else -x
    q = int(ax + 0.5 - DEADZONE)
    if q <= 0:
        return 0
    return q if x >= 0 else -q


def _pack_header(header: dict) -> bytes:
    body = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) > 0xFFFF:
        raise CodecError("the header is too long (is the prompt a novel?)")
    return MAGIC + struct.pack(">H", len(body)) + body


def read_header(blob: bytes) -> tuple[dict, int]:
    """``(header, offset of the first block)`` of an ``rdc`` stream."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < len(MAGIC) + 2 or bytes(blob[: len(MAGIC)]) != MAGIC:
        raise CodecError("not an rdc stream (bad magic)")
    (length,) = struct.unpack(">H", blob[len(MAGIC): len(MAGIC) + 2])
    start = len(MAGIC) + 2
    if len(blob) < start + length:
        raise CodecError("truncated rdc header")
    try:
        header = json.loads(bytes(blob[start: start + length]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CodecError(f"bad rdc header: {exc}") from exc
    for key in ("process", "width", "height", "step"):
        if key not in header:
            raise CodecError(f"rdc header without {key!r}")
    return header, start + length


def _check_step(step: float) -> float:
    step = float(step)
    if not (step > 0) or not math.isfinite(step):
        raise CodecError(f"step must be positive, got {step}")
    return step


def encode(data: bytes, process: str = "auto", step: float = DEFAULT_STEP, size: int | None = None,
           steps: int = DEFAULT_STEPS, prompt: str = "", seed: int = DEFAULT_SEED) -> dict:
    """Image bytes -> ``{"bytes", "process", "width", "height", "step", "blocks", "bits_per_pixel", "source_size", ...}``.

    ``step`` is the quantisation step of the last block (smaller = better,
    bigger); ``size`` caps the long side (``sd`` codes multiples of 8);
    ``steps``, ``prompt`` and ``seed`` are the ``sd`` process's.
    """
    step = _check_step(step)
    try:
        image = load_image(data)
    except VisionError as exc:
        raise CodecError(str(exc)) from exc
    source_size = [image.width, image.height]
    proc = get_process(process)
    if size is not None and (int(size) < 1 or int(size) > MAX_SIZE):
        raise CodecError(f"size must be between 1 and {MAX_SIZE}, got {size}")
    with proc.run_lock:
        return _encode(proc, image, source_size, step, size, steps, prompt, seed)


def _encode(proc: Any, image, source_size: list[int], step: float, size: int | None, steps: int, prompt: str, seed: int) -> dict:
    if proc.name == "sd":
        proc.steps, proc.prompt, proc.seed = max(1, int(steps)), prompt or "", int(seed)
        proc._step_hint = step
    width, height = proc.coded_size(image.width, image.height, int(size) if size else None)
    if (width, height) != (image.width, image.height):
        Image = vision._pil()
        image = image.resize((width, height), Image.LANCZOS)
    header = {"process": proc.name, "width": width, "height": height, "step": step, "source_size": source_size,
              **proc.header()}
    blocks = proc.blocks(width, height)
    truths = proc.truths(image, width, height)
    state = proc.start(width, height)
    packed = _pack_header(header)
    out = bytearray(packed)
    sizes: list[dict] = []
    for block in blocks:
        guess = proc.predict(state, block)
        delta = _delta(step, block)
        symbols: list[int] = []
        corrected: list[float] = []
        truth = truths[block.index]
        for i in range(block.count):
            q = _quantise((truth[i] - guess[i]) / delta)
            symbols.append(q)
            corrected.append(guess[i] + q * delta)
        coded = _write_symbols(symbols, block.width)
        out += _varint(len(coded)) + coded
        nonzero = sum(1 for q in symbols if q)
        sizes.append({"label": block.label, "count": block.count, "delta": delta, "bytes": len(coded), "nonzero": nonzero})
        state = proc.advance(state, block, corrected)
    total = len(out)
    return {
        "bytes": bytes(out),
        "process": proc.name,
        "width": width,
        "height": height,
        "step": step,
        "size": total,
        "header_bytes": len(packed),
        "bits_per_pixel": 8.0 * total / (width * height),
        "blocks": sizes,
        "source_size": source_size,
        **proc.header(),
    }


def decode(blob: bytes, process: str | None = None) -> dict:
    """``rdc`` bytes -> ``{"png", "image", "process", "width", "height", "step", "blocks", "blocks_decoded", "truncated"}``.

    A stream cut off after any block still decodes: the blocks that are there
    correct the generator, the rest of its steps run uncorrected, and
    ``truncated`` says so.  ``process`` overrides the header's.
    """
    header, pos = read_header(blob)
    name = (process or str(header["process"])).strip().lower()
    if name not in ("sd", "pyramid"):
        raise CodecError(f"unknown process {name!r} in the stream; expected 'sd' or 'pyramid'")
    step = _check_step(header["step"])
    width, height = int(header["width"]), int(header["height"])
    if width < 1 or height < 1 or width > MAX_SIZE or height > MAX_SIZE:
        raise CodecError(f"bad image size {width}x{height}")
    proc = get_process(name)
    with proc.run_lock:
        return _decode(proc, blob, pos, header, step, width, height)


def _decode(proc: Any, blob: bytes, pos: int, header: dict, step: float, width: int, height: int) -> dict:
    proc.configure(header)
    if proc.name == "sd":
        proc._step_hint = step
    blocks = proc.blocks(width, height)
    state = proc.start(width, height)
    decoded = 0
    truncated = False
    for block in blocks:
        guess = proc.predict(state, block)
        symbols = None
        if not truncated:
            if pos >= len(blob):
                truncated = True
            else:
                try:
                    length, after = _read_varint(blob, pos)
                except CodecError:
                    length, after = -1, pos
                if length < 0 or after + length > len(blob):
                    truncated = True
                else:
                    symbols = _read_symbols(bytes(blob[after: after + length]), block.count, block.width)
                    pos = after + length
        if symbols is None:
            corrected = guess  # no correction: the generator runs free
        else:
            delta = _delta(step, block)
            corrected = [guess[i] + symbols[i] * delta for i in range(block.count)]
            decoded += 1
        state = proc.advance(state, block, corrected)
    image = proc.finish(state, width, height)
    return {
        "png": vision._png_bytes(image),
        "image": image,
        "process": proc.name,
        "width": width,
        "height": height,
        "step": step,
        "blocks": len(blocks),
        "blocks_decoded": decoded,
        "truncated": truncated,
        "source_size": header.get("source_size"),
        "prompt": header.get("prompt", ""),
    }


# ---------------------------------------------------------------------------
# the text form, the measurements, the description
# ---------------------------------------------------------------------------


def encode_text(data: bytes, **options: Any) -> dict:
    """:func:`encode`, with the stream packed as ``img:rdc:<w>x<h>:<base64>`` under ``"text"`` as well."""
    result = encode(data, **options)
    result["text"] = vision.pack_text(TEXT_NAME, result["width"], result["height"], result["bytes"])
    return result


def decode_text(text: str, process: str | None = None) -> dict:
    """``img:rdc:<w>x<h>:<base64>`` -> :func:`decode`'s result (the base64 is repaired first, like any media text)."""
    try:
        name, _, _, payload, repaired = vision.parse_text(text)
    except VisionError as exc:
        raise CodecError(str(exc)) from exc
    if name != TEXT_NAME:
        raise CodecError(f"not an rdc text: the encoder is {name!r}, expected {TEXT_NAME!r}")
    result = decode(payload, process=process)
    result["repaired"] = repaired
    return result


def psnr(a, b) -> float:
    """Peak signal-to-noise ratio in dB between two RGB images of the same size (``inf`` when identical)."""
    if a.size != b.size:
        raise CodecError(f"images of different sizes: {a.size} and {b.size}")
    pa, pb = a.convert("RGB").tobytes(), b.convert("RGB").tobytes()
    if not pa:
        return math.inf
    sq = sum((x - y) * (x - y) for x, y in zip(pa, pb))
    if sq == 0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 * len(pa) / sq)


def compare_images(original: bytes, decoded, coded_size: int) -> dict:
    """How the codec did against the original, and against JPEG at the same byte budget.

    The original is resized to the decoded size when the codec changed it.
    JPEG's quality is searched for the largest file within ``coded_size``
    bytes (or the smallest JPEG Pillow makes, when even that is bigger).
    """
    Image = vision._pil()
    try:
        source = load_image(original)
    except VisionError as exc:
        raise CodecError(str(exc)) from exc
    if source.size != decoded.size:
        source = source.resize(decoded.size, Image.LANCZOS)
    result = {"psnr": psnr(source, decoded), "bytes": coded_size,
              "bits_per_pixel": 8.0 * coded_size / (decoded.width * decoded.height)}
    best = None
    for quality in range(1, 96):
        buf = io.BytesIO()
        try:
            source.save(buf, format="JPEG", quality=quality, optimize=True)
        except Exception:  # noqa: BLE001 - no JPEG support in this Pillow
            break
        n = len(buf.getvalue())
        if best is None or n <= coded_size:
            best = (quality, n, buf.getvalue())
        if n > coded_size:
            break
    if best is not None:
        q, n, data = best
        result["jpeg"] = {"quality": q, "bytes": n, "bits_per_pixel": 8.0 * n / (decoded.width * decoded.height),
                          "psnr": psnr(source, load_image(data))}
    return result


def describe() -> dict:
    """What is available: the dependencies, the configured diffusion model and which process ``auto`` resolves to."""
    def importable(module: str) -> bool:
        try:
            __import__(module)
        except ImportError:
            return False
        return True

    pillow = importable("PIL")
    sd = _PROCESSES.get("sd")
    torch_ok, diffusers_ok, transformers_ok = importable("torch"), importable("diffusers"), importable("transformers")
    loaded = bool(sd is not None and sd.unet is not None)
    return {
        "pillow": pillow,
        "torch": torch_ok,
        "diffusers": diffusers_ok,
        "transformers": transformers_ok,
        "sd_model": DEFAULT_SD_MODEL if sd is None else sd.model,
        "sd_vae": vision.DEFAULT_SD_VAE,
        "sd_loaded": loaded,
        "sd_error": None if sd is None else sd.error,
        "processes": list(PROCESSES),
        "default_step": DEFAULT_STEP,
        "default_steps": DEFAULT_STEPS,
        "auto": None if not pillow else ("sd" if loaded or (torch_ok and diffusers_ok and (sd is None or sd.error is None)) else "pyramid"),
        "format": "RDC\\x01 + u16 header length + JSON header + per step: varint length + arithmetic-coded corrections",
        "text_format": f"{vision.HEADER}:{TEXT_NAME}:<w>x<h>:<base64 of the rdc stream>",
    }


def iter_blocks(blob: bytes) -> Iterator[tuple[int, bytes]]:
    """``(index, coded bytes)`` of every complete block in a stream."""
    _, pos = read_header(blob)
    index = 0
    while pos < len(blob):
        length, after = _read_varint(blob, pos)
        if after + length > len(blob):
            return
        yield index, bytes(blob[after: after + length])
        pos = after + length
        index += 1


def truncate(blob: bytes, blocks: int) -> bytes:
    """The stream cut after its first ``blocks`` blocks (a progressive preview)."""
    _, pos = read_header(blob)
    end = pos
    for index, coded in iter_blocks(blob):
        if index >= blocks:
            break
        end += len(_varint(len(coded))) + len(coded)
    return bytes(blob[:end])
