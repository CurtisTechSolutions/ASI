"""Images as text: a Stable Diffusion image encoder (the inverse of image generation) plus base64.

Stable Diffusion *generates* an image by decoding a latent - ``4 x H/8 x W/8``
numbers - with its VAE.  This module runs that process backwards: the same
VAE's **encoder** turns an image into its compressed latent (an 8x spatial
reduction into 4 channels: 48 times fewer numbers than the RGB pixels), the
latent is quantised to one signed byte per number and base64-encoded, and the
resulting text::

    img:sd:128x128:AAECAwQF...

is what the trigram network trains on and predicts.  :func:`decode_text` runs
the forward process again (base64 -> latent -> VAE decoder -> PNG), so an
encoded or *predicted* text can be looked at.

Backends
--------
* ``"sd"`` - the diffusion VAE (:class:`SDVaeEncoder`): needs ``pillow``,
  ``torch`` and ``diffusers`` plus the VAE weights (``$RADIXNET_SD_VAE``, a
  Hugging Face model id or a local directory; default
  ``stabilityai/sd-vae-ft-mse``, the fine-tuned Stable Diffusion VAE).
* ``"tiny"`` - :class:`TinyEncoder`: ``pillow`` only; the same 8x spatial
  reduction as an RGB thumbnail (3 channels), so the text format, the API,
  the CLI and the Images tab all work without the diffusion weights.
* ``"auto"`` (default) - ``sd`` when it can be loaded, else ``tiny``.

Both encoders are deterministic (the VAE's latent *mean* is used, not a
sample), so the same image always gives the same text.
"""

from __future__ import annotations

import base64
import io
import os
import re
import struct
import threading
from typing import Any

from .encoding import repair_base64

__all__ = [
    "DEFAULT_SD_VAE",
    "DEFAULT_SIZE",
    "ENCODERS",
    "LATENT_SCALE",
    "SDVaeEncoder",
    "TinyEncoder",
    "VisionError",
    "decode_text",
    "describe",
    "encode_image",
    "get_encoder",
    "pack_text",
    "parse_text",
]

HEADER = "img"
ENCODERS = ("auto", "sd", "tiny")
DEFAULT_SIZE = 128
"""Images are resized to ``size x size`` before encoding (a multiple of 8; 128 -> a 4 x 16 x 16 latent)."""
MIN_SIZE, MAX_SIZE = 8, 1024
LATENT_SCALE = 4.0
"""Quantisation range of the scaled SD latents: ``[-4, 4]`` maps to one signed byte."""
DEFAULT_SD_VAE = os.environ.get("RADIXNET_SD_VAE", "stabilityai/sd-vae-ft-mse")

_TEXT_RE = re.compile(r"^\s*img:([a-z0-9_]+):(\d+)x(\d+):(.*)$", re.S)


class VisionError(ValueError):
    """A missing dependency, an unreadable image, an unknown encoder or a text that is not an encoded image."""


# ---------------------------------------------------------------------------
# the text format
# ---------------------------------------------------------------------------


def pack_text(encoder: str, width: int, height: int, payload: bytes) -> str:
    """``img:<encoder>:<w>x<h>:<base64 of the quantised latent>``."""
    return f"{HEADER}:{encoder}:{width}x{height}:" + base64.b64encode(payload).decode("ascii")


def parse_text(text: str) -> tuple[str, int, int, bytes, bool]:
    """``(encoder, width, height, payload, repaired)`` of an encoded-image text.

    The base64 part is repaired when it is not clean - whitespace, stray
    characters, a missing tail (a predicted text may be cut off or garbled):
    junk is dropped, the padding is completed and ``repaired`` says so.  The
    payload is *not* padded here (the decoder pads or truncates it to the
    latent size it needs).
    """
    match = _TEXT_RE.match(str(text or ""))
    if not match:
        raise VisionError("not an encoded image: expected 'img:<encoder>:<w>x<h>:<base64>'")
    encoder, width, height, body = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    try:
        payload, repaired = repair_base64(body)
    except ValueError as exc:  # pragma: no cover - the junk filter makes this rare
        raise VisionError(str(exc)) from exc
    if width <= 0 or height <= 0:
        raise VisionError(f"invalid image size {width}x{height}")
    return encoder, width, height, payload, repaired


def _fit_payload(payload: bytes, length: int) -> tuple[bytes, bool]:
    """Pad with zeros or truncate to ``length``; returns ``(payload, changed)``."""
    if len(payload) == length:
        return payload, False
    if len(payload) > length:
        return payload[:length], True
    return payload + bytes(length - len(payload)), True


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisionError("image encoding needs Pillow: pip install pillow") from exc
    return Image


def load_image(data: bytes):
    """Decode image bytes (PNG, JPEG, WebP, ... whatever Pillow reads) into an RGB image."""
    Image = _pil()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many kinds
        raise VisionError(f"not a readable image: {exc}") from exc
    return image.convert("RGB")


def _check_size(size: int) -> int:
    size = int(size)
    if size < MIN_SIZE or size > MAX_SIZE or size % 8:
        raise VisionError(f"size must be a multiple of 8 between {MIN_SIZE} and {MAX_SIZE}, got {size}")
    return size


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TinyEncoder:
    """The stand-in: an RGB thumbnail at 1/8 of the size (3 channels, one byte each) - no diffusion weights needed."""

    name = "tiny"
    channels = 3

    def latent_shape(self, width: int, height: int) -> tuple[int, int, int]:
        return (self.channels, height // 8, width // 8)

    def encode(self, image, size: int) -> tuple[bytes, tuple[int, int, int]]:
        Image = _pil()
        small = image.resize((size // 8, size // 8), Image.LANCZOS)
        return small.tobytes(), self.latent_shape(size, size)

    def decode(self, payload: bytes, width: int, height: int):
        Image = _pil()
        shape = self.latent_shape(width, height)
        small = Image.frombytes("RGB", (shape[2], shape[1]), payload)
        return small.resize((width, height), Image.BICUBIC)


class SDVaeEncoder:
    """The Stable Diffusion VAE run backwards: image -> latent mean (``4 x H/8 x W/8``), quantised to bytes.

    ``model`` is a Hugging Face model id or a local directory holding a
    ``diffusers`` ``AutoencoderKL``; the weights load on first use (kept for
    the process).  ``device`` defaults to CUDA / MPS when torch sees one.
    """

    name = "sd"
    channels = 4

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self.model = model or DEFAULT_SD_VAE
        self.device = device
        self.vae: Any = None
        self.scaling = 0.18215
        self.error: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def importable() -> bool:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def latent_shape(self, width: int, height: int) -> tuple[int, int, int]:
        return (self.channels, height // 8, width // 8)

    def load(self) -> None:
        """Load the VAE (idempotent); raises :class:`VisionError` when the dependencies or the weights are missing."""
        if self.vae is not None:
            return
        with self._lock:
            if self.vae is not None:
                return
            try:
                import torch
                from diffusers import AutoencoderKL
            except ImportError as exc:
                self.error = "the Stable Diffusion encoder needs torch and diffusers: pip install torch diffusers"
                raise VisionError(self.error) from exc
            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                )
            try:
                vae = AutoencoderKL.from_pretrained(self.model, torch_dtype=torch.float32)
            except Exception as exc:  # noqa: BLE001 - network, missing files, bad model ids
                self.error = f"cannot load the VAE {self.model!r}: {exc}"
                raise VisionError(self.error) from exc
            self.vae = vae.to(device).eval()
            self.device = device
            self.scaling = float(getattr(self.vae.config, "scaling_factor", 0.18215) or 0.18215)
            self.error = None

    def _to_tensor(self, image, size: int):
        import torch

        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = raw.to(torch.float32).view(size, size, 3).permute(2, 0, 1).unsqueeze(0)
        return (x / 127.5 - 1.0).to(self.device)

    def encode(self, image, size: int) -> tuple[bytes, tuple[int, int, int]]:
        import torch

        self.load()
        Image = _pil()
        image = image.resize((size, size), Image.LANCZOS)
        with torch.no_grad():
            posterior = self.vae.encode(self._to_tensor(image, size))
            latent = posterior.latent_dist.mean * self.scaling  # the deterministic inversion: no sampling noise
            q = torch.round(torch.clamp(latent / LATENT_SCALE, -1.0, 1.0) * 127.0).to(torch.int8)
        values = q.flatten().tolist()
        return struct.pack(f"{len(values)}b", *values), tuple(int(v) for v in latent.shape[1:])

    def decode(self, payload: bytes, width: int, height: int):
        import torch

        self.load()
        Image = _pil()
        shape = self.latent_shape(width, height)
        values = struct.unpack(f"{len(payload)}b", payload)
        z = torch.tensor(values, dtype=torch.float32, device=self.device).view(1, *shape) / 127.0 * LATENT_SCALE
        with torch.no_grad():
            sample = self.vae.decode(z / self.scaling).sample  # the forward image process
            pixels = ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        data = bytes(pixels[0].permute(1, 2, 0).contiguous().flatten().tolist())
        return Image.frombytes("RGB", (int(pixels.shape[3]), int(pixels.shape[2])), data)


_ENCODERS: dict[str, Any] = {}
_ENCODERS_LOCK = threading.Lock()


def _instance(name: str) -> Any:
    with _ENCODERS_LOCK:
        enc = _ENCODERS.get(name)
        if enc is None:
            enc = _ENCODERS[name] = SDVaeEncoder() if name == "sd" else TinyEncoder()
        return enc


def get_encoder(name: str = "auto") -> Any:
    """The encoder ``name`` (``sd`` / ``tiny``), or with ``auto`` the diffusion VAE when it loads, else the stand-in."""
    key = (name or "auto").strip().lower()
    if key not in ENCODERS:
        raise VisionError(f"unknown image encoder {name!r}; expected one of: {', '.join(ENCODERS)}")
    if key == "tiny":
        _pil()
        return _instance("tiny")
    if key == "sd":
        enc = _instance("sd")
        enc.load()
        return enc
    _pil()
    if SDVaeEncoder.importable():
        enc = _instance("sd")
        if enc.vae is not None:
            return enc
        if enc.error is None:  # try once per process; a failed download is not retried on every image
            try:
                enc.load()
                return enc
            except VisionError:
                pass  # the weights are not available: the stand-in takes over (describe() reports the error)
    return _instance("tiny")


def describe() -> dict:
    """What is available: the dependencies, the configured VAE and which encoder ``auto`` resolves to."""
    def importable(module: str) -> bool:
        try:
            __import__(module)
        except ImportError:
            return False
        return True

    pillow = importable("PIL")
    sd = _ENCODERS.get("sd")
    torch_ok = importable("torch")
    diffusers_ok = importable("diffusers")
    return {
        "pillow": pillow,
        "torch": torch_ok,
        "diffusers": diffusers_ok,
        "sd_model": DEFAULT_SD_VAE if sd is None else sd.model,
        "sd_loaded": bool(sd is not None and sd.vae is not None),
        "sd_error": None if sd is None else sd.error,
        "encoders": list(ENCODERS),
        "default_size": DEFAULT_SIZE,
        "auto": None if not pillow else (
            "sd" if (sd is not None and sd.vae is not None) or (sd is None and torch_ok and diffusers_ok)
            or (sd is not None and sd.error is None and torch_ok and diffusers_ok) else "tiny"
        ),
        "text_format": f"{HEADER}:<encoder>:<w>x<h>:<base64 of one signed byte per latent number>",
    }


def encode_image(data: bytes, size: int = DEFAULT_SIZE, encoder: str = "auto") -> dict:
    """Image bytes -> ``{"text", "encoder", "width", "height", "latent_shape", "bytes", "chars", "source_size"}``."""
    size = _check_size(size)
    image = load_image(data)
    enc = get_encoder(encoder)
    payload, shape = enc.encode(image, size)
    text = pack_text(enc.name, size, size, payload)
    return {
        "text": text,
        "encoder": enc.name,
        "width": size,
        "height": size,
        "latent_shape": list(shape),
        "bytes": len(payload),
        "chars": len(text),
        "source_size": [image.width, image.height],
    }


def decode_text(text: str, encoder: str | None = None) -> dict:
    """Encoded (or predicted) text -> ``{"png", "encoder", "width", "height", "bytes", "repaired"}``.

    ``encoder`` overrides the header (``sd`` latents need the diffusion VAE;
    ``tiny`` thumbnails only Pillow).  A payload that is too short or too
    long - a cut-off or rambling prediction - is padded with zeros or
    truncated, and ``repaired`` says so.
    """
    header_encoder, width, height, payload, repaired = parse_text(text)
    name = (encoder or header_encoder).strip().lower()
    if name not in ("sd", "tiny"):
        raise VisionError(f"unknown image encoder {name!r} in the text; expected 'sd' or 'tiny'")
    if width % 8 or height % 8:
        raise VisionError(f"the image size {width}x{height} is not a multiple of 8")
    enc = get_encoder(name)
    shape = enc.latent_shape(width, height)
    length = shape[0] * shape[1] * shape[2]
    payload, changed = _fit_payload(payload, length)
    image = enc.decode(payload, width, height)
    return {
        "png": _png_bytes(image),
        "encoder": enc.name,
        "width": width,
        "height": height,
        "bytes": length,
        "repaired": repaired or changed,
    }
