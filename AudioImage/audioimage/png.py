"""PNG in and out, written from scratch on :mod:`zlib` and :mod:`struct`.

The package needs a picture format that is lossless (a JPEG would smear the
spectrum and the decode would hear the smear), that can hold **16 bits per
pixel** (256 grey levels is a coarse ruler for amplitude - 65536 is not), and
that can carry the settings the encode used.  PNG does all three, and the
parts of it this package needs are small enough to implement directly.

Written: greyscale, RGB and RGBA, each at 8 or 16 bits, with adaptive scanline
filtering (each row is written with whichever of the five PNG filters predicts
it best, which is most of what makes a PNG small).  16-bit colour is what lets
a picture carry a full-precision magnitude *and* a phase, and the fourth
channel is what lets it carry the correction that makes the round trip exact
(see :mod:`audioimage.codec`).
Read: the same, plus palette and greyscale+alpha, at 8 or 16 bits.
Interlaced files are rejected rather than half-decoded.

``tEXt`` chunks survive the round trip, which is how :mod:`audioimage.codec`
keeps the sample rate and the window settings inside the picture itself.
"""

from __future__ import annotations

import os
import struct
import zlib
from array import array
from dataclasses import dataclass, field

__all__ = ["Image", "PngError", "read_png", "read_png_bytes", "write_png", "write_png_bytes"]

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MODES = ("L", "RGB", "RGBA")
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class PngError(ValueError):
    """A file that is not a PNG, or a PNG this module will not decode."""


@dataclass
class Image:
    """A raster: ``data`` is flat, row-major, ``width * height * channels`` samples."""

    width: int
    height: int
    mode: str = "L"
    depth: int = 8
    data: array = field(default_factory=lambda: array("H"))
    text: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise PngError(f"image must be at least 1x1, got {self.width}x{self.height}")
        if self.mode not in MODES:
            raise PngError(f"unknown mode {self.mode!r}; choose from {', '.join(MODES)}")
        if self.depth not in (8, 16):
            raise PngError(f"depth must be 8 or 16, got {self.depth}")

        want = self.width * self.height * self.channels
        if len(self.data) == 0:
            self.data = array("H", bytes(2 * want))
        elif len(self.data) != want:
            raise PngError(f"expected {want} samples for {self!r}, got {len(self.data)}")

    @property
    def channels(self) -> int:
        """1 for greyscale, 3 for RGB, 4 with an alpha channel."""
        return {"L": 1, "RGB": 3, "RGBA": 4}[self.mode]

    @property
    def maxval(self) -> int:
        """The largest sample value: 255 or 65535."""
        return (1 << self.depth) - 1

    def get(self, x: int, y: int) -> int:
        """The (first channel of the) pixel at ``x, y``."""
        return self.data[(y * self.width + x) * self.channels]

    def set(self, x: int, y: int, value: int) -> None:
        """Set a greyscale pixel."""
        self.data[(y * self.width + x) * self.channels] = value

    def get_rgb(self, x: int, y: int) -> tuple[int, int, int]:
        """The pixel at ``x, y`` as a triple (greys come back as ``(v, v, v)``)."""
        i = (y * self.width + x) * self.channels
        if self.channels == 1:
            v = self.data[i]
            return (v, v, v)
        return (self.data[i], self.data[i + 1], self.data[i + 2])

    def get_alpha(self, x: int, y: int) -> int:
        """The fourth channel, or fully opaque when there is not one."""
        if self.channels < 4:
            return self.maxval
        return self.data[(y * self.width + x) * 4 + 3]

    def set_alpha(self, x: int, y: int, value: int) -> None:
        """Set the fourth channel (ignored when there is not one)."""
        if self.channels == 4:
            self.data[(y * self.width + x) * 4 + 3] = value

    def set_rgb(self, x: int, y: int, rgb: tuple[int, int, int]) -> None:
        """Set an RGB pixel (on a greyscale image the red channel is used)."""
        i = (y * self.width + x) * self.channels
        if self.channels == 1:
            self.data[i] = rgb[0]
        else:
            self.data[i] = rgb[0]
            self.data[i + 1] = rgb[1]
            self.data[i + 2] = rgb[2]

    def row(self, y: int) -> array:
        """One scanline, as a slice of the flat buffer."""
        stride = self.width * self.channels
        return self.data[y * stride : (y + 1) * stride]

    def fill(self, value: int) -> None:
        """Set every sample."""
        for i in range(len(self.data)):
            self.data[i] = value

    def describe(self) -> dict[str, object]:
        """A summary for ``info`` / ``--json`` output."""
        return {"width": self.width, "height": self.height, "mode": self.mode, "depth": self.depth}

    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, mode={self.mode!r}, depth={self.depth})"


# ---------------------------------------------------------------------------
# scanline filters (PNG spec section 9)
# ---------------------------------------------------------------------------


def _paeth(a: int, b: int, c: int) -> int:
    """The PNG Paeth predictor: whichever of left / above / upper-left is closest."""
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter_row(raw: bytes, prev: bytes, bpp: int) -> bytes:
    """Pick the filter that makes this scanline flattest, and apply it."""
    n = len(raw)
    best: bytes | None = None
    best_score = -1
    for ftype in range(5):
        out = bytearray(n + 1)
        out[0] = ftype
        score = 0
        for i in range(n):
            a = raw[i - bpp] if i >= bpp else 0
            b = prev[i] if prev else 0
            if ftype == 0:
                v = raw[i]
            elif ftype == 1:
                v = raw[i] - a
            elif ftype == 2:
                v = raw[i] - b
            elif ftype == 3:
                v = raw[i] - ((a + b) >> 1)
            else:
                c = prev[i - bpp] if (prev and i >= bpp) else 0
                v = raw[i] - _paeth(a, b, c)
            v &= 0xFF
            out[i + 1] = v
            score += v if v < 128 else 256 - v  # sum of absolute signed values
        if best is None or score < best_score:
            best, best_score = bytes(out), score
    return best or b""


def _unfilter_row(line: bytearray, prev: bytes, ftype: int, bpp: int) -> bytes:
    """Undo one scanline filter in place."""
    n = len(line)
    if ftype == 0:
        return bytes(line)
    for i in range(n):
        a = line[i - bpp] if i >= bpp else 0
        b = prev[i] if prev else 0
        if ftype == 1:
            line[i] = (line[i] + a) & 0xFF
        elif ftype == 2:
            line[i] = (line[i] + b) & 0xFF
        elif ftype == 3:
            line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
        elif ftype == 4:
            c = prev[i - bpp] if (prev and i >= bpp) else 0
            line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise PngError(f"unknown scanline filter {ftype}")
    return bytes(line)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def write_png_bytes(image: Image, compression: int = 9) -> bytes:
    """Encode ``image`` as a complete PNG file."""
    color_type = {"L": 0, "RGB": 2, "RGBA": 6}[image.mode]
    channels = image.channels
    sample_bytes = image.depth // 8
    bpp = channels * sample_bytes
    stride = image.width * bpp
    maxval = image.maxval

    raw = bytearray()
    prev = b""
    flat = image.data
    for y in range(image.height):
        base = y * image.width * channels
        line = bytearray(stride)
        if image.depth == 8:
            for i in range(image.width * channels):
                v = flat[base + i]
                line[i] = 0 if v < 0 else (maxval if v > maxval else v)
        else:
            for i in range(image.width * channels):
                v = flat[base + i]
                v = 0 if v < 0 else (maxval if v > maxval else v)
                line[2 * i] = (v >> 8) & 0xFF
                line[2 * i + 1] = v & 0xFF
        raw += _filter_row(bytes(line), prev, bpp)
        prev = bytes(line)

    out = bytearray(SIGNATURE)
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", image.width, image.height, image.depth, color_type, 0, 0, 0))
    for key, value in image.text.items():
        key_b = key.encode("latin-1", "replace")[:79]
        out += _chunk(b"tEXt", key_b + b"\x00" + value.encode("latin-1", "replace"))
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), compression))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def write_png(path: str | os.PathLike[str], image: Image, compression: int = 9) -> str:
    """Write ``image`` to ``path`` and return the path."""
    blob = write_png_bytes(image, compression)
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return str(path)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_png_bytes(data: bytes) -> Image:
    """Decode a PNG held in memory into an :class:`Image` (mode ``L`` or ``RGB``)."""
    if len(data) < 8 or data[:8] != SIGNATURE:
        raise PngError("not a PNG file (bad signature)")
    pos = 8
    header: tuple[int, ...] | None = None
    idat = bytearray()
    palette: bytes = b""
    text: dict[str, str] = {}
    while pos + 8 <= len(data):
        (size,) = struct.unpack_from(">I", data, pos)
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + size]
        pos += 12 + size  # length + tag + body + crc
        if tag == b"IHDR":
            if size < 13:
                raise PngError("truncated IHDR")
            header = struct.unpack_from(">IIBBBBB", body, 0)
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat += body
        elif tag == b"tEXt":
            key, _, value = body.partition(b"\x00")
            text[key.decode("latin-1")] = value.decode("latin-1")
        elif tag == b"IEND":
            break
    if header is None:
        raise PngError("no IHDR chunk")
    width, height, depth, color_type, _comp, _filt, interlace = header
    if interlace:
        raise PngError("interlaced PNGs are not supported")
    if color_type not in _CHANNELS:
        raise PngError(f"unknown colour type {color_type}")
    if depth not in (1, 2, 4, 8, 16):
        raise PngError(f"unsupported bit depth {depth}")
    if depth < 8 and color_type != 3:
        raise PngError(f"bit depth {depth} is only supported for palette images")
    if width < 1 or height < 1:
        raise PngError(f"bad image size {width}x{height}")

    src_channels = _CHANNELS[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise PngError(f"the image data cannot be decompressed: {exc}") from exc

    bits_per_pixel = src_channels * depth
    stride = (width * bits_per_pixel + 7) // 8
    bpp = max(1, bits_per_pixel // 8)
    if len(raw) < height * (stride + 1):
        raise PngError(f"image data is short: {len(raw)} bytes, expected {height * (stride + 1)}")

    # RGBA keeps its fourth channel; greyscale+alpha and palette do not need one
    out_mode = "RGBA" if color_type == 6 else ("RGB" if color_type in (2, 3) else "L")
    # a palette is always 8-bit entries, and anything narrower than a byte is
    # widened to one; everything else keeps the depth it was written at, so a
    # 16-bit colour image survives the round trip
    out_depth = 8 if (color_type == 3 or depth < 8) else depth
    out_channels = {"L": 1, "RGB": 3, "RGBA": 4}[out_mode]
    image = Image(width, height, out_mode, out_depth)
    flat = image.data

    prev = b""
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        line = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        cur = _unfilter_row(line, prev, ftype, bpp)
        prev = cur
        base = y * width * out_channels
        if color_type == 3:  # palette lookup
            for x in range(width):
                if depth == 8:
                    idx = cur[x]
                else:
                    bit = x * depth
                    idx = (cur[bit >> 3] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1)
                off = idx * 3
                if off + 2 >= len(palette):
                    raise PngError(f"palette index {idx} is outside the PLTE chunk")
                flat[base + x * 3] = palette[off]
                flat[base + x * 3 + 1] = palette[off + 1]
                flat[base + x * 3 + 2] = palette[off + 2]
            continue
        for x in range(width):
            for c in range(out_channels):
                src = min(c, src_channels - 1) if out_channels > src_channels else c
                i = (x * src_channels + src) * (depth // 8)
                flat[base + x * out_channels + c] = (cur[i] << 8) | cur[i + 1] if depth == 16 else cur[i]
    image.text = text
    return image


def read_png(path: str | os.PathLike[str]) -> Image:
    """Read a PNG file from disk."""
    with open(path, "rb") as fh:
        return read_png_bytes(fh.read())
