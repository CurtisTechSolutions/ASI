"""Text, audio, images and video as symbols a network of machines can read.

A machine reads a string of symbols; this module turns other things into one,
standard library only, in exact integer arithmetic so the Rust crate
(``media.rs``) turns every file into the same symbols:

* **text** - lower-cased characters: ``a-z``, ``0-9``, ``_`` for a space and
  ``. , ? ! ' -``; anything else dropped (:func:`text_symbols`).
* **audio** - a WAV file heard as acoustic units by this repository's phonetic
  tokenizer (``../PhoneticTokenizer``, ``AcousticTokenizer``): 64 learned
  sound units, ``q0``-``q63``, a run of one unit folded into one.
* **images** - PNG (8-bit, any colour type, not interlaced), PPM / PGM (``P2``
  ``P3`` ``P5`` ``P6``) or BMP (24- or 32-bit, uncompressed); averaged down to
  a ``grid x grid`` of cells by brightness (``Y = (299 R + 587 G + 114 B) //
  1000``), each cell one of ``levels`` symbols ``L0`` (dark) .. ``L3``, read
  row by row or from the centre outward, ring by ring.
* **image differences** - two grids of the same size, cell by cell: ``D0`` for
  under 8 levels of 255 of change, ``D+1`` / ``D-1`` under 48, ``D+2`` /
  ``D-2`` beyond: what moved, and which way the light went.
* **video** - frames (a directory of images, or a video file ffmpeg splits
  into frames, if ffmpeg is installed) and its audio (``audio.wav`` in the
  directory, or the file's own track): ``|`` before every frame, the first
  frame (and every ``keyframe``-th) as an image, every other as its
  difference from the frame before, then the audio units that sound during
  it - a keyframe-and-delta stream, the shape video codecs have.
"""

from __future__ import annotations

import os
import random
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Sequence

__all__ = ["DIFFS", "IMAGE_EXTENSIONS", "TEXT_SYMBOLS", "audio_symbols", "audio_units", "cells", "diff_symbols",
           "image_symbols", "level_symbols", "make_dataset", "order", "read_image", "read_labels", "text_symbols",
           "unit_names", "video_frames", "video_symbols"]

TEXT_SYMBOLS = list("abcdefghijklmnopqrstuvwxyz0123456789") + ["_", ".", ",", "?", "!", "'", "-"]
"""What text is read as: 43 symbols."""

DIFFS = ["D-2", "D-1", "D0", "D+1", "D+2"]
"""What a change in a cell is read as, darker to lighter."""

IMAGE_EXTENSIONS = (".png", ".ppm", ".pgm", ".pnm", ".bmp")


# ---- text ----------------------------------------------------------------------------------------------------------

def text_symbols(text: str) -> list[str]:
    """``text`` lower-cased, a space as ``_``, every character not in :data:`TEXT_SYMBOLS` dropped."""
    known = set(TEXT_SYMBOLS)
    out = []
    for c in text.lower():
        c = "_" if c.isspace() else c
        if c in known:
            out.append(c)
    return out


# ---- images --------------------------------------------------------------------------------------------------------

def read_image(source: str | bytes) -> tuple[int, int, list[int]]:
    """An image file (a path, or its bytes) as ``(width, height, brightness)``: one 0-255 value per pixel, row by row,
    ``Y = (299 R + 587 G + 114 B) // 1000`` (alpha ignored)."""
    data = source if isinstance(source, bytes) else _read(source)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png(data)
    if data[:1] == b"P" and data[1:2] in b"2356":
        return _pnm(data)
    if data[:2] == b"BM":
        return _bmp(data)
    raise ValueError("not a PNG, PPM / PGM or BMP image")


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _y(r: int, g: int, b: int) -> int:
    return (299 * r + 587 * g + 114 * b) // 1000


def _png(data: bytes) -> tuple[int, int, list[int]]:
    pos, idat, palette = 8, b"", None
    width = height = depth = ctype = 0
    while pos + 8 <= len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or interlace:
                raise ValueError("only 8-bit, non-interlaced PNGs are read")
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(idat)
    stride = width * channels
    out, prev, at = [], bytearray(stride), 0
    for _ in range(height):
        ftype, line = raw[at], bytearray(raw[at + 1:at + 1 + stride])
        at += 1 + stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b, c = prev[i], prev[i - channels] if i >= channels else 0
            if ftype == 1:
                line[i] = (line[i] + a) & 255
            elif ftype == 2:
                line[i] = (line[i] + b) & 255
            elif ftype == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        for x in range(width):
            px = line[x * channels:(x + 1) * channels]
            if ctype == 3:
                k = px[0] * 3
                out.append(_y(palette[k], palette[k + 1], palette[k + 2]))
            elif channels >= 3:
                out.append(_y(px[0], px[1], px[2]))
            else:
                out.append(px[0])
        prev = line
    return width, height, out


def _pnm(data: bytes) -> tuple[int, int, list[int]]:
    kind = data[1:2]
    fields, pos = [], 2
    need = 3
    while len(fields) < need:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    width, height, top = fields
    pos += 1
    channels = 3 if kind in b"36" else 1
    if kind in b"56":
        if top > 255:
            raise ValueError("only 8-bit PPM / PGM files are read")
        values = list(data[pos:pos + width * height * channels])
    else:
        values = [int(v) for v in data[pos:].split()][:width * height * channels]
    if top != 255:
        values = [v * 255 // top for v in values]
    if channels == 1:
        return width, height, values
    return width, height, [_y(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]


def _bmp(data: bytes) -> tuple[int, int, list[int]]:
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bits, compression = struct.unpack_from("<HI", data, 28)
    if bits not in (24, 32) or compression not in (0, 3):
        raise ValueError("only 24- or 32-bit uncompressed BMPs are read")
    bpp = bits // 8
    stride = (width * bpp + 3) // 4 * 4
    rows = []
    for r in range(abs(height)):
        at = offset + r * stride
        rows.append([_y(data[at + x * bpp + 2], data[at + x * bpp + 1], data[at + x * bpp]) for x in range(width)])
    if height > 0:
        rows.reverse()
    return width, abs(height), [v for row in rows for v in row]


def cells(image: tuple[int, int, list[int]], grid: int) -> list[int]:
    """The image averaged down to ``grid x grid`` cells, row by row: each the integer mean of its pixels."""
    width, height, px = image
    out = []
    for cy in range(grid):
        y0, y1 = cy * height // grid, max(cy * height // grid + 1, (cy + 1) * height // grid)
        for cx in range(grid):
            x0, x1 = cx * width // grid, max(cx * width // grid + 1, (cx + 1) * width // grid)
            total = count = 0
            for y in range(y0, min(y1, height)):
                row = y * width
                for x in range(x0, min(x1, width)):
                    total += px[row + x]
                    count += 1
            out.append(total // count if count else 0)
    return out


def order(grid: int, how: str = "rows") -> list[int]:
    """The order cells are read in: ``rows``, or ``center-out`` - ring by ring from the middle, each ring top to
    bottom, left to right."""
    if how == "rows":
        return list(range(grid * grid))
    if how != "center-out":
        raise ValueError(f"order is rows or center-out, got {how!r}")
    c2 = grid - 1  # twice the centre, so even grids have a centre between cells
    return sorted(range(grid * grid),
                  key=lambda i: (max(abs(2 * (i // grid) - c2), abs(2 * (i % grid) - c2)), i // grid, i % grid))


def level_symbols(levels: int) -> list[str]:
    return [f"L{i}" for i in range(levels)]


def image_symbols(grid_values: Sequence[int], levels: int, how: str = "rows") -> list[str]:
    grid = int(round(len(grid_values) ** 0.5))
    return [f"L{grid_values[i] * levels // 256}" for i in order(grid, how)]


def diff_symbols(before: Sequence[int], after: Sequence[int], how: str = "rows") -> list[str]:
    grid = int(round(len(after) ** 0.5))
    out = []
    for i in order(grid, how):
        d = after[i] - before[i]
        m = abs(d)
        k = 0 if m < 8 else 1 if m < 48 else 2
        out.append(DIFFS[2 + (k if d > 0 else -k)])
    return out


# ---- audio ---------------------------------------------------------------------------------------------------------

_ACOUSTIC = None


def _acoustic():
    global _ACOUSTIC
    if _ACOUSTIC is None:
        from .phonetic import _phonetok
        _ACOUSTIC = _phonetok().AcousticTokenizer()
    return _ACOUSTIC


def unit_names() -> list[str]:
    """The acoustic units, ``q0`` .. ``q63``."""
    return list(_acoustic().vocab)


def audio_units(source: str | bytes) -> list[str]:
    """A WAV file (a path, or its bytes) heard as acoustic units."""
    data = source if isinstance(source, bytes) else _read(source)
    return list(_acoustic().hear(data))


def audio_symbols(source: str | bytes) -> list[str]:
    return audio_units(source)


# ---- video ---------------------------------------------------------------------------------------------------------

def video_frames(source: str, fps: int = 8, size: int = 64) -> tuple[list[str], str | None, str | None]:
    """A video as ``(frame image paths, audio WAV path or None, temporary directory to remove or None)``: a directory's
    images in name order and its ``audio.wav``; a video file split by ffmpeg into ``fps`` frames a second of ``size x
    size`` and a 16 kHz mono track."""
    if os.path.isdir(source):
        frames = sorted(os.path.join(source, f) for f in os.listdir(source)
                        if f.lower().endswith(IMAGE_EXTENSIONS))
        audio = os.path.join(source, "audio.wav")
        return frames, audio if os.path.exists(audio) else None, None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(f"{source}: a video file needs ffmpeg; or give a directory of frames (and audio.wav)")
    tmp = tempfile.mkdtemp(prefix="latticefsm-video-")
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", source, "-vf", f"fps={fps},scale={size}:{size}",
                    os.path.join(tmp, "frame%05d.ppm")], check=True)
    audio = os.path.join(tmp, "audio.wav")
    got = subprocess.run([ffmpeg, "-loglevel", "error", "-i", source, "-vn", "-ac", "1", "-ar", "16000",
                          "-c:a", "pcm_s16le", audio], capture_output=True)
    frames = sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".ppm"))
    return frames, audio if got.returncode == 0 and os.path.exists(audio) else None, tmp


def video_symbols(source: str, grid: int = 8, levels: int = 4, how: str = "rows", keyframe: int = 0,
                  frames: int = 16, audio: bool = True, fps: int = 8) -> list[str]:
    """``|`` before each of the first ``frames`` frames; the first (and every ``keyframe``-th) as an image, the rest
    as differences; after each frame the audio units that sound during it (the units split evenly over the frames)."""
    paths, wav, tmp = video_frames(source, fps=fps)
    try:
        paths = paths[:frames] if frames else paths
        grids = [cells(read_image(p), grid) for p in paths]
        units = audio_units(wav) if audio and wav else []
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    out, n = [], len(grids)
    for t, g in enumerate(grids):
        out.append("|")
        if t == 0 or (keyframe and t % keyframe == 0):
            out += image_symbols(g, levels, how)
        else:
            out += diff_symbols(grids[t - 1], g, how)
        out += units[t * len(units) // n:(t + 1) * len(units) // n]
    return out


# ---- labelled data -------------------------------------------------------------------------------------------------

_YES = {"1", "yes", "true", "accept", "y", "t"}
_NO = {"0", "no", "false", "reject", "n", "f"}


def read_labels(path: str, resolve: bool = True) -> list[tuple[str, str]]:
    """A ``input<TAB>label`` file: each line an input (a path relative to the file's directory, when ``resolve``) and
    its label (kept as text: a verdict ``1/0 yes/no true/false``, or an English target)."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" not in line:
                raise ValueError(f"{path}: each line is input<TAB>label, got {line!r}")
            x, y = line.split("\t", 1)
            x = x.strip()
            if resolve and not os.path.isabs(x):
                x = os.path.join(base, x)
            out.append((x, y.strip()))
    return out


def verdict(label: str) -> bool:
    v = label.strip().lower()
    if v in _YES:
        return True
    if v in _NO:
        return False
    raise ValueError(f"a verdict is 1/0, yes/no or true/false, got {label!r}")


# ---- synthetic data sets -------------------------------------------------------------------------------------------

def _ppm(path: str, width: int, height: int, values: Sequence[int]) -> None:
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode() + bytes(values))


def _speak(path: str, text: str) -> None:
    import importlib
    from .phonetic import _phonetok, tokenizer
    _phonetok()
    synth = importlib.import_module("phonetok.synth")
    pcm = synth.Synthesizer().speak(tokenizer(True).tokens(text))
    with open(path, "wb") as f:
        f.write(synth.wav_bytes(pcm))


def make_dataset(kind: str, out: str, count: int = 40, seed: int = 1) -> str:
    """Write a small labelled data set of ``kind`` into ``out`` with its ``labels.tsv``; returns that file's path.

    * ``text`` - short sentences; 1 when they hold a negation (*not*, *no*, *never*).
    * ``image`` - 32 x 32 stripes; 1 when they run up and down.
    * ``audio`` - spoken nouns (``phonetic.PLURALS``, both forms); 1 when plural.
    * ``video`` - 8 frames of a bright square crossing the picture; 1 when it moves right; each with a spoken
      number as its sound track, which says nothing about the label.
    """
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    rows: list[tuple[str, int]] = []
    if kind == "text":
        words = "the a cat dog sat ran saw big small red old house tree by in on".split()
        for i in range(count):
            s = [rng.choice(words) for _ in range(rng.randint(3, 6))]
            neg = i % 2 == 1
            if neg:
                s.insert(rng.randint(1, len(s)), rng.choice(["not", "no", "never"]))
            rows.append((" ".join(s), int(neg)))
        path = os.path.join(out, "labels.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(f"{x}\t{y}\n" for x, y in rows))
        return path
    if kind == "image":
        for i in range(count):
            vertical, period = i % 2 == 0, rng.randint(4, 8)
            phase = rng.randrange(period)
            px = []
            for y in range(32):
                for x in range(32):
                    on = ((x if vertical else y) + phase) % period < period // 2
                    px.append(max(0, min(255, (200 if on else 40) + rng.randint(-30, 30))))
            name = f"img{i:03d}.ppm"
            _ppm(os.path.join(out, name), 32, 32, px)
            rows.append((name, int(vertical)))
    elif kind == "audio":
        from .phonetic import PLURALS
        for i, (one, many) in enumerate(PLURALS[:max(1, count // 2)]):
            for word, label in ((one, 0), (many, 1)):
                name = f"{word}.wav"
                _speak(os.path.join(out, name), word)
                rows.append((name, label))
    elif kind == "video":
        numbers = ["one", "two", "three", "four"]
        for i in range(count):
            right = i % 2 == 0
            d = os.path.join(out, f"clip{i:03d}")
            os.makedirs(d, exist_ok=True)
            x, y, step = rng.randint(2, 6) if right else rng.randint(18, 22), rng.randint(4, 20), rng.randint(2, 3)
            for t in range(8):
                px = [20] * (32 * 32)
                for yy in range(y, y + 8):
                    for xx in range(x, x + 8):
                        if 0 <= xx < 32 and 0 <= yy < 32:
                            px[yy * 32 + xx] = 230
                _ppm(os.path.join(d, f"frame{t:02d}.ppm"), 32, 32, px)
                x += step if right else -step
            _speak(os.path.join(d, "audio.wav"), rng.choice(numbers))
            rows.append((f"clip{i:03d}", int(right)))
    else:
        raise ValueError(f"kind is text, image, audio or video, got {kind!r}")
    path = os.path.join(out, "labels.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(f"{x}\t{y}\n" for x, y in rows))
    return path
