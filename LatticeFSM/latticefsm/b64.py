"""Base64: everything as one 64-letter alphabet, so one N-gram model can learn text, images, audio and video alike.

Any bytes - a text's UTF-8, a PNG, a WAV, an MP4 - are written in the standard base64 alphabet (``A-Z a-z 0-9 + /``,
``=`` padding, RFC 4648), three bytes to four letters.  :class:`~latticefsm.ngram.NGram` with ``unit="base64"``
reads every letter as a token and learns which letter comes next; generated letters are decoded back to bytes.

* :func:`encode` / :func:`decode` - the standard encoding, and a lenient decoding (anything outside the alphabet
  ignored, padding optional, a trailing lone letter dropped).
* :func:`encode_prefix` - only the letters whose six bits are all known: a true prefix of the encoding of any
  continuation (``encode("Hel")`` is ``SGVs``, ``encode("Hell")`` ends ``bA==`` but ``encode("Hello")`` ends ``bG8=``;
  the prefix of ``Hell`` is ``SGVsb``), which is what a prompt must be for the model to continue it.
* :func:`file_sequences` - every file under some paths (all of them: text, pictures, sound, video) as base64
  sequences of ``width`` letters, cut at byte boundaries of ``width * 3 / 4`` so no group of three bytes is split.

Hand-written, standard library only, the same as ``b64.rs`` letter for letter.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

__all__ = ["ALPHABET", "CHUNK_BYTES", "decode", "encode", "encode_prefix", "file_sequences", "files_under"]

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
"""The 64 letters, in value order; ``=`` pads."""

CHUNK_BYTES = 57
"""Bytes per sequence when a file is cut up: 76 letters, a MIME line, never splitting a group of three bytes."""

_VALUE = {c: i for i, c in enumerate(ALPHABET)}


def encode(data: bytes) -> str:
    """Standard base64, padded."""
    out = []
    for i in range(0, len(data) - len(data) % 3, 3):
        n = data[i] << 16 | data[i + 1] << 8 | data[i + 2]
        out += [ALPHABET[n >> 18], ALPHABET[n >> 12 & 63], ALPHABET[n >> 6 & 63], ALPHABET[n & 63]]
    rest = len(data) % 3
    if rest == 1:
        n = data[-1] << 16
        out += [ALPHABET[n >> 18], ALPHABET[n >> 12 & 63], "=", "="]
    elif rest == 2:
        n = data[-2] << 16 | data[-1] << 8
        out += [ALPHABET[n >> 18], ALPHABET[n >> 12 & 63], ALPHABET[n >> 6 & 63], "="]
    return "".join(out)


def encode_prefix(data: bytes) -> str:
    """The letters whose six bits are all known (``8 * len(data) // 6`` of them): a prefix of the encoding of every
    continuation of ``data``."""
    return encode(data)[:8 * len(data) // 6]


def decode(text: str) -> bytes:
    """Base64 back to bytes, leniently: letters outside the alphabet (and ``=``) ignored, a last group of two or three
    letters read as one or two bytes, a last lone letter dropped."""
    values = [_VALUE[c] for c in text if c in _VALUE]
    out = bytearray()
    for i in range(0, len(values) - len(values) % 4, 4):
        n = values[i] << 18 | values[i + 1] << 12 | values[i + 2] << 6 | values[i + 3]
        out += bytes([n >> 16, n >> 8 & 255, n & 255])
    rest = values[len(values) - len(values) % 4:]
    if len(rest) >= 2:
        n = rest[0] << 18 | rest[1] << 12 | (rest[2] << 6 if len(rest) == 3 else 0)
        out.append(n >> 16)
        if len(rest) == 3:
            out.append(n >> 8 & 255)
    return bytes(out)


def files_under(paths: Iterable[str]) -> list[str]:
    """Every file under ``paths`` - any kind - files given as they are, directories walked in name order (hidden files
    and ``README.md`` left out)."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs.sort()
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                files += [os.path.join(root, n) for n in sorted(names) if not n.startswith(".") and n != "README.md"]
        elif os.path.exists(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def file_sequences(paths: Iterable[str], width: int = 76) -> list[tuple[str, list[str]]]:
    """Every file under ``paths`` as ``(path, sequences)``: its bytes base64-encoded, ``width`` letters a sequence
    (``width`` is rounded down to a multiple of 4), the last one shorter and padded."""
    step = max(1, width // 4) * 3
    out = []
    for path in files_under(paths):
        with open(path, "rb") as f:
            data = f.read()
        out.append((path, [encode(data[i:i + step]) for i in range(0, len(data), step)]))
    return out
