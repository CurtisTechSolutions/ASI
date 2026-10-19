"""Reading a corpus: every file under some paths - text, markup, documents and archives of any size - as lines, or
as bytes.

What a file is, is read off its name:

=========  ==========================================================================================================
kind       files, and what is read from them
=========  ==========================================================================================================
text       ``.txt .md .text .csv .tsv .log .rst .tex .org .srt .vtt`` - line by line, streamed
jsonl      ``.jsonl`` - one JSON value a line, its ``text`` (or ``content``, ``sentence``, ``body``), streamed
json       ``.json`` - the same of one JSON document
markup     ``.html .htm .xhtml .xml`` - the text between the tags (:func:`strip_markup`): a block tag (``p``, ``br``,
           ``div``, ``li``, ``h1``...) ends a line, scripts, styles and comments are dropped, entities decoded
docx       ``word/document.xml`` of a Word file, as markup; ``odt``: ``content.xml``; ``epub``: every ``.xhtml`` /
           ``.html`` / ``.htm`` inside, in the archive's order
zip        every member, in the central directory's order (ZIP64 too: any size; stored or deflated members)
tar        every regular member in order (``.tar``; ``.tar.gz`` / ``.tgz`` gzipped)
gz         the file inside, whose kind is the name without ``.gz`` (``notes.txt.gz`` is text)
=========  ==========================================================================================================

Archives are read member by member and text a line at a time (a line longer than :data:`MAX_LINE` bytes is cut
there), so neither the size of an archive nor of a file inside one is bounded by memory; markup, JSON documents
and the documents inside a docx / odt / epub are read whole, and an archive inside an archive is followed
:data:`MAX_DEPTH` deep (a zip inside one is read into memory first).  Directories are walked in name order;
hidden files and directories, ``README.md`` and files of no known kind are left out of a walk (a file named
directly is read as text whatever its name), and so are hidden members (``.``-anything, ``__MACOSX``) and
encrypted or otherwise-compressed zip members.

For ``base64`` (:func:`byte_documents`) every file is its bytes - pictures, sound, video, anything - except that
archives (``zip``, ``tar``, ``tar.gz``, ``tgz``, ``gz``) give their members' bytes instead.

The Rust crate's ``corpus.rs`` reads the same files into the same lines and bytes, member for member.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from typing import BinaryIO

__all__ = ["MAX_DEPTH", "MAX_LINE", "byte_documents", "file_kind", "files", "strip_markup", "text_documents",
           "text_lines"]

TEXT = (".txt", ".md", ".text", ".csv", ".tsv", ".log", ".rst", ".tex", ".org", ".srt", ".vtt")
MARKUP = (".html", ".htm", ".xhtml", ".xml")
MAX_LINE = 1 << 20
"""Bytes a line is read in at most: a longer one is cut there."""
MAX_DEPTH = 4
"""How deep archives inside archives are followed."""

BLOCK = frozenset("p h h1 h2 h3 h4 h5 h6 div br li tr title section article blockquote pre table ul ol dt dd hr "
                  "header footer nav aside caption figcaption".split())
SPACE = frozenset(("td", "th", "tab"))
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " ", "hellip": "\u2026",
             "mdash": "\u2014", "ndash": "\u2013", "lsquo": "\u2018", "rsquo": "\u2019", "ldquo": "\u201c",
             "rdquo": "\u201d", "copy": "\u00a9"}
_ENTITY = re.compile(r"&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);")


def file_kind(name: str) -> str | None:
    """What a file is, by its name: ``text``, ``jsonl``, ``json``, ``markup``, ``docx``, ``odt``, ``epub``, ``zip``,
    ``tar``, ``tgz``, ``gz``, or ``None``."""
    n = name.lower()
    if n.endswith((".tar.gz", ".tgz")):
        return "tgz"
    for ext, kind in ((".gz", "gz"), (".zip", "zip"), (".tar", "tar"), (".jsonl", "jsonl"), (".json", "json"),
                      (".docx", "docx"), (".odt", "odt"), (".epub", "epub")):
        if n.endswith(ext):
            return kind
    if n.endswith(MARKUP):
        return "markup"
    if n.endswith(TEXT):
        return "text"
    return None


def _hidden(member: str) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in member.split("/"))


def files(paths: Iterable[str], everything: bool = False) -> Iterator[tuple[str, bool]]:
    """Every file under ``paths`` as ``(path, named)``: a file named directly as it is (``named``), a directory walked in
    name order - hidden files and directories and ``README.md`` left out, and unless ``everything`` the files of no
    known kind too."""
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for n in sorted(names):
                    if n.startswith(".") or n == "README.md":
                        continue
                    if everything or file_kind(n) is not None:
                        yield os.path.join(root, n), False
        elif os.path.exists(p):
            yield p, True
        else:
            raise FileNotFoundError(p)


# ---- archives ------------------------------------------------------------------------------------------------------

def _zip_members(zf: zipfile.ZipFile) -> Iterator[zipfile.ZipInfo]:
    for info in zf.infolist():
        if (info.is_dir() or _hidden(info.filename) or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            continue
        yield info


def _tar_members(stream: BinaryIO) -> Iterator[tuple[str, BinaryIO]]:
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if m.isfile() and not _hidden(m.name):
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, f


def _open_zip(stream: BinaryIO) -> zipfile.ZipFile:
    if stream.seekable():
        return zipfile.ZipFile(stream)
    return zipfile.ZipFile(io.BytesIO(stream.read()))


def _expand(stream: BinaryIO, name: str, label: str, depth: int, kind: str | None,
            leaf) -> Iterator[tuple[str, str | None, BinaryIO]]:
    """``stream`` (a file called ``name``, known as ``label``) with its archives opened: ``leaf(kind)`` says whether a
    member of that kind is wanted as it is; yields ``(label, kind, stream)``."""
    if kind in ("zip", "tar", "tgz", "gz") and depth < MAX_DEPTH:
        if kind == "gz":
            with gzip.GzipFile(fileobj=stream, mode="rb") as g:
                inner = name[:-3]
                yield from _expand(g, inner, label, depth + 1, file_kind(inner), leaf)
        elif kind == "tgz":
            with gzip.GzipFile(fileobj=stream, mode="rb") as g:
                for member, f in _tar_members(g):
                    yield from _expand(f, member, f"{label}/{member}", depth + 1, file_kind(member), leaf)
        elif kind == "tar":
            for member, f in _tar_members(stream):
                yield from _expand(f, member, f"{label}/{member}", depth + 1, file_kind(member), leaf)
        else:
            with _open_zip(stream) as zf:
                for info in _zip_members(zf):
                    with zf.open(info) as f:
                        yield from _expand(f, info.filename, f"{label}/{info.filename}", depth + 1,
                                           file_kind(info.filename), leaf)
    elif kind not in ("zip", "tar", "tgz", "gz") and leaf(kind):
        yield label, kind, stream


def text_documents(paths: Iterable[str]) -> Iterator[tuple[str, str, str, BinaryIO]]:
    """Every text document under ``paths`` as ``(path, label, kind, stream)``: ``path`` the file it came from,
    ``label`` the file or, inside an archive, ``archive/member``.  A stream is good until the next document."""
    for path, named in files(paths):
        kind = file_kind(path)
        if kind is None and named:
            kind = "text"
        with open(path, "rb") as f:
            for label, k, stream in _expand(f, os.path.basename(path), path, 0, kind, lambda k: k is not None):
                yield path, label, k, stream


def byte_documents(paths: Iterable[str]) -> Iterator[tuple[str, str, BinaryIO]]:
    """Every file under ``paths`` as ``(path, label, stream)`` of its bytes, archives opened into their members."""
    for path, _ in files(paths, everything=True):
        with open(path, "rb") as f:
            for label, _, stream in _expand(f, os.path.basename(path), path, 0, file_kind(path), lambda k: True):
                yield path, label, stream


def read_chunks(stream: BinaryIO, size: int) -> Iterator[bytes]:
    """``stream`` ``size`` bytes at a time, the last chunk shorter."""
    while True:
        chunk = b""
        while len(chunk) < size:
            more = stream.read(size - len(chunk))
            if not more:
                break
            chunk += more
        if not chunk:
            return
        yield chunk
        if len(chunk) < size:
            return


# ---- text ----------------------------------------------------------------------------------------------------------

def _json_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "sentence", "body"):
            if isinstance(value.get(key), str):
                return value[key]
        return " ".join(v for v in value.values() if isinstance(v, str))
    if isinstance(value, list):
        return "\n".join(_json_text(v) for v in value)
    return ""


def _tag_name(tag: str) -> str:
    """A tag's local name, lower-cased (ASCII): ``</w:p>`` is ``p``."""
    t = tag.lstrip("/")
    end = 0
    while end < len(t) and t[end] not in " \t\n\r\f\v/":
        end += 1
    name = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in t[:end])
    return name.rsplit(":", 1)[-1]


def _entity(m: re.Match) -> str:
    body = m.group(1)
    if body[0] == "#":
        v = int(body[2:], 16) if body[1] in "xX" else int(body[1:])
        return chr(v) if 0 < v <= 0x10FFFF and not 0xD800 <= v <= 0xDFFF else m.group(0)
    return _ENTITIES.get(body, m.group(0))


def strip_markup(s: str) -> str:
    """The text of HTML or XML: a block tag ends a line (``\\n``), a table cell or tab is a space, any other tag is
    nothing; comments, declarations, processing instructions, scripts and styles dropped, CDATA kept, then the
    entities ``&amp; &lt; &gt; &quot; &apos; &nbsp; &hellip; &mdash; &ndash; &lsquo; &rsquo; &ldquo; &rdquo; &copy;`` and
    ``&#N;`` / ``&#xH;`` decoded (any other left as it is)."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        j = s.find("<", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        if s.startswith("<!--", j):
            k = s.find("-->", j + 4)
            i = n if k < 0 else k + 3
            continue
        if s.startswith("<![CDATA[", j):
            k = s.find("]]>", j + 9)
            out.append(s[j + 9:n if k < 0 else k])
            i = n if k < 0 else k + 3
            continue
        k = s.find(">", j + 1)
        if k < 0:
            out.append(s[j:])
            break
        tag = s[j + 1:k]
        i = k + 1
        if tag[:1] in ("!", "?"):
            continue
        name = _tag_name(tag)
        if name in ("script", "style") and not tag.startswith("/") and not tag.endswith("/"):
            # skip to the end of the matching close tag
            while True:
                c = s.find("</", i)
                if c < 0:
                    i = n
                    break
                e = s.find(">", c + 2)
                if e < 0:
                    i = n
                    break
                i = e + 1
                if _tag_name(s[c + 1:e]) == name:
                    break
            continue
        if name in BLOCK:
            out.append("\n")
        elif name in SPACE:
            out.append(" ")
    return _ENTITY.sub(_entity, "".join(out))


def _markup_of_zip(data: bytes, kind: str) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [i.filename for i in _zip_members(zf)]
        if kind == "docx":
            wanted = [n for n in names if n == "word/document.xml"]
        elif kind == "odt":
            wanted = [n for n in names if n == "content.xml"]
        else:
            wanted = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]
        return "\n".join(strip_markup(zf.read(n).decode("utf-8", "replace")) for n in wanted)


def _clean(lines: Iterable[str]) -> Iterator[str]:
    for ln in lines:
        ln = ln.strip()
        if ln:
            yield ln


def text_lines(kind: str, stream: BinaryIO) -> Iterator[str]:
    """The non-empty lines (stripped) of a document of ``kind``."""
    if kind in ("text", "jsonl"):
        while True:
            raw = stream.readline(MAX_LINE)
            if not raw:
                return
            for ln in raw.decode("utf-8", "replace").splitlines():
                if kind == "text":
                    ln = ln.strip()
                    if ln:
                        yield ln
                    continue
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield from _clean(_json_text(json.loads(ln)).splitlines())
                except ValueError:
                    yield ln
        return
    data = stream.read()
    if kind == "json":
        text = data.decode("utf-8", "replace")
        try:
            yield from _clean(_json_text(json.loads(text)).splitlines())
        except ValueError:
            yield from _clean(text.splitlines())
    elif kind == "markup":
        yield from _clean(strip_markup(data.decode("utf-8", "replace")).splitlines())
    elif kind in ("docx", "odt", "epub"):
        try:
            text = _markup_of_zip(data, kind)
        except zipfile.BadZipFile:
            return
        yield from _clean(text.splitlines())
