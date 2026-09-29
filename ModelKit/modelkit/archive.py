"""ZIP archives as training data: the text entries of an archive, unpacked safely.

An uploaded (or ``--data``) ``.zip`` is never written to disk as an archive:
its text entries become ordinary text files / texts.  Rules:

* entry paths never touch the file system - every entry gets a flat name
  ``<archive stem>__<dir>__<file>`` reduced to safe characters (long names
  keep their extension and end in a short hash of the full path);
* directories, ``__MACOSX`` metadata, ``._*`` / ``.DS_Store`` files, nested
  archives, encrypted entries, binary files (a NUL byte in the first 8 KB)
  and empty files are skipped, each with a reason;
* there is no size limit by default (:data:`MAX_ARCHIVE_ENTRIES` and
  :data:`MAX_UNPACKED_BYTES` are ``None``): a source tree with tens of
  thousands of files is fine.  A caller that wants a cap passes
  ``max_entries`` / ``max_bytes`` to :func:`extract_texts`; then the
  declared sizes are checked first and the bytes actually read are counted
  again, so a lying header cannot get past the limit.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass

__all__ = [
    "MAX_ARCHIVE_ENTRIES",
    "MAX_UNPACKED_BYTES",
    "Extracted",
    "Skipped",
    "extract_texts",
    "flat_name",
    "is_zip",
    "zip_texts_from_file",
]

MAX_ARCHIVE_ENTRIES: int | None = None
"""Default entry limit of :func:`extract_texts`: ``None`` = unlimited."""
MAX_UNPACKED_BYTES: int | None = None
"""Default unpacked-size limit of :func:`extract_texts`: ``None`` = unlimited."""
TEXT_SNIFF_BYTES = 8192
MAX_NAME = 128

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_ARCHIVE_SUFFIXES = (".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar")
_NAME_CHARS = frozenset("._- +@()")


@dataclass(slots=True)
class Extracted:
    """One text entry of an archive: its flat upload ``name``, the entry ``path`` and the decoded ``text``."""

    name: str
    path: str
    text: str
    bytes: int


@dataclass(slots=True)
class Skipped:
    path: str
    reason: str

    def to_dict(self) -> dict:
        return {"path": self.path, "reason": self.reason}


def is_zip(name: str, data: bytes) -> bool:
    """Whether an upload is a ZIP archive: the magic bytes decide, the name alone does not."""
    head = bytes(data[:4])
    if head in _ZIP_MAGIC:
        return True
    return False


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in _NAME_CHARS else "_" for ch in text).strip(" .")


def flat_name(archive_name: str, entry_path: str) -> str:
    """``<archive stem>__<dir>__<file>``: a flat, safe upload name for an entry (never a path)."""
    stem = os.path.basename(str(archive_name).replace("\\", "/")).strip()
    if stem.lower().endswith(".zip"):
        stem = stem[:-4]
    parts = [p for p in str(entry_path).replace("\\", "/").split("/") if p and p not in (".", "..")]
    pieces = [_safe(stem)] + [_safe(p) for p in parts]
    name = "__".join(p for p in pieces if p) or "entry"
    if len(name) > MAX_NAME:
        root, ext = os.path.splitext(name)
        ext = ext[:16]
        digest = hashlib.sha1(f"{archive_name}/{entry_path}".encode("utf-8", errors="replace")).hexdigest()[:8]
        name = root[: MAX_NAME - len(ext) - 9] + "-" + digest + ext
    return name


def _skip_reason(info: zipfile.ZipInfo) -> str | None:
    path = info.filename.replace("\\", "/")
    base = path.rstrip("/").rsplit("/", 1)[-1]
    if info.is_dir() or path.endswith("/"):
        return "directory"
    if path.startswith("__MACOSX/") or "/__MACOSX/" in path:
        return "macOS metadata"
    if base.startswith("._") or base == ".DS_Store" or base == "Thumbs.db":
        return "system file"
    if base.lower().endswith(_ARCHIVE_SUFFIXES):
        return "nested archive"
    if info.flag_bits & 0x1:
        return "encrypted"
    return None


def extract_texts(
    data: bytes,
    archive_name: str = "archive.zip",
    *,
    max_entries: int | None = None,
    max_bytes: int | None = None,
) -> tuple[list[Extracted], list[Skipped]]:
    """The text entries of a ZIP archive held in ``data`` (see the module docstring for the rules).

    ``max_entries`` / ``max_bytes`` default to the module constants
    (``None`` = no limit).
    """
    if max_entries is None:
        max_entries = MAX_ARCHIVE_ENTRIES
    if max_bytes is None:
        max_bytes = MAX_UNPACKED_BYTES
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid ZIP archive ({exc})") from None
    with archive:
        infos = archive.infolist()
        if max_entries is not None and len(infos) > max_entries:
            raise ValueError(f"archive holds {len(infos)} entries; the limit is {max_entries}")
        declared = sum(info.file_size for info in infos if not info.is_dir())
        if max_bytes is not None and declared > max_bytes:
            raise ValueError(f"archive would unpack to {declared} bytes; the limit is {max_bytes}")
        extracted: list[Extracted] = []
        skipped: list[Skipped] = []
        used = 0
        for info in infos:
            path = info.filename.replace("\\", "/")
            reason = _skip_reason(info)
            if reason is not None:
                skipped.append(Skipped(path, reason))
                continue
            try:
                with archive.open(info) as fh:
                    payload = fh.read() if max_bytes is None else fh.read(max_bytes - used + 1)
            except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError) as exc:
                skipped.append(Skipped(path, f"unreadable ({exc})"))
                continue
            used += len(payload)
            if max_bytes is not None and used > max_bytes:
                raise ValueError(f"archive unpacks to more than {max_bytes} bytes")
            if b"\x00" in payload[:TEXT_SNIFF_BYTES]:
                skipped.append(Skipped(path, "binary"))
                continue
            text = payload.decode("utf-8-sig", errors="replace")
            if not text.strip():
                skipped.append(Skipped(path, "empty"))
                continue
            extracted.append(Extracted(flat_name(archive_name, path), path, text, len(payload)))
    return extracted, skipped


def zip_texts_from_file(path: str) -> list[Extracted]:
    """The text entries of the ZIP file at ``path`` (``ValueError`` for a bad archive)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not is_zip(path, data):
        raise ValueError(f"{path} is not a ZIP archive")
    extracted, _skipped = extract_texts(data, os.path.basename(path))
    return extracted
