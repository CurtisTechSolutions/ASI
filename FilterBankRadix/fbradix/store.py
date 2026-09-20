"""Reading and writing model files: atomic, optionally gzipped, JSON.

Every persisted object in this package - a tree, a filter, a router, a whole
bank, a checkpoint index - is a plain JSON document, and every one of them is
written through :func:`write_json_atomic` and read through :func:`read_json`.
Same two functions, same contract, one place to change: that is what
"standardised" means here.

The rules are ``radixnet``'s, because a model file is a thing you lose work by
getting wrong and that package already settled it:

* **atomic** - write to a temporary file in the same directory, ``fsync`` it,
  then ``os.replace``.  A reader never sees a half-written file, and a crash
  mid-write leaves the previous version intact;
* **the content decides, not the suffix** - a file is gunzipped because it
  carries the gzip magic, so a plain-JSON file that happens to end in ``.gz``
  still loads.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile

__version__ = "1.0.0"
"""Format version, stamped into every file this package writes.

It lives here rather than in ``__init__`` because ``__init__`` imports the
model, and a model file has to be able to name its own format without an import
cycle."""

__all__ = ["__version__", "write_bytes_atomic", "write_json_atomic", "read_json", "read_bytes"]


def write_bytes_atomic(path: str, data: bytes, use_gzip: bool = False) -> None:
    """Write ``data`` to ``path`` via a temporary file and ``os.replace``."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            if use_gzip:
                with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                    gz.write(data)
            else:
                fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path: str, payload: dict, use_gzip: bool | None = None) -> str:
    """Write ``payload`` as JSON; gzip when ``path`` ends in ``.gz``.

    ``use_gzip`` overrides that guess.  Returns the path written.
    """
    if use_gzip is None:
        use_gzip = path.endswith(".gz")
    write_bytes_atomic(path, json.dumps(payload, separators=(",", ":")).encode("utf-8"), use_gzip)
    return path


def read_bytes(path: str) -> bytes:
    """Read a file, transparently gunzipping it when it carries the gzip magic."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def read_json(path: str) -> dict:
    """Read a JSON document written by :func:`write_json_atomic`."""
    return json.loads(read_bytes(path).decode("utf-8"))
