"""Checkpoint directory management: rotation, ``latest`` pointer, resume.

Checkpoints are ordinary model files (:meth:`RadixNet.save`) named
``ckpt-<tag>-<step:06d>.json[.gz]`` inside one directory.  ``latest.json``
points at the most recently written checkpoint and ``index.json`` keeps the
records (step, tag, metrics, timestamp, size) so listing never has to open
the model files; the index is reconciled with the directory on every listing,
so files removed or added by hand are handled.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone

from .model import RadixNet, read_json_file, write_bytes_atomic

__all__ = ["CheckpointManager"]

_PREFIX = "ckpt-"
_SUFFIXES = (".json.gz", ".json")
_LATEST = "latest.json"
_INDEX = "index.json"
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_NAME_RE = re.compile(r"^ckpt-(?P<tag>[A-Za-z0-9][A-Za-z0-9_\-]*)-(?P<step>\d{6,})\.json(?:\.gz)?$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sort_key(record: dict) -> tuple[int, str, str]:
    return (int(record["step"]), str(record.get("saved_at", "")), str(record["name"]))


class CheckpointManager:
    """Save / list / restore rotated checkpoints of a :class:`RadixNet`.

    ``keep`` is the number of checkpoints retained; the oldest (by
    ``(step, saved_at)``) are pruned after every save, except the one
    ``latest.json`` points at.  All operations are thread-safe.
    """

    def __init__(self, directory: str, keep: int = 5, compress: bool = True) -> None:
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        self.directory = os.path.abspath(directory)
        self.keep = int(keep)
        self.compress = bool(compress)
        self._lock = threading.RLock()
        os.makedirs(self.directory, exist_ok=True)

    # -- paths / files -------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def _read_json(self, name: str) -> dict | None:
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        try:
            data = read_json_file(path)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write_json(self, name: str, data: dict) -> None:
        payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        write_bytes_atomic(self._path(name), payload)

    def _record_from_file(self, name: str) -> dict | None:
        """Best-effort record for a checkpoint file that is not in the index."""
        m = _NAME_RE.match(name)
        path = self._path(name)
        if m is None or not os.path.isfile(path):
            return None
        st = os.stat(path)
        return {
            "name": name,
            "path": path,
            "step": int(m.group("step")),
            "tag": m.group("tag"),
            "metrics": None,
            "saved_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="microseconds"),
            "bytes": st.st_size,
        }

    def _index(self) -> dict[str, dict]:
        """Index reconciled with the directory contents (files win)."""
        stored = self._read_json(_INDEX) or {}
        index: dict[str, dict] = {}
        changed = False
        for name, record in stored.items():
            if isinstance(record, dict) and os.path.isfile(self._path(name)):
                record["path"] = self._path(name)
                index[name] = record
            else:
                changed = True
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            names = []
        for name in names:
            if name not in index and name.startswith(_PREFIX) and name.endswith(_SUFFIXES):
                record = self._record_from_file(name)
                if record is not None:
                    index[name] = record
                    changed = True
        if changed:
            self._write_json(_INDEX, index)
        return index

    def _prune(self, index: dict[str, dict], protect: str | None) -> None:
        records = sorted(index.values(), key=_sort_key)
        while len(records) > self.keep:
            victim = next((r for r in records if r["name"] != protect), None)
            if victim is None:
                break
            records.remove(victim)
            index.pop(victim["name"], None)
            try:
                os.unlink(self._path(victim["name"]))
            except FileNotFoundError:
                pass

    # -- public API ----------------------------------------------------------

    def save(self, model: RadixNet, step: int, tag: str = "epoch", metrics: dict | None = None) -> dict:
        """Write ``ckpt-<tag>-<step:06d>.json[.gz]``, update ``latest``, prune.

        Returns the record ``{"name", "path", "step", "tag", "metrics",
        "saved_at", "bytes"}``.  Saving the same ``(tag, step)`` twice
        overwrites the earlier file.
        """
        if not isinstance(tag, str) or not _TAG_RE.match(tag):
            raise ValueError(f"invalid checkpoint tag {tag!r} (letters, digits, '_' and '-' only)")
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        metrics = None if metrics is None else json.loads(json.dumps(metrics))
        name = f"{_PREFIX}{tag}-{step:06d}" + (".json.gz" if self.compress else ".json")
        path = self._path(name)
        with self._lock:
            model.save(path)
            record = {
                "name": name,
                "path": path,
                "step": step,
                "tag": tag,
                "metrics": metrics,
                "saved_at": _utc_now(),
                "bytes": os.path.getsize(path),
            }
            index = self._index()
            index[name] = record
            self._write_json(_LATEST, record)
            self._prune(index, protect=name)
            self._write_json(_INDEX, index)
        return dict(record)

    def list(self) -> list[dict]:
        """All checkpoint records sorted by ``(step, saved_at)``."""
        with self._lock:
            return [dict(r) for r in sorted(self._index().values(), key=_sort_key)]

    def latest(self) -> dict | None:
        """Record referenced by ``latest.json`` (``None`` if absent or its file is gone)."""
        with self._lock:
            record = self._read_json(_LATEST)
            if record is None or "name" not in record:
                return None
            path = self._path(str(record["name"]))
            if not os.path.isfile(path):
                return None
            record["path"] = path
            return record

    def load_latest(self, backend: str = "auto", device: str | None = None) -> RadixNet | None:
        """Load the checkpoint ``latest.json`` points at, or ``None``."""
        record = self.latest()
        if record is None:
            return None
        return RadixNet.load(record["path"], backend=backend, device=device)

    def resolve(self, name_or_path: str) -> str:
        """Absolute path of a checkpoint given its name, stem or path (``FileNotFoundError`` otherwise).

        A name carrying the other suffix (``.json`` for a ``.json.gz`` file or
        vice versa) is resolved too.
        """
        stem = name_or_path
        for suffix in _SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidates = [name_or_path, self._path(name_or_path)]
        candidates += [self._path(stem + suffix) for suffix in _SUFFIXES]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(f"no checkpoint {name_or_path!r} in {self.directory}")

    def load(self, name_or_path: str, backend: str = "auto", device: str | None = None) -> RadixNet:
        """Load a checkpoint by record name, bare stem or file path."""
        return RadixNet.load(self.resolve(name_or_path), backend=backend, device=device)

    def delete(self, name: str) -> bool:
        """Remove a checkpoint; ``latest`` moves to the newest remaining one.

        Returns ``False`` if ``name`` is absent, lies outside the directory or
        is not a checkpoint file (``latest.json`` / ``index.json`` cannot be
        deleted this way).
        """
        with self._lock:
            try:
                path = self.resolve(name)
            except FileNotFoundError:
                return False
            if os.path.dirname(path) != self.directory:
                return False
            target = os.path.basename(path)
            if _NAME_RE.match(target) is None:
                return False
            index = self._index()
            index.pop(target, None)
            try:
                os.unlink(path)
            except FileNotFoundError:
                return False
            latest = self._read_json(_LATEST)
            if latest is not None and latest.get("name") == target:
                remaining = sorted(index.values(), key=_sort_key)
                if remaining:
                    self._write_json(_LATEST, remaining[-1])
                else:
                    try:
                        os.unlink(self._path(_LATEST))
                    except FileNotFoundError:
                        pass
            self._write_json(_INDEX, index)
            return True

    def __repr__(self) -> str:
        return f"CheckpointManager(directory={self.directory!r}, keep={self.keep}, compress={self.compress})"
