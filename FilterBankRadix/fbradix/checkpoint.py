"""Checkpoint directory management: rotation, a ``latest`` pointer, resume.

A checkpoint is an ordinary bank file (:meth:`FilteredRadixBank.save`) named
``ckpt-<tag>-<step:06d>.json[.gz]`` inside one directory. ``latest.json``
points at the most recently written one and ``index.json`` keeps the records
(step, tag, metrics, timestamp, size) so listing never has to open a model
file. The index is reconciled with the directory on every listing, so files
removed or added by hand are handled.

This is ``RadixCyclicNN/radixnet/checkpoint.py``'s contract, deliberately: a
checkpoint layout is a thing you lose work by inventing twice, and a reader who
knows one of these directories knows the other.

What is worth saying about *this* architecture is what a checkpoint of it
contains. A bank is two layers that were fitted against each other - a filter
that routes and the trees its routing built - so a checkpoint carries both, and
the routing that joined them. It does not carry the corpus. A restored bank
therefore **predicts** exactly as it did, and **re-fits** rather than resumes:
:meth:`FilteredRadixBank.fit` rebuilds every expert from the segments it is
given, which is the same rule that makes a round of fitting honest.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone

from .bank import FilteredRadixBank
from .store import read_json, write_json_atomic

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
    """Save, list and restore rotated checkpoints of a :class:`FilteredRadixBank`.

    ``keep`` is how many are retained; the oldest by ``(step, saved_at)`` are
    pruned after every save, except the one ``latest.json`` points at. Every
    operation takes the same lock, so a background fit and a CLI listing can
    share a directory.
    """

    def __init__(self, directory: str, keep: int = 5, compress: bool = True) -> None:
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        self.directory = os.path.abspath(directory)
        self.keep = int(keep)
        self.compress = bool(compress)
        self._lock = threading.RLock()
        os.makedirs(self.directory, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def _read_json(self, name: str) -> dict | None:
        try:
            return read_json(self._path(name))
        except (OSError, ValueError):
            return None

    def _record_from_file(self, name: str) -> dict | None:
        """Rebuild an index record from a file the index does not know about."""
        m = _NAME_RE.match(name)
        if not m:
            return None
        path = self._path(name)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return {
            "name": name,
            "step": int(m.group("step")),
            "tag": m.group("tag"),
            "metrics": {},
            "saved_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "bytes": stat.st_size,
        }

    def _index(self) -> dict[str, dict]:
        """The index, reconciled with what is actually on disk."""
        stored = self._read_json(_INDEX) or {}
        records = {r["name"]: r for r in stored.get("checkpoints", []) if "name" in r}
        on_disk = {n for n in os.listdir(self.directory) if _NAME_RE.match(n)}
        for name in list(records):
            if name not in on_disk:
                del records[name]           # deleted by hand
        for name in on_disk:
            if name not in records:
                rec = self._record_from_file(name)
                if rec:
                    records[name] = rec     # dropped in by hand
        return records

    def _write_index(self, records: dict[str, dict]) -> None:
        write_json_atomic(
            self._path(_INDEX),
            {"checkpoints": sorted(records.values(), key=_sort_key)},
            use_gzip=False,
        )

    def _prune(self, records: dict[str, dict], protect: str | None) -> None:
        """Drop the oldest checkpoints past ``keep``, never the protected one."""
        ordered = sorted(records.values(), key=_sort_key)
        while len(ordered) > self.keep:
            victim = ordered.pop(0)
            if victim["name"] == protect:
                if not ordered:
                    break
                victim = ordered.pop(0)
            records.pop(victim["name"], None)
            try:
                os.unlink(self._path(victim["name"]))
            except OSError:
                pass

    # -- the interface -------------------------------------------------------

    def save(self, bank: FilteredRadixBank, step: int, tag: str = "round",
             metrics: dict | None = None) -> dict:
        """Write one checkpoint, update ``latest``, prune, and return its record."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if not _TAG_RE.match(tag):
            raise ValueError(f"tag must match {_TAG_RE.pattern}, got {tag!r}")
        name = f"{_PREFIX}{tag}-{int(step):06d}.json" + (".gz" if self.compress else "")
        with self._lock:
            path = bank.save(self._path(name))
            record = {
                "name": name,
                "step": int(step),
                "tag": tag,
                "metrics": dict(metrics or {}),
                "saved_at": _utc_now(),
                "bytes": os.path.getsize(path),
                "describe": bank.describe(),
            }
            records = self._index()
            records[name] = record
            write_json_atomic(self._path(_LATEST), {"name": name, **record}, use_gzip=False)
            self._prune(records, protect=name)
            self._write_index(records)
        return record

    def list(self) -> list[dict]:
        """Every checkpoint in the directory, oldest first."""
        with self._lock:
            return sorted(self._index().values(), key=_sort_key)

    def latest(self) -> dict | None:
        """The record ``latest.json`` points at, or the newest one on disk."""
        with self._lock:
            pointer = self._read_json(_LATEST)
            records = self._index()
            if pointer and pointer.get("name") in records:
                return records[pointer["name"]]
            ordered = sorted(records.values(), key=_sort_key)
            return ordered[-1] if ordered else None

    def resolve(self, name_or_path: str) -> str:
        """Turn a checkpoint name, a bare step, ``latest`` or a path into a path."""
        if name_or_path in ("latest", _LATEST):
            record = self.latest()
            if record is None:
                raise FileNotFoundError(f"no checkpoints in {self.directory}")
            return self._path(record["name"])
        if os.path.isabs(name_or_path) or os.path.exists(name_or_path):
            return name_or_path
        if _NAME_RE.match(name_or_path):
            return self._path(name_or_path)
        if name_or_path.isdigit():
            step = int(name_or_path)
            for record in sorted(self._index().values(), key=_sort_key):
                if record["step"] == step:
                    return self._path(record["name"])
            raise FileNotFoundError(f"no checkpoint at step {step} in {self.directory}")
        for suffix in _SUFFIXES:
            candidate = self._path(name_or_path + suffix)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"cannot resolve {name_or_path!r} in {self.directory}")

    def load(self, name_or_path: str = "latest") -> FilteredRadixBank:
        """Restore a bank by name, step, path, or ``latest`` (the default)."""
        return FilteredRadixBank.load(self.resolve(name_or_path))

    def load_latest(self) -> FilteredRadixBank | None:
        """The most recent bank, or ``None`` when the directory is empty."""
        record = self.latest()
        return None if record is None else self.load(record["name"])

    def delete(self, name: str) -> bool:
        """Remove one checkpoint; returns whether it was there."""
        with self._lock:
            records = self._index()
            if name not in records:
                return False
            try:
                os.unlink(self._path(name))
            except OSError:
                pass
            records.pop(name, None)
            pointer = self._read_json(_LATEST) or {}
            if pointer.get("name") == name:
                ordered = sorted(records.values(), key=_sort_key)
                if ordered:
                    write_json_atomic(self._path(_LATEST), dict(ordered[-1]), use_gzip=False)
                else:
                    try:
                        os.unlink(self._path(_LATEST))
                    except OSError:
                        pass
            self._write_index(records)
            return True

    def __repr__(self) -> str:
        return (f"CheckpointManager(directory={self.directory!r}, "
                f"keep={self.keep}, compress={self.compress})")
