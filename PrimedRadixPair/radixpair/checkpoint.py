"""Checkpoint directory management: rotation, a ``latest`` pointer, resume - a checkpoint is a model file.

Files are ``ckpt-<tag>-<step:06d>.json[.gz]`` in one directory; ``latest.json``
names the most recently written one and ``index.json`` keeps the records
(step, tag, metrics, timestamp, size), reconciled with the directory on every
listing so files removed or added by hand are handled.  The shape of
``RadixCyclicNN``'s ``CheckpointManager``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from .model import PairModel, read_json_file, write_bytes_atomic

__all__ = ["CheckpointManager"]

_PREFIX = "ckpt-"
_LATEST = "latest.json"
_INDEX = "index.json"
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_NAME_RE = re.compile(r"^ckpt-(?P<tag>[A-Za-z0-9][A-Za-z0-9_\-]*)-(?P<step>\d{6,})\.json(?:\.gz)?$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class CheckpointManager:
    def __init__(self, directory: str, keep: int = 5, compress: bool = True) -> None:
        if keep < 1:
            raise ValueError("keep >= 1")
        self.directory = directory
        self.keep = int(keep)
        self.compress = bool(compress)
        os.makedirs(directory, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def _read_json(self, name: str) -> dict | None:
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        try:
            return read_json_file(path)
        except (OSError, ValueError):
            return None

    def _write_json(self, name: str, data: dict) -> None:
        write_bytes_atomic(self._path(name), json.dumps(data, indent=1).encode("utf-8"))

    def _index(self) -> dict[str, dict]:
        index = (self._read_json(_INDEX) or {}).get("checkpoints", {})
        present = {n for n in os.listdir(self.directory) if _NAME_RE.match(n)}
        for name in list(index):
            if name not in present:
                del index[name]
        for name in present:
            if name not in index:
                m = _NAME_RE.match(name)
                index[name] = {"name": name, "tag": m.group("tag"), "step": int(m.group("step")),
                               "saved_at": "", "metrics": {}, "bytes": os.path.getsize(self._path(name))}
        return index

    def _prune(self, index: dict[str, dict], protect: str | None) -> None:
        records = sorted(index.values(), key=lambda r: (r["step"], r.get("saved_at", ""), r["name"]))
        while len(records) > self.keep:
            victim = records.pop(0)
            if victim["name"] == protect:
                records.append(victim)
                if len(records) <= self.keep:
                    break
                victim = records.pop(0)
            try:
                os.unlink(self._path(victim["name"]))
            except OSError:
                pass
            index.pop(victim["name"], None)

    def save(self, model: PairModel, step: int, tag: str = "epoch", metrics: dict | None = None) -> dict:
        if not _TAG_RE.match(tag):
            raise ValueError(f"tag must match {_TAG_RE.pattern}, got {tag!r}")
        name = f"{_PREFIX}{tag}-{int(step):06d}.json" + (".gz" if self.compress else "")
        model.save(self._path(name))
        record = {"name": name, "tag": tag, "step": int(step), "saved_at": _utc_now(),
                  "metrics": dict(metrics or {}), "bytes": os.path.getsize(self._path(name))}
        index = self._index()
        index[name] = record
        self._prune(index, protect=name)
        self._write_json(_INDEX, {"checkpoints": index})
        self._write_json(_LATEST, {"name": name, "step": record["step"], "saved_at": record["saved_at"]})
        return record

    def list(self) -> list[dict]:
        return sorted(self._index().values(), key=lambda r: (r["step"], r.get("saved_at", ""), r["name"]))

    def latest(self) -> dict | None:
        latest = self._read_json(_LATEST)
        index = self._index()
        if latest and latest.get("name") in index:
            return index[latest["name"]]
        records = self.list()
        return records[-1] if records else None

    def load_latest(self) -> PairModel | None:
        record = self.latest()
        return None if record is None else PairModel.load(self._path(record["name"]))

    def resolve(self, name_or_path: str) -> str:
        """A checkpoint name, a step number, ``latest``, or a path."""
        if name_or_path == "latest":
            record = self.latest()
            if record is None:
                raise FileNotFoundError(f"no checkpoints in {self.directory}")
            return self._path(record["name"])
        if os.path.isfile(name_or_path):
            return name_or_path
        index = self._index()
        if name_or_path in index:
            return self._path(name_or_path)
        if name_or_path.isdigit():
            step = int(name_or_path)
            for record in self.list():
                if record["step"] == step:
                    return self._path(record["name"])
        raise FileNotFoundError(f"no checkpoint {name_or_path!r} in {self.directory}")

    def load(self, name_or_path: str) -> PairModel:
        return PairModel.load(self.resolve(name_or_path))

    def delete(self, name: str) -> bool:
        index = self._index()
        if name not in index:
            return False
        try:
            os.unlink(self._path(name))
        except OSError:
            return False
        del index[name]
        self._write_json(_INDEX, {"checkpoints": index})
        latest = self._read_json(_LATEST)
        if latest and latest.get("name") == name:
            records = self.list()
            if records:
                self._write_json(_LATEST, {"name": records[-1]["name"], "step": records[-1]["step"],
                                           "saved_at": records[-1].get("saved_at", "")})
            else:
                try:
                    os.unlink(self._path(_LATEST))
                except OSError:
                    pass
        return True

    def __repr__(self) -> str:
        return f"CheckpointManager({self.directory!r}, keep={self.keep}, compress={self.compress})"
