"""PairModel: the four verbs, the feedback primitives, settings and persistence.

``train`` reads texts into the count tree and nothing else.  ``reward`` and
``punish`` credit the reward tree and nothing else - a punishment is a reward
with the sign reversed - with marks as weights (D-050); ``two_nrl`` is punish
then reward; ``feedback`` dispatches as D-026.  ``predict`` / ``generate``
walk the pair; ``score`` prices a text under both trees.  A model file holds
the codec whole and only the counts, rewards and penalties that are not zero;
priming is recomputed on load, so the file is the size of what was read and
judged, never of ``R**L`` (``DESIGN.md`` sections 11 and 13).
"""

from __future__ import annotations

import gzip
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone

from .codec import Codec, codec_from_dict, make_codec
from .pair import BACKOFFS, RadixPair, Score, Settings, TRAVERSALS
from .search import MODES, PathResult, dijkstra, greedy

__all__ = ["KINDS", "MODEL_FORMAT", "MODEL_FORMAT_VERSION", "PairModel", "load_model", "prime",
           "read_json_file", "write_bytes_atomic"]

MODEL_FORMAT = "radixpair"
MODEL_FORMAT_VERSION = 1
KINDS = ("count",)
"""The sine kind is phase 4 (DESIGN.md section 12)."""

ProgressFn = Callable[[dict], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes_atomic(path: str, data: bytes, use_gzip: bool = False) -> None:
    """Write through a temporary file and ``os.replace``: a reader never sees half a file."""
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


def read_json_file(path: str) -> dict:
    """A JSON document, gunzipped when it carries the gzip magic - the content decides, not the suffix."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


class _Half:
    """One half of the codec as an object, so ``model.encoder.encode`` reads as the family's."""

    def __init__(self, codec: Codec, which: str) -> None:
        self.codec = codec
        self.which = which

    def encode(self, text: str) -> list[int]:
        return self.codec.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        return self.codec.decode(ids)

    def __repr__(self) -> str:
        return f"<{self.which} of {self.codec.describe()}>"


class PairModel:
    def __init__(self, codec: Codec, L: int, kind: str = "count", seed: int = 0,
                 settings: Settings | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS} (the sine kind is phase 4), got {kind!r}")
        self.kind = kind
        self.codec = codec
        self.L = int(L)
        self.seed = int(seed)
        self.pair = RadixPair(codec, self.L, settings)
        self.rng = random.Random(self.seed)
        self.history: list[dict] = []
        self.created = _utc_now()
        self.encoder = _Half(codec, "encoder")
        self.decoder = _Half(codec, "decoder")

    @classmethod
    def prime(cls, codec: Codec, L: int, kind: str = "count", seed: int = 0,
              settings: Settings | None = None) -> "PairModel":
        return cls(codec, L, kind=kind, seed=seed, settings=settings)

    @property
    def settings(self) -> Settings:
        return self.pair.settings

    # -- reading ------------------------------------------------------------------------

    def train(self, texts: Iterable[str] | str, progress: ProgressFn | None = None) -> dict:
        """Count every text into the count tree; nothing else moves."""
        if isinstance(texts, str):
            texts = [texts]
        count = self.pair.count
        t0 = time.perf_counter()
        n = 0
        increments = 0
        unk0, units0 = count.unk, count.units
        for text in texts:
            increments += count.observe_text(text)
            n += 1
            if progress is not None and n % 100 == 0:
                progress({"call": "train", "texts": n, "increments": increments})
        units = count.units - units0
        unk = count.unk - unk0
        record = {"call": "train", "texts": n, "units": units, "unk": unk,
                  "unk_share": (unk / units) if units else 0.0, "increments": increments,
                  "seconds": round(time.perf_counter() - t0, 4), "at": _utc_now()}
        self.history.append(record)
        return record

    # -- outcomes ---------------------------------------------------------------------------

    def _credit(self, texts: Iterable[str] | str, sign: float, strength: float | None,
                weights: Sequence[float] | None, read: bool, call: str, prefix: str | None = None) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        base = abs(self.settings.strength if strength is None else float(strength))
        if weights is not None and len(weights) != len(texts):
            raise ValueError(f"{len(weights)} weights for {len(texts)} texts")
        reward, count, codec = self.pair.reward, self.pair.count, self.codec
        prefix_ids = codec.encode(prefix) if prefix else []
        skip = 1 + len(prefix_ids)
        t0 = time.perf_counter()
        entries = 0
        credited = 0
        skipped = 0
        for k, text in enumerate(texts):
            w = 1.0 if weights is None else float(weights[k])
            if w < 0:
                raise ValueError("a weight is a mark, >= 0")
            if w == 0.0 or base == 0.0:
                skipped += 1
                continue
            ids = [codec.start] + prefix_ids + codec.encode(text) + [codec.end]
            entries += reward.credit(ids, sign * base * w, self.settings.rungs, skip=skip if prefix else 0)
            if read:
                count.observe(ids)
            credited += 1
        record = {"call": call, "texts": credited, "skipped": skipped, "strength": base, "entries": entries,
                  "read": bool(read), "rungs": self.settings.rungs, "prefix": prefix or "",
                  "seconds": round(time.perf_counter() - t0, 4), "at": _utc_now()}
        self.history.append(record)
        return record

    def reward(self, texts: Iterable[str] | str, *, strength: float | None = None,
               weights: Sequence[float] | None = None, read: bool = False, prefix: str | None = None) -> dict:
        """Thumbs up: credit every step of every text at the model's ``rungs``.

        ``prefix`` is what the model was given: its units are context for the
        texts (the model's outputs) and are not credited themselves. ``read``
        also counts the texts into the count tree.
        """
        return self._credit(texts, +1.0, strength, weights, read, "reward", prefix)

    def punish(self, texts: Iterable[str] | str, *, strength: float | None = None,
               weights: Sequence[float] | None = None, prefix: str | None = None) -> dict:
        """Thumbs down: a reward with the sign reversed; never reads."""
        return self._credit(texts, -1.0, strength, weights, False, "punish", prefix)

    def two_nrl(self, bad: Iterable[str] | str, good: Iterable[str] | str, *, strength: float | None = None,
                bad_weights: Sequence[float] | None = None, good_weights: Sequence[float] | None = None,
                prefix: str | None = None) -> dict:
        """Punish the bad texts, then reward the good ones; no inversion (D-023)."""
        a = self.punish(bad, strength=strength, weights=bad_weights, prefix=prefix)
        b = self.reward(good, strength=strength, weights=good_weights, prefix=prefix)
        return {"call": "two_nrl", "punished": a, "rewarded": b}

    def feedback(self, good: Iterable[str] | str = (), bad: Iterable[str] | str = (), *,
                 good_weights: Sequence[float] | None = None, bad_weights: Sequence[float] | None = None,
                 strength: float | None = None, prefix: str | None = None) -> dict:
        """D-026's dispatch: both sets -> two_nrl; only good -> reward; only bad -> punish."""
        good = [good] if isinstance(good, str) else list(good)
        bad = [bad] if isinstance(bad, str) else list(bad)
        if good and bad:
            return self.two_nrl(bad, good, strength=strength, bad_weights=bad_weights, good_weights=good_weights,
                                prefix=prefix)
        if good:
            return self.reward(good, strength=strength, weights=good_weights, prefix=prefix)
        if bad:
            return self.punish(bad, strength=strength, weights=bad_weights, prefix=prefix)
        raise ValueError("feedback needs good texts, bad texts, or both")

    def invert(self) -> None:
        """The reward tree's swap: every reward a penalty of the same size and every penalty a reward."""
        self.pair.reward.invert()
        self.history.append({"call": "invert", "at": _utc_now()})

    # -- prediction ------------------------------------------------------------------------------

    def _context(self, prefix: str, start: bool) -> tuple[list[int], list[int]]:
        ids = self.codec.encode(prefix) if prefix else []
        context = ([self.codec.start] if start else []) + ids
        return ids, context

    def predict(self, prefix: str, length: int, mode: str = "greedy", traversal: str = "reward",
                start: bool = True, temperature: float = 1.0, to_end: bool = False,
                backoff: str | None = None, max_units: int | None = None) -> PathResult:
        """Continue a prefix by ``length`` units (``start``: the prefix begins a text).

        ``greedy`` - the exact fold, one unit at a time - is the default: on a
        primed tree the cheapest single path (``dijkstra``) can pay for a rare
        unit at the deepest level because the unread context it lands in then
        lets it fall to the root for nothing, where frequent units are cheap;
        the fold sums over every fall and does not. ``sample`` draws from the
        fold at ``temperature``.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if traversal not in TRAVERSALS:
            raise ValueError(f"traversal must be one of {TRAVERSALS}, got {traversal!r}")
        if backoff is not None and backoff not in BACKOFFS:
            raise ValueError(f"backoff must be one of {BACKOFFS}, got {backoff!r}")
        if length < 0:
            raise ValueError("length >= 0")
        prefix_ids, context = self._context(prefix, start)
        if mode == "dijkstra":
            if backoff is not None and backoff != self.settings.backoff:
                raise ValueError("the cheapest-path search walks the model's fold; backoff applies to greedy and sample")
            cap = max_units if max_units is not None else (length if to_end else None)
            result = dijkstra(self.pair, context, length, max_units=cap, to_end=to_end, traversal=traversal)
        else:
            result = greedy(self.pair, context, length, traversal=traversal, rng=self.rng,
                            temperature=0.0 if mode == "greedy" else temperature, to_end=to_end, backoff=backoff)
        result.full_text = self.codec.decode(prefix_ids + result.units)
        return result

    def generate(self, prefix: str = "", length: int = 60, mode: str = "greedy", **options) -> PathResult:
        """A whole text: the walk runs to the end mark, capped at ``length`` units.

        The greedy walk of the exact fold by default; ``mode="dijkstra"`` is the
        cheapest complete text, which - as the family's shortest-path
        generation - likes short ones; ``mode="sample"`` draws one.
        """
        return self.predict(prefix, length, mode=mode, to_end=True, **options)

    def score(self, text: str, traversal: str = "reward", backoff: str | None = None) -> Score:
        return self.pair.score(text, traversal, backoff)

    def distribution(self, context: Sequence[int], traversal: str = "reward",
                     backoff: str | None = None) -> list[tuple[int, float]]:
        """``[(unit id, probability)]`` of the next unit after a context of ids (the fold)."""
        P = self.pair.fold(context, traversal, backoff)
        return list(zip(self.pair.emits, P))

    def next_units(self, prefix: str, k: int = 10, traversal: str = "reward", start: bool = True,
                   backoff: str | None = None) -> list[tuple[str, float]]:
        """The ``k`` likeliest next units after a prefix, as ``(symbol, probability)``."""
        _, context = self._context(prefix, start)
        dist = self.distribution(context[-self.pair.D:] if self.pair.D > 0 else [], traversal, backoff)
        dist.sort(key=lambda t: (-t[1], t[0]))
        return [(self.codec.symbol(i), p) for i, p in dist[:k]]

    # -- settings and information ------------------------------------------------------------------

    def weights(self, **changes) -> dict:
        """Change the smoothing, the scales or the backoff at run time and report the settings."""
        self.pair.configure(**changes)
        return self.settings.to_dict()

    def info(self) -> dict:
        pair, count, reward = self.pair, self.pair.count, self.pair.reward
        return {
            "format": MODEL_FORMAT, "kind": self.kind, "codec": self.codec.describe(), "codec_name": self.codec.name,
            "R": self.codec.R, "R_out": pair.R_out, "L": self.L, "D": pair.D, "N": pair.address.N,
            "memory_bytes": pair.memory_bytes(), "seed": self.seed, "created": self.created,
            "read": {"texts": count.texts, "units": count.units, "unk": count.unk,
                     "unk_share": (count.unk / count.units) if count.units else 0.0},
            "judged": {"texts": reward.judged, "rewards_total": reward.rewards_total,
                       "penalties_total": reward.penalties_total},
            "nonzero_counts": sum(1 for _ in count.nonzero()),
            "nonzero_rewards": sum(1 for _ in reward.nonzero()),
            "settings": self.settings.to_dict(), "history": list(self.history),
        }

    # -- persistence ------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        count, reward = self.pair.count, self.pair.reward
        ids, values = [], []
        for i, c in count.nonzero():
            ids.append(i)
            values.append(c)
        rids, plus, minus = [], [], []
        for i, p, m in reward.nonzero():
            rids.append(i)
            plus.append(p)
            minus.append(m)
        return {
            "format": MODEL_FORMAT, "version": MODEL_FORMAT_VERSION,
            "codec": self.codec.to_dict(), "L": self.L, "kind": self.kind, "seed": self.seed,
            "settings": self.settings.to_dict(), "created": self.created,
            "read": {"texts": count.texts, "units": count.units, "unk": count.unk},
            "judged": {"texts": reward.judged, "rewards_total": reward.rewards_total,
                       "penalties_total": reward.penalties_total},
            "history": list(self.history),
            "rng": _rng_state(self.rng),
            "counts": {"ids": ids, "values": values},
            "rewards": {"ids": rids, "plus": plus, "minus": minus},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PairModel":
        if d.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a {MODEL_FORMAT} model file (format {d.get('format')!r})")
        if int(d.get("version", 0)) > MODEL_FORMAT_VERSION:
            raise ValueError(f"model file version {d.get('version')} is newer than this code ({MODEL_FORMAT_VERSION})")
        codec = codec_from_dict(d["codec"])
        settings = Settings.from_dict(d.get("settings"))
        model = cls(codec, int(d["L"]), kind=d.get("kind", "count"), seed=int(d.get("seed", 0)), settings=settings)
        model.created = d.get("created", model.created)
        model.history = list(d.get("history", []))
        count, reward = model.pair.count, model.pair.reward
        for i, c in zip(d["counts"]["ids"], d["counts"]["values"]):
            count.cnt[i] = int(c)
        rd = d.get("rewards", {"ids": [], "plus": [], "minus": []})
        for i, p, m in zip(rd["ids"], rd["plus"], rd["minus"]):
            reward.plus[i] = float(p)
            reward.minus[i] = float(m)
        read = d.get("read", {})
        count.texts, count.units, count.unk = int(read.get("texts", 0)), int(read.get("units", 0)), int(read.get("unk", 0))
        judged = d.get("judged", {})
        reward.judged = int(judged.get("texts", 0))
        reward.rewards_total = float(judged.get("rewards_total", 0.0))
        reward.penalties_total = float(judged.get("penalties_total", 0.0))
        count.version += 1
        reward.version += 1
        if d.get("rng"):
            _set_rng_state(model.rng, d["rng"])
        return model

    def save(self, path: str, use_gzip: bool | None = None) -> str:
        data = json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")
        gz = path.endswith(".gz") if use_gzip is None else bool(use_gzip)
        write_bytes_atomic(path, data, use_gzip=gz)
        return path

    @classmethod
    def load(cls, path: str) -> "PairModel":
        return cls.from_dict(read_json_file(path))

    def __repr__(self) -> str:
        return f"PairModel({self.kind}, {self.codec.describe()}, L={self.L}, N={self.pair.address.N:,})"


def _rng_state(rng: random.Random) -> list:
    version, state, gauss = rng.getstate()
    return [version, list(state), gauss]


def _set_rng_state(rng: random.Random, saved: list) -> None:
    version, state, gauss = saved
    rng.setstate((int(version), tuple(int(v) for v in state), gauss))


def prime(codec: Codec | str, L: int, kind: str = "count", seed: int = 0, settings: Settings | None = None,
          **codec_options) -> PairModel:
    """A primed model: ``codec`` a :class:`Codec` or a preset name (with its options)."""
    if isinstance(codec, str):
        codec = make_codec(codec, **codec_options)
    elif codec_options:
        raise ValueError("codec options go with a preset name, not a Codec object")
    return PairModel.prime(codec, L, kind=kind, seed=seed, settings=settings)


def load_model(path: str) -> PairModel:
    return PairModel.load(path)
