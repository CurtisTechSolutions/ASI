"""The corpus: text from four sources, and the labels the experiment is scored on.

The routing question - *can layer 1 discover that its input has kinds?* - needs
a corpus that really has kinds.  This one is built out of the repository's own
files: Python source, Go source, Markdown prose and JSON results, cut into
segments of a few lines each.  Four registers with genuinely different character
statistics, no download, no dependency, and nothing invented for the occasion.

**The labels are never shown to the network.**  They exist so that
:func:`fbradix.experiment.purity` can ask, afterwards, whether the addresses the
filter learned line up with the sources - and so the ``oracle`` arm can be given
the answer as a ceiling to measure the learned arms against.  Only that arm ever
sees them.

The file list is pinned rather than globbed, and :func:`build` writes what it
read to ``data/corpus.jsonl`` with a digest, so a rerun can tell whether the
input drifted when the repository changed under it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random

MIN_SEG = 48
"""Shortest segment: short lines are joined until a segment reaches this."""

MAX_SEG = 256
"""Longest segment: a longer line is cut into pieces of at most this."""

SOURCES: dict[str, tuple[str, ...]] = {
    "python": (
        "RadixCyclicNN/radixnet/graph.py",
        "RadixCyclicNN/radixnet/search.py",
        "RadixCyclicNN/radixnet/encoding.py",
        "RadixCyclicNN/radixnet/counter.py",
        "RadixCyclicNN/radixnet/schedule.py",
        "RadixCyclicNN/radixnet/model.py",
        "RadixCyclicNN/radixnet/countnet.py",
        "RadixCyclicNN/radixnet/resonance.py",
        "RadixCyclicNN/radixnet/negative.py",
        "RadixCyclicNN/radixnet/blame.py",
    ),
    "go": (
        "RadixCyclicNN/go/radixnet/nodes.go",
        "RadixCyclicNN/go/radixnet/weights.go",
        "RadixCyclicNN/go/radixnet/counter.go",
        "RadixCyclicNN/go/radixnet/encoding.go",
        "RadixCyclicNN/go/radixnet/model.go",
        "RadixCyclicNN/go/radixnet/chat.go",
        "RadixCyclicNN/go/radixnet/recall.go",
        "RadixCyclicNN/go/radixnet/search.go",
        "RadixCyclicNN/go/radixnet/beam.go",
    ),
    # The four papers, and deliberately not `Research/Insights.md`: that file is
    # a living index that this directory's own findings get written into, so a
    # corpus built from it changes every time a result is recorded - including
    # between two runs of the same sweep.  Measured the hard way.
    "prose": (
        "Research/CyclesAreAFeature.md",
        "Research/SineWaveActivationFunction.md",
        "Research/VanishingGradientIsAFeature.md",
        "Research/2NRL.md",
        "RadixCyclicNN/SPEC-EdgeDecay.md",
    ),
    "json": (
        "Experiments/TwoNRL_CartPole/results/twonrl_results.json",
        "Experiments/TwoNRL_CartPole/results/grading_speed_results.json",
        "Experiments/TwoNRL_CartPole/results/ablation_no_weights.json",
        "TwoNRL_Chess/results/main.json",
        "TwoNRL_Chess/results/entropy.json",
        "TwoNRL_Chess/results/per_round.json",
    ),
}

LABELS = tuple(SOURCES)

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(os.path.dirname(HERE), "data", "corpus.jsonl")


def repo_root(start: str = HERE) -> str:
    """Walk up until the directory holding ``Research/`` and ``RadixCyclicNN/``."""
    d = start
    while True:
        if all(os.path.isdir(os.path.join(d, x)) for x in ("Research", "RadixCyclicNN")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("could not find the repository root above " + start)
        d = parent


def segments_of(text: str, min_len: int = MIN_SEG, max_len: int = MAX_SEG) -> list[str]:
    """Cut ``text`` into segments on line boundaries, ``min_len..max_len`` long.

    Lines are joined until the segment is long enough and a single line longer
    than ``max_len`` is cut into pieces, so no segment is too short to have
    statistics or long enough to contain several registers.
    """
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        while len(line) > max_len:
            if buf:
                out.append("\n".join(buf))
                buf, size = [], 0
            out.append(line[:max_len])
            line = line[max_len:]
        if size + len(line) + 1 > max_len and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
        if size >= min_len:
            out.append("\n".join(buf))
            buf, size = [], 0
    if buf and size >= min_len // 2:
        out.append("\n".join(buf))
    return [s for s in out if len(s) >= min_len // 2]


def _thin(items: list[str], keep: int) -> list[str]:
    """Keep ``keep`` items spread evenly across ``items`` (deterministic)."""
    if keep >= len(items):
        return items
    step = len(items) / keep
    return [items[int(i * step)] for i in range(keep)]


def build(per_source: int = 200, root: str | None = None) -> list[dict]:
    """Read the pinned files and return ``[{"source": ..., "text": ...}, ...]``.

    Every source contributes the same number of segments, spread evenly over
    its files, so no register can win the routing metrics by sheer volume.
    """
    root = root or repo_root()
    out: list[dict] = []
    for label, paths in SOURCES.items():
        pools: list[list[str]] = []
        for p in paths:
            with open(os.path.join(root, p), encoding="utf-8") as fh:
                pools.append(segments_of(fh.read()))
        want = per_source
        got: list[str] = []
        # Round robin over the files, so a source's quota is spread across all
        # of its files and a short file cannot starve the count.
        share = [max(1, want // len(pools))] * len(pools)
        for _ in range(4):
            taken = [_thin(pool, sh) for pool, sh in zip(pools, share)]
            got = [t for group in zip(*[iter_pad(t, max(map(len, taken))) for t in taken]) for t in group if t]
            if len(got) >= want:
                break
            short = want - len(got)
            share = [sh + max(1, short // len(pools)) for sh in share]
        out.extend({"source": label, "text": t} for t in got[:want])
    return out


def iter_pad(items: list[str], n: int) -> list[str | None]:
    """``items`` padded with ``None`` to length ``n`` (for the round robin)."""
    return list(items) + [None] * (n - len(items))


def digest(samples: list[dict]) -> str:
    """SHA-256 of the corpus, so a result can be tied to the text it was got on."""
    hsh = hashlib.sha256()
    for s in samples:
        hsh.update(s["source"].encode())
        hsh.update(b"\0")
        hsh.update(s["text"].encode())
        hsh.update(b"\n")
    return hsh.hexdigest()


def save(samples: list[dict], path: str = SNAPSHOT) -> str:
    """Write the corpus as JSON lines; returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    return path


def load(path: str = SNAPSHOT, per_source: int = 200) -> list[dict]:
    """The committed snapshot if there is one, otherwise a fresh :func:`build`."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    return build(per_source)


def per_source_of(samples: list[dict]) -> int:
    """The largest number of segments any one source contributes."""
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["source"]] = counts.get(s["source"], 0) + 1
    return max(counts.values()) if counts else 0


def take(per_source: int = 200, path: str = SNAPSHOT) -> list[dict]:
    """The corpus at this size: the committed snapshot if it *is* this size.

    The snapshot is preferred over a fresh build for a reason that cost a run:
    the sources are files in this repository, so writing a finding into one of
    them changes the corpus under a sweep that is still running, and two tables
    in the same README stop being comparable.  A size the snapshot does not
    have is built - and its digest is recorded in the result file, which is how
    a reader can tell whether it drifted.
    """
    if per_source <= 0:
        return load(path)
    if os.path.exists(path):
        snap = load(path)
        if per_source_of(snap) == per_source:
            return snap
    return build(per_source)


# -- the synthetic corpus ------------------------------------------------------

def synthetic(per_source: int = 40, seed: int = 0) -> list[dict]:
    """Four made-up registers, for the tests: fast, tiny and label-perfect.

    ``words`` is lowercase prose, ``digits`` is numeric records, ``dna`` is a
    four-letter alphabet and ``brackets`` is nested punctuation.  They overlap
    in almost no characters, so a router that cannot separate *these* cannot
    separate anything.
    """
    rng = random.Random(seed)
    words = "the cat dog ran sat mat log hat far near green quiet".split()
    out: list[dict] = []
    for _ in range(per_source):
        out.append({"source": "words", "text": " ".join(rng.choice(words) for _ in range(14))})
        out.append({"source": "digits", "text": " ".join(
            f"{rng.randrange(10**5):05d}.{rng.randrange(100):02d}" for _ in range(8))})
        out.append({"source": "dna", "text": "".join(rng.choice("ACGT") for _ in range(70))})
        out.append({"source": "brackets", "text": "".join(
            rng.choice(["{}", "[]", "()", "<>", "::", "=>"]) for _ in range(35))})
    return out


# -- splitting -----------------------------------------------------------------

def split(samples: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic train/test split, stratified by source.

    Stratifying matters: an unstratified split can hand one register's segments
    almost entirely to the test set, and then every arm's bits/char is really a
    measurement of that one register.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["source"], []).append(s)
    train: list[dict] = []
    test: list[dict] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        cut = int(round(len(items) * (1.0 - test_frac)))
        train.extend(items[:cut])
        test.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def alphabet(samples: list[dict]) -> int:
    """Distinct characters in the corpus - the denominator of the smoothing floor."""
    seen: set[str] = set()
    for s in samples:
        seen.update(s["text"])
    return len(seen)


def summarise(samples: list[dict]) -> dict:
    """Counts and character totals per source, plus the digest."""
    per: dict[str, dict] = {}
    for s in samples:
        e = per.setdefault(s["source"], {"segments": 0, "chars": 0})
        e["segments"] += 1
        e["chars"] += len(s["text"])
    return {
        "segments": len(samples),
        "chars": sum(len(s["text"]) for s in samples),
        "alphabet": alphabet(samples),
        "per_source": per,
        "sha256": digest(samples),
    }
