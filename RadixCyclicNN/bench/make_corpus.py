#!/usr/bin/env python3
"""Write the corpus and the prefixes both ports are benchmarked on.

The comparison is only worth reading if the two builds did the *same* work, and
neither language can reproduce the other's random number generator, so neither
generates the corpus: this does, once, and both are handed the files.

The three recipes are the ones `radixnet/bench.py`, `go/radixnet/bench.go` and
`rust/src/bench.rs` use for their own synthetic corpora - lines of the sample
corpus verbatim, two lines spliced at a word boundary, and a line with a word or
two swapped for another word of the vocabulary - so the graph has the shape a
corpus gives it: unary chains to compress, shared prefixes to branch on and
realistic fan-out.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

MIN_PREFIX = 3
MAX_PREFIX = 12
FALLBACK = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a bird flew over the hill",
    "the cat chased the mouse",
    "the quick brown fox jumps over the lazy dog",
    "rain fell on the quiet town",
    "she walked to the river at dawn",
    "the old man read a book by the fire",
    "children played in the summer field",
    "a train passed through the empty station",
]


def base_lines(path: Path) -> list[str]:
    """The non-blank lines of at least three characters of a sample corpus."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError:
        lines = []
    lines = [line for line in lines if len(line) >= 3]
    return lines or list(FALLBACK)


def corpus(chars: int, seed: int, lines: list[str]) -> list[str]:
    """Training texts totalling at least `chars` characters."""
    vocab = sorted({word for line in lines for word in line.split()})
    rng = random.Random(seed)
    texts: list[str] = []
    total = 0
    while total < chars:
        roll = rng.random()
        if roll < 0.4 or len(vocab) < 2:
            text = rng.choice(lines)
        elif roll < 0.7:
            left = rng.choice(lines).split()
            right = rng.choice(lines).split()
            text = " ".join(left[: rng.randrange(1, len(left) + 1)] + right[rng.randrange(len(right)) :])
        else:
            parts = rng.choice(lines).split()
            for _ in range(rng.randrange(1, 3)):
                parts[rng.randrange(len(parts))] = rng.choice(vocab)
            text = " ".join(parts)
        if len(text) < 3:
            text = rng.choice(lines)
        texts.append(text)
        total += len(text)
    return texts


def prefixes(texts: list[str], count: int, seed: int) -> list[str]:
    """Prefixes of 3 to 12 characters cut out of the corpus, so every
    prediction starts at a node the trained graph knows."""
    rng = random.Random(seed + 1)
    out = []
    for _ in range(count):
        text = rng.choice(texts)
        high = min(MAX_PREFIX, len(text))
        length = MIN_PREFIX if high <= MIN_PREFIX else rng.randrange(MIN_PREFIX, high + 1)
        start = rng.randrange(len(text) - length + 1) if len(text) > length else 0
        out.append(text[start : start + length])
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", type=int, default=2_000_000, help="characters of training text (default: 2000000)")
    ap.add_argument("--predictions", type=int, default=3000, help="prefixes to write (default: 3000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=here.parent / "data" / "sample_corpus.txt")
    ap.add_argument("--out", type=Path, default=here / "corpus", help="the two files' stem (default: bench/corpus)")
    args = ap.parse_args()

    texts = corpus(args.chars, args.seed, base_lines(args.data))
    cuts = prefixes(texts, args.predictions, args.seed)
    texts_path = args.out.parent / f"{args.out.name}.txt"
    prefix_path = args.out.parent / f"{args.out.name}.prefixes.txt"
    texts_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    prefix_path.write_text("\n".join(cuts) + "\n", encoding="utf-8")
    print(f"{texts_path}: {len(texts)} texts, {sum(len(t) for t in texts)} characters")
    print(f"{prefix_path}: {len(cuts)} prefixes")


if __name__ == "__main__":
    main()
