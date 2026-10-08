"""Relearn the bundled merges (``radixnet/data/merges.txt``) of the ``token`` unit.

The corpus is the repository's own prose: every Markdown file tracked at one commit, read one text
per non-blank line - the way ``radixnet train --data`` reads a file.  The commit is pinned, so the
same command learns the same merges however the documents have changed since; learning from a
newer commit is a new vocabulary, and every model trained over the old one reads its tokens
wrongly under it (``../SPEC-Tokens.md`` §6), so the bundled file is relearned on purpose, not on a
whim.  The Go port keeps a copy (``go/radixnet/data/merges.txt``); ``make tokenizer-merges``
refreshes both.

    python3 tests/make_merges.py [--rev COMMIT] [--vocab 4096] [--out radixnet/data/merges.txt]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from radixnet.bpe import BPETokenizer  # noqa: E402

REV = "34a64a1"
"""The commit whose Markdown the bundled merges were learned from."""


def corpus(rev: str) -> tuple[list[str], list[str]]:
    """The non-blank lines of every Markdown file tracked at ``rev``, and the files' paths."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=top, check=True,
                           capture_output=True, text=True).stdout.split("\n")
    paths = sorted(n for n in names if n.endswith(".md"))
    texts: list[str] = []
    for path in paths:
        blob = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=top, check=True, capture_output=True).stdout
        texts.extend(line for line in blob.decode("utf-8").split("\n") if line.strip())
    return texts, paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rev", default=REV, help=f"the commit to read the Markdown of (default {REV})")
    parser.add_argument("--vocab", type=int, default=4096, help="ids in all: the bytes, the merges' tokens, <|endoftext|>")
    parser.add_argument("--out", default=os.path.join(ROOT, "radixnet", "data", "merges.txt"))
    args = parser.parse_args()
    texts, paths = corpus(args.rev)
    started = time.perf_counter()
    note = (f"learned by tests/make_merges.py from the {len(paths)} Markdown files of the ASI repository at "
            f"commit {args.rev}, {len(texts)} lines, one text per line; vocab_size {args.vocab}")
    tok = BPETokenizer.learn(texts, vocab_size=args.vocab, note=note)
    tok.save(args.out)
    print(f"{args.out}: {len(tok.merges)} merges, {tok.vocab_size} ids, from {len(texts)} lines of {len(paths)} files "
          f"in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
