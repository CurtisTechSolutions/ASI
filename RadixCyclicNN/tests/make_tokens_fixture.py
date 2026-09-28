"""Write ``tests/tokens_fixture.json``: what the Python tokenizer makes of a set of hard texts.

The Go and Rust ports of the tokenizer (``go/radixnet/bpe.go``, ``rust/src/bpe.rs``) read this file
in their own tests and must reproduce every pre-token, token, id, unit and decoded text in it, under
the bundled merges and under a small vocabulary learned here from the sample corpus.
``tests/test_tokens.py`` fails when the file is out of date; regenerate it with

    python3 tests/make_tokens_fixture.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from radixnet.bpe import BPETokenizer, GLUE, char_class, pretokenize  # noqa: E402

OUT = os.path.join(HERE, "tokens_fixture.json")

TEXTS = [
    "",
    "the cat sat on the mat",
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog!",
    "Hello, world!\nIt's 2026 - isn't it?",
    "He'll say they're here; we've seen I'M sure you'D",
    "'sdf x'sy don't 'S 'LL 're",
    "  two leading spaces and two trailing  ",
    "tabs\tand\t\ttwo tabs \t mixed",
    "lines\n\n\nthree newlines\r\nand a CRLF",
    "numbers 1 22 333 4444 55555 3.14159 1,000,000 v2 x86_64",
    "punctuation... !!! ?! (parens) [brackets] {braces} <angles> @#$%^&*_+=|\\/~`",
    "“curly quotes” — an em dash… and ‘single’ ones",
    "café naïve résumé Zürich Ångström",
    "日本語のテキスト、句読点。「括弧」",
    "Здравствуй, мир! Γειά σου Κόσμε",
    "emoji 😀👍🏽 🇺🇸 ❤️ ✨ family 👨‍👩‍👧",
    "combining é and ä marks",
    "non breaking　ideographic thin space",
    "controls \x00 \x07 \x1b[0m \x1c\x1d\x1e\x1f \x7f \x85 end",
    "a<|endoftext|>b",
    "walking talking running jumping",
    "Implement a new encoder and decoder layer that follows the traditional LLM tokenizer.",
    "tokenization ⁀glue and the ⁀ mark typed by hand",
    "zyxwvutsrq qqqqqqqq aaaaaaaaaa",
    "x" * 40,
    " ",
    "\n",
]

READS = [
    "the cat sat ⁀.",
    "The walk ⁀ing cat ⁀.",
    "⁀Ġcat ⁀. cat",
    "cat  sat",
    " cat sat",
    "cat sat ",
    "⁀nosuchtokenatall",
    "⁀",
    "⁀Ċ ⁀Ċ the",
    "the ca",
    "don 't",
    "caf ⁀Ã ⁀©",
]

CLASS_POINTS = [0x00, 0x09, 0x0A, 0x1C, 0x20, 0x21, 0x2F, 0x30, 0x39, 0x3A, 0x40, 0x41, 0x5A, 0x5B, 0x60, 0x61,
                0x7A, 0x7B, 0x7E, 0x7F, 0x85, 0x9F, 0xA0, 0xA1, 0xBF, 0xC0, 0xD7, 0xF7, 0xFF, 0x1680, 0x1FFF,
                0x2000, 0x200A, 0x200B, 0x2028, 0x202F, 0x2040, 0x205F, 0x206F, 0x2070, 0x20A0, 0x20CF, 0x2190,
                0x2BFF, 0x2C00, 0x2E00, 0x2E7F, 0x3000, 0x3001, 0x3004, 0x3008, 0x3012, 0x3014, 0x301F, 0x3020,
                0xFE00, 0xFE0F, 0xFF01, 0xFF10, 0xFF1A, 0xFF21, 0xFF3B, 0xFF5B, 0xFF65, 0xFF66, 0x1F000, 0x1F600,
                0x1FAFF, 0x1FB00, 0xE0020, 0xE007F, 0x10FFFF]

BROKEN = [
    [0xE2, 0x82],
    [0xE2, 0x82, 0x41],
    [0xF0, 0x9F, 0x98],
    [0xED, 0xA0, 0x80],
    [0xC0, 0xAF],
    [0x80, 0x80, 0x41],
    [0xF4, 0x90, 0x80, 0x80],
    [0xFF, 0xFE],
    [0x20, 0xC3],
    [0xE0, 0x80, 0x80],
    [0xF0, 0x80, 0x80, 0x80],
    [0xC3, 0xA9, 0xE9],
]
"""Byte strings that are not UTF-8: their decoded text pins the U+FFFD rule (one per maximal broken piece)."""


def cases(tok: BPETokenizer) -> dict:
    out = []
    for text in TEXTS:
        units = tok.units(text)
        form = " ".join(units)
        out.append({
            "text": text,
            "pretokens": pretokenize(" " + text),
            "tokens": tok.tokens(text),
            "ids": tok.encode(text),
            "special_ids": tok.encode(text, allowed_special=True),
            "units": units,
            "reread": tok.units(form),
            "spelled": tok.spell(form),
        })
    reads = []
    for text in READS:
        units = tok.units(text)
        reads.append({"text": text, "units": units, "spelled": tok.spell(text)})
    broken = []
    for data in BROKEN:
        # a byte string as ids: every byte is its own id; a leading space is added so the decoder's strip leaves the bytes
        ids = [0x20, *data]
        broken.append({"ids": ids, "text": tok.decode(ids)})
    return {"vocab_size": tok.vocab_size, "merges": len(tok.merges), "cases": out, "reads": reads, "broken": broken}


def build() -> dict:
    bundled = BPETokenizer.bundled()
    with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as fh:
        corpus = [line for line in fh.read().split("\n") if line.strip()]
    small = BPETokenizer.learn(corpus, vocab_size=320, note="the sample corpus, vocab_size 320")
    return {
        "about": "what radixnet/bpe.py makes of these texts; the Go and Rust ports must make the same "
                 "(regenerate with python3 tests/make_tokens_fixture.py)",
        "glue": GLUE,
        "classes": [[cp, char_class(chr(cp))] for cp in CLASS_POINTS],
        "bundled": cases(bundled),
        "small": {"merges_text": small.dumps(), **cases(small)},
    }


def render(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render(build()))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
