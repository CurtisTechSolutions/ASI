"""Write ``tests/parity.json``: what the Python tokenizer says about a battery of texts.

The Go and Rust ports' test suites read the same file and must say the same,
token for token, so that a model whose symbols are sounds can be trained in one
port and continued in another.  Regenerate it (``python3 tests/make_parity.py``)
only when the Python behaviour changes on purpose; ``tests/test_parity.py``
holds Python to it as well.
"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phonetok import PhoneticTokenizer, Lexicon, Phonotactics, LEVELS, letter_to_sound, respell  # noqa: E402
from phonetok.acoustic import AcousticTokenizer, default_codebook, frames_of, learn, pcm_samples, synthesize  # noqa: E402
from phonetok.synth import Synthesizer, Voice  # noqa: E402
from phonetok.g2p import Transcriber  # noqa: E402
from phonetok.phones import SYMBOLS, features, to_ipa  # noqa: E402
from phonetok.syllables import syllabify  # noqa: E402

TEXTS = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog!",
    "Strengths, butter? \"Hello\" (world) - 42 dogs; 3.14 is pi... Really?!",
    "I'll be there at 10:30 - don't be late, it's the 21st!",
    "well-known u.s. FBI mp3 xq NASA nasa e.g. a A I",
    "cats boxes cities walked loved stopped running carrying bigger happiest quickly unhappy toothbrush keyboard",
    "cat's dogs' fox's it'll would've they'd I'm can't didn't",
    "zebra tokenizer phonetic syllable radix cyclic photograph understand beautiful dictionary",
    "the K AE1 T sat DH.AH0 # K.AE1.T S.T.R- EH1 -NG.TH.S <s> </s>",
    "one - two -- three — four … five",
    "€10 $2.50 50% -7 1,000 007 b2b 3d",
    "practice makes perfect; knowledge is power: where there is a will, there is a way.",
    "water boils at one hundred degrees celsius",
    "grzeskowiak brettschneider dipaola villari",
    "",
    "   ",
    ", , .",
    "# K AE1 T",
    "DH AH0 #",
    "AH0 # K",
    ", DH AH0",
    "K AE1 T ,",
    "# #",
    ",",
    "# the cat . sat ,",
    "hello , world . K.AE1.T # #",
]

WORDS = ["cat", "strengths", "butter", "banana", "computer", "astro", "extra", "onion", "about", "happy", "texts",
         "apple", "athlete", "hmm", "zebra", "tokenizer", "photograph", "understand", "nation", "information",
         "specific", "rebuild", "tuna", "physics", "cheese", "knight", "quick", "boxes", "waited", "loved"]

RESPELL = ["K AE1 T", "M EY1 K", "S T R EH1 NG TH", "HH AE1 P IY0", "B AH1 T ER0", "T AY1 M", "G OW1", "S IY1",
           "P EY1 P ER0", "K Y UW1 T", "B UH1 K", "S K AY1", "N AY1 T", "AH0 B AW1 T", "JH AH1 M P", "K IY1 P",
           "B AE1 K", "M Y UW1 Z IH0 K", "F L AW1 ER0", "DH AH0", "Z IH1 B R AH0", "D IH1 K SH AH0 N EH2 R IY0",
           "HH Y UW1 JH", "S IH1 T IY0", "W EY1 SH D AH0 N", "AO1 Z"]

PAIRS = [("S", "T"), ("T", "L"), ("#", "S"), ("NG", "#"), ("#", "NG"), ("K", "W"), ("SH", "R"), ("M", "B"),
         ("AE", "T"), ("IY", "NG"), ("AE1", "T"), ("#", "AH"), ("Z", "#")]

SCORES = ["S T R EH1 NG TH", "NG S T R EH1 L K", "K AE1 T", "K T AE1", "K AH0 M P Y UW1 T ER0", "AH0"]

BLENDS = [("smoke", "fog"), ("breakfast", "lunch"), ("motor", "hotel"), ("spoon", "fork"), ("chill", "relax"),
          ("web", "seminar")]


def main() -> None:
    lex = Lexicon.core()
    doc = {
        "symbols": list(SYMBOLS),
        "levels": {},
        "explain": [],
        "syllables": {},
        "rules": {w: letter_to_sound(w) for w in WORDS + ["grzeskowiak", "brettschneider", "xylophone", "qwerty"]},
        "respell": {p: respell(p.split()) for p in RESPELL},
        "features": {p: list(features(p)) for p in ("K", "B", "AH0", "AH1", "IY1", "NG", "W", "ER0")},
        "ipa": {},
        "affinity": {},
        "score": {},
        "next": {},
        "coin": {},
        "blend": {},
    }
    for level in LEVELS:
        tok = PhoneticTokenizer(level=level, lexicon=Lexicon.core())
        rows = []
        for text in TEXTS:
            toks = tok.tokenize(text)
            rows.append({
                "text": text, "tokens": [t.text for t in toks], "kinds": [t.kind for t in toks],
                "words": [t.word for t in toks], "ids": tok.encode(text), "decoded": tok.decode(tok.tokens(text)),
            })
        doc["levels"][level] = {"rows": rows, "vocab": list(tok.vocab.tokens)}
        no = PhoneticTokenizer(level=level, lexicon=Lexicon.core(), stress=False, boundaries=False, pauses=False)
        doc["levels"][level]["bare"] = [no.text(t) for t in TEXTS]
    tok = PhoneticTokenizer(lexicon=Lexicon.core())
    for text in TEXTS:
        rows = tok.explain(text)
        doc["explain"].append([{k: r[k] for k in ("word", "phones", "how", "syllables", "ipa")} for r in rows])
        doc["ipa"][text] = tok.ipa(text)
    for w in WORDS:
        doc["syllables"][w] = [s.text() for s in syllabify(tok.pronounce(w))]
    p = Phonotactics.from_lexicon(lex)
    for a, b in PAIRS:
        doc["affinity"][f"{a} {b}"] = [p.affinity(a, b), p.prob(a, b)]
    for s in SCORES:
        doc["score"][s] = p.score(s.split())
    for a in ("#", "S", "AE", "NG", "K"):
        doc["next"][a] = p.next(a, 5)
    for seed in range(12):
        doc["coin"][str(seed)] = p.build(random.Random(seed), avoid=(ph for _, ph in lex.items()))
    for n in (1, 2, 3):
        doc["coin"][f"seed 5 syllables {n}"] = p.build(random.Random(5), syllables=n)
    for a, b in BLENDS:
        phones, i, j = p.blend(tok.pronounce(a), tok.pronounce(b))
        doc["blend"][f"{a} {b}"] = {"phones": phones, "kept": i, "dropped": j, "spelling": tok.blend(a, b)}
    doc["transcriber_spell"] = {}
    t = Transcriber(lex)
    for w in ("two", "two", "right", "zebra"):
        t.word(w)
    for phones in (["T", "UW1"], ["T", "UW0"], ["R", "AY1", "T"], ["Z", "EH1", "B", "R", "AH0"], ["K", "AE1", "T", "S"]):
        doc["transcriber_spell"][" ".join(phones)] = t.spell(phones)
    doc["acoustic"] = acoustic_block(tok)
    path = os.path.join(ROOT, "tests", "parity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}: {os.path.getsize(path)} bytes, {len(TEXTS)} texts, {len(LEVELS)} levels")


ACOUSTIC_TEXTS = ["the cat sat on the mat", "a bird in the hand"]
ACOUSTIC_VOICE = {"pitch": 130.0, "tempo": 1.0, "gain": 0.5}


def acoustic_block(tok: PhoneticTokenizer) -> dict:
    """The acoustic units: two utterances every port can synthesize bit-identically (their tokens are
    written down so no port needs the text tokenizer for this), their frames, their units under the bundled
    codebook, a small codebook learned from them, and slices of their resynthesis."""
    clips = []
    tokens = []
    for text in ACOUSTIC_TEXTS:
        toks = tok.tokens(text)
        tokens.append(toks)
        clips.append(pcm_samples(Synthesizer(voice_settings=Voice(**ACOUSTIC_VOICE)).speak(toks)))
    book = default_codebook()
    heard = AcousticTokenizer(book).listen(clips[0])
    frames = frames_of(clips[0], book.analysis)
    small = learn(clips, k=4, seed=3, iterations=10, note="parity")
    replay_units = heard.units[:12]

    def slice_of(pcm: bytes, at: int = 4000, count: int = 200) -> dict:
        import struct
        values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        return {"samples": len(values), "at": at, "values": list(values[at:at + count])}

    return {
        "voice": ACOUSTIC_VOICE, "texts": ACOUSTIC_TEXTS, "tokens": tokens, "samples": [len(c) for c in clips],
        "frames": {"count": len(frames), "rows": {"0": frames[0], "50": frames[50], "last": frames[-1]}},
        "units": heard.units, "codes": heard.codes,
        "learn": {"k": 4, "seed": 3, "iterations": 10, "centroids": small.centroids, "counts": small.counts,
                  "runs": small.runs, "mean": small.mean, "inertia": small.inertia},
        "replay": {"units": replay_units, "polish": 0, **slice_of(synthesize(replay_units, book))},
        "polished": {"units": replay_units, "polish": 2, **slice_of(synthesize(replay_units, book, polish=2))},
    }


if __name__ == "__main__":
    main()
