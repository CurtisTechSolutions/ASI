"""Learn the bundled codebook, ``phonetok/data/acoustic.tsv``, from the formant synthesizer's speech.

The package ships no recordings, so its default acoustic units are learned
from what it can make itself: the synthesizer reading a small corpus in two
voices.  That is speech the analysis can hear, and it is deterministic - the
synthesizer is bit-identical in every port - so the file this writes is
reproducible.  It is also the synthesizer's voice and not a person's: a
codebook learned from real recordings (``phonetok learn *.wav``) is the one to
use for real speech, and this one is the bootstrap that lets everything work
out of the box.

Regenerate it (``python3 tests/make_codebook.py``, a minute or two of pure
Python) only when the analysis or the synthesizer changes on purpose; the
parity fixture and the tests read the file as it is.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phonetok import PhoneticTokenizer, Lexicon  # noqa: E402
from phonetok.acoustic import DEFAULT_CODEBOOK, learn, pcm_samples  # noqa: E402
from phonetok.synth import Synthesizer, Voice  # noqa: E402

SENTENCES = [
    "the cat sat on the mat",
    "the quick brown fox jumps over the lazy dog",
    "a bird in the hand is worth two in the bush",
    "practice makes perfect",
    "water boils at one hundred degrees celsius",
    "she sells sea shells by the sea shore",
    "how much wood would a woodchuck chuck",
    "the five boxing wizards jump quickly",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "strength and thirst through thick and thin",
    "yesterday all my troubles seemed so far away",
    "please call stella and ask her to bring these things with her from the store",
    "six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother bob",
    "we also need a small plastic snake and a big toy frog for the kids",
    "she can scoop these things into three red bags, and we will go meet her wednesday at the train station",
    "one two three four five six seven eight nine ten",
    "zero eleven twelve thirteen twenty thirty forty fifty hundred thousand",
    "the north wind and the sun were disputing which was the stronger",
    "when a traveller came along wrapped in a warm cloak",
    "they agreed that the one who first succeeded in making the traveller take his cloak off should be considered stronger than the other",
    "then the north wind blew as hard as he could, but the more he blew the more closely did the traveller fold his cloak around him",
    "at last the north wind gave up the attempt",
    "then the sun shone out warmly, and immediately the traveller took off his cloak",
    "and so the north wind was obliged to confess that the sun was the stronger of the two",
    "yes. no? maybe, later. really!",
    "judge, church, measure, vision, thing, this, father, mother",
    "oil, boy, toy, cow, now, how, high, sky, day, say, go, so, you, too",
]

VOICES = [Voice(pitch=110.0, tempo=1.0), Voice(pitch=175.0, tempo=1.1)]
UNITS = 64
SEED = 1
ITERATIONS = 60


def main() -> int:
    tok = PhoneticTokenizer(lexicon=Lexicon.core())
    recordings = []
    seconds = 0.0
    for voice in VOICES:
        synth = Synthesizer(voice_settings=voice)
        for text in SENTENCES:
            pcm = synth.speak(tok.tokens(text))
            recordings.append(pcm_samples(pcm))
            seconds += len(pcm) / 2.0 / 16000
    print(f"{len(recordings)} utterances, {seconds:.1f} s of synthesized speech in {len(VOICES)} voices")
    started = time.perf_counter()
    note = (
        f"learned from the formant synthesizer's speech: {len(SENTENCES)} sentences in {len(VOICES)} voices "
        f"({seconds:.0f} s), tests/make_codebook.py; a codebook learned from recordings suits real speech better"
    )
    book = learn(recordings, k=UNITS, seed=SEED, iterations=ITERATIONS, note=note)
    print(f"{book.k} units in {time.perf_counter() - started:.0f} s; inertia {book.inertia:.1f}; "
          f"runs {min(book.runs):.1f} .. {max(book.runs):.1f} frames; counts {min(book.counts)} .. {max(book.counts)}")
    book.dump(DEFAULT_CODEBOOK)
    print(f"wrote {DEFAULT_CODEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
