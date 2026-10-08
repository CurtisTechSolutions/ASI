"""Train the bundled neural vocoder, ``phonetok/data/acoustic.vocoder.json``, on the synthesizer's speech.

The bundled codebook was learned from the formant synthesizer reading a small
corpus in two voices (``make_codebook.py``); the bundled vocoder is trained on
the same speech in more voices, so it hears the units back in the voice they
came from and knows every pitch the excitation may be asked for.  The last
sentence of every voice is held out and measured against the codebook's own
vocoder, and the report goes into the file.

Regenerate it (``python3 tests/make_vocoder.py``, ten minutes or so with numpy)
when the codebook or the network changes on purpose; the parity fixture and the
tests read the file as it is.  ``--steps``, ``--channels`` and ``--out`` vary the
recipe without editing it.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_codebook import SENTENCES  # noqa: E402
from phonetok import Lexicon, PhoneticTokenizer  # noqa: E402
from phonetok.acoustic import default_codebook, pcm_samples  # noqa: E402
from phonetok.neural import DEFAULT_VOCODER, train  # noqa: E402
from phonetok.synth import Synthesizer, Voice  # noqa: E402

VOICES = [Voice(pitch=110.0, tempo=1.0), Voice(pitch=175.0, tempo=1.1), Voice(pitch=95.0, tempo=0.9),
          Voice(pitch=140.0, tempo=1.0)]
"""The codebook's two voices, and two more so the excitation is heard at other pitches too."""
STEPS = 2000
CHANNELS = 64
SEED = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=DEFAULT_VOCODER)
    args = parser.parse_args()
    tok = PhoneticTokenizer(lexicon=Lexicon.core())
    book = default_codebook()
    started = time.perf_counter()
    recordings = []
    held = []
    seconds = 0.0
    for voice in VOICES:
        synth = Synthesizer(voice_settings=voice)
        for i, text in enumerate(SENTENCES):
            pcm = synth.speak(tok.tokens(text))
            (held if i == len(SENTENCES) - 1 else recordings).append(pcm_samples(pcm))
            seconds += len(pcm) / 2.0 / 16000
    recordings += held  # the held-out recordings are the last ones
    print(f"{len(recordings)} utterances, {seconds:.1f} s of synthesized speech in {len(VOICES)} voices, "
          f"{len(held)} held out; synthesized in {time.perf_counter() - started:.0f} s")
    note = (
        f"trained on the formant synthesizer's speech: {len(SENTENCES) - 1} sentences in {len(VOICES)} voices "
        f"({seconds:.0f} s), tests/make_vocoder.py; a vocoder trained on recordings suits real speech better"
    )
    vocoder, report = train(book, recordings, steps=args.steps, channels=args.channels, seed=args.seed,
                            hold_out=len(held), note=note, log=print)
    vocoder.dump(args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes) in {time.perf_counter() - started:.0f} s")
    print("held out:", report.get("held_out"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
