"""The command line: ``phonetok <command> ...`` (or ``python -m phonetok``).

    phonetok tokenize "The cat sat."            the tokens, one line
    phonetok tokenize --level syllable --ids     ... at another level, as ids
    phonetok explain "Strengths, butter?"        every word: phones, source, syllables, IPA
    phonetok ipa "The quick brown fox"           the IPA
    phonetok decode "DH AH0 # K AE1 T"           sounds back to words
    phonetok pronounce strength                  one word, every pronunciation the lexicon has
    phonetok rhymes cat                          rhymes in the lexicon (or: rhymes cat hat)
    phonetok affinity S T                        how well two sounds go together
    phonetok coin --count 5 --seed 1             new pronounceable words
    phonetok blend smoke fog                     a portmanteau
    phonetok lexicon                             what the lexicon holds and where it came from
    phonetok eval [--limit N]                    the rules against the full dictionary, if one is found
    phonetok say "The cat sat." --out say.wav    speak: the formant synthesizer, to a WAV file, a player or stdout

Every command takes ``--json`` for one JSON document on stdout, and the
tokenizer options ``--level``, ``--no-stress``, ``--no-boundaries``,
``--no-pauses``, ``--core`` (the bundled lexicon only) and ``--lexicon FILE``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from . import __version__
from .lexicon import Lexicon, find_full_dictionary, read_entries
from .phones import strip_stress, to_ipa
from .rules import letter_to_sound
from .syllables import rhymes, syllabify
from .synth import RATE, Synthesizer, Voice, duration, find_player, wav_bytes, wav_header, write_wav
from .tokenizer import LEVELS, PHONEME, PhoneticTokenizer


def _tokenizer(args: argparse.Namespace) -> PhoneticTokenizer:
    if args.lexicon:
        lexicon = Lexicon.core()
        lexicon.extend(read_entries(open(args.lexicon, encoding="utf-8")))
        lexicon.sources.append(f"file {args.lexicon}")
    else:
        lexicon = Lexicon.default(full=not args.core)
    return PhoneticTokenizer(
        level=args.level, lexicon=lexicon, stress=not args.no_stress,
        boundaries=not args.no_boundaries, pauses=not args.no_pauses,
    )


def _emit(args: argparse.Namespace, doc: dict, lines: list[str]) -> None:
    if args.json:
        json.dump(doc, sys.stdout, ensure_ascii=False, indent=1)
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))


def cmd_tokenize(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    text = " ".join(args.text)
    tokens = tok.tokenize(text)
    ids = tok.encode(text)
    doc = {"text": text, "level": tok.level, "tokens": [t.text for t in tokens], "ids": ids,
           "kinds": [t.kind for t in tokens], "vocab_size": tok.vocab_size}
    if args.ids:
        lines = [" ".join(str(i) for i in ids)]
    else:
        lines = [" ".join(t.text for t in tokens)]
    _emit(args, doc, lines)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    rows = tok.explain(" ".join(args.text))
    lines = [f"{'word':<16} {'how':<11} {'phones':<32} {'syllables':<32} ipa"]
    for r in rows:
        lines.append(f"{r['word']:<16} {r['how']:<11} {' '.join(r['phones']):<32} {' '.join(r['syllables']):<32} {r['ipa']}")
    _emit(args, {"words": rows, "tokenizer": tok.describe()}, lines)
    return 0


def cmd_ipa(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    text = " ".join(args.text)
    ipa = tok.ipa(text)
    _emit(args, {"text": text, "ipa": ipa}, [ipa])
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    tokens = " ".join(args.tokens).split()
    if tokens and all(t.lstrip("-").isdigit() for t in tokens):
        words = tok.decode([int(t) for t in tokens])
    else:
        words = tok.decode(tokens)
    _emit(args, {"tokens": tokens, "text": words}, [words])
    return 0


def cmd_pronounce(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    doc: dict = {"words": []}
    lines: list[str] = []
    for word in args.words:
        phones, how = tok.transcriber.explain(word)
        variants = tok.lexicon.pronunciations(word) or [tuple(phones)]
        syl = [s.text() for s in syllabify(phones)]
        doc["words"].append({"word": word, "how": how, "phones": phones, "variants": [list(v) for v in variants],
                             "syllables": syl, "ipa": tok.ipa(word), "rules": letter_to_sound(word)})
        lines.append(f"{word}: {' '.join(phones)}  [{how}]  /{tok.ipa(word)}/  {' | '.join(syl)}")
        for v in variants[1:]:
            lines.append(f"{' ' * len(word)}  {' '.join(v)}  [lexicon, alternative]")
        if how != "rules":
            lines.append(f"{' ' * len(word)}  {' '.join(letter_to_sound(word))}  [the rules alone would say]")
    _emit(args, doc, lines)
    return 0


def cmd_rhymes(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    if len(args.words) >= 2:
        a, b = args.words[0], args.words[1]
        yes = tok.rhymes(a, b)
        _emit(args, {"a": a, "b": b, "rhymes": yes}, [f"{a} / {b}: {'rhyme' if yes else 'do not rhyme'}"])
        return 0
    word = args.words[0]
    phones = tok.pronounce(word)
    found = sorted({w for w, p in tok.lexicon.items() if w != word.lower() and rhymes(phones, p)})
    if args.limit > 0:
        found = found[: args.limit]
    _emit(args, {"word": word, "phones": phones, "rhymes": found}, [" ".join(found) if found else "(none in the lexicon)"])
    return 0


def cmd_affinity(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    a, b = args.a.upper(), args.b.upper()
    value = tok.affinity(a, b)
    p = tok.phonotactics.prob(a, b)
    verdict = "go together" if value > 1 else "avoid each other" if value < -1 else "meet by chance"
    _emit(args, {"a": a, "b": b, "affinity_bits": value, "probability": p, "verdict": verdict},
          [f"{a} -> {b}: {value:+.2f} bits (P = {p:.3f}): they {verdict}"])
    return 0


def cmd_coin(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    rng = random.Random(args.seed)
    words = []
    for _ in range(args.count):
        phones = tok.phonotactics.build(rng, syllables=args.syllables, avoid=(p for _, p in tok.lexicon.items()))
        words.append({"phones": phones, "spelling": tok.transcriber.spell(phones), "ipa": to_ipa(phones)})
    _emit(args, {"words": words, "seed": args.seed},
          [f"{w['spelling']:<14} {' '.join(w['phones']):<24} /{w['ipa']}/" for w in words])
    return 0


def cmd_blend(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    pa, pb = tok.pronounce(args.a), tok.pronounce(args.b)
    phones, kept, dropped = tok.phonotactics.blend(pa, pb)
    spelling = tok.transcriber.spell(phones)
    _emit(args, {"a": args.a, "b": args.b, "phones": phones, "spelling": spelling, "ipa": to_ipa(phones)},
          [f"{args.a} + {args.b} = {spelling}  ({' '.join(phones)}, /{to_ipa(phones)}/)"])
    return 0


def cmd_lexicon(args: argparse.Namespace) -> int:
    tok = _tokenizer(args)
    lex = tok.lexicon
    full = find_full_dictionary()
    if full is not None and full[1] is not None:
        full[1].close()
    doc = {"words": len(lex), "sources": lex.sources, "full_dictionary": full[0] if full else None,
           "phonotactics": repr(tok.phonotactics)}
    lines = [f"{len(lex)} words from {', '.join(lex.sources)}",
             f"full dictionary: {full[0] if full else 'none found (pip install cmudict, or set PHONETOK_LEXICON)'}",
             f"phonotactics: {tok.phonotactics!r}"]
    if args.word:
        for w in args.word:
            lines.append(f"{w}: {' | '.join(' '.join(p) for p in lex.pronunciations(w)) or '(not in the lexicon)'}")
        doc["lookups"] = {w: [list(p) for p in lex.pronunciations(w)] for w in args.word}
    _emit(args, doc, lines)
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    """Speak a text (or a stream of tokens): the formant synthesizer, streamed as the tokens come."""
    tok = _tokenizer(args)
    synth = Synthesizer(rate=args.rate, voice_settings=Voice(pitch=args.pitch, tempo=args.tempo, gain=args.gain))
    if args.text:
        tokens = tok.tokens(" ".join(args.text))
    else:  # tokens on stdin, one line at a time: a model's walk can be piped in and heard as it goes
        def lines():
            for line in sys.stdin:
                yield from line.split()
        tokens = lines()
    if args.play:
        player = find_player()
        if player is None:
            sys.stderr.write("phonetok: no player found (aplay, paplay, ffplay, play or afplay)\n")
            return 2
        import subprocess

        proc = subprocess.Popen(player, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(wav_header(args.rate))
        total = 0
        for chunk in synth.stream(tokens):
            proc.stdin.write(chunk)
            proc.stdin.flush()
            total += len(chunk)
        proc.stdin.close()
        proc.wait()
        doc = {"seconds": total / 2.0 / args.rate, "player": player[0]}
        _emit(args, doc, [f"{doc['seconds']:.2f} s spoken through {player[0]}"])
        return 0
    if args.raw:  # 16-bit PCM straight out, as it is made: `phonetok say --raw ... | aplay -r 16000 -f S16_LE`
        out = sys.stdout.buffer
        for chunk in synth.stream(tokens):
            out.write(chunk)
            out.flush()
        return 0
    pcm = synth.speak(tokens)
    path = args.out or "say.wav"
    write_wav(path, pcm, args.rate)
    doc = {"path": path, "seconds": duration(pcm, args.rate), "rate": args.rate, "bytes": len(wav_bytes(pcm, args.rate))}
    _emit(args, doc, [f"{doc['seconds']:.2f} s of speech written to {path} ({args.rate} Hz)"])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """How the rules do against the full dictionary: word accuracy and phone error rate."""
    found = find_full_dictionary()
    if found is None:
        sys.stderr.write("no full dictionary found: pip install cmudict, or set PHONETOK_LEXICON\n")
        return 2
    name, stream = found
    entries = [(w, p) for w, p in read_entries(stream) if w.isalpha()]
    stream.close()
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    if args.limit > 0:
        entries = entries[: args.limit]
    seen: set[str] = set()
    words = right = right_stress = 0
    errors = total = 0
    for word, gold in entries:
        if word in seen:
            continue
        seen.add(word)
        got = letter_to_sound(word)
        g0, s0 = strip_stress(gold), strip_stress(got)
        words += 1
        right += g0 == s0
        right_stress += list(gold) == got
        errors += _edit_distance(g0, s0)
        total += len(g0)
    doc = {"dictionary": name, "words": words, "word_accuracy": right / max(words, 1),
           "word_accuracy_with_stress": right_stress / max(words, 1), "phone_error_rate": errors / max(total, 1)}
    _emit(args, doc, [f"{words} words of {name}: {doc['word_accuracy']:.1%} right, "
                      f"{doc['word_accuracy_with_stress']:.1%} with stress, phone error rate {doc['phone_error_rate']:.1%}"])
    return 0


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonetok", description="A phonetic tokenizer: text as the sounds it is made of.")
    parser.add_argument("--version", action="version", version=f"phonetok {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--level", choices=LEVELS, default=PHONEME, help="what one token is (default phoneme)")
    common.add_argument("--no-stress", action="store_true", help="drop the vowels' stress digits")
    common.add_argument("--no-boundaries", action="store_true", help="no # between words")
    common.add_argument("--no-pauses", action="store_true", help="drop punctuation instead of making pauses of it")
    common.add_argument("--core", action="store_true", help="the bundled core lexicon only, even if a full one is installed")
    common.add_argument("--lexicon", metavar="FILE", help="a CMU-format dictionary to read on top of the core")
    common.add_argument("--json", action="store_true", help="one JSON document on stdout")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokenize", parents=[common], help="the tokens of a text")
    p.add_argument("text", nargs="+")
    p.add_argument("--ids", action="store_true", help="print ids instead of tokens")
    p.set_defaults(func=cmd_tokenize)
    p = sub.add_parser("explain", parents=[common], help="every word: phones, where they came from, syllables, IPA")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_explain)
    p = sub.add_parser("ipa", parents=[common], help="the text in the IPA")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_ipa)
    p = sub.add_parser("decode", parents=[common], help="tokens (or ids) back to words")
    p.add_argument("tokens", nargs="+")
    p.set_defaults(func=cmd_decode)
    p = sub.add_parser("pronounce", parents=[common], help="how words are pronounced")
    p.add_argument("words", nargs="+")
    p.set_defaults(func=cmd_pronounce)
    p = sub.add_parser("rhymes", parents=[common], help="rhymes of a word in the lexicon, or whether two words rhyme")
    p.add_argument("words", nargs="+")
    p.add_argument("--limit", type=int, default=50, help="at most this many (0 = all)")
    p.set_defaults(func=cmd_rhymes)
    p = sub.add_parser("affinity", parents=[common], help="how well sound B follows sound A")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_affinity)
    p = sub.add_parser("coin", parents=[common], help="new pronounceable words")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--syllables", type=int, default=None)
    p.set_defaults(func=cmd_coin)
    p = sub.add_parser("blend", parents=[common], help="a portmanteau of two words")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_blend)
    p = sub.add_parser("lexicon", parents=[common], help="what the lexicon holds; look words up")
    p.add_argument("word", nargs="*")
    p.set_defaults(func=cmd_lexicon)
    p = sub.add_parser("say", parents=[common], help="speak a text with the formant synthesizer (tokens on stdin when no text)")
    p.add_argument("text", nargs="*")
    p.add_argument("--out", metavar="FILE", help="the WAV file to write (default say.wav)")
    p.add_argument("--play", action="store_true", help="play through aplay / paplay / ffplay / play / afplay as it is made")
    p.add_argument("--raw", action="store_true", help="16-bit mono PCM on stdout, streamed")
    p.add_argument("--rate", type=int, default=RATE, help=f"the sample rate (default {RATE})")
    p.add_argument("--pitch", type=float, default=120.0, help="the base pitch in Hz (default 120)")
    p.add_argument("--tempo", type=float, default=1.0, help="the pace, 1 = the table's (default 1)")
    p.add_argument("--gain", type=float, default=0.5, help="the peak level as a share of full scale (default 0.5)")
    p.set_defaults(func=cmd_say)
    p = sub.add_parser("eval", parents=[common], help="the rules against the full dictionary")
    p.add_argument("--limit", type=int, default=20000, help="words to sample (0 = all)")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"phonetok: {exc}\n")
        return 1
