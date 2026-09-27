"""Speech from a walk: the model is heard as it traverses its graph.

A model whose symbols are sounds (``--encoding phone:3:1``) emits phones as it
walks; a model of words or letters emits text.  :class:`Speaker` takes either,
one step at a time, turns it into the tokens of the phonetic tokenizer
(``../PhoneticTokenizer``) and feeds the formant synthesizer, which hands back
16-bit PCM as soon as a word can be committed.  So the walk is audible while it
is still walking.

The utterance is closed by the **final sentinel**: when the walk steps onto
END (``</s>``) - or is cut off by its length - :meth:`Speaker.end` flushes what
is pending with the closing intonation.  One walk, one utterance; ``count``
walks are ``count`` utterances, each ended by its own sentinel.

:func:`speak_walks` is the whole loop: locate the prefix, sample a walk with a
listener on every step, and yield the PCM chunks as they are made; the CLI's
``speak`` sits on it.

:func:`say` is the **output decoder**: any text a model produced - a
prediction, a generated sample, a turn of a conversation, in whatever units
the model is in - is fed to a :class:`Speaker` whole and closed by the
sentinel, so every output can be heard after the fact as well as while it is
made.  :func:`speak_texts` streams the same thing utterance by utterance; the
CLI's ``say`` and ``--speak``, the API's ``/api/say`` and the frontend's Hear
buttons sit on them.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from .encoding import ACOUSTIC, Encoding, acoustic_tokenizer, phonetic_tokenizer, phonetok_module
from .graph import END, FIRST, START, RadixCyclicGraph
from .penalty import DEFAULT_TRAVERSAL, traversal_costs
from .search import sample_walk

RATE = 16000


def _synth_module():
    """The synthesizer: ``phonetok.synth``, from wherever the package is found."""
    return phonetok_module("synth", "speaking")


class Speaker:
    """Turns what a model emits into speech, step by step.

    ``encoding`` says what the pieces are: sounds pass straight through; words are
    transcribed as they come; letters are gathered into words at whitespace and
    punctuation.  :meth:`feed` takes one emitted piece of text and returns the PCM
    that can be committed (often nothing yet); :meth:`end` closes the utterance.
    """

    def __init__(self, encoding: Encoding, rate: int = RATE, pitch: float = 120.0, tempo: float = 1.0,
                 gain: float = 0.5) -> None:
        self.encoding = encoding
        self.tokens: list[str] = []
        """Every token that reached the voice, for the record."""
        self._letters = ""  # the letters of the word being spelled out by a character model
        self._spoken_words = 0
        if encoding.unit == ACOUSTIC:
            # acoustic units are spoken through the codebook's vocoder, at the codebook's rate;
            # its gain is a multiplier on the level the units were learned at, so the voice's
            # default of half scale is the codebook's own level
            book = acoustic_tokenizer().codebook
            self.vocoder = phonetok_module("acoustic", "speaking").Vocoder(book, gain=gain * 2.0, pitch=pitch)
            self.rate = book.analysis.rate
            self.synth = None
            self.tokenizer = None
            return
        synth = _synth_module()
        self.vocoder = None
        self.rate = rate
        self.synth = synth.Synthesizer(rate=rate, voice_settings=synth.Voice(pitch=pitch, tempo=tempo, gain=gain))
        self.tokenizer = phonetic_tokenizer("phone")  # the words of a word or letter model are read through it

    def feed(self, piece: str) -> bytes:
        """One emitted piece: the units a step added.  Returns the PCM that is ready."""
        if self.vocoder is not None:
            out = bytearray()
            for unit in self.encoding.units(piece):
                self.tokens.append(unit)
                out += self.vocoder.feed(unit)
            return bytes(out)
        if self.encoding.phonetic:
            return self._feed_tokens(self.encoding.units(piece))
        if self.encoding.unit == "word":
            return self._feed_words(piece.split())
        # letters: a word ends at whitespace; punctuation ends it too and becomes a pause
        out = bytearray()
        for ch in piece:
            if ch.isspace():
                out += self._flush_letters()
            else:
                self._letters += ch
                if ch in ".,;:!?":
                    out += self._flush_letters()
        return bytes(out)

    def _flush_letters(self) -> bytes:
        if not self._letters:
            return b""
        word, self._letters = self._letters, ""
        return self._feed_words([word])

    def _feed_words(self, words: list[str]) -> bytes:
        out = bytearray()
        for word in words:
            tokens = self.tokenizer.tokens(word)
            if not tokens:
                continue
            if self._spoken_words and tokens[0] not in (",", ".", "?"):
                out += self._feed_tokens(["#"])
            self._spoken_words += 1
            out += self._feed_tokens(tokens)
        return bytes(out)

    def _feed_tokens(self, tokens: list[str]) -> bytes:
        out = bytearray()
        for t in tokens:
            self.tokens.append(t)
            out += self.synth.feed(t)
        return bytes(out)

    def end(self) -> bytes:
        """The final sentinel: the utterance is closed, and what was pending is spoken."""
        if self.vocoder is not None:
            self.tokens.append("</s>")
            return self.vocoder.end()
        out = bytearray(self._flush_letters())  # a letter model's last word, so the sentinel is recorded after it
        self.tokens.append("</s>")
        out += self.synth.end()
        self._spoken_words = 0
        return bytes(out)


def output_rate(encoding: Encoding, rate: int = RATE) -> int:
    """The sample rate a model is heard at: ``rate`` for the synthesizer, the codebook's for acoustic units."""
    if encoding.unit == ACOUSTIC:
        return acoustic_tokenizer().codebook.analysis.rate
    return rate


def speak_walks(
    model,
    prefix: str = "",
    count: int = 1,
    max_length: int = 60,
    temperature: float = 1.0,
    seed: int | None = None,
    rate: int = RATE,
    pitch: float = 120.0,
    tempo: float = 1.0,
    gain: float = 0.5,
    on_utterance: Callable[[int, str, str], None] | None = None,
) -> Iterator[bytes]:
    """Walk the model ``count`` times from ``prefix`` and yield the speech as each walk goes.

    Each walk is one utterance: the prefix is spoken first, then every step's
    units as the walk takes them, and the END sentinel (or the length cap) closes
    it.  ``on_utterance(i, text, spelled)`` is told what each walk said once it
    has ended - the text in the model's units and, for a model of sounds, the
    words they spell.
    """
    enc: Encoding = model.encoding
    graph: RadixCyclicGraph = model.graph
    rng = random.Random(seed) if seed is not None else None
    costs = traversal_costs(graph, DEFAULT_TRAVERSAL, 1.0, 1.0)
    overlap = enc.overlap
    for i in range(count):
        speaker = Speaker(enc, rate=rate, pitch=pitch, tempo=tempo, gain=gain)
        chunks: list[bytes] = []
        said: list[str] = []

        def emit(piece: str) -> None:
            if piece:
                said.append(piece)
                pcm = speaker.feed(piece)
                if pcm:
                    chunks.append(pcm)

        if prefix:
            emit(prefix)
        node, offset, lead = model._prefix_start(prefix)
        if lead:
            emit(lead)
        if node >= FIRST:  # a compressed node's label runs on past the located gram: that is the walk's first emission
            emit(enc.piece(graph.labels[node], offset + enc.n))
        # from START nothing precedes the first node stepped onto, so it is emitted whole; every node
        # after it - and every node after a located prefix - adds what lies past the overlap (the
        # rule :meth:`Encoding.decode_path` decodes a path by)
        whole_first = node < FIRST

        def on_step(c: int) -> None:
            nonlocal whole_first
            if c >= FIRST:
                emit(enc.piece(graph.labels[c], 0 if whole_first else overlap))
                whole_first = False
            elif c == END:
                chunks.append(speaker.end())

        walk = sample_walk(
            graph, node, offset, max_chars=max_length, temperature=temperature, rng=rng, costs=costs,
            on_step=on_step,
        )
        if not walk.reached_end:
            chunks.append(speaker.end())  # cut off by the length: the sentinel is ours to send
        for pcm in chunks:
            if pcm:
                yield pcm
        if on_utterance is not None:
            text = enc.join(*said)
            on_utterance(i, text, enc.spell(text))


# -- the output decoder ---------------------------------------------------------------


@dataclass
class Utterance:
    """One text of a model, spoken: what it said, what that spells, and how much audio it made."""

    text: str
    """The text as the model wrote it, in its units."""
    spelled: str
    """The words a text of sounds spells; the text itself for every other unit."""
    tokens: list[str] = field(default_factory=list)
    """Every token that reached the voice, the closing sentinel last."""
    samples: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text, "spelled": self.spelled, "tokens": list(self.tokens),
            "samples": self.samples, "seconds": self.seconds,
        }


@dataclass
class Spoken:
    """The speech of one or more texts: 16-bit mono PCM, and what each utterance was."""

    pcm: bytes
    rate: int
    encoding: str
    """The encoding the texts were read in (``str(model.encoding)``)."""
    decoder: str
    """``"voice"``: the formant synthesizer; ``"vocoder"``: the acoustic codebook's vocoder."""
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def seconds(self) -> float:
        return self.samples / self.rate if self.rate else 0.0

    def wav(self) -> bytes:
        """The speech as a WAV file."""
        return _synth_module().wav_bytes(self.pcm, self.rate)

    def to_dict(self) -> dict:
        """The record without the audio: what the CLI prints and the API returns beside ``wav_base64``."""
        return {
            "rate": self.rate, "samples": self.samples, "seconds": self.seconds,
            "encoding": self.encoding, "decoder": self.decoder, "count": len(self.utterances),
            "utterances": [u.to_dict() for u in self.utterances],
        }


def spelled(encoding: Encoding, text: str) -> str:
    """The words a text spells: through the tokenizer for a model of sounds (so ``"the cat"`` given to a
    phone model spells ``"the cat"`` too), the text itself for every other unit."""
    if not encoding.phonetic:
        return text
    return encoding.spell(" ".join(encoding.units(text)))


def decoder_name(encoding: Encoding) -> str:
    """What speaks a model's texts: the codebook's vocoder for acoustic units, the formant voice for the rest."""
    return "vocoder" if encoding.unit == ACOUSTIC else "voice"


def speak_texts(
    encoding: Encoding,
    texts: str | Iterable[str],
    rate: int = RATE,
    pitch: float = 120.0,
    tempo: float = 1.0,
    gain: float = 0.5,
    on_utterance: Callable[[int, Utterance], None] | None = None,
) -> Iterator[bytes]:
    """Speak texts as they are, one utterance each, and yield the PCM as it is made.

    This is the output decoder in its streaming form: every text - a
    prediction, a sample, a turn - is fed whole to a :class:`Speaker` for the
    encoding and closed by the END sentinel, exactly as a walk is, so a text
    of sounds is spoken directly, a text of acoustic units through the
    codebook's vocoder, and a text of words or letters is read through the
    tokenizer word by word.  ``on_utterance(i, utterance)`` is told what each
    text was once it has been spoken.  A token that is not a unit of the
    acoustic codebook is a :class:`ValueError`.
    """
    if isinstance(texts, str):
        texts = [texts]
    for i, text in enumerate(texts):
        speaker = Speaker(encoding, rate=rate, pitch=pitch, tempo=tempo, gain=gain)
        made = 0
        for pcm in (speaker.feed(text), speaker.end()):
            if pcm:
                made += len(pcm)
                yield pcm
        if on_utterance is not None:
            on_utterance(i, Utterance(text, spelled(encoding, text), list(speaker.tokens), made // 2,
                                      made / 2.0 / speaker.rate))


def say(
    encoding: Encoding,
    texts: str | Iterable[str],
    rate: int = RATE,
    pitch: float = 120.0,
    tempo: float = 1.0,
    gain: float = 0.5,
    polish: int = 0,
) -> Spoken:
    """The output decoder: texts in a model's units become speech, one utterance each.

    The same voice as :func:`speak_walks` reads them (``rate``, ``pitch``,
    ``tempo``, ``gain``), and the same rule closes each: the END sentinel.
    ``polish`` applies to acoustic units only - that many Griffin-Lim
    iterations over each whole utterance once it is known, which the streaming
    vocoder cannot do; ``0`` is exactly what the vocoder makes.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    out_rate = output_rate(encoding, rate)
    spoken = Spoken(b"", out_rate, str(encoding), decoder_name(encoding))
    chunks: list[bytes] = []
    if spoken.decoder == "vocoder" and polish > 0:
        # the whole utterance is known, so it can be polished: the vocoder's gain convention
        # is the Speaker's (the voice's half scale is the codebook's own level)
        acoustic = phonetok_module("acoustic", "speaking")
        book = acoustic_tokenizer().codebook
        for text in texts:
            units = encoding.units(text)
            pcm = acoustic.synthesize(units, book, polish=polish, gain=gain * 2.0, pitch=pitch)
            chunks.append(pcm)
            spoken.utterances.append(Utterance(text, spelled(encoding, text), [*units, "</s>"], len(pcm) // 2,
                                               len(pcm) / 2.0 / out_rate))
    else:
        chunks.extend(speak_texts(encoding, texts, rate=rate, pitch=pitch, tempo=tempo, gain=gain,
                                  on_utterance=lambda i, u: spoken.utterances.append(u)))
    spoken.pcm = b"".join(chunks)
    return spoken
