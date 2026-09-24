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
``speak`` and the API's ``/api/speak`` sit on it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

from .encoding import Encoding, phonetic_tokenizer, phonetok_module
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
        synth = _synth_module()
        self.encoding = encoding
        self.rate = rate
        self.synth = synth.Synthesizer(rate=rate, voice_settings=synth.Voice(pitch=pitch, tempo=tempo, gain=gain))
        self.tokenizer = phonetic_tokenizer("phone")  # the words of a word or letter model are read through it
        self._letters = ""  # the letters of the word being spelled out by a character model
        self._spoken_words = 0
        self.tokens: list[str] = []
        """Every token that reached the synthesizer, for the record."""

    def feed(self, piece: str) -> bytes:
        """One emitted piece: the units a step added.  Returns the PCM that is ready."""
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
        out = bytearray(self._flush_letters())
        self.tokens.append("</s>")
        out += self.synth.end()
        self._spoken_words = 0
        return bytes(out)


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
