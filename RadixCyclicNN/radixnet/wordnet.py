"""WordNGramNet - the count / reward model over an alphabet whose symbols are words.

``SPEC-WordNGrams.md``: not one of the graph's structural rules mentions a
*character*.  A node's label is a sequence of symbols, an edge exists where two
labels overlap by ``WINDOW - 1`` symbols, the index maps a ``WINDOW``-symbol key
to ``(node, offset)``, compression merges a unary chain at that seam.  All of
that is true of any alphabet, so a word n-gram model is not a different model -
it is :class:`~radixnet.countnet.CountRewardNet` with the encoder and the
decoder replaced::

    "the cat sat on the mat"
      characters ->  ["the", "he ", "e c", " ca", ...]
      words      ->  [("the","cat","sat"), ("cat","sat","on"), ...]

A word is a symbol, and a symbol is one code point
(:class:`~radixnet.encoding.Vocabulary`): word ``0`` is ``<unk>`` at
``U+0100`` and word ``i`` follows it, the surrogate block skipped.  Text
becomes symbols on the way in and symbols become text on the way out; between
those two points the graph, the weights, the counters, the paths, both
traversals and the search are the count model's, unchanged.

Three consequences, all of them deliberate:

* **the window is still three** - the overlap is two words and the pivot is the
  middle one, which is D-006's argument word for word.  Word bigrams need
  ``WINDOW = 2`` and are a separate change (``SPEC-WordNGrams.md`` §7);
* **whitespace is normalised** - ``decode(encode(t))`` joins the words with
  single spaces, so a corpus whose whitespace carries meaning (source code,
  base64, a waveform) belongs on the character model, which is why both kinds
  stay;
* **an unread word is ``<unk>``** at prediction and scoring time, and two
  different unread words are the same symbol.  That is what having a
  vocabulary costs; it is visible in ``unknown_transitions``.

Everything counted in symbols is counted in **words**: ``length`` emits words,
``max_length`` caps words, and ``score``'s ``chars`` / ``per_char`` are per
word.  ``stats()["units"]`` says so, because a number whose unit depends on the
model is a number that will be read wrong.
"""

from __future__ import annotations

from collections.abc import Iterable

from .countnet import CountRewardGraph, CountRewardNet
from .encoding import BACK_LABEL, END_LABEL, START_LABEL, WINDOW, Vocabulary, split_words, symbol_word, word_symbol
from .graph import FIRST
from .search import REWARD, PathResult

__all__ = ["WORD_MODEL_FORMAT", "WORD_UNITS", "WordGraph", "WordNGramNet"]

WORD_MODEL_FORMAT = "radixnet-word"
"""A format of its own, so every reader written before this existed refuses the file by the check it already makes."""

WORD_UNITS = "words"
_W = WINDOW
_SENTINELS = (START_LABEL, END_LABEL, BACK_LABEL)
"""A path carries the sentinels; they are labels, not words, and are reported as they are."""


class WordGraph(CountRewardGraph):
    """A :class:`~radixnet.countnet.CountRewardGraph` whose symbols are words.

    The only thing added is the alphabet: a :class:`~radixnet.encoding.Vocabulary`
    that the graph carries, saves in id order and loads back.  Nothing else about
    the graph knows or needs to know what a symbol stands for.
    """

    def __init__(
        self,
        seed: int = 0,
        count_scale: float = 0.0,
        reward_scale: float = 1.0,
        global_scale: float = 0.5,
        window_scale: float = 0.5,
        path_scale: float = 1.0,
        window: int = 10_000,
        vocabulary: Vocabulary | Iterable[str] | None = None,
    ) -> None:
        self.vocabulary = vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary(vocabulary)
        super().__init__(
            seed=seed, count_scale=count_scale, reward_scale=reward_scale, global_scale=global_scale,
            window_scale=window_scale, path_scale=path_scale, window=window,
        )

    def symbols_of(self, text: str) -> str:
        """Text as symbols: its words, an unread one as ``<unk>`` (the vocabulary only grows while training)."""
        return self.vocabulary.encode(text)

    def text_of(self, label: str) -> str:
        """A label as text: the words its code points stand for, joined by single spaces.

        A sentinel label (``<s>``, ``</s>``, ``<back>``) is not made of words
        and comes back as it is - a path carries them, and so does every view
        of the graph.
        """
        return label if label in _SENTINELS else self.vocabulary.decode(label)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["units"] = WORD_UNITS
        d["vocabulary"] = self.vocabulary.to_list()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WordGraph":
        units = d.get("units")
        if units is not None and units != WORD_UNITS:
            raise ValueError(f"not a {WORD_UNITS} graph document (units {units!r})")
        words = d.get("vocabulary")
        if words is None:
            raise ValueError("a word graph document carries its vocabulary")
        g = super().from_dict(d)
        g.vocabulary = Vocabulary(words)
        g._check_vocabulary()
        return g

    def _check_vocabulary(self) -> None:
        """Every symbol every label carries has to be a word this vocabulary holds."""
        size = len(self.vocabulary)
        for node in range(FIRST, len(self.labels)):
            if not self.alive[node]:
                continue
            for symbol in self.labels[node]:
                index = symbol_word(symbol)  # raises for a code point that carries no word at all
                if index >= size:
                    raise ValueError(f"node {node} holds word {index}, past the end of a vocabulary of {size}")


class WordNGramNet(CountRewardNet):
    """The word n-gram model: the count / reward model over words (see the module docstring).

    Training, feedback, 2NRL, both traversals and the beam search are the count
    model's; what differs is that a text is split into words on the way in and
    the words come back on the way out.
    """

    kind = "word"
    format = WORD_MODEL_FORMAT
    graph_class = WordGraph
    units = WORD_UNITS
    label = "Word n-gram"
    description = (
        "the count / reward model over an alphabet whose symbols are words: the same graph, weights and "
        "search, with a whitespace split for an encoder - lengths, counts and scores are per word, an "
        "unread word is <unk>, and whitespace is normalised"
    )

    # -- the alphabet --------------------------------------------------------

    @property
    def vocabulary(self) -> Vocabulary:
        """The words this model has read, in the order it first read them."""
        return self.graph.vocabulary

    def symbols(self, text: str, grow: bool = False) -> str:
        """``text`` as the graph's symbols; ``grow`` gives an unread word the next id (training)."""
        return self.vocabulary.encode(text, grow=grow)

    def words(self, symbols: str) -> str:
        """The words a symbol string stands for, joined by single spaces."""
        return self.vocabulary.decode(symbols)

    def normalise(self, text: str) -> str:
        """``text`` as this model can represent it: its words joined by single spaces.

        A word model's round trip costs the original whitespace and nothing
        else, and ``normalise`` is what it costs, so a caller can see it
        (``SPEC-WordNGrams.md`` §5).
        """
        return " ".join(split_words(text))

    # -- texts on the way in -------------------------------------------------

    def _clean_texts(self, texts: Iterable[str] | str) -> tuple[list[str], int]:
        """The texts as symbols, growing the vocabulary; a text of fewer than ``WINDOW`` words is skipped."""
        items = [texts] if isinstance(texts, str) else list(texts)
        vocabulary = self.vocabulary
        kept: list[str] = []
        skipped = 0
        for text in items:
            if not isinstance(text, str):
                raise TypeError(f"texts must be strings, got {type(text).__name__}")
            words = split_words(text)
            if len(words) < _W:
                skipped += 1
            else:
                kept.append("".join(word_symbol(vocabulary.add(word)) for word in words))
        return kept, skipped

    # -- texts on the way out ------------------------------------------------

    def _decode_result(self, result: PathResult) -> PathResult:
        """One walk's texts and labels, back in words."""
        result.text = self.words(result.text)
        result.full_text = self.words(result.full_text)
        result.labels = [self.graph.text_of(label) for label in result.labels]
        return result

    def _search(
        self,
        prefix: str,
        length: int,
        mode: str,
        k: int,
        beam: int | None,
        step_penalty: float,
        temperature: float,
        to_end: bool,
        max_length: int | None,
        rng=None,
        traversal: str = REWARD,
        **options,
    ):
        """The count model's search, run over symbols: words in, words out, ``length`` counted in words.

        ``options`` are the traversal's own (``penalty_scale`` / ``merit_scale``,
        :mod:`radixnet.penalty`): they price a step and know nothing about what a
        symbol stands for, so they ride through untouched.
        """
        found = super()._search(
            self.symbols(prefix), length, mode, k, beam, step_penalty, temperature, to_end, max_length,
            rng=rng, traversal=traversal, **options,
        )
        for result in found.top + found.bottom:
            self._decode_result(result)
        return self._decode_result(found)

    def locate(self, prefix: str) -> tuple[int, int]:
        """Node and window offset where a prediction for ``prefix`` starts (the prefix's *words*)."""
        return super().locate(self.symbols(prefix))

    def score(self, text: str) -> dict:
        """Log-probability of ``text``, its words walked through the structure.

        ``chars`` counts **words** and ``per_char`` is per word; an unread word
        is ``<unk>``, whose windows are not in the index, so it is charged what
        any unknown transition is charged and counted in
        ``unknown_transitions``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return super().score(self.symbols(text))

    def correct(self, wrong: str, right: str, **options) -> dict:
        """Teach one correction over **words**: the two sentences are aligned word by word.

        ``right`` is what the teacher wrote, so its words join the vocabulary;
        ``wrong`` is what the network wrote, so its words are already in it and
        anything else is ``<unk>``.  ``wrong_chars`` / ``right_chars`` count
        words, and the changes are reported as words.
        """
        result = super().correct(self.symbols(str(wrong or "")), self.symbols(str(right or ""), grow=True), **options)
        result["changes"] = [
            {**change, "wrong": self.words(change.get("wrong", "")), "right": self.words(change.get("right", ""))}
            for change in result.get("changes", [])
        ]
        return result

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict:
        return {**super().stats(), "units": WORD_UNITS, "vocabulary": len(self.vocabulary)}

    def top_words(self, limit: int = 20) -> list[dict]:
        """The most read words: ``[{"word", "id", "trigrams"}]``, ties in id order.

        ``trigrams`` is how many of the graph's windows the word appears in -
        what the graph actually knows about it, and the one count that survives
        compression.
        """
        counts = [0] * len(self.vocabulary)
        for trigram in self.graph.trigram_index:
            for symbol in trigram:
                index = symbol_word(symbol)
                if index < len(counts):
                    counts[index] += 1
        rows = [
            {"word": word, "id": i, "trigrams": counts[i]}
            for i, word in enumerate(self.vocabulary.words)
            if i or counts[i]  # <unk> only shows up where the graph really holds it
        ]
        rows.sort(key=lambda r: (-r["trigrams"], r["id"]))
        return rows[:limit] if limit > 0 else rows

    def __repr__(self) -> str:
        g = self.graph
        return (
            f"WordNGramNet(seed={self.seed}, nodes={g.num_nodes()}, edges={g.num_edges()}, "
            f"trigrams={g.num_trigrams()}, words={len(self.vocabulary)}, inverted={g.inverted})"
        )
