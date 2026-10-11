"""Train the N-gram model automatically: from a corpus, and from a local language model through Ollama.

:func:`autotrain` is the whole loop.  Round 0 trains on every text file under
the data paths (``data/corpus`` by default), every tenth line held out, and
measures the model on what was held out.  Each round after that, with an
:class:`Ollama` client:

1. **corpus** - the LLM writes ``lines`` new sentences about each of the
   round's topics (given, or the corpus's own most frequent words), and the
   model trains on them;
2. **continuations** - the model's own contexts handed to the LLM to finish:
   the first few words of training lines, continued by the LLM, trained on -
   the LLM shows what comes next exactly where the model is asking;
3. **judge** - the model writes ``judge`` texts of its own (greedy, from the
   first words of training lines) and the LLM says which are fluent, sensible
   English; every edge of an approved text is rewarded (widened), every edge of
   a rejected one punished (narrowed) - the lattice's credit, from a critic;

then the model is measured on the same held-out lines, saved if asked, and the
loop stops early once the held-out perplexity has not improved for
``patience`` rounds.  Every round is recorded in ``model.history``.

Ollama is reached over its HTTP API (``/api/tags``, ``/api/generate``),
standard library only: the address from ``OLLAMA_HOST`` (Ollama's own
convention; ``host:port`` is accepted) else ``http://127.0.0.1:11434``, the
model from ``LATTICEFSM_OLLAMA_MODEL`` else ``llama3.2``.  The prompts are fixed
strings, the same in the Rust crate, so the two ports ask the same questions.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from .ngram import NGram, corpus_sequences, detokenize, split_lines

__all__ = ["CONTINUE_PROMPT", "CORPUS_PROMPT", "DEFAULT_MODEL", "DEFAULT_URL", "JUDGE_PROMPT", "Ollama", "OllamaError",
           "autotrain", "parse_judgements", "parse_lines"]

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"

CORPUS_PROMPT = ("Write {n} different, natural English sentences about {topic}. Write one sentence per line, with no "
                 "numbering, no bullet points and nothing else.")
CONTINUE_PROMPT = ("Continue the following text with the next few words only. Reply with the continuation alone, "
                   "without repeating the text and without quotes.\n\nText: {text}")
JUDGE_PROMPT = ("Here are {n} numbered texts written by a small language model. For each one, decide whether it is "
                "fluent, sensible English. Reply only with JSON, a list like [{{\"id\": 1, \"ok\": true}}].\n\n{texts}")


class OllamaError(Exception):
    """Ollama is unreachable, refused the request, or answered something unusable."""


def normalise_url(url: str | None) -> str:
    url = (url or "").strip() or DEFAULT_URL
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    return url.replace("://0.0.0.0", "://127.0.0.1")


class Ollama:
    """A minimal client for Ollama's HTTP API."""

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 120.0) -> None:
        self.url = normalise_url(url or os.environ.get("OLLAMA_HOST"))
        self.model = model or os.environ.get("LATTICEFSM_OLLAMA_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout)

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise OllamaError(f"Ollama {path}: HTTP {e.code} {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise OllamaError(f"Ollama at {self.url} is not answering ({e}); start it with `ollama serve` and pull a "
                              f"model (`ollama pull {self.model}`)") from None

    def models(self) -> list[str]:
        data = self._request("GET", "/api/tags")
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise OllamaError("unexpected /api/tags response")
        return [m.get("name", "") for m in data["models"] if isinstance(m, dict)]

    def available(self) -> bool:
        try:
            self.models()
            return True
        except OllamaError:
            return False

    def generate(self, prompt: str, temperature: float = 0.8, num_predict: int | None = None,
                 seed: int | None = None) -> str:
        options: dict = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if seed is not None:
            options["seed"] = int(seed)
        data = self._request("POST", "/api/generate",
                             {"model": self.model, "prompt": prompt, "stream": False, "options": options})
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise OllamaError("unexpected /api/generate response (no 'response' text)")
        return _strip_thinking(data["response"])


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


_BULLET = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*")


def parse_lines(text: str, limit: int | None = None) -> list[str]:
    """An LLM's lines: numbering, bullets and wrapping quotes stripped; empty lines and lines ending in ``:`` (a
    preamble) dropped."""
    out = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip().strip("\"'“”").strip()
        if not line or line.endswith(":"):
            continue
        out.append(line)
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_judgements(text: str, n: int) -> list[bool | None]:
    """The judge's verdicts on ``n`` texts, by number: the JSON list it was asked for, or failing that lines like
    ``1: yes``; ``None`` for a text it said nothing about."""
    verdicts: list[bool | None] = [None] * n
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            items = json.loads(text[start:end + 1])
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and isinstance(item.get("id"), (int, float)):
                    i = int(item["id"]) - 1
                    ok = item.get("ok")
                    if 0 <= i < n and isinstance(ok, bool):
                        verdicts[i] = ok
            return verdicts
        except ValueError:
            pass
    for m in re.finditer(r"(\d+)\s*[:.)-]\s*(yes|no|true|false|ok|fail)", text, flags=re.I):
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            verdicts[i] = m.group(2).lower() in ("yes", "true", "ok")
    return verdicts


STOPWORDS = frozenset("""about above after again against along among another around because before being below between
both cannot could doing down during each every first further having however itself might never other others ought
since should still their theirs there these thing things those though through under until upon where whether which
while whose within without would yours always often rather quite""".split())
"""Words of five letters or more too common to be a topic."""


def _topics(model: NGram, count: int = 8) -> list[str]:
    """The corpus's own most frequent words of five letters or more (not :data:`STOPWORDS`): what to ask the LLM to
    write about."""
    seen: dict[int, int] = {}
    for row in model.grams.values():
        for w, c in row.items():
            seen[w] = seen.get(w, 0) + c
    words = [w for w in seen if model.vocab[w].isalpha() and len(model.vocab[w]) >= 5
             and model.vocab[w] not in STOPWORDS]
    words.sort(key=lambda w: (-seen[w], w))
    return [model.vocab[w] for w in words[:count]] or ["everyday life"]


def _prefixes(model: NGram, lines: Sequence[Sequence[str]], count: int, words: int, offset: int) -> list[str]:
    """``count`` beginnings of training sequences, ``words`` tokens each (for ``base64``, ``4 * words`` letters: as
    many whole bytes), picked by a fixed stride from ``offset``, written back as text."""
    out, n = [], len(lines)
    if n == 0:
        return out
    step = max(1, n // max(1, count))
    take = 4 * words if model.unit == "base64" else words
    for j in range(count):
        toks = list(lines[(offset + j * step) % n])[:take]
        if toks:
            out.append(detokenize(toks, model.unit))
    return out


def autotrain(model: NGram, data: Sequence[str] = ("data/corpus",), held_out: int = 10,
              ollama: Ollama | None = None, topics: Sequence[str] = (), rounds: int = 3, lines: int = 20,
              continuations: int = 20, judge: int = 10, prefix_words: int = 4, topics_per_round: int = 2,
              patience: int = 2, keep: str | None = None, save: str | None = None, seed: int = 1,
              log: Callable[[str], None] | None = None) -> list[dict]:
    """Train ``model`` on the corpus under ``data``, then for up to ``rounds`` rounds from Ollama (when given): new
    sentences, continuations of its own contexts, and a judge's credit on its own texts; measured on the held-out
    corpus lines every round.  Returns the rounds' records (also appended to ``model.history``)."""
    say = log or (lambda _msg: None)
    train: list[list[str]] = []
    test: list[list[str]] = []
    for path, file_lines in corpus_sequences(data, model.unit, model.lower):
        tr, te = split_lines(file_lines, held_out) if held_out else (file_lines, [])
        model.train_lines(tr, source=path)
        train += tr
        test += te
    records: list[dict] = []

    def record(r: int, added: dict) -> dict:
        ev = model.evaluate(test) if test else {"perplexity": None, "bits": None, "accuracy": None, "top5": None}
        st = model.stats()
        rec = {"round": r, "tokens": st["tokens"], "vocab": st["vocab"], "ngrams": st["ngrams"],
               "credited_edges": st["credited_edges"], "perplexity": ev["perplexity"], "bits": ev["bits"],
               "accuracy": ev["accuracy"], "top5": ev["top5"], "held_out": len(test), **added}
        records.append(rec)
        model.history.append(rec)
        ppl = "-" if rec["perplexity"] is None else f"{rec['perplexity']:.2f}"
        acc = "-" if rec["accuracy"] is None else f"{rec['accuracy']:.3f}"
        say(f"round {r}: {st['tokens']} tokens, {st['vocab']} words, {st['ngrams']} n-grams; held out: perplexity "
            f"{ppl}, next-token accuracy {acc}" + "".join(f", {k} {v}" for k, v in added.items()))
        if save:
            model.save(save)
        return rec

    record(0, {"corpus_lines": len(train)})
    if ollama is None or rounds <= 0:
        return records
    subjects = list(topics) or _topics(model)
    best = records[-1]["perplexity"]
    stale = 0
    for r in range(1, rounds + 1):
        written: list[str] = []
        for j in range(topics_per_round):
            topic = subjects[((r - 1) * topics_per_round + j) % len(subjects)]
            reply = ollama.generate(CORPUS_PROMPT.format(n=lines, topic=topic), temperature=0.8,
                                    seed=seed * 1000 + r * 10 + j)
            got = parse_lines(reply, lines)
            model.train_lines(got, source=f"ollama:{ollama.model}:corpus:{topic}")
            written += got
        continued: list[str] = []
        for j, prefix in enumerate(_prefixes(model, train, continuations, prefix_words, r * 7919)):
            reply = ollama.generate(CONTINUE_PROMPT.format(text=prefix), temperature=0.7, num_predict=32,
                                    seed=seed * 1000 + r * 10 + j)
            rest = parse_lines(reply, 1)
            if rest:
                continued.append(f"{prefix} {rest[0]}")
        model.train_lines(continued, source=f"ollama:{ollama.model}:continuations")
        approved = judged = 0
        if judge > 0:
            texts = [model.complete(p, max_tokens=24) for p in _prefixes(model, train, judge, 2, r * 104729)]
            texts = [t for t in texts if t.strip()]
            if texts:
                numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
                reply = ollama.generate(JUDGE_PROMPT.format(n=len(texts), texts=numbered), temperature=0.0,
                                        seed=seed * 1000 + r)
                for text, ok in zip(texts, parse_judgements(reply, len(texts))):
                    if ok is None:
                        continue
                    model.credit_text(text, 1.0 if ok else -1.0)
                    judged += 1
                    approved += ok
        if keep and (written or continued):
            os.makedirs(os.path.dirname(os.path.abspath(keep)), exist_ok=True)
            with open(keep, "a", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in written + continued))
        rec = record(r, {"ollama_lines": len(written), "continuations": len(continued), "judged": judged,
                         "approved": approved})
        if best is not None and rec["perplexity"] is not None:
            if rec["perplexity"] < best - 1e-9:
                best, stale = rec["perplexity"], 0
            else:
                stale += 1
                if stale >= patience:
                    say(f"held-out perplexity has not improved for {patience} rounds: stopping")
                    break
    return records
