"""The benches: throughput, ``compare`` (bits per unit under the three backoffs), ``feedback`` and ``rungs``.

Every bench is deterministic given its seed and returns a JSON-serialisable
dict; the CLI prints it.  ``PRD.md`` section 7 says which criterion each one
measures.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence

from .codec import Codec, make_codec
from .model import PairModel
from .pair import BACKOFFS, Settings

__all__ = ["compare", "feedback", "rungs", "split", "throughput"]


def _codec(codec: Codec | str, **options) -> Codec:
    return codec if isinstance(codec, Codec) else make_codec(codec, **options)


def split(texts: Sequence[str], holdout: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    """A deterministic train / held-out split, at least one text on each side when there are two."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return list(texts), list(texts)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    n_hold = max(1, int(round(len(texts) * holdout)))
    hold = set(order[:n_hold])
    return [t for i, t in enumerate(texts) if i not in hold], [t for i, t in enumerate(texts) if i in hold]


def throughput(codec: Codec | str = "chars", L: int = 4, units: int = 200_000, seed: int = 0, **options) -> dict:
    """Random units through ``observe`` and ``credit``: increments per second (PRD S-2)."""
    codec = _codec(codec, **options)
    model = PairModel.prime(codec, L)
    rng = random.Random(seed)
    emits = [x for x in codec.emits() if x != codec.end]
    ids = [codec.start] + [rng.choice(emits) for _ in range(units)] + [codec.end]
    t0 = time.perf_counter()
    increments = model.pair.count.observe(ids)
    t1 = time.perf_counter()
    entries = model.pair.reward.credit(ids, 1.0, "all")
    t2 = time.perf_counter()
    return {
        "codec": codec.describe(), "L": L, "N": model.pair.address.N, "units": units,
        "count": {"increments": increments, "seconds": round(t1 - t0, 4),
                  "per_second": round(increments / max(t1 - t0, 1e-9))},
        "credit": {"entries": entries, "seconds": round(t2 - t1, 4), "per_second": round(entries / max(t2 - t1, 1e-9))},
        "units_per_second": round(units / max(t1 - t0, 1e-9)),
    }


def _bits(model: PairModel, texts: Sequence[str], traversal: str = "reward", backoff: str | None = None) -> float:
    total = 0.0
    n = 0
    for text in texts:
        s = model.score(text, traversal, backoff)
        total += s.bits * (s.units + 1)
        n += s.units + 1
    return total / max(1, n)


def compare(texts: Sequence[str], codec: Codec | str = "chars", L: int = 4, holdout: float = 0.1, seed: int = 0,
            smoothing: float | None = None, **options) -> dict:
    """Bits per unit on a held-out split under ``backoff`` all / deepest / none (PRD S-4, S-5)."""
    train, held = split(texts, holdout, seed)
    codec = _codec(codec, texts=train, **options) if isinstance(codec, str) and codec in ("bpe", "syllables", "gpt2", "external") and "texts" not in options else _codec(codec, **options)
    settings = Settings() if smoothing is None else Settings(smoothing=smoothing)
    model = PairModel.prime(codec, L, settings=settings)
    t0 = time.perf_counter()
    record = model.train(train)
    trained = time.perf_counter() - t0
    result = {"codec": codec.describe(), "L": L, "N": model.pair.address.N, "train_texts": len(train),
              "held_texts": len(held), "units": record["units"], "unk_share": record["unk_share"],
              "smoothing": model.settings.smoothing, "train_seconds": round(trained, 4), "bits": {}}
    for backoff in BACKOFFS:
        t0 = time.perf_counter()
        result["bits"][backoff] = {"held": round(_bits(model, held, backoff=backoff), 4),
                                   "train": round(_bits(model, train, backoff=backoff), 4),
                                   "seconds": round(time.perf_counter() - t0, 4)}
    return result


def feedback(codec: Codec | str = "chars", L: int = 4, smoothing: float = 0.0, **options) -> dict:
    """The author's case: ``mat`` rewarded at 5 and punished at 1, ``log`` never judged (PRD S-6).

    The sentences are ``a cat sat on the mat`` / ``... log`` rather than the
    spec's ``the cat ...``: a primed tree of depth ``L`` sees ``L - 1`` units
    of context, and at four characters ``he `` must be followed by ``m`` or
    ``l`` alone for the branch to be the one the case is about.  It runs at
    ``smoothing = 0`` by default, as the grown graph the case comes from has no
    smoothing: on two sentences, Jeffreys smoothing gives the root enough say
    to turn a once-seen deep step.
    """
    codec = _codec(codec, **options)
    model = PairModel.prime(codec, L, settings=Settings(smoothing=smoothing))
    mat, log = "a cat sat on the mat", "a cat sat on the log"
    model.train([mat, log])
    prefix = "a cat sat on the "
    before = {t: model.predict(prefix, 3, traversal=t) for t in ("reward", "punishment")}
    model.reward("mat", strength=5.0, prefix=prefix)      # the outcome credits what was produced, not the prefix
    model.punish("mat", strength=1.0, prefix=prefix)
    after = {t: model.predict(prefix, 3, traversal=t) for t in ("reward", "punishment")}
    model.reward(["mat"] * 50, strength=5.0, prefix=prefix)
    bought = {t: model.predict(prefix, 3, traversal=t) for t in ("reward", "punishment")}

    def row(r):
        return {"continues": r.text, "cost": round(r.cost, 4)}

    return {
        "codec": codec.describe(), "L": L, "prefix": prefix, "smoothing": smoothing,
        "untouched": {t: row(r) for t, r in before.items()},
        "mat rewarded 5, punished 1": {t: row(r) for t, r in after.items()},
        "then rewarded 50 more times": {t: row(r) for t, r in bought.items()},
        "unbuyable": bought["punishment"].text == after["punishment"].text,
    }


def rungs(texts: Sequence[str], codec: Codec | str = "chars", L: int = 4, seed: int = 0, samples: int = 50,
          strength: float = 2.0, **options) -> dict:
    """Rewards at every level against the final nodes only: does a reward reach a sibling context? (PRD S-7)

    For ``samples`` steps ``(context, unit)`` drawn from the texts, a *sibling*
    is a different context in the texts that shares the step's last unit(s)
    but not its full context.  The step is rewarded under both settings and
    the change in the sibling's probability of the unit is reported, with the
    change on a context sharing nothing as the control.
    """
    codec = _codec(codec, **options)
    texts = [t for t in texts if t.strip()]
    rng = random.Random(seed)
    D = L - 1
    if D < 1:
        raise ValueError("rungs needs L >= 2: with one level there is no shorter context to reach")
    # every (context, unit) step of every text with a full context, keyed by its last unit
    steps: list[tuple[tuple[int, ...], int]] = []
    by_last: dict[int, list[tuple[int, ...]]] = {}
    for text in texts:
        ids = codec.padded(text)
        for t in range(D, len(ids)):
            ctx = tuple(ids[t - D:t])
            steps.append((ctx, ids[t]))
            by_last.setdefault(ctx[-1], []).append(ctx)
    if not steps:
        raise ValueError("no steps with a full context in the texts")
    results = {"all": [], "final": []}
    controls = {"all": [], "final": []}
    picked = 0
    for _ in range(samples * 4):
        if picked >= samples:
            break
        ctx, x = rng.choice(steps)
        if D >= 2:
            # a sibling shares the step's last unit and differs before it: the shorter suffix levels are shared
            siblings = [c for c in by_last.get(ctx[-1], []) if c != ctx and c[:-1] != ctx[:-1]]
        else:
            # one unit of context: the root is the only level two contexts can share
            siblings = [c for c, _ in steps if c != ctx]
        others = [c for c, _ in steps if c[-1] != ctx[-1] and c != ctx]
        if not siblings or not others:
            continue
        sib = rng.choice(siblings)
        other = rng.choice(others)
        picked += 1
        for setting in ("all", "final"):
            model = PairModel.prime(codec, L, settings=Settings(rungs=setting))
            model.train(texts)
            j = model.pair.index[x]
            before_sib = model.pair.fold(sib)[j]
            before_other = model.pair.fold(other)[j]
            model.pair.reward.credit(list(ctx) + [x], strength, setting)
            after_sib = model.pair.fold(sib)[j]
            after_other = model.pair.fold(other)[j]
            results[setting].append(math.log2(after_sib / before_sib))
            controls[setting].append(math.log2(after_other / before_other))

    def mean(v):
        return round(sum(v) / len(v), 4) if v else 0.0

    return {
        "codec": codec.describe(), "L": L, "samples": picked, "strength": strength,
        "sibling_lift_bits": {s: mean(v) for s, v in results.items()},
        "control_lift_bits": {s: mean(v) for s, v in controls.items()},
        "note": ("the lift is log2(P after / P before) of the rewarded unit under a context sharing the step's last unit "
                 "(sibling) and one with a different last unit (control); with rungs=final nothing reaches the sibling"
                 if D >= 2 else
                 "one unit of context: the root is the only level two contexts share, so the sibling is any other "
                 "context and the control is one with a different last unit - both reach the root"),
    }
