"""DecayNet - reading, saying, scoring and persistence over a :class:`DecayTree`.

Three verbs, and they are the model:

* :meth:`DecayNet.read` - a text goes in: every window of it, from START and
  from the root at every position, and every node a window passes through is
  **traversed** (``seen`` settles, gains one, the clock ticks).
* :meth:`DecayNet.say` (:meth:`predict`) - the model speaks token by token:
  one query of the tree per token, the deepest usable context of the last
  ``window`` grams of everything said so far, one decision by the shares of
  ``seen`` among that context's options, the token fed back; and every node
  the walk arrives at - the context it starts from, each option it takes,
  each context it relocates to - is **traversed**: a query adds to ``seen``
  exactly as a reading does.  ``quiet=True`` asks without arriving: a
  measurement.
* :meth:`DecayNet.score` - the log-probability of a text under the shares,
  the deepest context deciding and a miss costing ``UNKNOWN_PROB``, the
  convention of the two sibling models.  A measurement: nothing moves.

Between them the clock runs, and everything the model holds fades on it
(:class:`DecayTree`).  :meth:`tick` lets time pass without a traversal.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone

from .encoding import Encoding
from .search import PathResult, cheapest_path
from .tree import LIFE, MIN_SEEN, ROOT, START, DecayTree, Walks

__all__ = ["MAX_LEGS", "MAX_SLIDE", "MODEL_FORMAT", "MODEL_FORMAT_VERSION", "UNKNOWN_PROB", "DecayNet", "load_model"]

MODEL_FORMAT = "radixdecay"
MODEL_FORMAT_VERSION = 1

UNKNOWN_PROB = 1e-6
"""Probability charged by :meth:`DecayNet.score` for a token the deepest context was never followed by - the
sibling models' constant, so the three read on one scale."""

MAX_SLIDE = 2000
"""Units a walk to END may say before it is stopped: a window that forgets can go round, and this is its clock."""

MAX_LEGS = 64
"""How many times the cheapest path may re-enter a bounded tree before it stops."""

_LOG_UNKNOWN = math.log(UNKNOWN_PROB)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes_atomic(path: str, data: bytes, use_gzip: bool = False) -> None:
    """Write via a temporary file and ``os.replace``; a reader never sees a half-written file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            if use_gzip:
                with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                    gz.write(data)
            else:
                fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json_file(path: str) -> dict:
    """Load a JSON document, gunzipping it when it carries the gzip magic."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


class DecayNet:
    """The model: a :class:`DecayTree` and the three verbs over it.

    ``life`` and ``decay`` are the tree's clock (:mod:`radixdecay.tree`);
    ``min_seen`` how much of a visit a context must still hold to be
    consulted; ``depth`` bounds a root path; ``seed`` only draws samples.
    """

    kind = "radix-decay"
    label = "DecayNet (seen that fades, fed by reading and saying alike)"

    def __init__(
        self,
        encoding: Encoding | None = None,
        depth: int | None = None,
        life: float = LIFE,
        decay: str = "half-life",
        min_seen: float = MIN_SEEN,
        seed: int = 0,
    ) -> None:
        self.tree = DecayTree(encoding=encoding, depth=depth, life=life, decay=decay, min_seen=min_seen)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.history: list[dict] = []
        self.meta: dict = {
            "created": _utc_now(), "seed": self.seed, "texts_read": 0, "units_read": 0, "readings": 0,
            "traversals_said": 0, "ticks": 0,
        }

    @property
    def encoding(self) -> Encoding:
        return self.tree.encoding

    @property
    def depth(self) -> int | None:
        return self.tree.depth

    @property
    def clock(self) -> int:
        """The tree's clock: traversals so far, read or said, plus the time let pass."""
        return self.tree.traversals

    # -- reading -------------------------------------------------------------

    def _clean_texts(self, texts: Iterable[str] | str) -> tuple[list[str], int]:
        items = [texts] if isinstance(texts, str) else list(texts)
        enc = self.encoding
        kept: list[str] = []
        skipped = 0
        for t in items:
            if not isinstance(t, str):
                raise TypeError("texts must be strings")
            if enc.length(t) >= enc.n:
                kept.append(t)
            else:
                skipped += 1
        return kept, skipped

    def read(self, texts: Iterable[str] | str, times: int = 1) -> dict:
        """Read texts, ``times`` over: every window of every text, every node on its way traversed.

        Returns a record ``{texts, units, skipped_short, traversals, clock, nodes, ends, grams, remembered}``,
        also appended to :attr:`history`.
        """
        if times < 0:
            raise ValueError(f"times must be >= 0, got {times}")
        texts, skipped = self._clean_texts(texts)
        tree = self.tree
        encode = self.encoding.encode
        before = tree.traversals
        for _ in range(times):
            for text in texts:
                tree.read(encode(text))
        units = sum(self.encoding.length(t) for t in texts) * times
        self.meta["texts_read"] += len(texts) * times
        self.meta["units_read"] += units
        self.meta["readings"] += 1
        record = {
            "texts": len(texts), "times": times, "units": units, "skipped_short": skipped,
            "traversals": tree.traversals - before, "clock": tree.traversals,
            "nodes": tree.num_nodes(), "ends": tree.num_ends(), "grams": tree.num_grams(), "remembered": tree.remembered(),
        }
        self.history.append(record)
        return record

    def tick(self, traversals: int) -> int:
        """Let ``traversals`` of time pass with nothing arriving anywhere; returns the clock."""
        self.tree.tick(traversals)
        self.meta["ticks"] += int(traversals)
        return self.tree.traversals

    # -- one query, one decision ---------------------------------------------

    def context(self, grams: list[str], window: int | None = None) -> tuple[int, int] | None:
        """One query of the tree: the deepest usable context of the last ``window`` grams (all of them: ``None``).
        A window cut out of a longer history is never matched from START."""
        tree = self.tree
        if window is not None and len(grams) > window:
            loc = tree.locate(grams[len(grams) - window :], anchored=False)
        else:
            loc = tree.locate(grams)
        return None if loc is None else (loc[0], loc[1])

    def next_token(
        self, node: int, offset: int, temperature: float = 0.0, rng: random.Random | None = None
    ) -> tuple[int, str | None, float]:
        """The token that follows a located context: ``(node, gram, cost)``, ``gram`` ``None`` for END.

        Inside a run the run's next gram at cost 0; at a node's end the
        option with the largest share of ``seen`` at temperature 0, or a draw
        from ``share ** (1 / temperature)`` above it.  ``cost`` is
        ``-log share``.
        """
        tree = self.tree
        if offset < tree.held(node) - 1:
            return node, self.encoding.gram_at(tree.labels[node], offset + 1), 0.0
        options = tree.shares(node)
        if not options:
            return node, None, 0.0
        if temperature == 0 or len(options) == 1:
            child, share = max(options, key=lambda item: (item[1], -item[0]))
        else:
            if rng is None:
                rng = self.rng
            weights = [s ** (1.0 / temperature) for _, s in options]
            r = rng.random() * math.fsum(weights)
            child, share = options[-1]
            acc = 0.0
            for item, wgt in zip(options, weights):
                acc += wgt
                if r < acc:
                    child, share = item
                    break
        if tree.is_end(child):
            return child, None, -math.log(share)
        return child, tree.first_gram(child), -math.log(share)

    def _start(self, prefix: str) -> tuple[int, int, int]:
        """Where a walk begins for ``prefix``: ``(node, offset, cut)`` - the deepest usable context, or for a
        prefix shorter than a gram the most seen child of START whose label begins with it (``cut`` units of
        which the prefix already covers), or START for nothing known."""
        tree = self.tree
        enc = self.encoding
        grams = enc.encode(prefix)
        if grams:
            loc = tree.locate(grams)
            return (loc[0], loc[1], enc.n - 1) if loc is not None else (START, -1, 0)
        if prefix:
            best = -1
            for c in tree.children[START].values():
                if (enc.has_unit_prefix(tree.labels[c], prefix) and tree.usable(c, 0)
                        and (best < 0 or tree.seen(c) > tree.seen(best))):
                    best = c
            if best >= 0:
                return best, 0, enc.length(prefix)
        return START, -1, 0

    # -- saying --------------------------------------------------------------

    def say(
        self,
        prefix: str = "",
        length: int = 20,
        window: int | None = None,
        temperature: float = 0.0,
        to_end: bool = False,
        max_length: int | None = None,
        seed: int | None = None,
        quiet: bool = False,
    ) -> PathResult:
        """Continue ``prefix`` token by token, and mean it: every node the walk arrives at is traversed.

        Each step is one :meth:`context` of everything said so far (over the
        last ``window`` grams) and one :meth:`next_token` from it.  The
        context the walk starts from, every option it takes and every context
        it relocates to are traversed (``seen`` settles, gains one, the clock
        ticks): saying adds to ``seen`` exactly as reading does.  A run is one
        node, arrived at once.  ``quiet=True`` is the same walk as a
        measurement: nothing moves.  Stops at END, at ``length`` units (unless
        ``to_end``), at ``max_length``, or when the tree knows nothing about
        the context; says nothing at all when how a text begins has faded.
        ``expanded`` counts the queries, ``traversals`` what the walk added.
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 grams or None, got {window}")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        tree = self.tree
        enc = self.encoding
        n = enc.n
        rng = random.Random(seed) if seed is not None else self.rng
        emitted = ""
        n_emitted = 0
        labels: list[str] = []
        node_ids: list[int] = []
        step_costs: list[float] = []
        queries = 0
        traversals = 0
        reached_end = False
        grams = enc.encode(prefix)
        walks = Walks(tree, grams, window)  # every suffix of what has been said, walked once, moved per token
        if grams:
            found = walks.deepest()
            cut = n - 1 if found is not None else 0
            if found is None:
                loc = (START, -1)
                walks = Walks(tree, window=window)  # nothing known: the walk begins a text
            else:
                loc = (found[0], found[1])
        else:
            node, offset, cut = self._start(prefix)
            loc = (node, offset)
            if tree.is_real(node):
                gram = tree.first_gram(node)
                piece = enc.piece_of(gram, cut)
                emitted = enc.join_units(emitted, piece)
                n_emitted += len(enc.view(piece))
                walks.push(gram)
                labels.append(tree.labels[node])
                node_ids.append(node)
                step_costs.append(0.0)
                cut = n - 1
        if to_end and max_length is None:
            max_length = max(length, MAX_SLIDE)
        if loc[0] == START and not tree.usable(START, -1):
            return self._spelled(PathResult(full_text=enc.join(prefix, "")))  # how a text begins has faded
        current = loc[0]
        if not quiet:
            tree.touch(current)  # the context the walk starts from is consulted: traversed
            traversals += 1
        while True:
            if max_length is not None and n_emitted >= max_length:
                break
            if not to_end and n_emitted >= length:
                break
            node, offset = loc
            queries += 1
            child, gram, cost = self.next_token(node, offset, temperature, rng)
            if child != node:
                current = child
                if not quiet:
                    tree.touch(child)  # the option taken is arrived at: traversed
                    traversals += 1
            labels.append(tree.labels[child])
            node_ids.append(child)
            step_costs.append(cost)
            if gram is None:
                reached_end = tree.is_end(child)
                break
            piece = enc.piece_of(gram, cut) if cut else gram
            cut = n - 1
            units = len(enc.view(piece))
            if max_length is not None and n_emitted + units > max_length:
                piece = enc.truncate(piece, max_length - n_emitted)
                units = max_length - n_emitted
            emitted = enc.join_units(emitted, piece)
            n_emitted += units
            walks.push(gram)
            found = walks.deepest()
            if found is None:
                break
            loc = (found[0], found[1])
            if loc[0] != current:
                current = loc[0]
                if not quiet:
                    tree.touch(current)  # a context arrived at by relocating: traversed
                    traversals += 1
        if not quiet:
            self.meta["traversals_said"] += traversals
        result = PathResult(
            text=emitted, labels=labels, node_ids=node_ids, cost=math.fsum(step_costs), step_costs=step_costs,
            expanded=queries, reached_end=reached_end, full_text=enc.join(prefix, emitted), traversals=traversals,
        )
        return self._spelled(result)

    predict = say

    def cheapest(
        self,
        prefix: str = "",
        length: int = 20,
        to_end: bool = False,
        max_length: int | None = None,
        quiet: bool = False,
        max_legs: int = MAX_LEGS,
    ) -> PathResult:
        """The cheapest continuation by ``-log share`` - one search instead of a query per token - and then, unless
        ``quiet``, every node on the path it returns is traversed, the context it starts from included: what the
        model said, it has said.  A bounded tree re-enters at the deepest context of what was said when a path
        runs out, at most ``max_legs`` times."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if max_legs < 1:
            raise ValueError(f"max_legs must be >= 1, got {max_legs}")
        tree = self.tree
        enc = self.encoding
        node, offset, cut = self._start(prefix)
        if node == START and not tree.usable(START, -1):
            return self._spelled(PathResult(full_text=enc.join(prefix, "")))  # how a text begins has faded
        include_context = not tree.is_real(node) or cut != enc.n - 1
        cut = 0 if not tree.is_real(node) or cut == enc.n - 1 else cut
        emitted = ""
        n_emitted = 0
        labels: list[str] = []
        node_ids: list[int] = []
        step_costs: list[float] = []
        expanded = 0
        legs = 0
        reached_end = False
        while True:
            need = max(0, length - n_emitted)
            cap = None if max_length is None else max(0, max_length - n_emitted)
            leg = cheapest_path(tree, node, offset, need, cap, to_end, include_context)
            text = enc.piece_of(leg.text, cut) if cut else leg.text
            cut = 0
            legs += 1
            emitted = enc.join_units(emitted, text)
            n_emitted += len(enc.view(text))
            labels.extend(leg.labels)
            node_ids.extend(leg.node_ids)
            step_costs.extend(leg.step_costs)
            expanded += leg.expanded
            if leg.reached_end:
                reached_end = True
                break
            if (not to_end and n_emitted >= length) or (cap is not None and n_emitted >= max_length):
                break
            if not text or legs >= max_legs:
                break
            loc = tree.locate(enc.encode(enc.join(prefix, emitted)))
            if loc is None:
                break
            node, offset, include_context = loc[0], loc[1], False
        traversals = 0
        if not quiet:
            last = -1
            for nid in node_ids:
                if nid != last:  # every node the path arrives at, the context it starts from included
                    tree.touch(nid)
                    traversals += 1
                last = nid
            self.meta["traversals_said"] += traversals
        if max_length is not None:
            emitted = enc.truncate(emitted, max_length)
        return self._spelled(PathResult(
            text=emitted, labels=labels, node_ids=node_ids, cost=math.fsum(step_costs), step_costs=step_costs,
            expanded=expanded, reached_end=reached_end, full_text=enc.join(prefix, emitted), traversals=traversals,
        ))

    def generate(self, count: int = 1, max_length: int = 40, temperature: float = 0.0, seed: int | None = None,
                 window: int | None = None, quiet: bool = False) -> list[PathResult]:
        """``count`` texts said from START.  At temperature 0 a saying model may still differ from one text to the
        next: each one is traversed, and the next is said by a tree that has changed."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        return [
            self.say("", length=max_length, window=window, temperature=temperature, max_length=max_length,
                     seed=None if seed is None else seed + i, quiet=quiet)
            for i in range(count)
        ]

    def _spelled(self, result: PathResult) -> PathResult:
        enc = self.encoding
        result.full_spelled = enc.spell(result.full_text)
        result.spelled = enc.spell_tail(result.full_text, result.text)
        return result

    # -- scoring: a measurement ----------------------------------------------

    def _relocate(self, history: list[str], ending: bool = False, longest: int | None = None) -> tuple[int, int]:
        if history and (longest is None or longest > 0):
            loc = self.tree.locate(history, ending, longest=longest)
            if loc is not None:
                return loc[0], loc[1]
        return ROOT, -1

    def score(self, text: str) -> dict:
        """Log-probability of ``text`` under the shares; nothing moves.

        The walk follows the context it is in while that context is usable,
        backs off to the longest usable shorter suffix when it is not (a
        context that has faded below ``min_seen``, one whose options have all
        faded, or one at the depth bound), and on a miss relocates to the
        deepest usable context of the whole history.  A step inside a run
        costs 0 and is not a transition; a token the context was never
        followed by, or whose ``seen`` has faded to nothing, costs
        ``log(UNKNOWN_PROB)`` and counts as unknown.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tree = self.tree
        enc = self.encoding
        grams = enc.encode(text)
        units = enc.length(text)
        if not grams:
            return {"log_prob": 0.0, "per_unit": 0.0, "units": units, "transitions": 0, "unknown_transitions": 0,
                    "units_name": enc.units_name}
        labels, children, end_leaf, held = tree.labels, tree.children, tree.end_leaf, tree.held
        view, last, ov = enc.view, enc.last_unit, enc.overlap
        depth, symbols_of, min_seen = tree.depth, tree.symbols_of, tree.min_seen
        log_prob = 0.0
        transitions = unknown = 0
        node, offset = START, -1
        history: list[str] = []

        def settle(node: int, offset: int, ending: bool = False) -> tuple[int, int]:
            if (
                tree.seen(node) < min_seen
                or (offset == held(node) - 1 and not tree.has_options(node))
                or (not ending and depth is not None and symbols_of(node, offset) >= depth)
            ):
                return self._relocate(history, ending, longest=symbols_of(node, offset) - 1)  # back off: shorter only
            return node, offset

        for g in grams:
            node, offset = settle(node, offset)
            ok = False
            if offset < held(node) - 1:
                if view(labels[node])[offset + 1 + ov] == last(g):
                    offset += 1
                    ok = True
            else:
                c = children[node].get(g)
                if c is not None:
                    lp = tree.log_share(node, c)
                    if lp is not None:
                        log_prob += lp
                        transitions += 1
                        node, offset = c, 0
                        ok = True
            history.append(g)
            if not ok:
                transitions += 1
                unknown += 1
                log_prob += _LOG_UNKNOWN
                node, offset = self._relocate(history)
        node, offset = settle(node, offset, ending=True)
        transitions += 1
        lp = tree.log_share(node, end_leaf[node]) if offset == held(node) - 1 and end_leaf[node] >= 0 else None
        if lp is None:
            unknown += 1
            log_prob += _LOG_UNKNOWN
        else:
            log_prob += lp
        return {
            "log_prob": log_prob, "per_unit": log_prob / max(1, units), "units": units, "transitions": transitions,
            "unknown_transitions": unknown, "units_name": enc.units_name,
        }

    def bits_per_unit(self, texts: Iterable[str] | str) -> dict:
        """``-log2 P`` per unit over texts, with the miss rate."""
        items = [texts] if isinstance(texts, str) else list(texts)
        total = 0.0
        units = transitions = unknown = 0
        for t in items:
            s = self.score(t)
            total -= s["log_prob"]
            units += s["units"]
            transitions += s["transitions"]
            unknown += s["unknown_transitions"]
        return {
            "bits_per_unit": total / math.log(2.0) / units if units else 0.0, "units": units,
            "transitions": transitions, "unknown_transitions": unknown,
            "miss_rate": unknown / transitions if transitions else 0.0, "units_name": self.encoding.units_name,
        }

    def next_token_accuracy(self, texts: Iterable[str] | str, window: int | None = None) -> dict:
        """How often the walk's most likely token is the text's next one, over texts; quiet.  ``known`` is how often
        the actual token was among the context's options."""
        items = [texts] if isinstance(texts, str) else list(texts)
        enc = self.encoding
        tree = self.tree
        correct = known = total = 0
        for text in items:
            grams = enc.encode(text)
            if not grams:
                continue
            walks = Walks(tree, window=window)
            for actual in grams + [None]:
                if walks.history:
                    found = walks.deepest()
                    loc = None if found is None else (found[0], found[1])
                else:
                    loc = (START, -1) if tree.usable(START, -1) else None
                total += 1
                if loc is not None:
                    node, offset = loc
                    _child, gram, _cost = self.next_token(node, offset)
                    correct += gram == actual
                    if offset < tree.held(node) - 1:
                        known += enc.gram_at(tree.labels[node], offset + 1) == actual
                    elif actual is None:
                        known += tree.end_leaf[node] >= 0 and tree.seen(tree.end_leaf[node]) > 0.0
                    else:
                        c = tree.children[node].get(actual)
                        known += c is not None and tree.seen(c) > 0.0
                if actual is not None:
                    walks.push(actual)
        return {"window": window, "tokens": total, "accuracy": correct / total if total else 0.0,
                "known": known / total if total else 0.0}

    def recites(self, texts: Iterable[str] | str, prefix_units: int = 8) -> dict:
        """How many texts the model says whole, quietly, from their first ``prefix_units`` units."""
        items = [texts] if isinstance(texts, str) else list(texts)
        enc = self.encoding
        tried = recited = 0
        for text in items:
            units = enc.length(text)
            if units <= prefix_units:
                continue
            tried += 1
            head = enc.piece(text, 0, prefix_units)
            out = self.say(head, length=units, to_end=True, quiet=True)
            if enc.join(head, out.text) == enc.normalize(text):
                recited += 1
        return {"texts": tried, "recited": recited, "fraction": recited / tried if tried else 0.0}

    # -- stats and persistence -----------------------------------------------

    def stats(self) -> dict:
        t = self.tree
        return {
            "kind": self.kind, "nodes": t.num_nodes(), "ends": t.num_ends(), "grams": t.num_grams(),
            "label_chars": t.label_chars(), "branches": t.branches(), "max_depth": t.max_depth(),
            "remembered": t.remembered(), "total_seen": t.total_seen(), "clock": t.traversals,
            "life": t.life, "decay": t.decay, "min_seen": t.min_seen, "depth": t.depth,
            "encoding": str(self.encoding), "unit": self.encoding.unit, "units": self.encoding.units_name,
            **{k: self.meta[k] for k in ("texts_read", "units_read", "readings", "traversals_said", "ticks")},
        }

    def to_dict(self) -> dict:
        state = self.rng.getstate()
        return {
            "format": MODEL_FORMAT, "version": MODEL_FORMAT_VERSION, "kind": self.kind, "saved_at": _utc_now(),
            "seed": self.seed, "rng_state": [state[0], list(state[1]), state[2]],
            "meta": dict(self.meta), "history": [dict(r) for r in self.history], "tree": self.tree.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecayNet":
        if not isinstance(d, dict) or d.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a {MODEL_FORMAT} model document")
        if int(d.get("version", 1)) > MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported {MODEL_FORMAT} model version {d.get('version')}")
        model = cls.__new__(cls)
        model.tree = DecayTree.from_dict(d["tree"])
        model.seed = int(d.get("seed", 0))
        model.rng = random.Random(model.seed)
        state = d.get("rng_state")
        if state:
            model.rng.setstate((int(state[0]), tuple(int(x) for x in state[1]), state[2]))
        model.history = [dict(r) for r in d.get("history") or []]
        meta = dict(d.get("meta") or {})
        for key in ("texts_read", "units_read", "readings", "traversals_said", "ticks"):
            meta.setdefault(key, 0)
        meta.setdefault("seed", model.seed)
        meta.setdefault("created", _utc_now())
        model.meta = meta
        return model

    def save(self, path: str) -> None:
        payload = json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        write_bytes_atomic(path, payload, use_gzip=path.endswith(".gz"))

    @classmethod
    def load(cls, path: str) -> "DecayNet":
        return cls.from_dict(read_json_file(path))

    def __repr__(self) -> str:
        t = self.tree
        return (
            f"DecayNet(decay={t.decay!r}, life={t.life:g}, nodes={t.num_nodes()}, remembered={t.remembered()}, "
            f"clock={t.traversals})"
        )


def load_model(path: str) -> DecayNet:
    """Load a model file written by :meth:`DecayNet.save`."""
    return DecayNet.load(path)
