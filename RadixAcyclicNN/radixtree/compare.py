"""The comparison: this tree against ``RadixCyclicNN``'s graph, on one corpus, under one rule.

The two models share the encoding, the activation, the learning rule, the
cost and the shortest-path search; they differ in what a node is.  So every
number below is a measurement of that one difference: what a cycle buys, and
what it costs, on the same text.

The cyclic model is imported from the sibling checkout (``../RadixCyclicNN``)
and nothing here is skipped silently: when it cannot be imported the result
says so and only the tree's half is filled in.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time

from . import corpus
from .model import RadixTreeNet

__all__ = ["PREFIXES", "REPETITIONS", "compare", "load_radixnet", "markdown"]

PREFIXES = ("the quick", "the cat", "knowledge", "two plus", "the moon", "th", "zzzq")
"""Prefixes every run predicts from: openings of the sample corpus, a partial gram, and one nothing knows."""

REPETITIONS = (("a", 4), ("a", 10), ("a", 100), ("a", 1000), ("abc", 2), ("abc", 5), ("abc", 50), ("abc", 500))
"""``(motif, times)``: the texts the counting argument of ``Research/CyclesAreAFeature.md`` §3 is measured on."""

RECALL_PREFIX = 8
"""Characters of a training text a model is given before it must recite the rest."""


def load_radixnet():
    """The ``radixnet`` package from the sibling ``RadixCyclicNN`` directory, or ``None``."""
    try:
        return importlib.import_module("radixnet")
    except ImportError:
        pass
    sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN")
    if os.path.isdir(os.path.join(sibling, "radixnet")) and sibling not in sys.path:
        sys.path.append(sibling)
    try:
        return importlib.import_module("radixnet")
    except ImportError:
        return None


def _cyclic_stats(model) -> dict:
    """The cyclic graph's sizes in the tree's terms: its sentinels are not counted as nodes."""
    g = model.graph
    first = importlib.import_module("radixnet.graph").FIRST
    real = [i for i in range(first, len(g.labels)) if g.alive[i]]
    return {
        "nodes": g.num_nodes() - first,
        "edges": g.num_edges(),
        "grams": g.num_trigrams(),
        "label_chars": sum(len(g.labels[i]) for i in real),
        "compression_ratio": g.compression_ratio(),
    }


def _tree_stats(model: RadixTreeNet) -> dict:
    s = model.stats()
    return {key: s[key] for key in ("nodes", "ends", "edges", "grams", "label_chars", "branches", "max_depth", "compression_ratio")}


def _bits(model, texts) -> dict:
    """Bits per character and the miss rate under a model's own ``score``."""
    import math

    total = 0.0
    chars = transitions = unknown = 0
    for t in texts:
        s = model.score(t)
        total -= s["log_prob"]
        chars += s["chars"]
        transitions += s["transitions"]
        unknown += s["unknown_transitions"]
    return {
        "bits_per_char": total / math.log(2.0) / chars if chars else 0.0,
        "chars": chars,
        "transitions": transitions,
        "unknown_transitions": unknown,
        "miss_rate": unknown / transitions if transitions else 0.0,
    }


def _recall(model, texts, prefix_len: int = RECALL_PREFIX) -> dict:
    """How many training texts the model recites whole from their first ``prefix_len`` characters."""
    recited = 0
    tried = 0
    for text in texts:
        if len(text) <= prefix_len:
            continue
        tried += 1
        result = model.predict(text[:prefix_len], length=len(text), to_end=True)
        if text[:prefix_len] + result.text == text:
            recited += 1
    return {"texts": tried, "recited": recited, "fraction": recited / tried if tried else 0.0}


def _predictions(model, prefixes, length: int = 30) -> list[dict]:
    out = []
    for prefix in prefixes:
        t0 = time.perf_counter()
        result = model.predict(prefix, length=length, to_end=True)
        out.append({
            "prefix": prefix, "text": result.text, "cost": result.cost, "expanded": result.expanded,
            "reached_end": result.reached_end, "seconds": time.perf_counter() - t0,
        })
    return out


def _train_cyclic(radixnet, texts, settings) -> tuple[object, list[dict], float]:
    model = radixnet.RadixNet(seed=settings["seed"], backend="python")
    t0 = time.perf_counter()
    records = model.train(
        texts, epochs=settings["epochs"], lr=settings["lr"], act_lr=settings["act_lr"], batch_size=settings["batch_size"],
    )
    return model, records, time.perf_counter() - t0


def _train_tree(texts, settings) -> tuple[RadixTreeNet, list[dict], float]:
    model = RadixTreeNet(seed=settings["seed"], depth=settings["depth"], min_count=settings["min_count"])
    t0 = time.perf_counter()
    records = model.train(
        texts, epochs=settings["epochs"], lr=settings["lr"], act_lr=settings["act_lr"], batch_size=settings["batch_size"],
    )
    return model, records, time.perf_counter() - t0


def _repetition(radixnet, settings) -> list[dict]:
    """Nodes each structure needs for a repeated motif - the table of the paper's §3, measured."""
    rows = []
    for motif, times in REPETITIONS:
        text = motif * times
        row: dict = {"motif": motif, "times": times, "chars": len(text)}
        tree = RadixTreeNet(seed=settings["seed"], depth=settings["depth"])
        tree.train([text], epochs=0)
        row["tree_nodes"] = tree.tree.num_nodes()
        row["tree_label_chars"] = tree.tree.label_chars()
        if radixnet is not None:
            cyc = radixnet.RadixNet(seed=settings["seed"], backend="python")
            cyc.train([text], epochs=0)
            row["cyclic_nodes"] = _cyclic_stats(cyc)["nodes"]
            row["cyclic_label_chars"] = _cyclic_stats(cyc)["label_chars"]
        rows.append(row)
    return rows


def _generalisation(radixnet, settings) -> dict:
    """Shown ``aaaa`` once, what each structure says about ``aaaaaaa`` - the paper's §3 claim, measured."""
    out: dict = {"trained_on": "aaaa", "scored": "aaaaaaa"}
    tree = RadixTreeNet(seed=settings["seed"], depth=settings["depth"])
    tree.train(["aaaa"], epochs=1)
    out["tree"] = tree.score("aaaaaaa")
    out["tree_representable"] = tree.tree.node_path(tree.encoding.encode("aaaaaaa")) is not None
    if radixnet is not None:
        cyc = radixnet.RadixNet(seed=settings["seed"], backend="python")
        cyc.train(["aaaa"], epochs=1)
        out["cyclic"] = cyc.score("aaaaaaa")
        out["cyclic_representable"] = cyc.graph.node_path(cyc.encoder.encode("aaaaaaa")) is not None
    return out


def compare(
    train_texts,
    test_texts=(),
    *,
    seed: int = 1,
    epochs: int = 10,
    lr: float = 0.5,
    act_lr: float = 0.05,
    batch_size: int = 32,
    depth: int | None = None,
    min_count: int = 1,
    prefixes=PREFIXES,
    corpus_info: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Train both models on ``train_texts`` under one setting and measure everything that differs."""
    train_texts = list(train_texts)
    test_texts = list(test_texts)
    settings = {
        "seed": seed, "epochs": epochs, "lr": lr, "act_lr": act_lr, "batch_size": batch_size,
        "depth": depth, "min_count": min_count,
    }
    radixnet = load_radixnet()
    started = time.perf_counter()
    result: dict = {
        "corpus": dict(corpus_info or {}),
        "settings": settings,
        "models": {},
    }
    result["corpus"].update({
        "train_texts": len(train_texts), "train_chars": sum(len(t) for t in train_texts),
        "test_texts": len(test_texts), "test_chars": sum(len(t) for t in test_texts),
        "digest": corpus.digest(train_texts + test_texts),
    })

    def say(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    say(f"tree: training on {len(train_texts)} texts, {epochs} epochs")
    tree, records, seconds = _train_tree(train_texts, settings)
    result["models"]["tree"] = {
        "available": True, "kind": tree.kind, "stats": _tree_stats(tree), "train_seconds": seconds,
        "epoch_loss": [r["loss"] for r in records], "transitions_per_epoch": records[-1]["transitions"] if records else 0,
        "train_bits": _bits(tree, train_texts), "test_bits": _bits(tree, test_texts) if test_texts else None,
        "recall": _recall(tree, train_texts), "predictions": _predictions(tree, prefixes),
    }
    if radixnet is None:
        say("cyclic: radixnet not importable (expected ../RadixCyclicNN); its half is left empty")
        result["models"]["cyclic"] = {"available": False}
    else:
        say(f"cyclic: training on {len(train_texts)} texts, {epochs} epochs")
        cyc, records, seconds = _train_cyclic(radixnet, train_texts, settings)
        result["models"]["cyclic"] = {
            "available": True, "kind": cyc.kind, "stats": _cyclic_stats(cyc), "train_seconds": seconds,
            "epoch_loss": [r["loss"] for r in records], "transitions_per_epoch": records[-1]["transitions"] if records else 0,
            "train_bits": _bits(cyc, train_texts), "test_bits": _bits(cyc, test_texts) if test_texts else None,
            "recall": _recall(cyc, train_texts), "predictions": _predictions(cyc, prefixes),
        }
    say("repetition and generalisation")
    result["repetition"] = _repetition(radixnet, settings)
    result["generalisation"] = _generalisation(radixnet, settings)
    result["seconds"] = time.perf_counter() - started
    return result


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def markdown(result: dict) -> str:
    """The result as the tables ``README.md`` carries."""
    models = result["models"]
    tree = models["tree"]
    cyc = models.get("cyclic", {"available": False})
    c = result["corpus"]
    s = result["settings"]
    lines = [
        f"corpus: {c.get('path', '?')} - {c['train_texts']} texts / {c['train_chars']} chars to train, "
        f"{c['test_texts']} texts / {c['test_chars']} chars held out (sha256 {c['digest'][:12]})",
        f"settings: seed {s['seed']}, {s['epochs']} epochs, lr {s['lr']}, act_lr {s['act_lr']}, batch {s['batch_size']}, "
        f"depth {s['depth']}, min_count {s['min_count']}",
        "",
        "| | cyclic graph | tree |",
        "|---|---:|---:|",
    ]

    def row(name, key_c, key_t, digits=3):
        left = _fmt(key_c(cyc), digits) if cyc.get("available") else "-"
        lines.append(f"| {name} | {left} | {_fmt(key_t(tree), digits)} |")

    row("real nodes", lambda m: m["stats"]["nodes"], lambda m: m["stats"]["nodes"])
    row("END leaves", lambda m: 1, lambda m: m["stats"]["ends"])
    row("edges", lambda m: m["stats"]["edges"], lambda m: m["stats"]["edges"])
    row("distinct grams", lambda m: m["stats"]["grams"], lambda m: m["stats"]["grams"])
    row("label characters", lambda m: m["stats"]["label_chars"], lambda m: m["stats"]["label_chars"])
    row("grams per node", lambda m: m["stats"]["compression_ratio"], lambda m: m["stats"]["compression_ratio"])
    row("transitions per epoch", lambda m: m["transitions_per_epoch"], lambda m: m["transitions_per_epoch"])
    row("training seconds", lambda m: m["train_seconds"], lambda m: m["train_seconds"], 2)
    row("final epoch loss (nats)", lambda m: m["epoch_loss"][-1] if m["epoch_loss"] else None, lambda m: m["epoch_loss"][-1] if m["epoch_loss"] else None)
    row("train bits/char", lambda m: m["train_bits"]["bits_per_char"], lambda m: m["train_bits"]["bits_per_char"])
    row("train miss rate", lambda m: m["train_bits"]["miss_rate"], lambda m: m["train_bits"]["miss_rate"])
    if tree.get("test_bits"):
        row("held-out bits/char", lambda m: m["test_bits"]["bits_per_char"], lambda m: m["test_bits"]["bits_per_char"])
        row("held-out miss rate", lambda m: m["test_bits"]["miss_rate"], lambda m: m["test_bits"]["miss_rate"])
    row("texts recited whole", lambda m: f"{m['recall']['recited']}/{m['recall']['texts']}", lambda m: f"{m['recall']['recited']}/{m['recall']['texts']}")
    lines += ["", "| prefix | cyclic graph says | tree says |", "|---|---|---|"]
    tp = {p["prefix"]: p for p in tree["predictions"]}
    cp = {p["prefix"]: p for p in cyc.get("predictions", [])}
    for prefix in tp:
        left = repr(cp[prefix]["text"]) if prefix in cp else "-"
        lines.append(f"| `{prefix}` | {left} | {tp[prefix]['text']!r} |")
    lines += ["", "| text | chars | cyclic nodes | tree nodes | cyclic label chars | tree label chars |", "|---|---:|---:|---:|---:|---:|"]
    for r in result["repetition"]:
        lines.append(
            f"| `{r['motif']}` × {r['times']} | {r['chars']} | {_fmt(r.get('cyclic_nodes'))} | {r['tree_nodes']} | "
            f"{_fmt(r.get('cyclic_label_chars'))} | {r['tree_label_chars']} |"
        )
    g = result["generalisation"]
    lines += ["", f"shown `{g['trained_on']}` once, scoring `{g['scored']}`:"]
    if "cyclic" in g:
        lines.append(
            f"- cyclic graph: representable {g['cyclic_representable']}, {g['cyclic']['unknown_transitions']} unknown of "
            f"{g['cyclic']['transitions']} transitions, log-prob {g['cyclic']['log_prob']:.2f}"
        )
    lines.append(
        f"- tree: representable {g['tree_representable']}, {g['tree']['unknown_transitions']} unknown of "
        f"{g['tree']['transitions']} transitions, log-prob {g['tree']['log_prob']:.2f}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    """``python3 -m radixtree.compare``: run the comparison and write the result."""
    import argparse

    parser = argparse.ArgumentParser(description="RadixAcyclicNN against RadixCyclicNN on one corpus")
    parser.add_argument("--data", help="corpus file, one text per line (default: the prose snapshot, built if missing)")
    parser.add_argument("--heldout-every", type=int, default=5, help="hold out every Nth text (0: none)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--act-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", help="write the JSON result here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.data:
        texts = corpus.read_texts(args.data)
        path = args.data
    else:
        texts = corpus.load()
        path = corpus.SNAPSHOT
    train, test = corpus.split(texts, args.heldout_every)
    result = compare(
        train, test, seed=args.seed, epochs=args.epochs, lr=args.lr, act_lr=args.act_lr, batch_size=args.batch_size,
        depth=args.depth, min_count=args.min_count, corpus_info={"path": os.path.relpath(path)}, verbose=not args.quiet,
    )
    print(markdown(result))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
