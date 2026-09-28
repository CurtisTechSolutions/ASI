"""``python3 -m radixtree <command>`` - every command this model has.

    train      build the tree from a corpus, train it, save it
    predict    the cheapest, a sampled, or a token-by-token continuation of a prefix
    generate   texts from START
    score      log-probability and bits per unit of texts
    accuracy   next-token accuracy of the token-by-token walk, over windows
    stats      the size and shape of a saved model
    invert     flip a saved model (2NRL's negation) and save it
    2nrl       train on the bad texts, invert, fine-tune on the good ones
    check      the finite-difference gradient checks
    corpus     build the prose corpus snapshot from the pinned files
    compare    this tree against RadixCyclicNN's graph on one corpus
    demo       train on the sample corpus and show what the tree does
    test       run the test suite

``predict``, ``generate``, ``score`` and ``stats`` take ``--load`` (a model
file written by ``train``); without it they train a model from ``--data``
first, so every command runs straight out of a checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import corpus
from .check import check_all
from .encoding import UNIT_KINDS, Encoding
from .model import RadixTreeNet, load_model
from .search import COSTS

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(os.path.dirname(HERE), "data", "sample_corpus.txt")
GARBAGE = os.path.join(os.path.dirname(HERE), "data", "sample_garbage.txt")


def _add_train_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=SAMPLE, help="corpus file, one text per line (default: the sample corpus)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--act-lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--depth", type=int, default=None, help="symbols a root path may hold (default: every whole suffix)")
    p.add_argument("--n", type=int, default=3, help="units per gram")
    p.add_argument("--unit", choices=UNIT_KINDS, default="char",
                   help="what a unit is: a character, or a phone or syllable read through the phonetic tokenizer")
    p.add_argument("--min-count", type=int, default=1, help="how often a context must have been seen to be trusted")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-compress", action="store_true", help="do not merge unary chains")
    p.add_argument("--verbose", action="store_true", help="print one line per epoch")


def _add_load(p: argparse.ArgumentParser) -> None:
    p.add_argument("--load", help="a model file written by train")


def _encoding(args) -> Encoding:
    return Encoding(unit=args.unit, n=args.n)


def _train(args) -> RadixTreeNet:
    texts = corpus.read_texts(args.data)
    model = RadixTreeNet(seed=args.seed, depth=args.depth, encoding=_encoding(args), min_count=args.min_count)
    model.train(
        texts, epochs=args.epochs, lr=args.lr, act_lr=args.act_lr, batch_size=args.batch_size,
        auto_compress=not args.no_compress, verbose=args.verbose,
    )
    return model


def _model(args) -> RadixTreeNet:
    """The model a command works with: loaded when asked, else trained from ``--data``."""
    if getattr(args, "load", None):
        return load_model(args.load)
    return _train(args)


def _save(model: RadixTreeNet, path: str | None) -> None:
    if path:
        model.save(path)
        print(f"  saved {path}", file=sys.stderr)


def cmd_train(args) -> int:
    model = _train(args)
    last = model.history[-1] if model.history else None
    print(json.dumps({"stats": model.stats(), "last_epoch": last}, indent=1))
    _save(model, args.save)
    return 0


def _mode(args) -> str:
    if args.slide:
        return "slide"
    return "sample" if args.sample else "dijkstra"


def cmd_predict(args) -> int:
    model = _model(args)
    result = model.predict(
        args.prefix, length=args.length, mode=_mode(args), step_penalty=args.step_penalty,
        temperature=args.temperature, to_end=args.to_end, max_length=args.max_length, costs=args.costs,
        seed=args.sample_seed, window=args.window,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        print(result.full_spelled)
        if model.encoding.phonetic:
            print(f"  sounds {result.full_text}", file=sys.stderr)
        print(f"  continuation {result.spelled!r}  cost {result.cost:.3f}  expanded {result.expanded}  "
              f"legs {result.legs}  reached END {result.reached_end}", file=sys.stderr)
    return 0


def cmd_generate(args) -> int:
    model = _model(args)
    results = model.generate(
        max_length=args.max_length, count=args.count, mode="slide" if args.slide else ("dijkstra" if args.dijkstra else "sample"),
        temperature=args.temperature, seed=args.sample_seed, step_penalty=args.step_penalty, to_end=args.to_end,
        costs=args.costs, window=args.window,
    )
    for r in results:
        print(r.full_spelled)
    return 0


def cmd_score(args) -> int:
    model = _model(args)
    texts = list(args.text)
    if args.texts:
        texts += corpus.read_texts(args.texts)
    if not texts:
        print("nothing to score: give texts, or --texts FILE", file=sys.stderr)
        return 2
    for t in texts:
        s = model.score(t)
        print(f"{s['log_prob']:10.3f}  {s['unknown_transitions']:3d}/{s['transitions']:<3d} unknown  {t}")
    print(json.dumps(model.bits_per_char(texts), indent=1))
    return 0


def cmd_accuracy(args) -> int:
    """Next-token accuracy of the token-by-token walk, over windows."""
    model = _model(args)
    texts = corpus.read_texts(args.texts) if args.texts else corpus.read_texts(args.data)
    windows = [None if w in ("all", "none", "0") else int(w) for w in args.windows.split(",")]
    rows = [model.next_token_accuracy(texts, window=w) for w in windows]
    print(f"{'window':>8} {'tokens':>8} {'accuracy':>9} {'known':>7}")
    for r in rows:
        print(f"{('all' if r['window'] is None else r['window']):>8} {r['tokens']:8d} {r['accuracy']:9.3f} {r['known']:7.3f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"texts": args.texts or args.data, "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


def cmd_stats(args) -> int:
    model = _model(args)
    print(json.dumps(model.stats(), indent=1))
    return 0


def cmd_invert(args) -> int:
    model = _model(args)
    model.invert()
    print(f"inverted: {model.tree.inverted}")
    _save(model, args.save)
    return 0


def cmd_2nrl(args) -> int:
    model = _model(args) if args.load else RadixTreeNet(seed=args.seed, depth=args.depth, encoding=_encoding(args), min_count=args.min_count)
    bad = corpus.read_texts(args.bad)
    good = corpus.read_texts(args.good)
    out = model.two_nrl(
        bad, good, neg_epochs=args.neg_epochs, pos_epochs=args.pos_epochs, neg_lr=args.neg_lr, pos_lr=args.pos_lr,
        batch_size=args.batch_size,
    )
    summary = {
        "negative_loss": [r["loss"] for r in out["negative"]],
        "positive_loss": [r["loss"] for r in out["positive"]],
        "inverted": out["inverted"],
        "good_bits": model.bits_per_char(good),
        "bad_bits": model.bits_per_char(bad),
    }
    print(json.dumps(summary, indent=1))
    _save(model, args.save)
    return 0


def cmd_check(args) -> int:
    worst = max(check_all().values())
    print(f"  worst error overall {worst:.2e}")
    return 0 if worst < 1e-5 else 1


def cmd_corpus(args) -> int:
    texts = corpus.build(args.lines)
    path = corpus.save(texts, args.out or corpus.SNAPSHOT)
    print(json.dumps({"texts": len(texts), "chars": sum(len(t) for t in texts), "digest": corpus.digest(texts)}, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_compare(args) -> int:
    from .compare import main as compare_main

    argv = []
    if args.data:
        argv += ["--data", args.data]
    argv += ["--heldout-every", str(args.heldout_every), "--epochs", str(args.epochs), "--lr", str(args.lr),
             "--act-lr", str(args.act_lr), "--batch-size", str(args.batch_size), "--min-count", str(args.min_count),
             "--seed", str(args.seed)]
    if args.depth is not None:
        argv += ["--depth", str(args.depth)]
    if args.out:
        argv += ["--out", args.out]
    return compare_main(argv)


def cmd_demo(args) -> int:
    texts = corpus.read_texts(args.data)
    print(f"{len(texts)} texts, {sum(len(t) for t in texts)} characters")
    check_all()
    model = RadixTreeNet(seed=args.seed, depth=args.depth, encoding=_encoding(args), min_count=args.min_count)
    for r in model.train(texts, epochs=args.epochs, lr=args.lr, act_lr=args.act_lr, batch_size=args.batch_size):
        print(f"  epoch {r['epoch']:2d}  loss {r['loss']:.4f}  nodes {r['nodes']}  ends {r['ends']}  "
              f"transitions {r['transitions']}  ({r['seconds']:.2f}s)")
    s = model.stats()
    print(f"\n  {s['nodes']} real nodes, {s['ends']} END leaves, {s['grams']} distinct grams, "
          f"{s['label_chars']} label characters, {s['branches']} branches, depth {s['max_depth']}")
    print(f"  train bits/char {model.bits_per_char(texts)['bits_per_char']:.3f}\n")
    for prefix in ("the quick", "the cat", "knowledge", "th", "zzzq"):
        r = model.predict(prefix, length=30, to_end=True)
        print(f"  {prefix!r:14} -> {r.spelled!r}  (cost {r.cost:.3f}, {r.expanded} nodes visited, END {r.reached_end})")
    print("\n  token by token, a window of four grams:")
    for prefix in ("the quick", "the cat"):
        r = model.predict(prefix, length=40, to_end=True, mode="slide", window=4)
        print(f"  {prefix!r:14} -> {r.spelled!r}  ({r.expanded} queries)")
    print("\n  three sampled texts:")
    for r in model.generate(max_length=50, count=3, seed=7):
        print(f"    {r.full_spelled}")
    print("\n  what the tree cannot do: shown 'aaaa', it does not represent 'aaaaaaa'")
    small = RadixTreeNet(seed=args.seed)
    small.train(["aaaa"], epochs=1)
    print(f"    {small.score('aaaaaaa')}")
    return 0


def cmd_test(args) -> int:
    import unittest

    root = os.path.dirname(HERE)
    if root not in sys.path:
        sys.path.insert(0, root)
    suite = unittest.defaultTestLoader.discover(os.path.join(root, "tests"), top_level_dir=root)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radixtree", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="build and train a tree, save it")
    _add_train_options(p)
    p.add_argument("--save", help="write the model here (.json or .json.gz)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="continue a prefix")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("prefix")
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--to-end", action="store_true", help="run to an END leaf")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--sample", action="store_true", help="a sampled walk instead of the cheapest path")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=0.0, help="added to every step; may be negative here")
    p.add_argument("--costs", choices=COSTS, default="logprob")
    p.add_argument("--slide", action="store_true", help="token by token: one query of the tree per token, fed back")
    p.add_argument("--window", type=int, default=None, help="grams of context per query in --slide (default: all)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("generate", help="texts from START")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--max-length", type=int, default=60)
    p.add_argument("--dijkstra", action="store_true", help="the one cheapest text instead of samples")
    p.add_argument("--slide", action="store_true", help="token by token, one query per token")
    p.add_argument("--window", type=int, default=None, help="grams of context per query in --slide")
    p.add_argument("--to-end", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--costs", choices=COSTS, default="logprob")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("score", help="log-probability of texts")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("text", nargs="*")
    p.add_argument("--texts", help="a file of texts to score, one per line")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("accuracy", help="next-token accuracy of the token-by-token walk, over windows")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("--texts", help="texts to measure on, one per line (default: the training data)")
    p.add_argument("--windows", default="1,2,3,4,6,8,all", help="comma-separated grams of context; 'all' is the whole history")
    p.add_argument("--out")
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser("stats", help="size and shape")
    _add_train_options(p)
    _add_load(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("invert", help="negate every weight and activation")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("--save")
    p.set_defaults(func=cmd_invert)

    p = sub.add_parser("2nrl", help="train on bad texts, invert, fine-tune on good ones")
    _add_train_options(p)
    _add_load(p)
    p.add_argument("--bad", default=GARBAGE)
    p.add_argument("--good", default=SAMPLE)
    p.add_argument("--neg-epochs", type=int, default=3)
    p.add_argument("--pos-epochs", type=int, default=3)
    p.add_argument("--neg-lr", type=float, default=0.5)
    p.add_argument("--pos-lr", type=float, default=0.1)
    p.add_argument("--save")
    p.set_defaults(func=cmd_2nrl)

    p = sub.add_parser("check", help="finite-difference gradient checks")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("corpus", help="build the prose corpus snapshot")
    p.add_argument("--lines", type=int, default=corpus.DEFAULT_LINES)
    p.add_argument("--out")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("compare", help="against RadixCyclicNN on one corpus")
    p.add_argument("--data", help="corpus file (default: the prose snapshot)")
    p.add_argument("--heldout-every", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--act-lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("demo", help="train on the sample corpus and look at it")
    _add_train_options(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("test", help="run the tests")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_test)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
