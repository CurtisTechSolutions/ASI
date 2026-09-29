"""``python3 -m radixdecay <command>`` - every command this model has.

    read        read a corpus into a model - every window, every node on its way traversed - and save it
    say         continue a prefix token by token, and remember having said it
    cheapest    the cheapest continuation in one search, then remember having said it
    generate    texts said from START
    score       log-probability and bits per unit of texts (a measurement: nothing moves)
    accuracy    next-token accuracy of the token-by-token walk, over windows (a measurement)
    stats       the size, the memory and the clock of a model
    tick        let time pass: advance the clock without arriving anywhere
    experiment  the habit, forgetting, use and recovery experiments
    demo        read the sample corpus and watch it remember, say, forget and recover
    test        run the test suite

``train`` and ``predict`` are ``read`` and ``say`` under the sibling models'
names.  Every command but ``read`` takes ``--load`` (a model file written by
``read`` or by any command's ``--save``); without it the command reads
``--data`` into a fresh model first, so everything runs straight out of a
checkout.  A command that says something changes the model - that is the
point - and ``--save`` writes the changed model back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import corpus
from .encoding import UNIT_KINDS, Encoding
from .model import MAX_LEGS, DecayNet, load_model
from .search import PathResult
from .tree import DECAYS, LIFE, MIN_SEEN

HERE = os.path.dirname(os.path.abspath(__file__))


def _add_model_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=corpus.SAMPLE, help="corpus file, one text per line (default: the sample corpus)")
    p.add_argument("--readings", type=int, default=1, help="how many times the corpus is read")
    p.add_argument("--depth", type=int, default=None, help="symbols a root path may hold (default: every whole suffix)")
    p.add_argument("--n", type=int, default=3, help="units per gram")
    p.add_argument("--unit", choices=UNIT_KINDS, default="char",
                   help="what a unit is: a character, or a phone or syllable read through the phonetic tokenizer")
    p.add_argument("--life", type=float, default=LIFE,
                   help="traversals for a lone visit to fade to a half (half-life) or to nothing (linear)")
    p.add_argument("--decay", choices=DECAYS, default="half-life", help="how seen fades on the clock")
    p.add_argument("--min-seen", type=float, default=MIN_SEEN,
                   help="how much of a visit a context must still hold to be consulted")
    p.add_argument("--seed", type=int, default=1, help="seeds the sampled walks only")


def _add_load(p: argparse.ArgumentParser) -> None:
    p.add_argument("--load", help="a saved model file; without it the command reads --data first")


def _add_save(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save", help="write the model, as it is after this command, here (.json or .json.gz)")


def _encoding(args) -> Encoding:
    return Encoding(unit=args.unit, n=args.n)


def _new(args) -> DecayNet:
    return DecayNet(encoding=_encoding(args), depth=args.depth, life=args.life, decay=args.decay,
                    min_seen=args.min_seen, seed=args.seed)


def _read(args) -> DecayNet:
    model = _new(args)
    model.read(corpus.read_texts(args.data), times=args.readings)
    return model


def _model(args) -> DecayNet:
    """The model a command works with: loaded when asked, else read from ``--data``."""
    if getattr(args, "load", None):
        return load_model(args.load)
    return _read(args)


def _save(model: DecayNet, path: str | None) -> None:
    if path:
        model.save(path)
        print(f"  saved {path}", file=sys.stderr)


def _show(model: DecayNet, result: PathResult) -> None:
    print(result.full_spelled)
    if model.encoding.phonetic:
        print(f"  sounds {result.full_text}", file=sys.stderr)
    print(f"  continuation {result.spelled!r}  cost {result.cost:.3f}  queries {result.expanded}  "
          f"traversals {result.traversals}  reached END {result.reached_end}  clock {model.clock}", file=sys.stderr)


def cmd_read(args) -> int:
    """Read ``--data`` into a fresh model, or into ``--load``."""
    if args.load:
        model = load_model(args.load)
        record = model.read(corpus.read_texts(args.data), times=args.readings)
    else:
        model = _read(args)
        record = model.history[-1]
    print(json.dumps({"read": record, "stats": model.stats()}, indent=1))
    _save(model, args.save)
    return 0


def cmd_say(args) -> int:
    model = _model(args)
    results = [
        model.say(args.prefix, length=args.length, window=args.window, temperature=args.temperature, to_end=args.to_end,
                  max_length=args.max_length, seed=args.sample_seed, quiet=args.quiet)
        for _ in range(args.times)
    ]
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=1))
    else:
        for r in results:
            _show(model, r)
    _save(model, args.save)
    return 0


def cmd_cheapest(args) -> int:
    model = _model(args)
    result = model.cheapest(args.prefix, length=args.length, to_end=args.to_end, max_length=args.max_length,
                            quiet=args.quiet, max_legs=args.max_legs)
    if args.json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        _show(model, result)
    _save(model, args.save)
    return 0


def cmd_generate(args) -> int:
    model = _model(args)
    for r in model.generate(count=args.count, max_length=args.max_length, temperature=args.temperature,
                            seed=args.sample_seed, window=args.window, quiet=args.quiet):
        print(r.full_spelled)
    _save(model, args.save)
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
    print(json.dumps(model.bits_per_unit(texts), indent=1))
    return 0


def cmd_accuracy(args) -> int:
    """Next-token accuracy of the token-by-token walk, over windows; ``--heldout-every N`` reads all but every
    Nth text of ``--data`` and measures on those."""
    if args.heldout_every and not args.load:
        kept, held = corpus.split(corpus.read_texts(args.data), args.heldout_every)
        model = _new(args)
        model.read(kept, times=args.readings)
        texts = corpus.read_texts(args.texts) if args.texts else held
        measured = args.texts or f"every {args.heldout_every}th text of {args.data}, held out"
    else:
        model = _model(args)
        texts = corpus.read_texts(args.texts) if args.texts else corpus.read_texts(args.data)
        measured = args.texts or args.data
    windows = [None if w in ("all", "none", "0") else int(w) for w in args.windows.split(",")]
    rows = [model.next_token_accuracy(texts, window=w) for w in windows]
    print(f"{'window':>8} {'tokens':>8} {'accuracy':>9} {'known':>7}")
    for r in rows:
        print(f"{('all' if r['window'] is None else r['window']):>8} {r['tokens']:8d} {r['accuracy']:9.3f} {r['known']:7.3f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"measured_on": measured, "texts": len(texts), "model": model.stats(), "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


def cmd_stats(args) -> int:
    model = _model(args)
    print(json.dumps(model.stats(), indent=1))
    return 0


def cmd_tick(args) -> int:
    model = _model(args)
    traversals = args.traversals if args.traversals is not None else int(round(args.lives * model.tree.life))
    keys = ("clock", "remembered", "total_seen")
    before = {k: model.stats()[k] for k in keys}
    model.tick(traversals)
    after = {k: model.stats()[k] for k in keys}
    print(json.dumps({"ticked": traversals, "lives": traversals / model.tree.life, "before": before, "after": after}, indent=1))
    _save(model, args.save)
    return 0


def cmd_experiment(args) -> int:
    from . import experiment

    results = experiment.run(
        args.which, corpus.read_texts(args.data), out_dir=args.out, life=args.life, decay=args.decay,
        min_seen=args.min_seen, depth=args.depth, encoding=_encoding(args), seed=args.seed,
    )
    print(experiment.tables(results))
    for name in results:
        if "written" in results[name]:
            print(f"wrote {results[name]['written']}", file=sys.stderr)
    return 0


def cmd_demo(args) -> int:
    texts = corpus.read_texts(args.data)
    model = _new(args)
    enc = model.encoding
    print(f"{len(texts)} texts, {sum(enc.length(t) for t in texts)} {enc.units_name}; life {model.tree.life:g} traversals, "
          f"decay {model.tree.decay}")
    r = model.read(texts, times=args.readings)
    s = model.stats()
    print(f"  read {r['times']}×: {r['traversals']} traversals; {s['nodes']} nodes, {s['ends']} END leaves, "
          f"{s['grams']} grams, {s['branches']} branches")
    print(f"  recites {model.recites(texts)['recited']}/{len(texts)} texts from their first eight {enc.units_name}; "
          f"{model.bits_per_unit(texts)['bits_per_unit']:.3f} bits per {enc.unit}\n")
    print("  saying 'the' three times - every saying is a traversal of what it says:")
    for _ in range(3):
        out = model.say("the", to_end=True)
        print(f"    {out.full_spelled!r}  probability {__import__('math').exp(-out.cost):.3f}  traversals {out.traversals}  clock {model.clock}")
    print("\n  a few prefixes, said token by token, a window of four grams:")
    for prefix in ("the quick", "the cat", "knowledge", "th", "zzzq"):
        out = model.say(prefix, to_end=True, window=4)
        print(f"    {prefix!r:12} -> {out.spelled!r}  ({out.expanded} queries, {out.traversals} traversals)")
    print("\n  two lives pass with nothing read or said:")
    model.tick(int(2 * model.tree.life))
    s = model.stats()
    print(f"    remembered {s['remembered']}/{s['nodes']} nodes; recites {model.recites(texts)['recited']}/{len(texts)}; "
          f"{model.bits_per_unit(texts)['bits_per_unit']:.2f} bits per {enc.unit}")
    out = model.say("the", to_end=True)
    print(f"    'the' -> {out.spelled!r}  (what it said three times is what it still says)")
    print("\n  the corpus read once more:")
    model.read(texts)
    s = model.stats()
    print(f"    remembered {s['remembered']}/{s['nodes']} nodes; recites {model.recites(texts)['recited']}/{len(texts)}; "
          f"{model.bits_per_unit(texts)['bits_per_unit']:.3f} bits per {enc.unit}; clock {model.clock}")
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
    parser = argparse.ArgumentParser(prog="radixdecay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("read", aliases=["train"], help="read a corpus into a model, save it")
    _add_model_options(p)
    _add_load(p)
    _add_save(p)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("say", aliases=["predict"], help="continue a prefix token by token, and remember it")
    _add_model_options(p)
    _add_load(p)
    _add_save(p)
    p.add_argument("prefix")
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--to-end", action="store_true", help="run to an END leaf")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--window", type=int, default=None, help="grams of context per query (default: everything said)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 takes the largest share; above it, a draw")
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--times", type=int, default=1, help="say it this many times over")
    p.add_argument("--quiet", action="store_true", help="ask without arriving: nothing is traversed")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("cheapest", help="the cheapest continuation in one search, then remember it")
    _add_model_options(p)
    _add_load(p)
    _add_save(p)
    p.add_argument("prefix")
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--to-end", action="store_true")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-legs", type=int, default=MAX_LEGS)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cheapest)

    p = sub.add_parser("generate", help="texts said from START")
    _add_model_options(p)
    _add_load(p)
    _add_save(p)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--max-length", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("score", help="log-probability of texts; nothing moves")
    _add_model_options(p)
    _add_load(p)
    p.add_argument("text", nargs="*")
    p.add_argument("--texts", help="a file of texts to score, one per line")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("accuracy", help="next-token accuracy over windows; nothing moves")
    _add_model_options(p)
    _add_load(p)
    p.add_argument("--texts", help="texts to measure on, one per line (default: --data)")
    p.add_argument("--heldout-every", type=int, default=0, help="read all but every Nth text of --data, measure on those")
    p.add_argument("--windows", default="1,2,3,4,6,8,all")
    p.add_argument("--out")
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser("stats", help="size, memory and clock")
    _add_model_options(p)
    _add_load(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("tick", help="let time pass")
    _add_model_options(p)
    _add_load(p)
    _add_save(p)
    p.add_argument("--traversals", type=int, default=None, help="how many traversals of time pass")
    p.add_argument("--lives", type=float, default=1.0, help="or how many lives (default: one)")
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("experiment", help="habit, forgetting, use, recovery")
    _add_model_options(p)
    p.add_argument("--which", default="all", choices=("habit", "forgetting", "use", "recovery", "all"))
    p.add_argument("--out", help="directory for <name>_results.json")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("demo", help="read the sample corpus and watch it")
    _add_model_options(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("test", help="run the tests")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_test)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
