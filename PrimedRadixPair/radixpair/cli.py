"""``python3 -m radixpair <command>`` - every command this model has.

    prime      make a primed model file: the codec, L, the settings
    train      count texts into the count tree
    reward     credit texts to the reward tree (a punishment is a negative reward: see punish)
    punish     the same, with the sign reversed
    2nrl       punish the bad texts, then reward the good
    feedback   good and / or bad texts with ratings, dispatched as D-026
    predict    continue a prefix
    generate   a whole text
    score      bits per unit, and the reward tree's readings of a text
    weights    show or change the smoothing, the scales and the backoff
    info       what the model is and what it holds
    checkpoints  list a checkpoint directory
    invert     swap the reward tree's rewards and penalties
    check      the brute-force oracle: prime a small tree literally and compare
    bench      throughput | compare | feedback | rungs

``--json`` on every command prints one JSON document to stdout; progress and
notes go to stderr.  ``--data FILE`` reads one text per line, blank lines
skipped (``-`` reads stdin).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from . import bench as benches
from .check import DEFAULT_SIZES, check_address, check_all
from .checkpoint import CheckpointManager
from .codec import CODECS, DEFAULT_CODEC, DEFAULT_L, make_codec
from .model import PairModel, load_model
from .pair import BACKOFFS, Settings, TRAVERSALS
from .search import MODES


class CliError(Exception):
    pass


def read_texts(paths: Sequence[str] | None, what: str = "training") -> list[str]:
    texts: list[str] = []
    for path in paths or ():
        if path == "-":
            blob = sys.stdin.read()
        else:
            if not os.path.isfile(path):
                raise CliError(f"no such {what} file: {path}")
            with open(path, encoding="utf-8", errors="replace") as fh:
                blob = fh.read()
        texts.extend(line for line in blob.splitlines() if line.strip())
    return texts


def _texts_arg(args, attr: str = "data", text_attr: str = "text", what: str = "training") -> list[str]:
    texts = list(getattr(args, text_attr, None) or [])
    texts.extend(read_texts(getattr(args, attr, None), what))
    if not texts:
        raise CliError(f"nothing to read: give --text or --data FILE")
    return texts


def _ratings(values: Sequence[float] | None, n: int) -> list[float] | None:
    if values is None:
        return None
    if len(values) != n:
        raise CliError(f"{len(values)} ratings for {n} texts")
    for v in values:
        if not 0 <= v <= 10:
            raise CliError("a rating is a mark out of 10")
    return [v / 10.0 for v in values]


def _out(args, payload: dict, lines: Sequence[str] = ()) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for line in lines:
            print(line)


def _load(args) -> PairModel:
    if not os.path.isfile(args.model):
        raise CliError(f"no model file {args.model}; prime one first")
    return load_model(args.model)


# -- commands ------------------------------------------------------------------------

def _resolve(args, feedback: bool = False) -> tuple[str, int]:
    """The codec preset and depth a command runs at: phones at L=3 unless told otherwise (letters for the mat / log case)."""
    codec = args.codec or ("chars" if feedback else DEFAULT_CODEC)
    L = args.L if args.L is not None else DEFAULT_L.get(codec, 2)
    return codec, L


def cmd_prime(args) -> None:
    args.codec, args.L = _resolve(args)
    options: dict = {}
    if args.codec in ("bpe", "syllables", "gpt2", "external"):
        texts = read_texts(args.tokenizer_data, "tokenizer data") if args.tokenizer_data else None
        if args.codec == "bpe":
            options = {"vocab_size": args.vocab_size, "texts": texts}
        elif args.codec == "syllables":
            options = {"stress": not args.no_stress, "vocabulary": args.vocabulary, "texts": texts, "top": args.top}
        elif args.codec == "gpt2":
            files = args.gpt2_files or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "data", "gpt2")
            options = {"files": files if isinstance(files, str) or len(files) == 1 and False else files, "top": args.top, "texts": texts}
            if isinstance(files, list):
                options["files"] = files[0] if len(files) == 1 else tuple(files)
        else:
            if not args.external:
                raise CliError("--external SPEC is required with --codec external")
            options = {"spec": args.external, "top": args.top, "texts": texts}
    elif args.codec == "phones":
        options = {"stress": not args.no_stress}
    try:
        codec = make_codec(args.codec, **options)
        settings = Settings(rungs=args.rungs, node_ceiling=args.ceiling if args.ceiling else Settings().node_ceiling)
        model = PairModel.prime(codec, args.L, kind=args.kind, seed=args.seed, settings=settings)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    model.save(args.model)
    info = model.info()
    _out(args, {"saved": args.model, **{k: info[k] for k in ("codec", "R", "L", "N", "memory_bytes")}},
         [f"primed {info['codec']} at L={args.L}: {info['N']:,} nodes, {info['memory_bytes'] / 2**20:.1f} MiB",
          f"saved {args.model}"])


def cmd_train(args) -> None:
    model = _load(args)
    texts = _texts_arg(args)
    manager = CheckpointManager(args.checkpoint_dir) if args.checkpoint_dir else None
    record = model.train(texts, progress=(lambda p: print(f"  {p['texts']} texts...", file=sys.stderr)) if not args.json else None)
    model.save(args.model)
    if manager is not None:
        manager.save(model, step=len(model.history), tag="train", metrics={"units": record["units"]})
    _out(args, record, [f"read {record['texts']} texts, {record['units']} units ({record['unk_share']:.1%} unk), "
                        f"{record['increments']} increments in {record['seconds']}s", f"saved {args.model}"])


def _credit_cmd(args, call: str) -> None:
    model = _load(args)
    texts = _texts_arg(args, what="judged")
    weights = _ratings(args.ratings, len(texts))
    if call == "reward":
        record = model.reward(texts, strength=args.strength, weights=weights, read=args.read, prefix=args.prefix)
    else:
        record = model.punish(texts, strength=args.strength, weights=weights, prefix=args.prefix)
    model.save(args.model)
    _out(args, record, [f"{call}: {record['texts']} texts at strength {record['strength']}, "
                        f"{record['entries']} entries at rungs={record['rungs']}", f"saved {args.model}"])


def cmd_reward(args) -> None:
    _credit_cmd(args, "reward")


def cmd_punish(args) -> None:
    _credit_cmd(args, "punish")


def cmd_2nrl(args) -> None:
    model = _load(args)
    bad = read_texts(args.bad, "bad")
    good = read_texts(args.good, "good")
    if not bad or not good:
        raise CliError("2nrl needs --bad FILE and --good FILE")
    record = model.two_nrl(bad, good, strength=args.strength, prefix=args.prefix)
    model.save(args.model)
    _out(args, record, [f"punished {record['punished']['texts']} bad texts, rewarded {record['rewarded']['texts']} good",
                        f"saved {args.model}"])


def cmd_feedback(args) -> None:
    model = _load(args)
    good = list(args.good or []) + read_texts(args.good_data, "good")
    bad = list(args.bad or []) + read_texts(args.bad_data, "bad")
    try:
        record = model.feedback(good, bad, good_weights=_ratings(args.good_ratings, len(good)) if good else None,
                                bad_weights=_ratings(args.bad_ratings, len(bad)) if bad else None,
                                strength=args.strength, prefix=args.prefix)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    model.save(args.model)
    _out(args, record, [f"feedback: {record['call']}", f"saved {args.model}"])


def _walk_lines(result, args) -> list[str]:
    lines = [f"{result.full_text}"]
    if result.spelled != result.full_text:
        lines.append(f"  spelled: {result.spelled}")
    lines.append(f"  continuation: {result.text!r}  cost {result.cost:.3f}  "
                 f"mode {result.mode}  traversal {result.traversal}  {'reached the end' if result.reached_end else ''}")
    if args.hops:
        lines.append("  hops: " + " ".join(f"{k}:{i}" for k, i in result.hops))
    return lines


def cmd_predict(args) -> None:
    model = _load(args)
    try:
        result = model.predict(args.prefix, args.length, mode=args.mode, traversal=args.traversal,
                               start=not args.fragment, temperature=args.temperature, to_end=args.to_end,
                               backoff=args.backoff)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    _out(args, result.to_dict(), _walk_lines(result, args))


def cmd_generate(args) -> None:
    model = _load(args)
    try:
        result = model.generate(args.prefix, args.length, mode=args.mode, traversal=args.traversal,
                                temperature=args.temperature, backoff=args.backoff)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    _out(args, result.to_dict(), _walk_lines(result, args))


def cmd_score(args) -> None:
    model = _load(args)
    texts = _texts_arg(args, what="scored")
    rows = []
    lines = []
    for text in texts:
        s = model.score(text, traversal=args.traversal)
        rows.append({"text": text, **s.to_dict()})
        lines.append(f"{s.bits:.3f} bits/unit  mean reward {s.mean_reward:+.3f}  worst penalty {s.worst_penalty:.3f}  {text!r}")
        if args.per_unit:
            for unit, bits, r, p in s.per_unit:
                lines.append(f"    {unit!r:>10}  {bits:6.3f} bits  reward {r:+.3f}  penalty {p:.3f}")
    _out(args, {"scores": rows}, lines)


def cmd_weights(args) -> None:
    model = _load(args)
    changes = {k: getattr(args, k) for k in ("smoothing", "share_scale", "reward_scale", "merit_scale", "penalty_scale",
                                             "backoff", "alpha", "floor", "strength")}
    if any(v is not None for v in changes.values()):
        try:
            settings = model.weights(**changes)
        except ValueError as exc:
            raise CliError(str(exc)) from None
        model.save(args.model)
    else:
        settings = model.settings.to_dict()
    _out(args, settings, [f"{k} = {v}" for k, v in settings.items()])


def cmd_info(args) -> None:
    model = _load(args)
    info = model.info()
    lines = [f"{info['codec']}  L={info['L']} (contexts up to {info['D']})  {info['N']:,} nodes  "
             f"{info['memory_bytes'] / 2**20:.1f} MiB  kind {info['kind']}",
             f"read: {info['read']['texts']} texts, {info['read']['units']} units, {info['read']['unk_share']:.1%} unk; "
             f"{info['nonzero_counts']:,} non-zero counts",
             f"judged: {info['judged']['texts']} texts, rewards {info['judged']['rewards_total']:g}, "
             f"penalties {info['judged']['penalties_total']:g}; {info['nonzero_rewards']:,} credited nodes",
             "settings: " + ", ".join(f"{k}={v}" for k, v in info["settings"].items())]
    _out(args, info, lines)


def cmd_checkpoints(args) -> None:
    manager = CheckpointManager(args.checkpoint_dir)
    records = manager.list()
    _out(args, {"checkpoints": records}, [f"{r['name']}  step {r['step']}  {r['bytes']} bytes  {r['saved_at']}" for r in records]
         or ["(none)"])


def cmd_invert(args) -> None:
    model = _load(args)
    model.invert()
    model.save(args.model)
    _out(args, {"inverted": True}, ["rewards and penalties swapped", f"saved {args.model}"])


def cmd_check(args) -> None:
    if args.R or args.L:
        if not (args.R and args.L):
            raise CliError("give both --R and --L")
        results = [check_address(args.R, args.L)]
    else:
        results = check_all(DEFAULT_SIZES)
    _out(args, {"checks": results}, [f"R={r['R']} L={r['L']}: {r['nodes']:,} nodes, {r['splits']} splits, "
                                     f"{r['unary']} unary - the brute-force tree is the arithmetic tree ({r['seconds']}s)"
                                     for r in results])


def _codec_options(args) -> dict:
    options: dict = {}
    if args.codec == "gpt2":
        options["files"] = args.gpt2_files if args.gpt2_files else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpt2")
        if isinstance(options["files"], list):
            options["files"] = options["files"][0] if len(options["files"]) == 1 else tuple(options["files"])
    if args.codec in ("gpt2", "syllables", "external", "bpe") and getattr(args, "top", None):
        options["top"] = args.top
    if args.codec == "bpe":
        options["vocab_size"] = args.vocab_size
    if args.codec == "syllables":
        options["vocabulary"] = args.vocabulary
    if args.codec == "external":
        options["spec"] = args.external
    return options


def cmd_bench(args) -> None:
    args.codec, args.L = _resolve(args, feedback=args.which == "feedback")
    try:
        if args.which == "throughput":
            result = benches.throughput(args.codec, args.L, args.units, args.seed, **_codec_options(args))
            lines = [f"{result['codec']} L={args.L}: {result['count']['per_second']:,} increments/s counting, "
                     f"{result['credit']['per_second']:,} entries/s crediting, {result['units_per_second']:,} units/s"]
        elif args.which == "compare":
            texts = read_texts(args.data)
            result = benches.compare(texts, args.codec, args.L, args.holdout, args.seed, args.smoothing, **_codec_options(args))
            lines = [f"{result['codec']} L={args.L}: {result['train_texts']} train / {result['held_texts']} held-out texts, "
                     f"smoothing {result['smoothing']}, {result['unk_share']:.1%} unk"]
            lines += [f"  backoff {b:8s}  held-out {v['held']:.4f} bits/unit   train {v['train']:.4f}" for b, v in result["bits"].items()]
        elif args.which == "feedback":
            result = benches.feedback(args.codec, args.L, **_codec_options(args))
            lines = [f"{result['codec']} L={args.L}, prefix {result['prefix']!r}"]
            for stage in ("untouched", "mat rewarded 5, punished 1", "then rewarded 50 more times"):
                lines.append(f"  {stage}: " + "; ".join(f"{t} -> {r['continues']!r} at {r['cost']}" for t, r in result[stage].items()))
            lines.append(f"  unbuyable: {result['unbuyable']}")
        else:
            texts = read_texts(args.data)
            result = benches.rungs(texts, args.codec, args.L, args.seed, args.samples, args.strength, **_codec_options(args))
            lines = [f"{result['codec']} L={args.L}: {result['samples']} rewarded steps at strength {result['strength']}",
                     f"  sibling lift (bits): all {result['sibling_lift_bits']['all']:+.4f}   final {result['sibling_lift_bits']['final']:+.4f}",
                     f"  control lift (bits): all {result['control_lift_bits']['all']:+.4f}   final {result['control_lift_bits']['final']:+.4f}"]
    except ValueError as exc:
        raise CliError(str(exc)) from None
    _out(args, result, lines)


# -- the parser ------------------------------------------------------------------------------------

def _add_codec_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--codec", default=None, choices=sorted(CODECS),
                   help="the tokenizer (default: phones - the phonetic tokenizer, phones in and phones out)")
    p.add_argument("--L", type=int, default=None,
                   help="the longest sequence held; contexts up to L-1 (default: the depth the codec primes to - phones 3, chars 4, syllables and gpt2 2)")
    p.add_argument("--vocab-size", type=int, default=1024, help="bpe: the vocabulary size")
    p.add_argument("--vocabulary", default="lexicon", choices=("lexicon", "corpus"), help="syllables: the source")
    p.add_argument("--top", type=int, default=None, help="gpt2 / syllables / external: keep the most frequent tokens")
    p.add_argument("--gpt2-files", nargs="+", default=None, metavar="PATH",
                   help="gpt2: the directory holding encoder.json + vocab.bpe, or the two paths (default: data/gpt2)")
    p.add_argument("--external", default=None, metavar="SPEC", help="external: tiktoken:<encoding> or hf:<tokenizer.json>")
    p.add_argument("--no-stress", action="store_true", help="phones / syllables: drop the vowels' stress")
    p.add_argument("--tokenizer-data", nargs="+", default=None, metavar="FILE",
                   help="texts a codec closes its vocabulary from (bpe, syllables --vocabulary corpus, a capped gpt2)")


def _add_walk_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="greedy", choices=MODES)
    p.add_argument("--traversal", default="reward", choices=TRAVERSALS)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--backoff", default=None, choices=BACKOFFS, help="greedy / sample: how far to fall")
    p.add_argument("--hops", action="store_true", help="print every hop of the walk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radixpair", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print one JSON document")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prime", help="make a primed model file")
    p.add_argument("--model", required=True)
    _add_codec_flags(p)
    p.add_argument("--kind", default="count", choices=("count",))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rungs", default="all", choices=("all", "final"))
    p.add_argument("--ceiling", type=int, default=None, help="the node ceiling (default 4,194,304)")
    p.set_defaults(func=cmd_prime)

    p = sub.add_parser("train", help="count texts into the count tree")
    p.add_argument("--model", required=True)
    p.add_argument("--text", action="append")
    p.add_argument("--data", action="append", metavar="FILE")
    p.add_argument("--checkpoint-dir", default=None)
    p.set_defaults(func=cmd_train)

    for name, func in (("reward", cmd_reward), ("punish", cmd_punish)):
        p = sub.add_parser(name, help=f"{name} texts in the reward tree")
        p.add_argument("--model", required=True)
        p.add_argument("--text", action="append")
        p.add_argument("--data", action="append", metavar="FILE")
        p.add_argument("--strength", type=float, default=None)
        p.add_argument("--ratings", type=float, nargs="+", default=None, help="marks out of 10, one per text")
        p.add_argument("--prefix", default=None, help="what the model was given: context for the texts, not credited")
        if name == "reward":
            p.add_argument("--read", action="store_true", help="also count the texts (the family's count model's behaviour)")
        p.set_defaults(func=func)

    p = sub.add_parser("2nrl", help="punish the bad texts, then reward the good")
    p.add_argument("--model", required=True)
    p.add_argument("--bad", nargs="+", required=True, metavar="FILE")
    p.add_argument("--good", nargs="+", required=True, metavar="FILE")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--prefix", default=None)
    p.set_defaults(func=cmd_2nrl)

    p = sub.add_parser("feedback", help="good and / or bad texts with ratings")
    p.add_argument("--model", required=True)
    p.add_argument("--good", action="append")
    p.add_argument("--bad", action="append")
    p.add_argument("--good-data", nargs="+", default=None, metavar="FILE")
    p.add_argument("--bad-data", nargs="+", default=None, metavar="FILE")
    p.add_argument("--good-ratings", type=float, nargs="+", default=None)
    p.add_argument("--bad-ratings", type=float, nargs="+", default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--prefix", default=None)
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("predict", help="continue a prefix")
    p.add_argument("--model", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--length", type=int, default=20)
    p.add_argument("--fragment", action="store_true", help="the prefix does not begin a text")
    p.add_argument("--to-end", action="store_true")
    _add_walk_flags(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("generate", help="a whole text")
    p.add_argument("--model", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--length", type=int, default=60)
    _add_walk_flags(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("score", help="bits per unit and the reward tree's readings")
    p.add_argument("--model", required=True)
    p.add_argument("--text", action="append")
    p.add_argument("--data", action="append", metavar="FILE")
    p.add_argument("--traversal", default="reward", choices=TRAVERSALS)
    p.add_argument("--per-unit", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("weights", help="show or change the settings a prediction reads")
    p.add_argument("--model", required=True)
    for name in ("smoothing", "share_scale", "reward_scale", "merit_scale", "penalty_scale", "alpha", "floor", "strength"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float, default=None)
    p.add_argument("--backoff", default=None, choices=BACKOFFS)
    p.set_defaults(func=cmd_weights)

    p = sub.add_parser("info", help="what the model is and what it holds")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("checkpoints", help="list a checkpoint directory")
    p.add_argument("--checkpoint-dir", required=True)
    p.set_defaults(func=cmd_checkpoints)

    p = sub.add_parser("invert", help="swap rewards and penalties")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_invert)

    p = sub.add_parser("check", help="the brute-force oracle")
    p.add_argument("--R", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("bench", help="throughput | compare | feedback | rungs")
    p.add_argument("which", nargs="?", default="throughput", choices=("throughput", "compare", "feedback", "rungs"))
    _add_codec_flags(p)
    p.add_argument("--data", nargs="+", default=None, metavar="FILE")
    p.add_argument("--units", type=int, default=200_000)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=None)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except CliError as exc:
        print(f"radixpair: {exc}", file=sys.stderr)
        return 2
    return 0
