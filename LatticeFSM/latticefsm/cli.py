"""``python3 -m latticefsm <command>`` - the reference implementation's command line.

    demo        learn a language, be stimulated, let time pass
    train       teach a language by credit over random strings, and save the machine
    run         read a string, traversing it (or --quiet), optionally --credit it, and save what that did
    teach       traverse one edge deliberately and credit it
    accuracy    how many random strings of a language the machine classifies right (nothing moves)
    table       the greedy transition table (nothing moves)
    stats       size, memory, clock and stimulation
    tick        let time pass
    stimulate   raise the stimulation
    focus       set the machine's focus (a node of the central vertical vector to start from), or learn it
    skip        let runs skip a node when a two-edge path is more efficient
    rearrange   let the nodes rearrange themselves toward the centre (one pass, or --full), or swap two
    compress    fold the matrix into its central node (exact, float32 or float16) and save the code
    expand      rebuild a machine from a code, from the central node outward (all shells, or --shells k)
    core-run    walk a string straight from a code, decoding only the cells the walk reaches
    network     machines connected into a network: new, fit, evaluate, predict, translate, encode, summary, make-data
    ngram       predict the next thing: train an N-gram model on a corpus or with Ollama, predict, generate
    chain       one input to a node, the output node predicts the next: walks only as deep as confidence needs
    base64      encode anything (text, an image, a video) as base64, or decode it back
    experiment  the learning, stimulation, adaptation and compression experiments
    test        run the test suite

The Rust binary (``../rust``) has the same commands and ``serve`` besides; the
two read and write the same machine files.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from . import experiment
from .compress import PRECISIONS, compress, fidelity, load_core
from .languages import LANGUAGES, all_strings, examples, language
from .machine import BASELINE, DEFAULT_ALPHABET, DEFAULT_STATES, DISCOUNT, LIFE, Machine, load_machine


def _add_machine_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--states", type=int, default=DEFAULT_STATES,
                   help=f"how many states (default {DEFAULT_STATES}: with the default alphabet, a 13 x 13 x 13 matrix)")
    p.add_argument("--alphabet", default=DEFAULT_ALPHABET, help=f"one character per symbol (default {DEFAULT_ALPHABET})")
    p.add_argument("--accepting", default=None, help="accepting states, comma-separated (default: the language's, or 0)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--life", type=float, default=LIFE, help="ticks for a trace, a width or the stimulation to fade by half")
    p.add_argument("--baseline", type=float, default=BASELINE, help="the stimulation the machine rests at")
    p.add_argument("--calm", type=float, default=None, help="ticks for a surge of stimulation to fade by half (default: life)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=DISCOUNT, help="credit each step back from a run's end receives")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--language", default="even-b", choices=sorted(LANGUAGES), help="the language train / demo / accuracy use")
    p.add_argument("--focus", type=float, default=None,
                   help="0..1: the node of the central vertical vector runs start from (0 or none the top, 1 the bottom)")
    p.add_argument("--learn-focus", action="store_true", help="read the focus off the input, learned from credit")
    p.add_argument("--skip", action="store_true", help="skip a node when a two-edge path is more efficient")
    p.add_argument("--skip-margin", type=float, default=0.0, help="how much more efficient (log-probability)")
    p.add_argument("--rearrange-every", type=int, default=0,
                   help="let the nodes rearrange themselves every N transitions (0: never)")
    p.add_argument("--compress-every", type=int, default=0,
                   help="fold the matrix into its central node every N transitions (0: never)")
    p.add_argument("--compress-precision", choices=list(PRECISIONS), default="exact")
    p.add_argument("--compress-rebuild", action="store_true",
                   help="rebuild the matrix from the code after each automatic compression, applying its loss")


def _add_load_save(p: argparse.ArgumentParser) -> None:
    p.add_argument("--load", help="a saved machine (.json or .json.gz); without it a fresh machine is made")
    p.add_argument("--save", help="write the machine, as it is after this command, here")


def _fresh(args) -> Machine:
    if args.accepting is not None:
        accepting = [int(s) for s in args.accepting.split(",") if s.strip()]
    else:
        accepting = list(language(args.language).accepting)
    return Machine(args.states, list(args.alphabet), accepting=accepting, start=args.start, life=args.life,
                   baseline=args.baseline, calm=args.calm, temperature=args.temperature, discount=args.discount,
                   seed=args.seed, compress_every=args.compress_every, compress_precision=args.compress_precision,
                   compress_rebuild=args.compress_rebuild, focus=args.focus, learn_focus=args.learn_focus,
                   skip=args.skip, skip_margin=args.skip_margin, rearrange_every=args.rearrange_every)


def _machine(args) -> Machine:
    return load_machine(args.load) if getattr(args, "load", None) else _fresh(args)


def _save(m: Machine, args) -> None:
    if getattr(args, "save", None):
        m.save(args.save)
        print(f"saved {args.save}", file=sys.stderr)


def _table(m: Machine) -> None:
    table = m.transition_table()
    acc = m.accepting
    # a symbol no edge was ever traversed on is a row of ties: listed once, not thirteen times
    used = [sym for a, sym in enumerate(m.alphabet) if any(e.symbol == a and e.touched for e in m.lattice)]
    unused = [sym for sym in m.alphabet if sym not in used]
    print(f"  greedy table (accepting: {acc}):")
    for s in range(m.n_states):
        cells = "   ".join(f"{sym} -> {table[(s, sym)]}" for sym in used)
        print(f"    {s:>2}{'*' if s in acc else ' '}: {cells}")
    if unused and used:
        print(f"    (never traversed on {' '.join(unused)}: every next state still ties)")


def cmd_train(args) -> int:
    lang = language(args.language)
    m = _machine(args)
    for st in m.lattice.states:
        st.accepting = st.index in lang.accepting
    rng = random.Random(args.seed)
    test = examples(lang, args.tests, rng, args.max_length)
    print(f"{lang.name} ({lang.description}): {m.n_states} states, {args.episodes} episodes"
          f"{', quietly' if args.quiet else ''}")
    print(f"  accuracy before {m.accuracy(test):.3f}")
    for ep, acc in experiment.teach_language(m, lang.name, args.episodes, rng, args.max_length, quiet=args.quiet,
                                             test=test, every=max(1, args.episodes // 8)):
        print(f"  after {ep:>6}: {acc:.3f}")
    _table(m)
    _save(m, args)
    return 0


def cmd_run(args) -> int:
    m = _machine(args)
    for _ in range(max(1, args.times)):
        r = m.run(args.text, stimulation=args.stimulation, temperature=args.temperature_run, quiet=args.quiet,
                  from_middle=args.from_middle, focus=args.focus if args.load else None,
                  skip=True if args.skip else None)
        steps = "  ".join(f"{t.source} -{t.symbol}({t.probability:.2f})-> {t.target}" if t.skipped is None else
                          f"{t.source} ={t.symbol}({t.probability:.2f})=> {t.target} [skipping {t.skipped}]"
                          for t in r.transitions)
        print(f"{'quietly: ' if args.quiet else ''}{steps or '(empty)'}  => {r.final} "
              f"({'accepted' if r.accepted else 'rejected'}), log p {r.log_probability:.3f}, "
              f"stimulation {m.stimulation:.2f}")
        if args.credit is not None:
            n = m.credit(args.credit)
            print(f"  credited {n} edges with {args.credit:+}")
    _save(m, args)
    return 0


def _print_core(core, m) -> None:
    s = core.summary()
    f = fidelity(m, core.decompress())
    print(f"the {s['shape'][0]} x {s['shape'][1]} x {s['shape'][2]} matrix folded into its central node "
          f"({s['center_label'][0]}, {s['center_label'][1]}, {s['center_label'][2]}), {s['precision']}:")
    print(f"  {s['bytes']} bytes for {s['touched']} touched edges of {s['cells']} "
          f"(the dense matrix is {s['dense_bytes']} bytes: {s['ratio']:.1f}x smaller), plus a "
          f"{s['record_bytes']}-byte record of the rest")
    print(f"  loss: {'none - every field of every edge comes back bit for bit' if f['lossless'] else 'lossy'}; "
          f"largest relative error {f['max_relative_error']:.2e}; KL mean {f['mean_kl']:.2e}, max {f['max_kl']:.2e}; "
          f"greedy choices changed {f['greedy_changed']} of {f['rows']}")
    print("  from the central node outward:")
    for row in s["shell_table"]:
        print(f"    shell {row['shell']}: {row['cells']:>4} cells, {row['touched']:>4} touched, {row['bytes']:>6} bytes")


def cmd_compress(args) -> int:
    m = _machine(args)
    core = compress(m, precision=args.precision, budget=args.budget)
    _print_core(core, m)
    if args.core:
        core.save(args.core)
        print(f"saved {args.core}", file=sys.stderr)
    return 0


def cmd_expand(args) -> int:
    core = load_core(args.core)
    m = core.decompress(shells=args.shells)
    shown = "every shell" if args.shells is None else f"{args.shells} shell{'s' if args.shells != 1 else ''}"
    st = m.stats()
    print(f"rebuilt from the central node outward, {shown}: {st['touched']} of {st['edges']} edges written, "
          f"clock {st['clock']}")
    _table(m)
    _save(m, args)
    return 0


def _focus_line(m) -> str:
    if m.learn_focus:
        return "focus learned from the input: each run starts from the node of the central vertical vector it picks"
    if m.focus is None:
        return f"no focus: runs start from the start state, {m.start}"
    s, a, t = m.stats()["focus_node"]
    return f"focus {m.focus:g}: runs start from node ({s}, {m.alphabet[a]}, {t}) of the central vertical vector, state {s}"


def cmd_focus(args) -> int:
    m = _machine(args)
    if args.none:
        m.focus = None
    elif args.level is not None:
        m.focus = args.level
    if args.learn is not None:
        m.learn_focus = args.learn == "on"
    print(_focus_line(m))
    if m.learn_focus and args.text:
        f = m.focus_learner.predict([c for c in args.text if not c.isspace()])
        print(f"  the learner reads {args.text!r} as focus {f:.3f}: state {m.focus_state(f)}")
    _save(m, args)
    return 0


def cmd_skip(args) -> int:
    m = _machine(args)
    m.skip = args.state == "on"
    if args.margin is not None:
        if args.margin < 0:
            raise SystemExit("--margin must be >= 0")
        m.skip_margin = args.margin
    print(f"skips {'on' if m.skip else 'off'}, margin {m.skip_margin:g}; {m.skips} taken so far")
    _save(m, args)
    return 0


def cmd_rearrange(args) -> int:
    m = _machine(args)
    if args.swap:
        kind, i, j = args.swap
        if kind == "states":
            m.swap_states(int(i), int(j))
        else:
            m.swap_symbols(i, j)
        print(f"swapped {kind} {i} and {j}")
    else:
        made = m.rearrange(full=args.full)
        print(f"{len(made)} swaps toward the centre{' (until settled)' if args.full else ' (one pass)'}")
    st = m.stats()
    print(f"  states, top to bottom: {st['state_order']}  (start state {m.start})")
    print(f"  symbols, in order: {''.join(st['symbol_order'])}")
    _save(m, args)
    return 0


def cmd_core_run(args) -> int:
    core = load_core(args.core)
    states, accepted = core.run(args.text, from_middle=args.from_middle)
    print(f"{'from the middle: ' if args.from_middle else ''}{' -> '.join(map(str, states))}  "
          f"({'accepted' if accepted else 'rejected'}), read straight from the code")
    return 0


def cmd_teach(args) -> int:
    m = _machine(args)
    e = m.teach(args.source, args.symbol, args.target, args.amount)
    print(e)
    _save(m, args)
    return 0


def cmd_accuracy(args) -> int:
    lang = language(args.language)
    m = _machine(args)
    test = examples(lang, args.tests, random.Random(args.seed + 1000), args.max_length)
    print(f"{lang.name} on {len(test)} strings: {m.accuracy(test, temperature=args.temperature_run):.3f}")
    return 0


def cmd_table(args) -> int:
    _table(_machine(args))
    return 0


def cmd_stats(args) -> int:
    for k, v in _machine(args).stats().items():
        print(f"  {k}: {v}")
    return 0


def cmd_tick(args) -> int:
    m = _machine(args)
    m.tick(args.ticks)
    print(f"clock {m.clock}  stimulation {m.stimulation:.3f}")
    _save(m, args)
    return 0


def cmd_stimulate(args) -> int:
    m = _machine(args)
    if args.level is not None:
        m.stimulation = args.level
    else:
        m.stimulate(args.amount)
    print(f"stimulation {m.stimulation:.3f}")
    _save(m, args)
    return 0


def _network(args):
    from .network import Network, load_network
    if args.load:
        return load_network(args.load)
    options = {"life": args.life, "temperature": args.temperature, "discount": args.discount}
    if args.skip:
        options["skip"] = True
    if args.learn_focus:
        options["learn_focus"] = True
    net = Network(args.arch, alphabet=list(args.alphabet), seed=args.seed, merge=args.merge, vote=args.vote,
                  layer_options=options)
    if args.accepting is not None:
        accepting = [int(s) for s in args.accepting.split(",") if s.strip()]
    elif net.decoder is not None:
        accepting = None
    else:
        accepting = list(language(args.language).accepting)
    return net.compile(accepting=accepting, layer_discount=args.layer_discount, credit=args.credit)


def _network_accuracy(net, lang, max_length: int) -> tuple[float, int]:
    strings = all_strings(max_length)
    return net.evaluate(strings, [lang.member(s) for s in strings]), len(strings)


def _reads_files(net) -> bool:
    from .network import PhoneticEncoder, TextEncoder
    return net.encoder is not None and not isinstance(net.encoder, (PhoneticEncoder, TextEncoder))


def _pairs(args, net=None) -> tuple[str, list[tuple[str, str]]]:
    from .media import read_labels
    from .phonetic import TASKS
    if args.pairs:
        return args.pairs, read_labels(args.pairs, resolve=net is not None and _reads_files(net))
    name = args.task or "plural"
    if name not in TASKS:
        raise SystemExit(f"no task {name!r}: plural or past")
    return name, list(TASKS[name])


def _data(args, net) -> tuple[list[str], list[bool]]:
    from .media import read_labels, verdict
    rows = read_labels(args.data, resolve=_reads_files(net))
    return [x for x, _ in rows], [verdict(y) for _, y in rows]


def cmd_network(args) -> int:
    from .phonetic import split
    if args.action == "make-data":
        from .media import make_dataset
        path = make_dataset(args.kind, args.out, count=args.count, seed=args.seed)
        print(f"wrote {path}")
        return 0
    net = _network(args)
    by_pairs = bool(args.task or args.pairs)
    if args.action in ("new", "summary"):
        print(net.summary())
    elif args.action == "encode":
        for x in args.text:
            symbols = net._symbols(x)
            print(f"{x}: {len(symbols)} symbols")
            print(f"  {' '.join(symbols)}")
    elif args.action == "fit" and args.data:
        x, y = _data(args, net)
        tr = [i for i in range(len(x)) if i % 5 != 4]
        te = [i for i in range(len(x)) if i % 5 == 4]
        xs, ys, vx, vy = [x[i] for i in tr], [y[i] for i in tr], [x[i] for i in te], [y[i] for i in te]
        epochs = args.epochs if args.epochs is not None else 10
        print(f"{args.data}: {len(xs)} examples taught x {epochs} epochs, {len(vx)} held out")
        print(f"  accuracy before: taught {net.evaluate(xs, ys):.6f}, held out {net.evaluate(vx, vy):.6f}")
        for _ in range(epochs):
            h = net.fit(xs, ys, epochs=1)
            print(f"  epoch {net.epochs}: right while taught {h['accuracy'][0]:.6f}, taught {net.evaluate(xs, ys):.6f}, "
                  f"held out {net.evaluate(vx, vy):.6f}")
        print(net.summary())
    elif args.action == "evaluate" and args.data:
        x, y = _data(args, net)
        tr = [i for i in range(len(x)) if i % 5 != 4]
        te = [i for i in range(len(x)) if i % 5 == 4]
        print(f"accuracy {args.data}: taught {net.evaluate([x[i] for i in tr], [y[i] for i in tr]):.6f}, "
              f"held out {net.evaluate([x[i] for i in te], [y[i] for i in te]):.6f}")
    elif args.action == "fit" and by_pairs:
        name, pairs = _pairs(args, net)
        train, test = split(pairs)
        epochs = args.epochs if args.epochs is not None else 20
        print(f"{name}: {len(train)} pairs taught x {epochs} epochs, {len(test)} held out")
        print(f"  exact before: taught {net.evaluate_pairs(train):.6f}, held out {net.evaluate_pairs(test):.6f}")
        for _ in range(epochs):
            h = net.fit_pairs(train, epochs=1, validation_data=test)
            print(f"  epoch {net.epochs}: sounds right while taught {h['sounds'][0]:.6f}, exact {h['exact'][0]:.6f}, "
                  f"held out {h['val_exact'][0]:.6f}")
        print(net.summary())
    elif args.action == "fit":
        lang = language(args.language)
        epochs = args.epochs if args.epochs is not None else 1
        rng = random.Random(args.seed)
        train = examples(lang, args.episodes, rng, args.max_length)
        acc, n = _network_accuracy(net, lang, args.max_length)
        print(f"{lang.name} ({lang.description}): {args.episodes} strings x {epochs} epochs")
        print(f"  accuracy on every string up to {args.max_length} symbols ({n}): before {acc:.6f}")
        for _ in range(epochs):
            hist = net.fit([s for s, _ in train], [m for _, m in train], epochs=1)
            acc, _ = _network_accuracy(net, lang, args.max_length)
            print(f"  epoch {net.epochs}: right while taught {hist['accuracy'][0]:.6f}, after {acc:.6f}")
        print(net.summary())
    elif args.action == "evaluate" and by_pairs:
        name, pairs = _pairs(args, net)
        train, test = split(pairs)
        print(f"exact {name}: taught {net.evaluate_pairs(train):.6f}, held out {net.evaluate_pairs(test):.6f}")
    elif args.action == "evaluate":
        lang = language(args.language)
        acc, n = _network_accuracy(net, lang, args.max_length)
        print(f"accuracy {lang.name} on every string up to {args.max_length} symbols ({n}): {acc:.6f}")
    elif args.action == "predict":
        for text in args.text or [""]:
            t = net.trace(text)
            print(f"{text or '(empty)'}: {'accepted' if t['accepted'] else 'rejected'}")
            for step in t["layers"]:
                english = f' = "{step["english"]}"' if "english" in step else ""
                print(f"  {step['layer']} read {step['read'] or '-'}, walked {' '.join(map(str, step['states'])) or '-'}, "
                      f"wrote {step['wrote'] or '-'}{english}{' (accepting)' if step['accepted'] else ''}")
    elif args.action == "translate":
        for text in args.text:
            print(f"{text} -> {net.translate(text)}  ({' '.join(net.sounds(text))})")
    if args.save and args.action in ("new", "fit", "summary"):
        net.save(args.save)
        print(f"saved {args.save}", file=sys.stderr)
    return 0


def _ngram(args):
    from .ngram import NGram, load_ngram
    if getattr(args, "load", None) and os.path.exists(args.load):
        return load_ngram(args.load)
    if getattr(args, "load", None) and args.action not in ("train", "autotrain"):
        raise SystemExit(f"no model at {args.load}")
    return NGram(args.order, args.unit, lower=not args.keep_case, stimulation=args.stimulation)


def _fmt(v, digits=3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def cmd_ngram(args) -> int:
    from .autotrain import Ollama, OllamaError, autotrain
    from .ngram import corpus_sequences, detokenize, split_lines
    if args.action == "ollama":
        client = Ollama(args.url, args.model)
        try:
            names = client.models()
        except OllamaError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Ollama at {client.url}: {len(names)} models" + "".join(f"\n  {n}" for n in names))
        print(f"autotrain would use {client.model}" + ("" if client.model in names or
                                                       any(n.split(':')[0] == client.model for n in names)
                                                       else " (not pulled: `ollama pull " + client.model + "`)"))
        return 0
    m = _ngram(args)
    data = [p for d in (args.data or ["data/corpus"]) for p in d.split(",") if p]
    if args.action == "train":
        _, test = m.train_paths(data, held_out=args.held_out, predict=not args.no_predict, credit=args.self_credit)
        learned = m.reset_online()
        st = m.stats()
        what = "base64 letters" if st["unit"] == "base64" else st["unit"] + "s"
        print(f"{st['order']}-grams of {what}: {st['tokens']} tokens, {st['vocab']} distinct, {st['ngrams']} "
              f"n-grams in {st['contexts']} contexts, from {st['sources']} sources")
        if learned["predicted"]:
            print(f"while training: predicted {learned['predicted']} tokens before learning each, first guess right "
                  f"{_fmt(learned['accuracy'])}, perplexity {_fmt(learned['perplexity'], 2)}")
        if test:
            ev = m.evaluate(test)
            print(f"held out ({len(test)} lines, {ev['tokens']} tokens): perplexity {_fmt(ev['perplexity'], 2)}, "
                  f"{_fmt(ev['bits'])} bits a token, next-token accuracy {_fmt(ev['accuracy'])}, top 5 {_fmt(ev['top5'])}")
    elif args.action == "autotrain":
        client = Ollama(args.url, args.model) if args.ollama else None
        topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()]
        try:
            autotrain(m, data, held_out=args.held_out, ollama=client, topics=topics, rounds=args.rounds,
                      lines=args.lines, continuations=args.continuations, judge=args.judge, patience=args.patience,
                      keep=args.keep, save=args.save, seed=args.seed, log=print, predict=not args.no_predict,
                      self_credit=args.self_credit)
        except OllamaError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    elif args.action == "predict":
        for text in args.text or [""]:
            print(f"after {text!r}:" if text else "at the start:")
            for tok, p in m.predict(text, args.k):
                print(f"  {tok:<16} {p:.6f}")
    elif args.action == "generate":
        import random as _random
        rng = _random.Random(args.seed)
        if args.out:
            if m.unit != "base64":
                raise SystemExit("--out writes bytes: it needs a base64 model (--unit base64)")
            start = b""
            if args.prefix_from:
                with open(args.prefix_from, "rb") as f:
                    start = f.read(args.prefix_bytes)
            elif args.text:
                start = args.text[0].encode("utf-8")
            data_out = m.generate_bytes(start, args.max_tokens, args.temperature, rng)
            with open(args.out, "wb") as f:
                f.write(data_out)
            print(f"wrote {len(data_out)} bytes to {args.out}")
            return 0
        for text in args.text or [""]:
            if m.unit == "base64":
                print(m.complete(text, args.max_tokens, args.temperature, rng))
                continue
            toks = m.generate(text, args.max_tokens, args.temperature, rng)
            print(f"{text}{' ' if text and toks and m.unit == 'word' else ''}{detokenize(toks, m.unit)}")
    elif args.action == "evaluate":
        test = []
        for _, seqs in corpus_sequences(data, m.unit, m.lower):
            test += split_lines(seqs, args.held_out)[1] if args.held_out else seqs
        ev = m.evaluate(test)
        print(f"{len(test)} lines, {ev['tokens']} tokens: perplexity {_fmt(ev['perplexity'], 6)}, bits "
              f"{_fmt(ev['bits'], 6)}, accuracy {_fmt(ev['accuracy'], 6)}, top 5 {_fmt(ev['top5'], 6)}")
    elif args.action == "stats":
        for k, v in m.stats().items():
            print(f"{k}: {v}")
    elif args.action == "credit":
        for text in args.text:
            print(f"credited {m.credit_text(text, args.amount)} edges of {text!r} with {args.amount:+}")
    if args.save and args.action in ("train", "credit"):
        m.save(args.save)
        print(f"saved {args.save}", file=sys.stderr)
    return 0


def _chain_data(args) -> list[str]:
    return [p for d in (args.data or ["data/corpus"]) for p in d.split(",") if p]


def _chain_test(m, args) -> list:
    from .ngram import corpus_sequences, split_lines
    test = []
    for _, seqs in corpus_sequences(_chain_data(args), m.unit, m.lower):
        test += split_lines(seqs, args.held_out)[1] if args.held_out else seqs
    return test


def _show(token) -> str:
    return "\u00b7" if token is None else token


def cmd_chain(args) -> int:
    from .chain import Chain, load_chain
    from .ngram import detokenize
    if args.load and os.path.exists(args.load):
        m = load_chain(args.load)
    elif args.load and args.action != "train":
        raise SystemExit(f"no chain at {args.load}")
    else:
        m = Chain(args.nodes, args.unit, lower=not args.keep_case, confidence=args.confidence)
        if args.action != "train":
            # nothing to load: learn the data first (every --held-out-th line kept back), quietly
            m.train_paths(_chain_data(args), held_out=args.held_out, predict=False)
    if args.set_confidence is not None:
        m.confidence = args.set_confidence
    what = "base64 letters" if m.unit == "base64" else m.unit + "s"
    if args.action == "train":
        _, test = m.train_paths(_chain_data(args), held_out=args.held_out, predict=not args.no_predict,
                                credit=args.self_credit)
        learned = m.reset_online()
        st = m.stats()
        print(f"a chain of {m.nodes} nodes ({m.depth} input nodes and the output node) over {what}: {st['tokens']} "
              f"inputs, {st['vocab']} distinct, {st['contexts']} node contexts, from {st['sources']} sources")
        if learned["predicted"]:
            print(f"while training: predicted {learned['predicted']} inputs before learning each, first guess right "
                  f"{_fmt(learned['accuracy'])}, perplexity {_fmt(learned['perplexity'], 2)}, "
                  f"{_fmt(learned['depth'], 2)} input nodes traversed")
        if test:
            ev = m.evaluate(test)
            print(f"held out ({len(test)} lines, {ev['tokens']} inputs) at confidence {m.confidence:.2f}: perplexity "
                  f"{_fmt(ev['perplexity'], 2)}, accuracy {_fmt(ev['accuracy'])}, {_fmt(ev['depth'], 2)} of "
                  f"{m.depth} input nodes traversed")
    elif args.action == "predict":
        for text in args.text or [""]:
            e = m.explain(text, args.k)
            print(f"after {text!r}:" if text else "at the start:")
            print("  input nodes: " + " ".join(_show(t) for t in e["inputs"]))
            steps = " > ".join(f"node {st['node']} " + ("(output)" if st["input"] is None else repr(st["input"])) +
                               f" {st['confidence']:.3f}" for st in e["walk"])
            gathered = e["walk"][-1]["gathered"] if e["walk"] else 0.0
            print(f"  walk: {steps or '(nothing learned)'}; {e['depth']} of {m.depth} input nodes, "
                  f"confidence gathered {gathered:.3f}")
            for tok, p in e["next"]:
                print(f"  {tok:<16} {p:.6f}")
    elif args.action == "generate":
        import random as _random
        rng = _random.Random(args.seed)
        for text in args.text or [""]:
            start = m.prompt_tokens(text)
            toks, depths = m.continue_tokens(start, args.max_tokens, args.temperature, rng)
            if m.unit == "base64":
                line = detokenize(start + toks, m.unit)
            else:
                line = f"{text}{' ' if text and toks and m.unit == 'word' else ''}{detokenize(toks, m.unit)}"
            mean = sum(depths) / len(depths) if depths else 0.0
            print(f"{line}\n  ({len(depths)} walks, {mean:.2f} input nodes traversed on average: "
                  f"{' '.join(str(d) for d in depths)})")
    elif args.action == "evaluate":
        test = _chain_test(m, args)
        ev = m.evaluate(test)
        print(f"{len(test)} lines, {ev['tokens']} inputs at confidence {m.confidence:.2f}: perplexity "
              f"{_fmt(ev['perplexity'], 6)}, bits {_fmt(ev['bits'], 6)}, accuracy {_fmt(ev['accuracy'], 6)}, "
              f"input nodes traversed {_fmt(ev['depth'], 6)}; walks by depth: {' '.join(map(str, ev['depths']))}")
    elif args.action == "sweep":
        test = _chain_test(m, args)
        thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
        print(f"{len(test)} held-out lines; the less confidence a walk needs, the fewer of the {m.depth} input nodes "
              f"it traverses:")
        for r in m.sweep(test, thresholds):
            print(f"  confidence {r['confidence']:.2f}: accuracy {_fmt(r['accuracy'], 4)}, perplexity "
                  f"{_fmt(r['perplexity'], 3)}, {_fmt(r['depth'], 3)} input nodes traversed")
    elif args.action == "stats":
        st = m.stats()
        print(f"a chain of {m.nodes} nodes over {what}, confidence {m.confidence:.2f}: {st['tokens']} inputs, "
              f"{st['vocab']} distinct, {st['contexts']} node contexts, {st['credited']} credited")
        for d in st["depths"]:
            print(f"  depth {d['depth']}: {d['contexts']} contexts, tested {d['tested']} times, right "
                  f"{_fmt(d['accuracy'])}")
    elif args.action == "credit":
        for text in args.text:
            print(f"credited {m.credit_text(text, args.amount)} nodes on the walks of {text!r} with {args.amount:+}")
    if args.save and args.action in ("train", "credit"):
        m.save(args.save)
        print(f"saved {args.save}", file=sys.stderr)
    return 0


def cmd_base64(args) -> int:
    from . import b64
    for item in args.input or ["-"]:
        if item == "-":
            data = sys.stdin.buffer.read()
        elif os.path.isfile(item):
            with open(item, "rb") as f:
                data = f.read()
        else:
            data = item.encode("utf-8")
        if args.action == "encode":
            text = b64.encode(data)
            if args.out:
                with open(args.out, "w", encoding="ascii") as f:
                    f.write(text + "\n")
                print(f"wrote {len(text)} letters ({len(data)} bytes encoded) to {args.out}", file=sys.stderr)
            else:
                print(text)
        else:
            raw = b64.decode(data.decode("ascii", "ignore"))
            if args.out:
                with open(args.out, "wb") as f:
                    f.write(raw)
                print(f"wrote {len(raw)} bytes to {args.out}", file=sys.stderr)
            else:
                sys.stdout.write(raw.decode("utf-8", "replace") + ("" if raw.endswith(b"\n") else "\n"))
    return 0


def cmd_experiment(args) -> int:
    for rec in experiment.run(args.which, args.out, episodes=args.episodes, life=args.life):
        print(experiment.tables(rec))
        print()
    return 0


def cmd_demo(args) -> int:
    lang = language(args.language)
    m = _fresh(args)
    for st in m.lattice.states:
        st.accepting = st.index in lang.accepting
    rng = random.Random(args.seed)
    test = examples(lang, 300, rng, 6)
    print(f"a machine of {m.n_states} states over {m.alphabet}: {len(m.lattice)} edges, each a record of its own; "
          f"life {m.life:g} ticks, baseline stimulation {m.baseline:g}")
    print(f"\nlearning {lang.name!r} ({lang.description}) by reward and punishment over random strings:")
    print(f"  accuracy before: {m.accuracy(test):.3f}")
    for ep, acc in experiment.teach_language(m, lang.name, args.episodes, rng, 6, test=test,
                                             every=max(1, args.episodes // 4)):
        print(f"  after {ep:>5} episodes: {acc:.3f}")
    _table(m)
    s = m.stats()
    print(f"  clock {s['clock']}, {s['touched']} of {s['edges']} edges touched, widest channel {s['widest']:.2f}, "
          f"narrowest {s['narrowest']:.2f}")
    print("\nstimulation prefers the wide channel.  A fresh fork of three edges, widths 4, 1 and 0.25:")
    fork = Machine(3, "a", life=m.life)
    for t, w in enumerate((4.0, 1.0, 0.25)):
        fork.edge(0, "a", t).set_width(w, 0)
    for level in (0.0, 1.0, 2.0, 4.0):
        print(f"  stimulation {level:<4g}: [{', '.join(f'{p:.3f}' for p in fork.probabilities(0, 'a', stimulation=level))}]")
    m.stimulate(3.0)
    print(f"\nstimulated by +3: level {m.stimulation:.2f}; a life of silence later:")
    m.tick(int(m.life))
    print(f"  level {m.stimulation:.2f}, the widest channel now {m.stats()['widest']:.2f}; accuracy still {m.accuracy(test):.3f}")
    m.tick(int(4 * m.life))
    print(f"  four more lives: level {m.stimulation:.2f}, widest {m.stats()['widest']:.2f}; accuracy {m.accuracy(test):.3f}"
          " - the verdicts never fade, the widths and traces do")
    core = compress(m)
    f = fidelity(m, core.decompress())
    half = compress(m, "float16")
    fh = fidelity(m, half.decompress())
    cs, ca, ct = core.center
    print(f"\nfolded into its central node ({cs}, {m.alphabet[ca]}, {ct}): {core.bytes} bytes exact, "
          f"{'lossless' if f['lossless'] else 'lossy'} - {core.summary()['ratio']:.1f}x smaller than the dense matrix")
    print(f"  float16: {half.bytes} bytes, max KL {fh['max_kl']:.1e}, {fh['greedy_changed']} greedy choices changed")
    states, accepted = core.run("abba", from_middle=True)
    print(f"  'abba' walked straight from the code, from the middle state outward: {' -> '.join(map(str, states))} "
          f"({'accepted' if accepted else 'rejected'})")
    _save(m, args)
    return 0


def cmd_test(args) -> int:
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_latticefsm")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m latticefsm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help, machine=True, load=True):
        q = sub.add_parser(name, help=help)
        if machine:
            _add_machine_options(q)
        if load:
            _add_load_save(q)
        q.set_defaults(fn=fn)
        return q

    q = add("demo", cmd_demo, "learn a language, be stimulated, let time pass", load=False)
    q.add_argument("--episodes", type=int, default=4000)
    q.add_argument("--save")
    q = add("train", cmd_train, "teach a language by credit over random strings")
    q.add_argument("--episodes", type=int, default=4000)
    q.add_argument("--max-length", type=int, default=6)
    q.add_argument("--tests", type=int, default=300)
    q.add_argument("--quiet", action="store_true", help="run the strings without traversing: the control")
    q = add("run", cmd_run, "read a string from the start state")
    q.add_argument("text")
    q.add_argument("--stimulation", type=float, default=None, help="run at this level instead of the machine's")
    q.add_argument("--temperature-run", type=float, default=None, help="draw at this temperature (0: greedy)")
    q.add_argument("--quiet", action="store_true", help="ask only: nothing moves")
    q.add_argument("--credit", type=float, default=None, help="credit the run: positive rewards, negative punishes")
    q.add_argument("--from-middle", action="store_true", help="start from the middle state, the central node's")
    q.add_argument("--times", type=int, default=1)
    q = add("teach", cmd_teach, "traverse one edge deliberately and credit it")
    q.add_argument("source", type=int)
    q.add_argument("symbol")
    q.add_argument("target", type=int)
    q.add_argument("--amount", type=float, default=1.0)
    q = add("accuracy", cmd_accuracy, "classification accuracy on random strings of --language (nothing moves)")
    q.add_argument("--tests", type=int, default=300)
    q.add_argument("--max-length", type=int, default=6)
    q.add_argument("--temperature-run", type=float, default=0.0)
    add("table", cmd_table, "the greedy transition table (nothing moves)")
    add("stats", cmd_stats, "size, memory, clock and stimulation")
    q = add("tick", cmd_tick, "let time pass")
    q.add_argument("--ticks", type=int, default=1)
    q = add("stimulate", cmd_stimulate, "raise the stimulation by --amount, or set --level")
    q.add_argument("--amount", type=float, default=1.0)
    q.add_argument("--level", type=float, default=None)
    q = add("focus", cmd_focus, "set the focus, or learn it from the input")
    q.add_argument("--level", type=float, default=None, help="0..1")
    q.add_argument("--none", action="store_true", help="no focus: runs start from the start state")
    q.add_argument("--learn", choices=("on", "off"), default=None)
    q.add_argument("--text", default=None, help="show the focus the learner reads off this input")
    q = add("skip", cmd_skip, "let runs skip a node when a two-edge path is more efficient")
    q.add_argument("state", choices=("on", "off"))
    q.add_argument("--margin", type=float, default=None)
    q = add("rearrange", cmd_rearrange, "let the nodes rearrange themselves toward the centre, or swap two")
    q.add_argument("--full", action="store_true", help="passes until no swap is left")
    q.add_argument("--swap", nargs=3, metavar=("states|symbols", "I", "J"), default=None)
    q = add("compress", cmd_compress, "fold the matrix into its central node and save the code")
    q.add_argument("--precision", choices=list(PRECISIONS), default=None, help="default exact: no loss")
    q.add_argument("--budget", type=int, default=None, help="bytes: the least lossy precision that fits")
    q.add_argument("--core", default=None, help="write the code here (.json or .json.gz)")
    q = add("expand", cmd_expand, "rebuild a machine from a code, from the central node outward", machine=False)
    q.add_argument("--core", required=True)
    q.add_argument("--shells", type=int, default=None, help="rebuild only this many shells (0 is the central node)")
    q = add("core-run", cmd_core_run, "walk a string straight from a code", machine=False, load=False)
    q.add_argument("text")
    q.add_argument("--core", required=True)
    q.add_argument("--from-middle", action="store_true")
    q = add("experiment", cmd_experiment, "the learning, stimulation and adaptation experiments", machine=False, load=False)
    q.add_argument("--which", nargs="+", default=["all"], choices=list(experiment.EXPERIMENTS) + ["all"])
    q.add_argument("--out", default=None, help="write <name>_results.json files here")
    q.add_argument("--episodes", type=int, default=4000)
    q.add_argument("--life", type=float, default=LIFE)
    q = add("network", cmd_network, "machines connected into a network: new, fit, evaluate, predict, translate, summary",
            machine=False)
    q.add_argument("action", choices=("new", "fit", "evaluate", "predict", "translate", "encode", "summary",
                                      "make-data"))
    q.add_argument("text", nargs="*", help="predict / translate / encode: the inputs to read (texts or files)")
    q.add_argument("--data", default=None, help="fit / evaluate: a file of input<TAB>verdict lines (1/0, yes/no); "
                   "paths relative to it; every 5th held out")
    q.add_argument("--kind", choices=("text", "image", "audio", "video"), default="image",
                   help="make-data: which synthetic data set")
    q.add_argument("--out", default="data", help="make-data: the directory to write it to")
    q.add_argument("--count", type=int, default=40, help="make-data: how many examples")
    q.add_argument("--arch", default="13>13", help="e.g. 13>(13|13)>13: > in a row, (a|b) side by side, 13f final only; "
                   "enc / dec: English in / out through the phonetic tokenizer (enc>13>dec); text, audio, image, diff, "
                   "video: an encoder of that kind first (image>13)")
    q.add_argument("--task", choices=("plural", "past"), default=None, help="fit / evaluate English to English pairs")
    q.add_argument("--pairs", default=None, help="a file of input<TAB>target English pairs (every 4th held out)")
    q.add_argument("--alphabet", default=DEFAULT_ALPHABET, help="the input's symbols, one character each")
    q.add_argument("--accepting", default=None, help="the output layers' accepting states (default: the language's)")
    q.add_argument("--language", default="even-b", choices=sorted(LANGUAGES))
    q.add_argument("--credit", choices=("shared", "own"), default="shared",
                   help="output layers credited by the network's verdict or their own")
    q.add_argument("--merge", choices=("concat", "interleave"), default="concat")
    q.add_argument("--vote", choices=("majority", "any", "all"), default="majority")
    q.add_argument("--layer-discount", type=float, default=0.8)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--life", type=float, default=LIFE)
    q.add_argument("--temperature", type=float, default=1.0)
    q.add_argument("--discount", type=float, default=DISCOUNT)
    q.add_argument("--skip", action="store_true")
    q.add_argument("--learn-focus", action="store_true")
    q.add_argument("--episodes", type=int, default=4000)
    q.add_argument("--epochs", type=int, default=None, help="default 1 for a language, 20 for pairs")
    q.add_argument("--max-length", type=int, default=6)
    q = add("ngram", cmd_ngram, "predict the next thing: an N-gram model trained on a corpus or with Ollama",
            machine=False, load=False)
    q.add_argument("action", choices=("train", "autotrain", "predict", "generate", "evaluate", "stats", "credit",
                                      "ollama"))
    q.add_argument("text", nargs="*", help="predict / generate / credit: the texts")
    q.add_argument("--data", nargs="+", default=None, help="text files or directories (default data/corpus)")
    q.add_argument("--load", default=None, help="a saved model (train / autotrain continue it when it exists)")
    q.add_argument("--save", default=None, help="write the model here (.json or .json.gz)")
    q.add_argument("--order", type=int, default=3, help="N: the context is the last N-1 tokens (default 3)")
    q.add_argument("--unit", choices=("word", "char", "sound", "base64"), default="word",
                   help="base64: any file (text, image, audio, video) as base64 letters")
    q.add_argument("--keep-case", action="store_true", help="do not lower-case")
    q.add_argument("--stimulation", type=float, default=1.0, help="how much a credited edge's width counts")
    q.add_argument("--held-out", type=int, default=10, help="hold out every N-th line to measure on (0: none)")
    q.add_argument("--no-predict", action="store_true",
                   help="train / autotrain: only count (by default every token is predicted first, then learned)")
    q.add_argument("--self-credit", type=float, default=0.0,
                   help="while predicting: reward a right first guess, punish a wrong one by this much (default 0)")
    q.add_argument("-k", type=int, default=10, help="predict: how many next tokens")
    q.add_argument("--max-tokens", type=int, default=30)
    q.add_argument("--temperature", type=float, default=0.0, help="generate: 0 is the likeliest every time")
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--amount", type=float, default=1.0, help="credit: positive rewards, negative punishes")
    q.add_argument("--ollama", action="store_true", help="autotrain: use Ollama (else the corpus only)")
    q.add_argument("--url", default=None, help="Ollama's address (default $OLLAMA_HOST or http://127.0.0.1:11434)")
    q.add_argument("--model", default=None, help="the Ollama model (default $LATTICEFSM_OLLAMA_MODEL or llama3.2)")
    q.add_argument("--topics", default=None, help="autotrain: comma-separated topics (default: the corpus's own words)")
    q.add_argument("--rounds", type=int, default=3)
    q.add_argument("--lines", type=int, default=20, help="autotrain: sentences the LLM writes per topic")
    q.add_argument("--continuations", type=int, default=20, help="autotrain: contexts the LLM continues per round")
    q.add_argument("--judge", type=int, default=10, help="autotrain: own texts the LLM judges per round")
    q.add_argument("--patience", type=int, default=2, help="autotrain: rounds without improvement before stopping")
    q.add_argument("--keep", default=None, help="autotrain: append what the LLM wrote to this file")
    q.add_argument("--out", default=None, help="generate: write a base64 model's output, decoded, to this file")
    q.add_argument("--prefix-from", default=None, help="generate --out: start from the first bytes of this file")
    q.add_argument("--prefix-bytes", type=int, default=64)
    q = add("chain", cmd_chain, "one input to a node, the output node predicts the next: walks only as deep as "
            "confidence needs", machine=False, load=False)
    q.add_argument("action", choices=("train", "predict", "generate", "evaluate", "sweep", "stats", "credit"))
    q.add_argument("text", nargs="*", help="predict / generate / credit: the texts")
    q.add_argument("--data", nargs="+", default=None, help="files or directories (default data/corpus)")
    q.add_argument("--load", default=None, help="a saved chain (without one, the data is learned first)")
    q.add_argument("--save", default=None, help="write the chain here (.json or .json.gz)")
    q.add_argument("--nodes", type=int, default=8, help="input nodes + the output node (default 8)")
    q.add_argument("--unit", choices=("word", "char", "sound", "base64"), default="char",
                   help="what one node holds (default char: a letter)")
    q.add_argument("--keep-case", action="store_true", help="do not lower-case")
    q.add_argument("--confidence", type=float, default=0.9,
                   help="the confidence a walk gathers before it stops (0: the output node alone, 1: as deep as it "
                        "can; default 0.9)")
    q.add_argument("--set-confidence", type=float, default=None, help="use this threshold with a loaded chain")
    q.add_argument("--held-out", type=int, default=10, help="hold out every N-th line to measure on (0: none)")
    q.add_argument("--no-predict", action="store_true", help="train: only learn (by default every input is "
                   "predicted first)")
    q.add_argument("--self-credit", type=float, default=0.0,
                   help="train: reward the nodes behind a right first guess, punish those behind a wrong one")
    q.add_argument("--thresholds", default="0,0.3,0.5,0.7,0.8,0.9,0.95,1", help="sweep: the confidences")
    q.add_argument("-k", type=int, default=10, help="predict: how many next inputs")
    q.add_argument("--max-tokens", type=int, default=40)
    q.add_argument("--temperature", type=float, default=0.0, help="generate: 0 is the likeliest every time")
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--amount", type=float, default=1.0, help="credit: positive rewards, negative punishes")
    q = add("base64", cmd_base64, "encode anything (a file or a text) as base64, or decode it back", machine=False,
            load=False)
    q.add_argument("action", choices=("encode", "decode"))
    q.add_argument("input", nargs="*", help="files, or texts (- or nothing: standard input)")
    q.add_argument("--out", default=None, help="write the result here instead of printing it")
    add("test", cmd_test, "run the test suite", machine=False, load=False)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
