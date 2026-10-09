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
    experiment  the learning, stimulation, adaptation and compression experiments
    test        run the test suite

The Rust binary (``../rust``) has the same commands and ``serve`` besides; the
two read and write the same machine files.
"""

from __future__ import annotations

import argparse
import random
import sys

from . import experiment
from .compress import PRECISIONS, compress, fidelity, load_core
from .languages import LANGUAGES, examples, language
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
    add("test", cmd_test, "run the test suite", machine=False, load=False)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
