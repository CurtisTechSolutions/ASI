"""``python3 -m fbradix <command>`` - every command this architecture has.

    check        the finite-difference gradient checks, both learning rules
    corpus       build the four-source corpus and write the snapshot
    demo         fit a small bank and show what each address ended up holding
    train        fit a bank and print the round-by-round table
    route        which expert a piece of text is filtered into, and why
    generate     sample characters, re-routing after every one
    experiment   the arms and the sweeps (see fbradix/experiment.py)
    test         run the test module
"""

from __future__ import annotations

import argparse
import json
import sys

from . import corpus
from .bank import FilteredRadixBank, FilterRouter
from .check import check_all
from .experiment import ARMS, contingency, make_router, nmi, purity, run_arm


def _corpus(args):
    """Load or build the corpus, then split it."""
    data = corpus.build(args.per_source) if args.per_source else corpus.load()
    train, test = corpus.split(data, seed=args.seed)
    return data, train, test, corpus.alphabet(data)


def cmd_check(args) -> int:
    """Both gradient rules against central differences."""
    worst = max(check_all().values())
    print(f"  worst error overall {worst:.2e}")
    return 0 if worst < 1e-5 else 1


def cmd_corpus(args) -> int:
    """Build the corpus from the pinned files and write the snapshot."""
    data = corpus.build(args.per_source or 200)
    path = corpus.save(data)
    print(json.dumps(corpus.summarise(data), indent=1))
    print(f"wrote {path}")
    return 0


def cmd_demo(args) -> int:
    """Fit a small bank and show what each address ended up holding."""
    data, train, test, alpha = _corpus(args)
    print(f"{len(train)} train / {len(test)} test segments, alphabet {alpha}")
    check_all()
    print()
    router = FilterRouter(bits=args.bits, seed=args.seed)
    bank = FilteredRadixBank(router, depth=args.depth, alphabet=alpha, seed=args.seed)
    for rec in bank.fit(train, rounds=args.rounds, epochs=args.epochs, test=test):
        print(f"  round {rec['round']}  train={rec['train_bits']:.3f}  test={rec['test_bits']:.3f} "
              f"bits/char  load={rec['load']}  nodes={rec['nodes']}")
    routes = bank.assign(test)
    labels = [s["source"] for s in test]
    print(f"\n  purity {purity(routes, labels):.3f}   nmi {nmi(routes, labels):.3f}")
    print("  address -> what it holds (held-out segments)")
    for r, c in sorted(contingency(routes, labels).items()):
        print(f"    {r:>3b}  {dict(c)}")
    print("\n  filter:", bank.router.stats())
    return 0


def cmd_train(args) -> int:
    """Fit one arm and print its table."""
    data, train, test, alpha = _corpus(args)
    rec = run_arm(args.arm, train, test, alpha, bits=args.bits, depth=args.depth,
                  rounds=args.rounds, epochs=args.epochs, seed=args.seed)
    for h in rec["history"]:
        print(f"  round {h['round']}  train={h['train_bits']:.4f}  "
              f"test={h.get('test_bits', float('nan')):.4f}  load={h['load']}")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("history", "contingency")}, indent=1))
    return 0


def cmd_route(args) -> int:
    """Show the filter's responses and the address they spell for one text."""
    data, train, test, alpha = _corpus(args)
    router = FilterRouter(bits=args.bits, seed=args.seed)
    bank = FilteredRadixBank(router, depth=args.depth, alphabet=alpha, seed=args.seed)
    bank.fit(train, rounds=args.rounds, epochs=args.epochs)
    from .filter import features
    text = args.text or "for i in range(10):\n    total += i * i"
    x = features(text)
    route, resp = router.filter.address(x)
    print(f"  text      {text!r}")
    print(f"  responses {[round(v, 4) for v in resp]}")
    print(f"  address   {route} ({route:0{args.bits}b})  -> expert {route}: "
          f"{bank.experts[route].num_nodes} nodes, {bank.experts[route].chars} chars")
    print(f"  bits/char under every expert: "
          f"{[round(e.bits_per_char([text])[0], 2) for e in bank.experts]}")
    return 0


def cmd_generate(args) -> int:
    """Sample characters from the bank, re-routing after each one."""
    data, train, test, alpha = _corpus(args)
    router = FilterRouter(bits=args.bits, seed=args.seed)
    bank = FilteredRadixBank(router, depth=args.depth, alphabet=alpha, seed=args.seed)
    bank.fit(train, rounds=args.rounds, epochs=args.epochs)
    prefix = args.text or "def "
    print(repr(bank.generate(prefix, length=args.length, seed=args.seed)))
    return 0


def cmd_experiment(argv: list[str]) -> int:
    """Hand over to the experiment driver, flags and all."""
    from .experiment import main as run
    return run(argv)


def cmd_test(args) -> int:
    """Run the test module in-process."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests import test_fbradix
    return test_fbradix.main()


def main(argv=None) -> int:
    """Parse and dispatch."""
    ap = argparse.ArgumentParser(prog="fbradix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, default=2, help="filter units; 2**bits experts")
    ap.add_argument("--depth", type=int, default=5, help="context depth of every expert")
    ap.add_argument("--rounds", type=int, default=3, help="route/build/train/refilter rounds")
    ap.add_argument("--epochs", type=int, default=3, help="one-hop epochs per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-source", type=int, default=0,
                    help="segments per source; 0 uses the committed snapshot")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("check", help=cmd_check.__doc__).set_defaults(fn=cmd_check)
    sub.add_parser("corpus", help=cmd_corpus.__doc__).set_defaults(fn=cmd_corpus)
    sub.add_parser("demo", help=cmd_demo.__doc__).set_defaults(fn=cmd_demo)
    p = sub.add_parser("train", help=cmd_train.__doc__)
    p.add_argument("--arm", default="learned", choices=ARMS)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("route", help=cmd_route.__doc__)
    p.add_argument("text", nargs="?")
    p.set_defaults(fn=cmd_route)
    p = sub.add_parser("generate", help=cmd_generate.__doc__)
    p.add_argument("text", nargs="?")
    p.add_argument("--length", type=int, default=80)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("experiment", help=cmd_experiment.__doc__)
    p.add_argument("--sweep", default="all", help="all | main | depth | scale")
    p.add_argument("--quick", action="store_true")
    sub.add_parser("test", help=cmd_test.__doc__).set_defaults(fn=cmd_test)

    argv = list(sys.argv[1:] if argv is None else argv)
    if "experiment" in argv:
        # The experiment driver owns its own flags; handing it the tail
        # verbatim beats teaching this parser every sweep option twice.
        return cmd_experiment(argv[argv.index("experiment") + 1:])
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
