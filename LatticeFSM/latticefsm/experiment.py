"""The three experiments: a language is learned by credit, stimulation prefers the wide channel, an edge adapts.

Each is a function of the machine's settings that returns a JSON-serialisable
record; :func:`run` runs the ones asked for and writes them, and
:func:`tables` renders them as the markdown ``README.md`` quotes.  Everything
is deterministic given the seed.

* :func:`learning` - for each language and each machine size, the machine
  runs random strings, is rewarded when it ends in the right kind of state
  and punished when it does not, and its accuracy on held-out strings is
  measured as it goes.  The control runs the same strings quietly and
  credits them the same way: credit without a traversal lands on nothing,
  so nothing is learned.
* :func:`stimulation` - a fork of three channels of different widths, and
  the probability of each against the stimulation level; then a machine
  stimulated once, and the preference relaxing on the clock as the
  stimulation does.
* :func:`adaptation` - two edges that start from the same prototype and live
  different lives: one is used and rewarded, one is used and punished.  Use
  widens, punishment narrows, the verdict never fades, the width and the
  trace do; and the two weighting functions end up different.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Sequence

from .edge import Weighting
from .languages import ALPHABET, LANGUAGES, examples, language
from .machine import BASELINE, DISCOUNT, LIFE, Machine

__all__ = ["EXPERIMENTS", "adaptation", "learning", "main", "run", "stimulation", "tables", "teach_language"]

EXPERIMENTS = ("learning", "stimulation", "adaptation")


def _settings(**kw) -> dict:
    return {k: v for k, v in kw.items()}


def teach_language(machine: Machine, name: str, episodes: int, rng: random.Random, max_length: int = 6,
                   quiet: bool = False, reward: float = 1.0, test: Sequence[tuple[str, bool]] | None = None,
                   every: int = 0) -> list[tuple[int, float]]:
    """Run ``episodes`` random strings through ``machine`` and credit each by whether it was classified right.

    Returns ``(episode, accuracy on test)`` every ``every`` episodes, when
    ``test`` and ``every`` are given.  Quiet, the runs traverse nothing and
    the credit has nothing to land on.
    """
    lang = language(name)
    curve = []
    for ep in range(1, episodes + 1):
        s = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_length)))
        result = machine.run(s, quiet=quiet)
        if result.accepted == lang.member(s):
            machine.reward(reward)
        else:
            machine.punish(reward)
        if test is not None and every and ep % every == 0:
            curve.append((ep, machine.accuracy(test)))
    return curve


def learning(episodes: int = 4000, sizes: Sequence[int] = (3, 4, 6), seeds: Sequence[int] = (1, 2, 3, 4, 5),
             life: float = LIFE, discount: float = DISCOUNT, max_length: int = 6, tests: int = 300) -> dict:
    rows = []
    for name, lang in LANGUAGES.items():
        for states in sizes:
            arms = {"credited": [], "quiet": []}
            curves = []
            for seed in seeds:
                for arm in arms:
                    rng = random.Random(seed)
                    test = examples(lang, tests, rng, max_length)
                    m = Machine(states, ALPHABET, accepting=lang.accepting, life=life, discount=discount, seed=seed)
                    before = m.accuracy(test)
                    curve = teach_language(m, name, episodes, rng, max_length, quiet=(arm == "quiet"), test=test,
                                           every=max(1, episodes // 8))
                    after = m.accuracy(test)
                    arms[arm].append({"seed": seed, "before": before, "after": after, "curve": curve,
                                      "touched": m.stats()["touched"], "clock": m.clock})
                    if arm == "credited":
                        curves.append(curve)
            credited = [a["after"] for a in arms["credited"]]
            rows.append({
                "language": name, "description": lang.description, "min_states": lang.min_states, "states": states,
                "before": sum(a["before"] for a in arms["credited"]) / len(seeds),
                "after_mean": sum(credited) / len(seeds),
                "after_min": min(credited), "after_max": max(credited),
                "solved": sum(a >= 0.999 for a in credited),
                "quiet_after": sum(a["after"] for a in arms["quiet"]) / len(seeds),
                "quiet_touched": max(a["touched"] for a in arms["quiet"]),
                "arms": arms,
            })
    return {
        "experiment": "learning",
        "settings": _settings(episodes=episodes, sizes=list(sizes), seeds=list(seeds), life=life, discount=discount,
                              max_length=max_length, tests=tests),
        "rows": rows,
    }


def stimulation(widths: Sequence[float] = (4.0, 1.0, 0.25), levels: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
                life: float = LIFE, baseline: float = BASELINE, surge: float = 3.0,
                waits: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0)) -> dict:
    n = len(widths)
    m = Machine(n, "a", life=life, baseline=baseline, seed=1)
    for t, w in enumerate(widths):
        m.edge(0, "a", t).set_width(w, 0)
    curve = [{"stimulation": level, "probabilities": m.probabilities(0, "a", stimulation=level)} for level in levels]
    # a surge, and its relaxation on the clock
    m.stimulate(surge)
    relax = []
    for lives in waits:
        m2 = Machine(n, "a", life=life, baseline=baseline, seed=1)
        for t, w in enumerate(widths):
            m2.edge(0, "a", t).set_width(w, 0)
        m2.stimulate(surge)
        m2.tick(int(lives * life))
        relax.append({"lives": lives, "stimulation": m2.stimulation,
                      "probabilities": m2.probabilities(0, "a"), "widths": [e.width_at(m2.clock, m2.life)
                                                                             for e in m2.lattice.row(0, "a")]})
    return {
        "experiment": "stimulation",
        "settings": _settings(widths=list(widths), levels=list(levels), life=life, baseline=baseline, surge=surge,
                              waits=list(waits)),
        "curve": curve,
        "relaxation": relax,
    }


def adaptation(uses: int = 20, life: float = LIFE, waits: Sequence[float] = (0.0, 1.0, 4.0)) -> dict:
    """Two edges, one life each: rewarded on every use, and punished on every use."""
    m = Machine(3, "a", life=life, seed=1)
    good, bad = m.edge(0, "a", 1), m.edge(0, "a", 2)
    stages = []

    def snapshot(label: str) -> None:
        stages.append({
            "stage": label, "clock": m.clock,
            "probabilities": m.probabilities(0, "a"),
            "edges": {
                "rewarded": _edge_view(good, m), "punished": _edge_view(bad, m),
                "untouched": _edge_view(m.edge(0, "a", 0), m),
            },
        })

    snapshot("fresh")
    for i in range(uses):
        m.teach(0, "a", 1, +1.0)
        m.teach(0, "a", 2, -1.0)
        if i + 1 in (1, 5, uses):
            snapshot(f"after {i + 1} uses each")
    end = m.clock
    for lives in waits[1:]:
        m.tick(end + int(lives * life) - m.clock)
        snapshot(f"{lives:g} {'life' if lives == 1 else 'lives'} later")
    return {
        "experiment": "adaptation",
        "settings": _settings(uses=uses, life=life, waits=list(waits), prototype=Weighting().to_list()),
        "stages": stages,
    }


def _edge_view(e, m: Machine) -> dict:
    """Every field of an edge by name, the fading ones read at the clock - the Rust crate's ``Edge::describe``."""
    return {
        "source": e.source, "symbol": e.symbol, "target": e.target,
        "seen": e.seen, "first_seen": e.first_seen, "last_seen": e.last_seen,
        "recent": e.recent_at(m.clock, m.life), "age": e.age_at(m.clock, m.life),
        "rewarded": e.rewarded, "punished": e.punished, "net": e.net,
        "last_rewarded": e.last_rewarded, "last_punished": e.last_punished,
        "width": e.width_at(m.clock, m.life),
        "weighting": e.weighting.to_list(), "features": list(e.features),
        "log_weight": e.log_weight(m.clock, m.life, m.stimulation),
    }


# ---- rendering ---------------------------------------------------------------------------------------------------

def tables(record: dict) -> str:
    kind = record["experiment"]
    if kind == "learning":
        return _learning_table(record)
    if kind == "stimulation":
        return _stimulation_table(record)
    return _adaptation_table(record)


def _learning_table(r: dict) -> str:
    s = r["settings"]
    lines = [f"learning: {s['episodes']} episodes, strings up to {s['max_length']} symbols, {len(s['seeds'])} seeds, "
             f"{s['tests']} held-out strings; the quiet arm runs the same strings without traversing", "",
             "| language | states (min) | before | after: mean | min | max | solved | quiet after |",
             "|---|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        lines.append(f"| {row['language']} ({row['description']}) | {row['states']} ({row['min_states']}) | "
                     f"{row['before']:.2f} | **{row['after_mean']:.2f}** | {row['after_min']:.2f} | "
                     f"{row['after_max']:.2f} | {row['solved']}/{len(s['seeds'])} | {row['quiet_after']:.2f} |")
    return "\n".join(lines)


def _stimulation_table(r: dict) -> str:
    s = r["settings"]
    widths = s["widths"]
    head = " | ".join(f"width {w:g}" for w in widths)
    lines = [f"stimulation: a fork of {len(widths)} channels; probability of each by stimulation level", "",
             f"| stimulation | {head} |", "|---|" + "---|" * len(widths)]
    for c in r["curve"]:
        lines.append(f"| {c['stimulation']:g} | " + " | ".join(f"{p:.3f}" for p in c["probabilities"]) + " |")
    lines += ["", f"a surge of +{s['surge']:g} over the baseline {s['baseline']:g}, relaxing on the clock", "",
              f"| lives since | stimulation | {head} |", "|---|---|" + "---|" * len(widths)]
    for c in r["relaxation"]:
        lines.append(f"| {c['lives']:g} | {c['stimulation']:.2f} | "
                     + " | ".join(f"{p:.3f}" for p in c["probabilities"]) + " |")
    return "\n".join(lines)


def _adaptation_table(r: dict) -> str:
    lines = ["adaptation: two edges from one prototype, one rewarded on every use and one punished", "",
             "| stage | clock | P(rewarded) | P(punished) | P(untouched) | width + | width − | net + | net − | "
             "recent + | bias + | bias − |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in r["stages"]:
        g, b = st["edges"]["rewarded"], st["edges"]["punished"]
        p = st["probabilities"]
        lines.append(f"| {st['stage']} | {st['clock']} | {p[1]:.3f} | {p[2]:.3f} | {p[0]:.3f} | {g['width']:.2f} | "
                     f"{b['width']:.2f} | {g['net']:+.2f} | {b['net']:+.2f} | {g['recent']:.2f} | "
                     f"{g['weighting'][0]:+.2f} | {b['weighting'][0]:+.2f} |")
    last = r["stages"][-1]["edges"]
    lines += ["", "the two weighting functions at the end (bias, seen, recent, net, age, width, rate):",
              f"  rewarded: {[round(v, 3) for v in last['rewarded']['weighting']]}",
              f"  punished: {[round(v, 3) for v in last['punished']['weighting']]}"]
    return "\n".join(lines)


# ---- running -----------------------------------------------------------------------------------------------------

def run(which: Sequence[str], out: str | None = None, **kw) -> list[dict]:
    """Run the experiments named (or ``all``) and write each as ``<out>/<name>_results.json``."""
    names = EXPERIMENTS if "all" in which else tuple(which)
    records = []
    for name in names:
        if name == "learning":
            rec = learning(episodes=kw.get("episodes", 4000), sizes=kw.get("sizes", (3, 4, 6)),
                           seeds=kw.get("seeds", (1, 2, 3, 4, 5)), life=kw.get("life", LIFE),
                           discount=kw.get("discount", DISCOUNT))
        elif name == "stimulation":
            rec = stimulation(life=kw.get("life", LIFE), baseline=kw.get("baseline", BASELINE))
        elif name == "adaptation":
            rec = adaptation(life=kw.get("life", LIFE))
        else:
            raise ValueError(f"no experiment named {name!r}; choose from {EXPERIMENTS} or all")
        records.append(rec)
        if out:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"{name}_results.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="the LatticeFSM experiments")
    p.add_argument("--which", nargs="+", default=["all"], choices=list(EXPERIMENTS) + ["all"])
    p.add_argument("--out", default=None)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--life", type=float, default=LIFE)
    args = p.parse_args(argv)
    for rec in run(args.which, args.out, episodes=args.episodes, life=args.life):
        print(tables(rec))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
