"""The four experiments: a habit forms, a memory fades, use keeps it, reading brings it back.

Each is a function of a corpus and the model's settings that returns a
JSON-serialisable record; :func:`run` runs the ones asked for and writes
them, and :func:`tables` renders them as the markdown ``README.md`` quotes.
Everything is deterministic given the seed: reading is deterministic, and
the one sampled arm draws from a seeded generator.

* :func:`habit` - the same prefix said over and over.  Every saying is a
  traversal of what it says, so the probability of the text it says climbs
  and the branch it decides at sharpens; a quiet control asks the same
  question the same number of times and moves nothing; a sampled arm shows
  a habit forming by chance and then feeding itself.
* :func:`forgetting` - read once, then time passes.  The curve of what the
  model still holds, still recites, and how many bits a text costs, against
  elapsed lives, under each decay.
* :func:`use` - after reading, a long silence broken only by saying one text.
  The said text is kept while the rest fade; the same schedule asked quietly
  keeps nothing.
* :func:`recovery` - fade, then re-read.  Every traversal adds, so one
  reading of a text restores it, and only the texts re-read come back.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence

from .encoding import Encoding
from .model import DecayNet
from .tree import DECAYS, LIFE, MIN_SEEN, START

__all__ = ["EXPERIMENTS", "forgetting", "habit", "main", "new_model", "recovery", "run", "tables", "use"]

EXPERIMENTS = ("habit", "forgetting", "use", "recovery")


def new_model(
    texts: Sequence[str],
    life: float = LIFE,
    decay: str = "half-life",
    min_seen: float = MIN_SEEN,
    depth: int | None = None,
    encoding: Encoding | None = None,
    seed: int = 1,
    readings: int = 1,
) -> DecayNet:
    """A fresh model that has read ``texts`` ``readings`` times."""
    model = DecayNet(encoding=encoding, depth=depth, life=life, decay=decay, min_seen=min_seen, seed=seed)
    model.read(texts, times=readings)
    return model


def _settings(**kw) -> dict:
    enc = kw.get("encoding")
    out = {k: v for k, v in kw.items() if k != "encoding"}
    out["encoding"] = str(enc) if enc is not None else str(Encoding())
    return out


def _fraction(a: int, b: int) -> float:
    return a / b if b else 0.0


def entropy(shares: Sequence[float]) -> float:
    """Bits of uncertainty in a distribution."""
    return -sum(s * math.log2(s) for s in shares if s > 0.0)


# -- habit -------------------------------------------------------------------

def first_branch(model: DecayNet, prefix: str) -> int | None:
    """The first node with a choice on the walk from ``prefix``: where a saying decides."""
    tree = model.tree
    grams = model.encoding.encode(prefix)
    if grams:
        loc = tree.locate(grams)
        if loc is None:
            return None
        node = loc[0]
    else:
        node = START
    while True:
        options = tree.shares(node)
        if len(options) != 1:
            return node
        node = options[0][0]
        if tree.is_end(node):
            return None


def branch_shares(model: DecayNet, node: int | None, units: int = 12) -> list[dict]:
    """The options at ``node`` as ``[{option, share, seen}]``, the option named by its first ``units``."""
    if node is None:
        return []
    tree = model.tree
    enc = model.encoding
    rows = []
    for c, share in sorted(tree.shares(node), key=lambda item: -item[1]):
        name = "</s>" if tree.is_end(c) else enc.spell(enc.piece_of(tree.labels[c], 0, units))
        rows.append({"option": name, "share": share, "seen": tree.seen(c)})
    return rows


def habit(
    texts: Sequence[str], prefix: str = "the", times: int = 10, sampled: int = 30, temperature: float = 1.0,
    seed: int = 1, **settings,
) -> dict:
    """Say ``prefix`` ``times`` over and watch the said text's probability climb; quietly, and it does not;
    sampled at ``temperature``, and a habit forms by chance."""

    def arm(quiet: bool, temp: float, n: int) -> dict:
        model = new_model(texts, seed=seed, **settings)
        branch = first_branch(model, prefix)
        before = branch_shares(model, branch)
        rows = []
        for i in range(n):
            r = model.say(prefix, to_end=True, temperature=temp, seed=seed + i if temp > 0 else None, quiet=quiet)
            rows.append({
                "repeat": i + 1, "says": r.full_spelled, "probability": math.exp(-r.cost), "cost": r.cost,
                "traversals": r.traversals, "clock": model.clock,
            })
        after = branch_shares(model, branch)
        return {
            "quiet": quiet, "temperature": temp, "times": n, "rows": rows,
            "distinct": len({r["says"] for r in rows}),
            "before": before, "after": after,
            "entropy_before": entropy([b["share"] for b in before]), "entropy_after": entropy([a["share"] for a in after]),
            "top_share_before": before[0]["share"] if before else 0.0, "top_share_after": after[0]["share"] if after else 0.0,
        }

    return {
        "experiment": "habit", "prefix": prefix, "settings": _settings(seed=seed, **settings),
        "arms": {"said": arm(False, 0.0, times), "quiet": arm(True, 0.0, times), "sampled": arm(False, temperature, sampled)},
    }


# -- forgetting ---------------------------------------------------------------

def _measure(model: DecayNet, texts: Sequence[str], nodes: int) -> dict:
    rec = model.recites(texts)
    bits = model.bits_per_unit(texts)
    remembered = model.tree.remembered()
    return {
        "clock": model.clock, "remembered": remembered, "remembered_fraction": _fraction(remembered, nodes),
        "recited": rec["recited"], "texts": rec["texts"], "recited_fraction": rec["fraction"],
        "bits_per_unit": bits["bits_per_unit"], "miss_rate": bits["miss_rate"], "total_seen": model.tree.total_seen(),
    }


def forgetting(
    texts: Sequence[str], lives: Sequence[float] = (0, 0.25, 0.5, 1, 2, 4, 8), decays: Sequence[str] = DECAYS,
    seed: int = 1, **settings,
) -> dict:
    """Read once, then let ``lives`` of time pass: what is remembered, recited and how much a text costs, per decay."""
    settings = {k: v for k, v in settings.items() if k != "decay"}
    curves = {}
    for decay in decays:
        model = new_model(texts, decay=decay, seed=seed, **settings)
        read_clock = model.clock
        life = model.tree.life
        nodes = model.tree.num_nodes()
        rows = []
        for x in lives:
            target = read_clock + int(round(x * life))
            if target > model.clock:
                model.tick(target - model.clock)
            rows.append({"lives": x, **_measure(model, texts, nodes)})
        curves[decay] = {"reading_traversals": read_clock, "reading_lives": read_clock / life, "nodes": nodes, "rows": rows}
    return {"experiment": "forgetting", "settings": _settings(seed=seed, **settings), "lives": list(lives), "curves": curves}


# -- use ----------------------------------------------------------------------

def use(
    texts: Sequence[str], used: str | None = None, rounds: int = 20, gap: int | None = None, prefix_units: int = 8,
    seed: int = 1, **settings,
) -> dict:
    """After reading, ``rounds`` silences of ``gap`` traversals (a tenth of a life by default), each broken by
    saying one text from its first ``prefix_units`` units: said, said quietly, or not at all.  The text is the
    first one the model recites right after reading (use keeps alive what is alive), unless ``used`` is given."""
    if used is None:
        probe = new_model(texts, seed=seed, **settings)
        used = next((t for t in texts if probe.recites([t], prefix_units=prefix_units)["recited"]), texts[0])
    used_text = used
    arms = {}
    for name in ("said", "quiet", "silent"):
        model = new_model(texts, seed=seed, **settings)
        enc = model.encoding
        prefix = enc.piece(used_text, 0, prefix_units)
        life = model.tree.life
        nodes = model.tree.num_nodes()
        gap_ = gap if gap is not None else max(1, int(life // 10))
        said = 0
        for _ in range(rounds):
            model.tick(gap_)
            if name == "said":
                said += model.say(prefix, to_end=True).traversals
            elif name == "quiet":
                model.say(prefix, to_end=True, quiet=True)
        rec_used = model.recites([used_text], prefix_units=prefix_units)
        arms[name] = {
            "rounds": rounds, "gap": gap_, "elapsed_lives": rounds * gap_ / life, "traversals_said": said,
            "used_recited": bool(rec_used["recited"]), "says": model.say(prefix, to_end=True, quiet=True).full_spelled,
            **_measure(model, texts, nodes),
        }
    return {
        "experiment": "use", "used": used_text, "prefix_units": prefix_units, "settings": _settings(seed=seed, **settings),
        "arms": arms,
    }


# -- recovery -----------------------------------------------------------------

def recovery(texts: Sequence[str], fade_lives: float = 4.0, every: int = 3, seed: int = 1, **settings) -> dict:
    """Read, fade ``fade_lives`` lives, re-read every ``every``-th text once, then all of them once."""
    model = new_model(texts, seed=seed, **settings)
    life = model.tree.life
    nodes = model.tree.num_nodes()
    subset = [t for i, t in enumerate(texts) if i % every == every - 1]
    others = [t for i, t in enumerate(texts) if i % every != every - 1]

    def stage(name: str) -> dict:
        row = {"stage": name, **_measure(model, texts, nodes)}
        row["re_read_recited"] = model.recites(subset)["recited"]
        row["re_read_texts"] = model.recites(subset)["texts"]
        row["others_recited"] = model.recites(others)["recited"]
        row["others_texts"] = model.recites(others)["texts"]
        return row

    rows = [stage("read once")]
    model.tick(int(fade_lives * life))
    rows.append(stage(f"{fade_lives:g} lives later"))
    model.read(subset)
    rows.append(stage(f"every {every}th text re-read"))
    model.read(texts)
    rows.append(stage("all re-read"))
    return {
        "experiment": "recovery", "fade_lives": fade_lives, "every": every, "settings": _settings(seed=seed, **settings),
        "rows": rows,
    }


# -- running and reporting ----------------------------------------------------

def run(which: str | Sequence[str], texts: Sequence[str], out_dir: str | None = None, **settings) -> dict:
    """Run the experiments named (``"all"`` for every one); write ``<name>_results.json`` under ``out_dir``."""
    names = list(EXPERIMENTS) if which == "all" else ([which] if isinstance(which, str) else list(which))
    for name in names:
        if name not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {name!r}; choose from {EXPERIMENTS} or 'all'")
    fns = {"habit": habit, "forgetting": forgetting, "use": use, "recovery": recovery}
    results = {}
    for name in names:
        results[name] = fns[name](texts, **settings)
        results[name]["corpus"] = {"texts": len(texts), "units": sum(len(t) for t in texts)}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{name}_results.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(results[name], fh, indent=1)
            results[name]["written"] = path
    return results


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _shares_line(rows: Sequence[dict]) -> str:
    return ", ".join(f"{r['option']!r} {r['share']:.2f}" for r in rows[:4]) + (" ..." if len(rows) > 4 else "")


def tables(results: dict) -> str:
    """The markdown tables for a :func:`run` record."""
    out: list[str] = []
    if "habit" in results:
        h = results["habit"]
        out.append(f"### habit - saying {h['prefix']!r} over and over\n")
        out.append("| arm | says | probability at first | at last | distinct texts | top share at the branch, before → after | entropy before → after |")
        out.append("|---|---|---|---|---|---|---|")
        for name, arm in h["arms"].items():
            rows = arm["rows"]
            label = {"said": "said", "quiet": "asked quietly", "sampled": f"sampled at T={arm['temperature']:g}"}[name]
            out.append(f"| {label} ({arm['times']}×) | {rows[0]['says']!r} | {rows[0]['probability']:.3f} | {rows[-1]['probability']:.3f} "
                       f"| {arm['distinct']} | {arm['top_share_before']:.2f} → {arm['top_share_after']:.2f} "
                       f"| {arm['entropy_before']:.2f} → {arm['entropy_after']:.2f} bits |")
        said = h["arms"]["said"]
        out.append(f"\nthe branch before: {_shares_line(said['before'])}")
        out.append(f"after saying: {_shares_line(said['after'])}")
        out.append("")
    if "forgetting" in results:
        f = results["forgetting"]
        out.append("### forgetting - read once, then time passes\n")
        for decay, curve in f["curves"].items():
            out.append(f"`{decay}` (the reading itself took {curve['reading_traversals']} traversals, {curve['reading_lives']:.2f} lives)\n")
            out.append("| lives since | remembered | recited | bits/unit | misses |")
            out.append("|---|---|---|---|---|")
            for r in curve["rows"]:
                out.append(f"| {r['lives']:g} | {r['remembered']}/{curve['nodes']} ({_pct(r['remembered_fraction'])}) "
                           f"| {r['recited']}/{r['texts']} | {r['bits_per_unit']:.2f} | {_pct(r['miss_rate'])} |")
            out.append("")
    if "use" in results:
        u = results["use"]
        out.append(f"### use - a silence broken only by saying {u['used']!r}\n")
        out.append("| arm | lives elapsed | traversals said | the used text recited | recited | remembered | bits/unit |")
        out.append("|---|---|---|---|---|---|---|")
        for name, arm in u["arms"].items():
            out.append(f"| {name} | {arm['elapsed_lives']:.1f} | {arm['traversals_said']} | {'yes' if arm['used_recited'] else 'no'} "
                       f"| {arm['recited']}/{arm['texts']} | {_pct(arm['remembered_fraction'])} | {arm['bits_per_unit']:.2f} |")
        out.append("")
    if "recovery" in results:
        r = results["recovery"]
        out.append(f"### recovery - fade {r['fade_lives']:g} lives, re-read every {r['every']}th text, then all\n")
        out.append("| stage | remembered | recited | of the re-read | of the others | bits/unit | misses |")
        out.append("|---|---|---|---|---|---|---|")
        for row in r["rows"]:
            out.append(f"| {row['stage']} | {_pct(row['remembered_fraction'])} | {row['recited']}/{row['texts']} "
                       f"| {row['re_read_recited']}/{row['re_read_texts']} | {row['others_recited']}/{row['others_texts']} "
                       f"| {row['bits_per_unit']:.2f} | {_pct(row['miss_rate'])} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    from . import corpus

    p = argparse.ArgumentParser(prog="radixdecay.experiment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", default="all", choices=EXPERIMENTS + ("all",))
    p.add_argument("--data", default=corpus.SAMPLE)
    p.add_argument("--life", type=float, default=LIFE)
    p.add_argument("--decay", choices=DECAYS, default="half-life")
    p.add_argument("--min-seen", type=float, default=MIN_SEEN)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None, help="directory for <name>_results.json")
    args = p.parse_args(argv)
    results = run(args.which, corpus.read_texts(args.data), out_dir=args.out, life=args.life, decay=args.decay,
                  min_seen=args.min_seen, depth=args.depth, seed=args.seed)
    print(tables(results))
    for name in results:
        if "written" in results[name]:
            print(f"wrote {results[name]['written']}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
