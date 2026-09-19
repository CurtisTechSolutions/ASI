"""Build the tables in README.md straight from the committed result JSON.

    python3 summarize.py            # every table it can find in results/
    python3 summarize.py main       # one of: main, depth, scale

Standard library only, like everything else here.  If a table in README.md and
this script's output ever disagree, the script is right - it reads the numbers,
the README repeats them.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ARM_LABEL = {
    "single": "`single` - one radix tree, no filter",
    "roundrobin": "`roundrobin` - split at random",
    "oracle": "`oracle` - routed by the true source",
    "frozen": "`frozen` - calibrated random filter",
    "learned": "**`learned`** - the architecture",
    "monotone": "`monotone` - tanh units, same knobs",
    "unbalanced": "`unbalanced` - no load pressure",
    "fixedwave": "`fixedwave` - only the projection learns",
    "deep": "`deep` - learned, deep shared prior",
    "deep-oracle": "`deep-oracle` - oracle, deep shared prior",
}


def load(name: str) -> dict | None:
    """Read ``results/<name>_results.json`` if it is there."""
    path = os.path.join(RESULTS, f"{name}_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def table_main(doc: dict) -> str:
    """Every arm at one depth and one corpus size, averaged over the seeds."""
    res = doc.get("main") or doc
    by_arm = defaultdict(list)
    for r in res["runs"]:
        by_arm[r["arm"]].append(r)
    rows = [
        "| arm | bits/char (held out) | nodes | live | purity | nmi |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(by_arm, key=lambda a: mean(r["test_bits_per_char"] for r in by_arm[a]))
    for arm in order:
        rs = by_arm[arm]
        b = [r["test_bits_per_char"] for r in rs]
        rows.append(
            f"| {ARM_LABEL.get(arm, arm)} | **{mean(b):.3f}** ± {sd(b):.3f} | "
            f"{mean(r['nodes'] for r in rs):,.0f} | "
            f"{mean(r['live_experts'] for r in rs):.1f} | "
            f"{mean(r['purity'] for r in rs):.3f} | "
            f"{mean(r['nmi'] for r in rs):.3f} |"
        )
    c = res.get("corpus", {})
    rows.append("")
    rows.append(
        f"{c.get('segments', '?')} segments / {c.get('chars', '?')} characters, "
        f"alphabet {c.get('alphabet', '?')}, depth {res.get('depth')}, "
        f"{1 << res.get('bits', 0)} addresses, "
        f"{len({r['seed'] for r in res['runs']})} seeds."
    )
    return "\n".join(rows)


def table_by(doc: dict, key: str, label: str, fmt="{}") -> str:
    """One row per value of ``key`` (depth or corpus size), one column per arm."""
    res = doc.get(key if key != "per_source" else "scale") or doc
    runs = res["runs"]
    arms, values = [], []
    for r in runs:
        if r["arm"] not in arms:
            arms.append(r["arm"])
        v = r[key]
        if v not in values:
            values.append(v)
    cell = {(r[key], r["arm"]): r for r in runs}
    rows = [
        f"| {label} | " + " | ".join(f"`{a}`" for a in arms) + " | best |",
        "|---|" + "---|" * (len(arms) + 1),
    ]
    for v in values:
        got = [(a, cell[(v, a)]["test_bits_per_char"]) for a in arms if (v, a) in cell]
        best = min(got, key=lambda t: t[1])[0] if got else "-"
        cells = []
        for a, b in got:
            cells.append(f"**{b:.3f}**" if a == best else f"{b:.3f}")
        extra = cell.get((v, arms[0]))
        note = ""
        if key == "per_source" and extra:
            note = f" ({extra['corpus_chars']:,} chars)"
        rows.append(f"| {fmt.format(v)}{note} | " + " | ".join(cells) + f" | `{best}` |")
    return "\n".join(rows)


def main(argv=None) -> int:
    which = (argv or sys.argv[1:]) or ["main", "depth", "scale"]
    printed = 0
    for name in which:
        doc = load(name)
        if not doc:
            print(f"(no results/{name}_results.json - run `make {name}`)")
            continue
        printed += 1
        if name == "main":
            print("### The headline table\n")
            print(table_main(doc))
        elif name == "depth":
            print("\n### Context depth\n")
            print(table_by(doc, "depth", "depth", "{}"))
        elif name == "scale":
            print("\n### Corpus size\n")
            print(table_by(doc, "per_source", "segments per source", "{}"))
        print()
    return 0 if printed else 1


if __name__ == "__main__":
    sys.exit(main())
