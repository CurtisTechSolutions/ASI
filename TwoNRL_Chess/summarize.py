"""Rebuild the README's tables from whatever is in ``results/``.

Reads the JSON written by ``experiment.py --out`` and ``benchmark.py --out``
and prints the tables in markdown, so the README and the runs cannot drift
apart.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["2nrl", "2nrl-worst", "2nrl-random", "positive", "repulsion"]


DEFAULT_SCHEDULE = "phased"
DEFAULT_NEG_FRACTION = 0.3


def label_of(run: dict) -> str:
    """The arm, plus whatever it changed - the ablations reuse the arm names.

    ``per_round.json`` and the phase-depth files all run the arm called
    ``2nrl``, so pooling on ``run["arm"]`` alone would silently average four
    different configurations into one row.
    """
    cfg = run.get("config", {})
    label = run["arm"]
    # The rule curriculum is on by default, so the ablation is what gets named:
    # pooling it with the arm it ablates would average the two things the whole
    # comparison is about.
    if not cfg.get("rule_updates", 0):
        return f"{label} (no rule heads)"
    if cfg.get("schedule", DEFAULT_SCHEDULE) != DEFAULT_SCHEDULE:
        return f"{label} (per-round)"
    if cfg.get("invert_mode", "unit") != "unit":
        return f"{label} (read-out flip)"
    # neg_fraction only means anything under the `schedule` trigger; under the
    # default it is inert, and labelling by it alone would name two identical runs
    # as if they were different arms.
    if cfg.get("invert_trigger", "plateau") == "schedule":
        fraction = cfg.get("neg_fraction", DEFAULT_NEG_FRACTION)
        return f"{label} (fixed date, phase 1 = {fraction:.0%})"
    return label


def load_runs(paths: list[str]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for path in paths:
        with open(path) as fh:
            blob = json.load(fh)
        for run in blob.get("runs", []):
            arms.setdefault(label_of(run), []).append(run)
    ordered = {a: arms[a] for a in ORDER if a in arms}
    return ordered | {a: rs for a, rs in sorted(arms.items()) if a not in ordered}


def negative_entropy(runs: list[dict]) -> list[float]:
    """H(q) over the rounds that actually trained on a negative set.

    The positive-only control has no negative set, so it gets no H(q) rather
    than a misleading zero.
    """
    if runs and runs[0]["arm"] == "positive":
        return []
    return [h["train"]["negative_entropy"] for r in runs for h in r["history"][1:]
            if h["train"].get("phase") in ("negative", "both") and h["train"].get("n")]


def inversions(runs: list[dict]) -> list[dict]:
    """Every recorded sign flip, with the held-out scores either side of it."""
    out = []
    for run in runs:
        for h in run["history"][1:]:
            t = h["train"]
            if t.get("inverted") and "inversion_before" in t:
                out.append({"seed": run["seed"], "round": h["round"],
                            "before": t["inversion_before"], "after": t["inversion_after"],
                            "error": t.get("inversion_error", 0.0)})
    return out


def stat(runs: list[dict], path: tuple[str, ...]) -> tuple[float, float]:
    vals = []
    for run in runs:
        node = run
        for key in path:
            node = node[key]
        vals.append(float(node))
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean: float, std: float, places: int = 2) -> str:
    return f"{mean:.{places}f} ± {std:.{places}f}"


def table_headline(arms: dict[str, list[dict]]) -> None:
    print("\n### Final, on the held-out positions "
          f"(mean ± sd over {len(next(iter(arms.values())))} seeds)\n")
    print("| arm | H(q) | legal first try | refusals per move | centipawn loss | "
          "agrees with Stockfish |")
    print("|---|---|---|---|---|---|")
    for arm, runs in arms.items():
        ent = negative_entropy(runs)
        legal = stat(runs, ("heldout", "top1_legal"))
        refus = stat(runs, ("heldout", "refusals"))
        cpl = stat(runs, ("heldout", "cp_loss"))
        agree = stat(runs, ("heldout", "agreement"))
        # H(q) is a property of a negative set, so the positive-only arm has none.
        hq = f"{np.mean(ent):.2f}" if ent else "—"
        print(f"| `{arm}` | {hq} | {fmt(*legal, 3)} | "
              f"{fmt(*refus, 1)} | {fmt(*cpl, 1)} | {fmt(*agree, 3)} |")


def table_rules(arms: dict[str, list[dict]]) -> None:
    """What the heads actually know, asked of them directly.

    ``legal AUC`` is the one to read: accuracy is meaningless against a 97:3
    class split, and this is not.
    """
    keys = [("legal_auc", "legal AUC", 3), ("pseudo_auc", "pseudo-legal AUC", 3),
            ("capture_acc", "capture", 3), ("check_acc", "gives check", 3),
            ("safe_acc", "lands safely", 3), ("threat_acc", "piece is loose", 3)]
    present = [k for k, _, _ in keys
               if any(k in r["heldout"] for rs in arms.values() for r in rs)]
    if not present:
        return
    print("\n### What the heads know, on the held-out exam\n")
    print("| arm | " + " | ".join(l for k, l, _ in keys if k in present)
          + " | refusals per move |")
    print("|---" * (len(present) + 2) + "|")
    for arm, runs in arms.items():
        cells = []
        for key, _, places in keys:
            if key not in present:
                continue
            vals = [r["heldout"][key] for r in runs if key in r["heldout"]]
            cells.append(fmt(float(np.mean(vals)), float(np.std(vals)), places)
                         if vals else "—")
        cells.append(fmt(*stat(runs, ("heldout", "refusals")), 1))
        print(f"| `{arm}` | " + " | ".join(cells) + " |")
    print("\n0.5 is no knowledge of the rules and 1.0 is the rules. A network and "
          "its\ninversion score `a` and `1 - a`, exactly.")


def table_inversion(arms: dict[str, list[dict]]) -> None:
    """What the sign flip alone buys: no training between the two columns."""
    rows = []
    for arm, runs in arms.items():
        flips = inversions(runs)
        if not flips:
            continue
        for key, label, places in (("refusals", "refusals per move", 1),
                                   ("top1_legal", "legal first try", 3),
                                   ("legal_auc", "legal AUC", 3),
                                   ("pseudo_auc", "pseudo-legal AUC", 3),
                                   ("capture_acc", "capture", 3),
                                   ("check_acc", "gives check", 3),
                                   ("cp_loss", "centipawn loss", 1)):
            if key not in flips[0]["before"]:
                continue
            before = np.array([f["before"][key] for f in flips])
            after = np.array([f["after"][key] for f in flips])
            rows.append((arm, label, before.mean(), before.std(), after.mean(),
                         after.std(), places, len(flips),
                         max(f["error"] for f in flips)))
    if not rows:
        return
    print("\n### Phase 2 on its own: one sign flip, no training\n")
    print("| arm | metric | before the flip | after the flip | change |")
    print("|---|---|---|---|---|")
    for arm, label, bm, bs, am, asd, places, n, _ in rows:
        if label in ("legal first try", "legal AUC", "pseudo-legal AUC",
                     "capture", "gives check"):   # higher is better; these complement
            change = f"{am - bm:+.3f}"
        elif am <= 0 or bm <= 0:
            change = "—"
        elif bm > am:
            change = f"×{bm / am:.1f} better"
        else:
            change = f"×{am / bm:.1f} worse"
        print(f"| `{arm}` | {label} | {fmt(bm, bs, places)} | {fmt(am, asd, places)} | {change} |")
    worst = max(r[8] for r in rows)
    print(f"\nMeasured negation error over every flip: **{worst:.1e}**.")


def table_games(arms: dict[str, list[dict]]) -> None:
    print("\n### Final games against Stockfish (greedy, no exploration)\n")
    print("| arm | score | centipawn loss per move | refusals per move | plies survived |")
    print("|---|---|---|---|---|")
    for arm, runs in arms.items():
        print(f"| `{arm}` | {fmt(*stat(runs, ('vs_stockfish', 'score')), 2)} | "
              f"{fmt(*stat(runs, ('vs_stockfish', 'acpl')), 0)} | "
              f"{fmt(*stat(runs, ('vs_stockfish', 'refusals')), 1)} | "
              f"{fmt(*stat(runs, ('vs_stockfish', 'plies')), 0)} |")


def table_curve(arms: dict[str, list[dict]], every: int = 4) -> None:
    print("\n### The curve: refusals and centipawn loss, by round\n")
    rounds = len(next(iter(arms.values()))[0]["history"])
    picks = [0] + [r for r in range(every, rounds, every)] + [rounds - 1]
    picks = sorted(set(p for p in picks if p < rounds))
    header = "| arm | metric | " + " | ".join(f"r{p}" for p in picks) + " |"
    print(header)
    print("|---" * (len(picks) + 2) + "|")
    for arm, runs in arms.items():
        for key, label, places in (("refusals", "refusals/move", 1),
                                   ("cp_loss", "cp loss", 0)):
            cells = []
            for p in picks:
                cells.append(f"{np.mean([r['history'][p][key] for r in runs]):.{places}f}")
            print(f"| `{arm}` | {label} | " + " | ".join(cells) + " |")


def table_growth(arms: dict[str, list[dict]]) -> None:
    print("\n### Self-building: where the network ended up\n")
    print("| arm | hidden layers | parameters | growth steps |")
    print("|---|---|---|---|")
    for arm, runs in arms.items():
        hidden = runs[0]["hidden"]
        grew = np.mean([len(r["growth"]) for r in runs])
        params = np.mean([r["parameters"] for r in runs])
        print(f"| `{arm}` | {hidden} | {params:,.0f} | {grew:.1f} |")


def table_bench(path: str) -> None:
    with open(path) as fh:
        blob = json.load(fh)
    print("\n### Head to head\n")
    print("| match | score for the first | W–D–L | games | 95% CI | Elo |")
    print("|---|---|---|---|---|---|")
    for name, m in blob.get("head_to_head", {}).items():
        elo = "—" if m["elo"] is None else f"{m['elo']:+.0f}"
        print(f"| {name} | {m['score']:.3f} | {m['wins']}–{m['draws']}–{m['losses']} | "
              f"{m['games']} | ±{m['ci95']:.3f} | {elo} |")
    print("\n### Common opponent\n")
    print("| player | vs Stockfish | vs a random *legal* mover | legal first try | "
          "refusals per move |")
    print("|---|---|---|---|---|")
    for arm, m in blob.get("common_opponent", {}).items():
        sf = m["vs_stockfish"]
        rnd = m.get("vs_random")
        rnd_txt = f"{rnd['score']:.3f} ± {rnd['ci95']:.3f}" if rnd else "—"
        print(f"| `{arm}` | {sf['score']:.3f} ± {sf['ci95']:.3f} | {rnd_txt} | "
              f"{sf['top1_legal']:.3f} | {sf['refusals']:.1f} |")


def table_exposure(path: str) -> None:
    """The same weights ranked two ways - what the rating was measuring."""
    with open(path) as fh:
        blob = json.load(fh)
    rows = blob.get("rows", {})
    if not rows:
        return
    print("\n### The same networks, ranked two ways "
          f"({blob.get('openings', 0) * 2} games a row, against a random legal mover)\n")
    print("| network | ranked by | score vs random | centipawn loss | refusals per move |")
    print("|---|---|---|---|---|")
    for name, r in rows.items():
        arm, _, weight = name.partition(" (rule_weight=")
        w = weight.rstrip(")")
        how = "quality alone" if w in ("0", "0.0") else f"quality + {w} × legal"
        print(f"| `{arm}` | {how} | {r['score']:.3f} ± {r['stderr']:.3f} | "
              f"{r['acpl']:.0f} | {r['refusals']:.1f} |")
    print("\nNothing was retrained between the two rows of a pair and not one weight\n"
          "differs. Only the ordering the moves are proposed in changes.")


def fill_readme(readme: str, sections: dict[str, str]) -> None:
    """Replace each ``<!--NAME-->`` placeholder with its table, in place.

    The README and the runs cannot then drift apart, which is the whole reason
    this file exists.  A section is wrapped in ``<!--NAME-->`` / ``<!--/NAME-->``
    so a second run replaces what the first wrote instead of stacking a copy
    underneath it, and a placeholder with no table keeps its bare marker, so a
    partial results directory leaves the rest of the document alone.
    """
    with open(readme) as fh:
        text = fh.read()
    filled = []
    for name, body in sections.items():
        body = body.strip()
        if not body:
            continue
        block = f"<!--{name}-->\n{body}\n<!--/{name}-->"
        pattern = re.compile(rf"<!--{name}-->.*?<!--/{name}-->", re.S)
        if pattern.search(text):
            text = pattern.sub(lambda _: block, text, count=1)
        elif f"<!--{name}-->" in text:
            text = text.replace(f"<!--{name}-->", block, 1)
        else:
            continue
        filled.append(name)
    with open(readme, "w") as fh:
        fh.write(text)
    print(f"filled {len(filled)} README section(s): {', '.join(filled)}")


def capture(fn, *args) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--bench", default=None)
    p.add_argument("--exposure", default=None,
                   help="ranking_ablation.py output; fills the EXPOSURE section")
    p.add_argument("--write-readme", default=None,
                   help="a README to fill the <!--NAME--> placeholders in, in place")
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(args.results, "*.json")))
    runs = [x for x in paths if "benchmark" not in os.path.basename(x)]
    arms = load_runs(runs)
    if not arms:
        raise SystemExit(f"no run JSON in {args.results}/ - run `make run` first")
    print(f"# Tables rebuilt from {len(runs)} result file(s)")
    table_headline(arms)
    table_rules(arms)
    table_inversion(arms)
    table_games(arms)
    table_curve(arms)
    table_growth(arms)
    bench = args.bench or os.path.join(args.results, "benchmark.json")
    if os.path.exists(bench):
        table_bench(bench)
    exposure = args.exposure or os.path.join(args.results, "ranking_ablation.json")
    if os.path.exists(exposure):
        table_exposure(exposure)

    if args.write_readme:
        sections = {
            "HEADLINE": capture(table_headline, arms),
            "RULES": capture(table_rules, arms),
            "CURVE": capture(table_curve, arms),
            "ARMS": capture(table_inversion, arms),
            "GROWTH": capture(table_growth, arms),
            "BENCH": capture(table_bench, bench) if os.path.exists(bench) else "",
            "EXPOSURE": capture(table_exposure, exposure) if os.path.exists(exposure) else "",
        }
        fill_readme(args.write_readme, sections)


if __name__ == "__main__":
    main()
