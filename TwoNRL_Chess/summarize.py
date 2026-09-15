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

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["2nrl", "2nrl-worst", "2nrl-random", "positive", "repulsion"]


def load_runs(paths: list[str]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for path in paths:
        with open(path) as fh:
            blob = json.load(fh)
        for run in blob.get("runs", []):
            arms.setdefault(run["arm"], []).append(run)
    return {arm: arms[arm] for arm in ORDER if arm in arms} | \
           {arm: rs for arm, rs in arms.items() if arm not in ORDER}


def negative_entropy(runs: list[dict]) -> list[float]:
    """H(q) is only defined on the rounds that actually ran a negative block."""
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
        print(f"| `{arm}` | {np.mean(ent) if ent else 0:.2f} | {fmt(*legal, 3)} | "
              f"{fmt(*refus, 1)} | {fmt(*cpl, 1)} | {fmt(*agree, 3)} |")


def table_inversion(arms: dict[str, list[dict]]) -> None:
    """What the sign flip alone buys: no training between the two columns."""
    rows = []
    for arm, runs in arms.items():
        flips = inversions(runs)
        if not flips:
            continue
        for key, label, places in (("refusals", "refusals per move", 1),
                                   ("top1_legal", "legal first try", 3),
                                   ("cp_loss", "centipawn loss", 1)):
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
        if label == "legal first try":
            change = f"×{am / bm:.1f}" if bm > 0 else "—"
        else:
            change = f"×{bm / am:.1f} better" if am > 0 and bm > am else f"×{am / bm:.1f} worse"
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--bench", default=None)
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(args.results, "*.json")))
    runs = [x for x in paths if "benchmark" not in os.path.basename(x)]
    arms = load_runs(runs)
    if not arms:
        raise SystemExit(f"no run JSON in {args.results}/ - run `make run` first")
    print(f"# Tables rebuilt from {len(runs)} result file(s)")
    table_headline(arms)
    table_inversion(arms)
    table_games(arms)
    table_curve(arms)
    table_growth(arms)
    bench = args.bench or os.path.join(args.results, "benchmark.json")
    if os.path.exists(bench):
        table_bench(bench)


if __name__ == "__main__":
    main()
