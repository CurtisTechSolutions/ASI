#!/usr/bin/env python3
"""Run the Go port and the Rust port over the same corpus and tabulate them.

Both builds are handed the same texts and the same prefixes (`make_corpus.py`),
both train for the same epochs, both punish the same texts before predicting -
so the only thing that differs is the implementation.  Before any timing is
reported the two runs are checked against each other: same nodes, same edges,
same trigrams, same transitions, same loss, same prediction at the same cost.
A speed comparison between two programs that computed different things is not a
comparison, so a parity failure stops the run.

    python3 bench/compare.py                       # the default matrix
    python3 bench/compare.py --chars 1000000 --repeat 5
    python3 bench/compare.py --no-build --out bench/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GO_BIN = ROOT / "go" / "bin" / "radixnet-count"
RUST_BIN = ROOT / "rust" / "target" / "release" / "radixnet-bench"

# The numbers both ports must agree on before any timing is worth reading.
EXACT_KEYS = [
    "chars",
    "texts",
    "nodes",
    "edges",
    "trigrams",
    "transitions",
    "dijkstra_expansions",
    "punished_texts",
    "punished_edges",
]
CLOSE_KEYS = ["compression_ratio", "loss", "edge_reward_negative"]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def build() -> None:
    print("building the Go CLI ...", file=sys.stderr)
    run(["go", "build", "-o", str(GO_BIN), "./cmd/radixnet-count"], cwd=ROOT / "go")
    print("building the Rust binary ...", file=sys.stderr)
    run(["cargo", "build", "--release"], cwd=ROOT / "rust")


def go_command(args: argparse.Namespace, traversal: str, workers: int, exact: bool) -> list[str]:
    cmd = [str(GO_BIN), "--json", "--workers", str(workers)]
    if exact:
        cmd.append("--exact")
    return cmd + [
        "bench",
        "--texts", str(args.texts),
        "--prefixes", str(args.prefixes),
        "--epochs", str(args.epochs),
        "--punish-every", str(args.punish_every),
        "--traversal", traversal,
    ]


def rust_command(args: argparse.Namespace, traversal: str, workers: int, dump: Path | None = None) -> list[str]:
    cmd = [
        str(RUST_BIN), "--json",
        "--workers", str(workers),
        "--texts", str(args.texts),
        "--prefixes", str(args.prefixes),
        "--epochs", str(args.epochs),
        "--punish-every", str(args.punish_every),
        "--traversal", traversal,
    ]
    return cmd + (["--dump", str(dump)] if dump else [])


def measure(cmd: list[str], repeat: int) -> dict:
    """Runs a build `repeat` times and keeps the median rates (and the best)."""
    runs = [json.loads(run(cmd).stdout) for _ in range(repeat)]
    out = dict(runs[-1])
    for key in ("train_seconds", "predict_seconds", "transitions_per_sec", "chars_per_sec", "predictions_per_sec"):
        values = [r[key] for r in runs]
        out[key] = statistics.median(values)
        out[key + "_best"] = min(values) if key.endswith("seconds") else max(values)
    return out


def parity(go: dict, rust: dict) -> list[str]:
    """What the two runs disagree on - empty when they did the same work."""
    problems = []
    for key in EXACT_KEYS:
        if go.get(key) != rust.get(key):
            problems.append(f"{key}: go={go.get(key)!r} rust={rust.get(key)!r}")
    for key in CLOSE_KEYS:
        a, b = go.get(key), rust.get(key)
        if a is None or b is None or abs(a - b) > 1e-9 * max(1.0, abs(a)):
            problems.append(f"{key}: go={a!r} rust={b!r}")
    a, b = go.get("sample_prediction", {}), rust.get("sample_prediction", {})
    if a.get("continuation") != b.get("continuation"):
        problems.append(f"sample continuation: go={a.get('continuation')!r} rust={b.get('continuation')!r}")
    elif abs(a.get("cost", 0) - b.get("cost", 0)) > 1e-9:
        problems.append(f"sample cost: go={a.get('cost')!r} rust={b.get('cost')!r}")
    return problems


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def version(cmd: list[str]) -> str:
    try:
        return run(cmd).stdout.strip().splitlines()[0]
    except Exception:
        return "?"


def table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """A markdown table: columns are (header, key, format)."""
    out = ["| " + " | ".join(h for h, _, _ in columns) + " |"]
    out.append("|" + "|".join("---" if i == 0 else "--:" for i, _ in enumerate(columns)) + "|")
    for row in rows:
        cells = []
        for _, key, fmt in columns:
            value = row.get(key)
            if value is None:
                cells.append("-")
            elif fmt == "s":
                cells.append(str(value))
            elif fmt == "int":
                cells.append(f"{value:,.0f}")
            elif fmt == "x":
                cells.append(f"{value:.2f}x")
            else:
                cells.append(f"{value:.4f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", type=int, default=2_000_000, help="corpus size to generate (default: 2000000)")
    ap.add_argument("--predictions", type=int, default=3000, help="prefixes to predict (default: 3000)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--punish-every", type=int, default=7, help="punish every Nth text before predicting")
    ap.add_argument("--repeat", type=int, default=3, help="runs per row; the median is reported")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep-corpus", action="store_true", help="reuse bench/corpus*.txt if they are there")
    ap.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    args = ap.parse_args()
    args.texts = HERE / "corpus.txt"
    args.prefixes = HERE / "corpus.prefixes.txt"

    if not args.no_build:
        build()
    if not (args.keep_corpus and args.texts.exists() and args.prefixes.exists()):
        run([sys.executable, str(HERE / "make_corpus.py"), "--chars", str(args.chars),
             "--predictions", str(args.predictions), "--seed", str(args.seed), "--out", str(HERE / "corpus")])

    builds = [
        ("Go (racy, its default)", lambda t, w: go_command(args, t, w, exact=False)),
        ("Go (exact)", lambda t, w: go_command(args, t, w, exact=True)),
        ("Rust", lambda t, w: rust_command(args, t, w)),
    ]
    results: dict[tuple[str, str, int], dict] = {}
    for traversal in ("reward", "least-punished"):
        for workers in (1, 0):
            for name, command in builds:
                print(f"running {name:<24} traversal={traversal:<15} workers={workers}", file=sys.stderr)
                results[(traversal, name, workers)] = measure(command(traversal, workers), args.repeat)

    # the two ports must have done the same work before any of this means anything
    problems = []
    for traversal in ("reward", "least-punished"):
        for workers in (1, 0):
            for label in ("Go (exact)",):
                found = parity(results[(traversal, label, workers)], results[(traversal, "Rust", workers)])
                problems += [f"{traversal}, {workers or 'all'} worker(s): {p}" for p in found]
    if problems:
        print("PARITY FAILED - the two ports did not do the same work:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1

    rows = []
    for traversal in ("reward", "least-punished"):
        for workers in (1, 0):
            base = results[(traversal, "Go (racy, its default)", workers)]
            for name, _ in builds:
                r = results[(traversal, name, workers)]
                rows.append({
                    "build": name,
                    "traversal": traversal,
                    "workers": "1" if workers == 1 else "all cores",
                    "train_seconds": r["train_seconds"],
                    "transitions_per_sec": r["transitions_per_sec"],
                    "predict_seconds": r["predict_seconds"],
                    "predictions_per_sec": r["predictions_per_sec"],
                    "expansions": r["dijkstra_expansions"],
                    "train_speedup": base["train_seconds"] / r["train_seconds"] if r["train_seconds"] else 0,
                    "predict_speedup": base["predict_seconds"] / r["predict_seconds"] if r["predict_seconds"] else 0,
                })

    # how much of the difference is the traversal rather than the language: the
    # answers the two searches disagree on, over the same model
    dumps = {}
    for traversal in ("reward", "least-punished"):
        path = HERE / f"predictions.{traversal}.txt"
        run(rust_command(args, traversal, 1, dump=path))
        dumps[traversal] = path.read_text(encoding="utf-8").splitlines()
        path.unlink()
    changed = sum(1 for a, b in zip(dumps["reward"], dumps["least-punished"]) if a != b)
    shape = results[("reward", "Rust", 1)]
    columns = [
        ("build", "build", "s"),
        ("traversal", "traversal", "s"),
        ("workers", "workers", "s"),
        ("train s", "train_seconds", "f"),
        ("transitions/s", "transitions_per_sec", "int"),
        ("predict s", "predict_seconds", "f"),
        ("predictions/s", "predictions_per_sec", "int"),
        ("expansions", "expansions", "int"),
        ("train vs Go", "train_speedup", "x"),
        ("predict vs Go", "predict_speedup", "x"),
    ]
    head = [
        "# Rust against Go: the count / reward model, measured",
        "",
        "Written by `bench/compare.py`; rerun it to replace this file.  Every row",
        "trained on the same corpus and predicted the same prefixes, and the Go and",
        "Rust runs were checked against each other before being timed (same graph,",
        "same transitions, same loss, same prediction at the same cost).",
        "",
        "The last two columns are against **Go's own default counting at the same",
        "worker count** - the row above each group of three.",
        "",
        f"- **when** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **machine** {cpu_model()}, {platform.system()} {platform.machine()}",
        f"- **go** {version(['go', 'version'])}",
        f"- **rust** {version(['rustc', '--version'])}",
        f"- **corpus** {shape['texts']:,} texts / {shape['chars']:,} characters, {args.epochs} epochs,"
        f" {shape['predict_count']:,} predictions of {shape.get('dijkstra_expansions', 0):,} expansions",
        f"- **graph** {shape['nodes']:,} nodes, {shape['edges']:,} edges, {shape['trigrams']:,} trigrams,"
        f" compression {shape['compression_ratio']:.4f}",
        f"- **blame** every {args.punish_every}th text punished before the predictions"
        f" ({shape['punished_texts']:,} texts, {shape['punished_edges']:,} edges)",
        f"- **repeats** {args.repeat} runs per row, median reported",
        f"- **the two searches** differ on {changed:,} of {len(dumps['reward']):,} continuations"
        f" ({100 * changed / max(1, len(dumps['reward'])):.1f}%)",
        "",
        table(rows, columns),
        "",
    ]
    text = "\n".join(head)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if shutil.which("go") is None or shutil.which("cargo") is None:
        print("both go and cargo have to be on PATH", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
