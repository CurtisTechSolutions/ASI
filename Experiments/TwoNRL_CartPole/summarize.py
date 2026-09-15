"""Build the results tables in README.md straight from the run JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STAGES = [("untrained", "untrained"),
          ("after_negative", "after negative phase (trained to fail)"),
          ("after_invert", "**after inversion — no training**"),
          ("after_finetune", "after fine-tune")]


def load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def table(runs: list[dict], controls: list[dict]) -> str:
    seeds = [r["seed"] for r in runs]
    head = "| stage | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |"
    rule = "|---|" + "---|" * (len(seeds) + 1)
    rows = [head, rule]
    for key, label in STAGES:
        vals = [r[key]["mean"] for r in runs]
        rows.append(f"| {label} | " + " | ".join(f"{v:.1f}" for v in vals)
                    + f" | **{np.mean(vals):.1f}** |")
    if controls:
        vals = [c["after_training"]["mean"] for c in controls]
        rows.append("| control: same net, same env steps, no 2NRL | "
                    + " | ".join(f"{v:.1f}" for v in vals) + f" | {np.mean(vals):.1f} |")
    return "\n".join(rows)


def failure_table(runs: list[dict]) -> str:
    rows = ["| seed | steps to fail | graded loss | venture radius | majority action | anti-balance |",
            "|---|---|---|---|---|---|"]
    for r in runs:
        sd = r["failure_state_dependence"]
        rows.append(f"| {r['seed']} | {r['failure_episode_length']:.1f} | "
                    f"{r.get('failure_graded_loss', float('nan')):.2f} | "
                    f"{r['final_start_radius']:.2f} | "
                    f"{sd['majority_action_share']:.0%} | {sd['anti_balance_agreement']:.0%} |")
    return "\n".join(rows)


def main() -> None:
    main_r = load("results/twonrl_results.json")
    abl = load("results/ablation_no_venture.json")
    speed = load("results/grading_speed_results.json")
    noweights = load("results/ablation_no_weights.json")
    if not main_r:
        sys.exit("results/twonrl_results.json not found - run the experiment first")

    print("## Results\n")
    print(table(main_r["runs"], main_r.get("controls", [])))
    print("\n### What the negative phase learned\n")
    print(failure_table(main_r["runs"]))

    print("\n### Ablations\n")
    rows = ["| variant | inverted, no training | failure policy |", "|---|---|---|"]
    inv = np.mean([r["after_invert"]["mean"] for r in main_r["runs"]])
    maj = np.mean([r["failure_state_dependence"]["majority_action_share"] for r in main_r["runs"]])
    anti = np.mean([r["failure_state_dependence"]["anti_balance_agreement"] for r in main_r["runs"]])
    rows.append(f"| **graded loss `|target - achieved|` drives the radius** | **{inv:.1f}** | "
                f"majority action {maj:.0%}, anti-balance {anti:.0%} |")
    if speed:
        i = np.mean([r["after_invert"]["mean"] for r in speed["runs"]])
        m = np.mean([r["failure_state_dependence"]["majority_action_share"] for r in speed["runs"]])
        a = np.mean([r["failure_state_dependence"]["anti_balance_agreement"] for r in speed["runs"]])
        rows.append(f"| ratio delta `L_min/L` (saturates at the max radius) | {i:.1f} | "
                    f"majority action {m:.0%}, anti-balance {a:.0%} |")
    if noweights:
        i = np.mean([r["after_invert"]["mean"] for r in noweights["runs"]])
        m = np.mean([r["failure_state_dependence"]["majority_action_share"] for r in noweights["runs"]])
        a = np.mean([r["failure_state_dependence"]["anti_balance_agreement"] for r in noweights["runs"]])
        rows.append(f"| graded loss, but no `bad_weights` (every failure trains equally) | {i:.1f} | "
                    f"majority action {m:.0%}, anti-balance {a:.0%} |")
    if abl:
        i = np.mean([r["after_invert"]["mean"] for r in abl["runs"]])
        m = np.mean([r["failure_state_dependence"]["majority_action_share"] for r in abl["runs"]])
        a = np.mean([r["failure_state_dependence"]["anti_balance_agreement"] for r in abl["runs"]])
        rows.append(f"| no venturing out at all (standard resets) | {i:.1f} | "
                    f"majority action {m:.0%}, anti-balance {a:.0%} |")
    print("\n".join(rows))

    steps = [r["env_steps"]["total"] for r in main_r["runs"]]
    print(f"\nEnvironment steps per run: {int(np.mean(steps)):,} "
          f"(negative {int(np.mean([r['env_steps']['negative'] for r in main_r['runs']])):,}, "
          f"positive {int(np.mean([r['env_steps']['positive'] for r in main_r['runs']])):,}); "
          f"the control is given the same budget.")
    err = max(r["inversion_max_error"] for r in main_r["runs"])
    print(f"\nLargest inversion error over every run: `{err:.1e}`.")


if __name__ == "__main__":
    main()
