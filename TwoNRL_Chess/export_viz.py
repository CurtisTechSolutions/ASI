"""Turn the runs and the phase-boundary checkpoints into one JSON the page reads.

The interesting picture in this experiment is not a bar chart.  It is the
**action space**: all 4096 from-to pairs, scored by the network, with the legal
ones marked.  Laid out as a 64x64 grid it shows directly what
``Research/2NRL.md`` §6.5 is about - the network being refused until it works
out which moves the board accepts - and then shows phase 2 reversing it in one
operation.

Four network states are scored, which are the four states 2NRL has:

    untrained      before anything; the ranking is arbitrary
    phase1_end     trained as hard as it can be to break the rules
    after_invert   the same network, one sign flip later, nothing else
    final          after phase 3 has fine-tuned on Stockfish's moves

``after_invert`` minus ``phase1_end`` is the whole claim, and putting the two
grids side by side is the clearest way to see whether it holds.

Run ``make viz`` (which captures the checkpoints first) rather than this
directly.
"""

from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import glob
import json
import os

import chess
import numpy as np

import dataset
from agent import Agent
from moves import ACTION_FROM, ACTION_TO, N_ACTIONS, to_move
from sbnn import SineNet
from summarize import label_of

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ("untrained", "phase1_end", "after_invert", "final")
STATE_LABEL = {
    "untrained": "Untrained",
    "phase1_end": "Trained to fail",
    "after_invert": "After the sign flip",
    "final": "After fine-tuning",
}
STATE_NOTE = {
    "untrained": "Before anything. The ranking over the 4096 moves is arbitrary, "
                 "and the board refuses most of what it proposes.",
    "phase1_end": "Phase 1, at the full learning rate, trained toward the moves the "
                  "board refused. It now proposes illegal moves first, on purpose.",
    "after_invert": "Phase 2. The same network with every weight and amplitude sign "
                    "flipped - one closed-form operation, no training, no data.",
    "final": "Phase 3, fine-tuned toward Stockfish's move at a fifth of the rate.",
}


def score_grid(agent: Agent, board: chess.Board) -> dict:
    """The network's ranking over all 4096 actions, plus which of them are legal.

    Scores are turned into a **percentile of the ranking** rather than kept raw,
    because the policy is an argmax - what matters is the order, and percentiles
    are comparable between a network and its own negation.
    """
    scores, _, _ = agent.action_scores(board)
    order = np.argsort(-scores)
    rank = np.empty(N_ACTIONS, dtype=np.int32)
    rank[order] = np.arange(N_ACTIONS)
    pct = 1.0 - rank / (N_ACTIONS - 1)             # 1.0 = the move it would play
    legal_actions = {int(a) for a in range(N_ACTIONS)
                     if board.is_legal(to_move(board, a, board.turn))}
    refusals = int(np.argmax([int(a) in legal_actions for a in order]))
    return {"percentile": [round(float(v), 4) for v in pct],
            "legal": sorted(legal_actions),
            "refusals": refusals,
            "top": [int(a) for a in order[:8]],
            "top_legal": [bool(int(a) in legal_actions) for a in order[:8]]}


def board_view(board: chess.Board) -> dict:
    """The position itself, for the little board beside the grid."""
    pieces = {}
    for square, piece in board.piece_map().items():
        pieces[str(square)] = piece.symbol()
    return {"fen": board.fen(), "pieces": pieces, "turn": "w" if board.turn else "b",
            "legal_count": board.legal_moves.count()}


def action_spaces(ckpt_dir: str, positions: list[dict], arm: str, seed: int) -> list[dict]:
    out = []
    for item in positions:
        board = item["board"]
        entry = {"board": board_view(board), "note": item.get("note", ""), "states": {}}
        for state in STATES:
            path = os.path.join(ckpt_dir, f"{arm}_seed{seed}_{state}.npz")
            if not os.path.exists(path):
                continue
            entry["states"][state] = score_grid(Agent(SineNet.load(path)), board)
        if entry["states"]:
            out.append(entry)
    return out


def curves(arms: dict[str, list[dict]]) -> dict:
    """Per-round refusals and centipawn loss, averaged over seeds, per arm."""
    out = {}
    for label, runs in arms.items():
        rounds = min(len(r["history"]) for r in runs)
        series = {"round": list(range(rounds)), "refusals": [], "cp_loss": [],
                  "legal": [], "invert_round": None, "phase1_rounds": runs[0].get("negative_rounds")}
        for i in range(rounds):
            series["refusals"].append(round(float(np.mean([r["history"][i]["refusals"] for r in runs])), 2))
            series["cp_loss"].append(round(float(np.mean([r["history"][i]["cp_loss"] for r in runs])), 1))
            series["legal"].append(round(float(np.mean([r["history"][i]["top1_legal"] for r in runs])), 4))
        for h in runs[0]["history"][1:]:
            if h["train"].get("inverted"):
                series["invert_round"] = h["round"]
                break
        out[label] = series
    return out


def finals(arms: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for label, runs in arms.items():
        def col(key, where="heldout"):
            return np.array([r[where][key] for r in runs], dtype=float)
        ent = [h["train"]["negative_entropy"] for r in runs for h in r["history"][1:]
               if h["train"].get("phase") in ("negative", "both") and h["train"].get("n")]
        rows.append({
            "arm": label, "seeds": len(runs),
            "entropy": None if (not ent or runs[0]["arm"] == "positive") else round(float(np.mean(ent)), 2),
            "refusals": round(float(col("refusals").mean()), 1),
            "refusals_sd": round(float(col("refusals").std()), 1),
            "legal": round(float(col("top1_legal").mean()), 4),
            "legal_sd": round(float(col("top1_legal").std()), 4),
            "cp_loss": round(float(col("cp_loss").mean()), 1),
            "cp_loss_sd": round(float(col("cp_loss").std()), 1),
            "agreement": round(float(col("agreement").mean()), 4),
            "hidden": runs[0]["hidden"], "parameters": runs[0]["parameters"],
            "growth": round(float(np.mean([len(r["growth"]) for r in runs])), 1),
        })
    return rows


def inversion_effect(arms: dict[str, list[dict]]) -> list[dict]:
    """What phase 2 did on its own, with nothing between the two readings."""
    rows = []
    for label, runs in arms.items():
        flips = [h["train"] for r in runs for h in r["history"][1:]
                 if h["train"].get("inverted") and "inversion_before" in h["train"]]
        if not flips:
            continue
        row = {"arm": label, "flips": len(flips),
               "error": max(f.get("inversion_error", 0.0) for f in flips)}
        for key in ("refusals", "top1_legal", "cp_loss"):
            before = np.array([f["inversion_before"][key] for f in flips], dtype=float)
            after = np.array([f["inversion_after"][key] for f in flips], dtype=float)
            row[key] = {"before": round(float(before.mean()), 3),
                        "before_sd": round(float(before.std()), 3),
                        "after": round(float(after.mean()), 3),
                        "after_sd": round(float(after.std()), 3)}
        rows.append(row)
    return rows


def load_arms(results: str) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(results, "*.json"))):
        if "benchmark" in os.path.basename(path):
            continue
        with open(path) as fh:
            for run in json.load(fh).get("runs", []):
                arms.setdefault(label_of(run), []).append(run)
    return arms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--checkpoints", default=os.path.join(HERE, "viz", "checkpoints"))
    p.add_argument("--heldout", default=dataset.DEFAULT_PATH)
    p.add_argument("--arm", default="2nrl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--positions", type=int, default=3)
    p.add_argument("--out", default=os.path.join(HERE, "viz", "data.json"))
    args = p.parse_args()

    arms = load_arms(args.results)
    if not arms:
        raise SystemExit(f"no run JSON in {args.results}/ - run `make run` first")

    held = dataset.load(args.heldout)
    picked = []
    step = max(1, len(held) // max(args.positions, 1))
    for i in range(0, len(held), step):
        item = held[i]
        if item["board"].legal_moves.count() >= 20:
            picked.append({"board": item["board"],
                           "note": f"{item['board'].legal_moves.count()} of 4096 moves are legal here"})
        if len(picked) >= args.positions:
            break

    blob = {
        "arms": finals(arms),
        "curves": curves(arms),
        "inversion": inversion_effect(arms),
        "action_space": action_spaces(args.checkpoints, picked, args.arm, args.seed),
        "state_order": list(STATES),
        "state_label": STATE_LABEL,
        "state_note": STATE_NOTE,
        "action_from": [int(v) for v in ACTION_FROM],
        "action_to": [int(v) for v in ACTION_TO],
        "config": next(iter(arms.values()))[0]["config"],
        "opponent": next(iter(arms.values()))[0]["opponent"],
    }
    bench = os.path.join(args.results, "benchmark.json")
    if os.path.exists(bench):
        with open(bench) as fh:
            blob["benchmark"] = json.load(fh)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(blob, fh, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({size:.0f} KB)")
    print(f"  arms          {len(blob['arms'])}")
    print(f"  action spaces {len(blob['action_space'])} positions "
          f"x {len(blob['action_space'][0]['states']) if blob['action_space'] else 0} states")
    print(f"  benchmark     {'yes' if 'benchmark' in blob else 'not yet'}")


if __name__ == "__main__":
    main()
