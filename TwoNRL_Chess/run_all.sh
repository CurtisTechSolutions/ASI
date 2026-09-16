#!/usr/bin/env bash
# Every run behind the README, in order.  `make all` calls the same targets one
# at a time; this is the whole suite in one go, which is what produced
# `results/`.  Ablations write their networks to their own directory so they do
# not overwrite the main run's.
set -u
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2"}
ROUNDS=${ROUNDS:-40}
GAMES=${GAMES:-10}
JOBS=${JOBS:-4}
COMMON="--seeds $SEEDS --rounds $ROUNDS --games $GAMES --jobs $JOBS"
mkdir -p results

echo "### 1/7  the headline: 2NRL against the same thing without it"
python3 experiment.py $COMMON --arms 2nrl positive \
    --weights weights --out results/main.json

echo "### 2/7  the H(q) ladder of §11"
python3 experiment.py $COMMON --arms 2nrl-worst 2nrl-random \
    --weights weights --out results/entropy.json

echo "### 3/7  arm E: repulsion instead of inversion"
python3 experiment.py $COMMON --arms repulsion \
    --weights weights --out results/repulsion.json

echo "### 4/7  ablation: §9.3's perpetual loop, inverting inside every round"
python3 experiment.py $COMMON --arms 2nrl --schedule per-round \
    --weights weights_per_round --out results/per_round.json

echo "### 5/7  ablation: negate only the read-out instead of every unit"
python3 experiment.py $COMMON --arms 2nrl --invert-mode readout \
    --weights weights_readout --out results/invert_readout.json

echo "### 6/7  ablation (§12 item 4): a fixed date for phase 1 instead of the trigger"
# --neg-fraction only means anything under the `schedule` trigger; with the
# default `plateau` trigger it is ignored, and passing it alone silently
# reproduces the headline run.
python3 experiment.py $COMMON --arms 2nrl --invert-trigger schedule --neg-fraction 0.15 \
    --weights weights_depth15 --out results/phase_depth_15.json
python3 experiment.py $COMMON --arms 2nrl --invert-trigger schedule --neg-fraction 0.50 \
    --weights weights_depth50 --out results/phase_depth_50.json

echo "### 7/7  the benchmark: head to head, common opponent, and the floor"
python3 benchmark.py --weights weights --seeds $SEEDS --openings 12 \
    --arms 2nrl positive 2nrl-worst 2nrl-random repulsion \
    --out results/benchmark.json

echo "### tables"
python3 summarize.py --results results
