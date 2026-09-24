#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p results_r400 weights_r400
python3 experiment.py --seeds 0 1 2 --rounds 400 --games 10 --jobs 4 \
  --arms 2nrl positive --weights weights_r400 --out results_r400/main.json
python3 experiment.py --seeds 0 1 2 --rounds 400 --games 10 --jobs 4 \
  --arms 2nrl-worst 2nrl-random repulsion --weights weights_r400 --out results_r400/others.json
echo R400_DONE
