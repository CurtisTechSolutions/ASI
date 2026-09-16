#!/usr/bin/env bash
# The long run: five arms, three seeds, five times the rounds of the standard
# suite. Everything else is the default configuration.
set -u
cd "$(dirname "$0")"
ROUNDS=${ROUNDS:-200}
GAMES=${GAMES:-10}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-4}
mkdir -p results_long weights_long
python3 experiment.py --seeds $SEEDS --rounds $ROUNDS --games $GAMES --jobs $JOBS \
    --arms 2nrl positive --weights weights_long --out results_long/main.json
python3 experiment.py --seeds $SEEDS --rounds $ROUNDS --games $GAMES --jobs $JOBS \
    --arms 2nrl-worst 2nrl-random repulsion --weights weights_long \
    --out results_long/others.json
echo LONG_RUN_DONE
