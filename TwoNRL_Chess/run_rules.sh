#!/usr/bin/env bash
# The run behind the rule-curriculum numbers.
#
# Two configurations at the identical round count, so the curriculum is the
# only thing that differs:
#
#   results_rules/main.json   every arm WITH the seven heads and the mined
#                             curriculum of `rules.py`
#   results_rules/norules.json
#                             the same arms with `--rule-updates 0`, which is
#                             the single-score network this repository had
#                             before - legality learned incidentally, two rows
#                             per decision, out of a softmax about move quality
#
# Then the ladder, on the selected networks of the first.
set -u
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2"}
ROUNDS=${ROUNDS:-60}
GAMES=${GAMES:-10}
JOBS=${JOBS:-4}
COMMON="--seeds $SEEDS --rounds $ROUNDS --games $GAMES --jobs $JOBS"
mkdir -p results_rules

echo "### 1/3  every arm, with the rule curriculum"
python3 experiment.py $COMMON --arms 2nrl positive 2nrl-worst 2nrl-random repulsion \
    --weights weights_rules --out results_rules/main.json

echo "### 2/3  the ablation: the same arms without it"
python3 experiment.py $COMMON --arms 2nrl positive --rule-updates 0 --rule-weight 0 \
    --weights weights_norules --out results_rules/norules.json

echo "### 3/5  the ladder, on the selected networks"
python3 elo.py --weights weights_rules --openings 14 --bootstrap 300 \
    --out results_rules/elo.json

echo "### 4/5  head to head, common opponent, and the floor"
python3 benchmark.py --weights weights_rules --seeds $SEEDS --openings 12 \
    --arms 2nrl positive 2nrl-worst 2nrl-random repulsion \
    --out results_rules/benchmark.json

# The ladder above rates networks that are now ranking with the legal head.
# This asks what that head is worth by re-ranking the identical weights
# without it - and it is the measurement that says the older ratings in this
# repository were largely ratings of the refusal walk.
echo "### 5/5  the same weights, ranked two ways"
python3 ranking_ablation.py --weights weights_rules --openings 40 \
    --out results_rules/ranking_ablation.json

echo "### tables"
python3 summarize.py --results results_rules --write-readme README.md

echo "RULES_DONE"
