#!/usr/bin/env bash
# The inversion operator, three ways, everything else held fixed.
#
#   complement   W -> 1 - W        the shipped operator
#   unit         a -> -a, k -> -k  negates every unit exactly
#   negate       W -> -W           the sign flip
#
# All on the sine activation throughout, rule heads included.  Ordered so the
# comparison that decides the question finishes first - the three operators on
# `2nrl`, plus the `positive` control they are all measured against - and the
# remaining arms after.  Every step skips itself if its output is already on
# disk, so a run cut short by a restart resumes where it stopped instead of
# starting again.
set -u
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"0 1 2"}
ROUNDS=${ROUNDS:-60}
GAMES=${GAMES:-10}
JOBS=${JOBS:-4}
OUT=${OUT:-results_sine}
COMMON="--seeds $SEEDS --rounds $ROUNDS --games $GAMES --jobs $JOBS"
mkdir -p "$OUT"

step() {                      # step <output> <label> <experiment.py args...>
    local out="$1" label="$2"; shift 2
    if [ -s "$out" ]; then
        echo "### $label - already done, skipping ($out)"
        return
    fi
    echo "### $label"
    python3 experiment.py $COMMON "$@" --out "$out.part" && mv "$out.part" "$out"
}

step "$OUT/main.json"   "1/4  2nrl and positive, complement W -> 1 - W" \
     --arms 2nrl positive --weights weights_sine
step "$OUT/unit.json"   "2/4  2nrl, per-unit flip" \
     --arms 2nrl --invert-mode unit --weights weights_sine_unit
step "$OUT/negate.json" "3/4  2nrl, sign flip W -> -W" \
     --arms 2nrl --invert-mode negate --weights weights_sine_negate
step "$OUT/others.json" "4/4  the other arms, complement" \
     --arms 2nrl-worst 2nrl-random repulsion --weights weights_sine

echo "### tables"
python3 summarize.py --results "$OUT"
echo OPERATORS_DONE
