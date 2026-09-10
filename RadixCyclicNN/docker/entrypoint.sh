#!/bin/sh
# Container entrypoint for RadixCyclicNN.
#
# Runs the `radixnet` CLI with the given arguments.  Before that, when
# RADIXNET_RESUME=1, the newest checkpoint is restored into the model file so a
# restarted container carries on from where a long training / evolve job left
# off instead of from the last explicit save.
#
#   RADIXNET_RESUME           1 = restore <checkpoint dir>/latest into the model file first (default 0)
#   RADIXNET_MODEL            model file (default /data/model.json)
#   RADIXNET_CHECKPOINT_DIR   checkpoint directory (default /data/checkpoints)
set -e

MODEL="${RADIXNET_MODEL:-/data/model.json}"
CKPT="${RADIXNET_CHECKPOINT_DIR:-/data/checkpoints}"

if [ "${RADIXNET_RESUME:-0}" = "1" ]; then
  if [ -f "$CKPT/latest.json" ]; then
    echo "entrypoint: restoring the latest checkpoint from $CKPT into $MODEL" >&2
    radixnet --model "$MODEL" checkpoints --dir "$CKPT" --restore latest \
      || echo "entrypoint: restore failed, continuing with $MODEL" >&2
  else
    echo "entrypoint: RADIXNET_RESUME=1 but $CKPT has no checkpoint yet" >&2
  fi
fi

exec radixnet "$@"
