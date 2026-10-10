#!/usr/bin/env bash
# Keep uploading runs' metrics.jsonl to Weights & Biases, detached, restarting
# the uploader if it dies (e.g. network trouble). Training is unaffected either way.
#
#   training/zero/wandb_follow.sh <run> [<run> ...]       # e.g. az3-net az3-sims
#   kill -- -$(cat runs/<run>/wandb_pid)                  # stop following one run
#
# Set GPUS=1 for runs on a single GPU started before train.py recorded it.
set -u
cd "$(dirname "$0")/../.."
for RUN in "$@"; do
  nohup setsid bash -c '
    echo $$ > "runs/'"$RUN"'/wandb_pid"
    until .venv/bin/python -m training.zero.wandb_sync "'"$RUN"'" --follow --gpus '"${GPUS:-2}"'; do
      echo "[wandb_follow] exited with $? at $(date); restarting in 60s" >&2
      sleep 60
    done
  ' >> "runs/$RUN/wandb_sync.log" 2>&1 &
  echo "following $RUN (log: runs/$RUN/wandb_sync.log)"
done
