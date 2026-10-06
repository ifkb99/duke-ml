#!/usr/bin/env bash
# Run training detached, restarting from the latest checkpoint if it crashes.
#
#   training/zero/run.sh <run-name> [extra train.py flags...]
#   tail -f runs/<run-name>/train.log        # watch progress
#   kill -- -$(cat runs/<run-name>/pid)      # stop (the whole process group)
set -u
cd "$(dirname "$0")/../.."
RUN=${1:?usage: run.sh <run-name> [flags...]}
shift
mkdir -p "runs/$RUN"

# Leave some GPU memory for other work (tests, evaluation) on the same cards.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

nohup setsid bash -c '
  echo $$ > "runs/'"$RUN"'/pid"
  until .venv/bin/python -m training.zero.train --run "'"$RUN"'" '"$*"'; do
    echo "[run.sh] train.py exited with $? at $(date); restarting in 30s" >&2
    sleep 30
  done
' >> "runs/$RUN/train.log" 2>&1 &
echo "started $RUN (log: runs/$RUN/train.log)"
