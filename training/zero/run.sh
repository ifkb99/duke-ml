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

# CPUs to run on. This machine's core 20 (logical CPUs 10 and 11) is faulty:
# every random segfault (python, ptxas, nvrtc, node) landed there, and none
# happened once it was excluded. Child processes inherit the restriction.
DUKE_CPUS=${DUKE_CPUS:-0-9,12-31}

# Print a Python stack trace if the process dies from a native crash (segfault).
export PYTHONFAULTHANDLER=1
# Leave some GPU memory for other work (tests, evaluation) on the same cards.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}

nohup setsid bash -c '
  echo $$ > "runs/'"$RUN"'/pid"
  until taskset -c '"$DUKE_CPUS"' .venv/bin/python -m training.zero.train --run "'"$RUN"'" '"$*"'; do
    echo "[run.sh] train.py exited with $? at $(date); restarting in 30s" >&2
    sleep 30
  done
' >> "runs/$RUN/train.log" 2>&1 &
echo "started $RUN (log: runs/$RUN/train.log)"
