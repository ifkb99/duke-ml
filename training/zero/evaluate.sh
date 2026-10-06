#!/usr/bin/env bash
# Run the checkpoint evaluator (ratings, benchmark, calibration) detached,
# restarting it if it crashes. It shares the GPUs with training, alternating
# between them, so it needs some free GPU memory (training's
# XLA_PYTHON_CLIENT_MEM_FRACTION should leave ~3 GB per GPU).
#
#   training/zero/evaluate.sh az2 az3-net az3-sims --stride az1=1000
#   tail -f runs/ratings/evaluator.log
#   kill -- -$(cat runs/ratings/pid)
set -u
cd "$(dirname "$0")/../.."
mkdir -p runs/ratings
DUKE_CPUS=${DUKE_CPUS:-0-9,12-31}
export PYTHONFAULTHANDLER=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup setsid bash -c '
  echo $$ > runs/ratings/pid
  until taskset -c '"$DUKE_CPUS"' .venv/bin/python -m training.zero.evaluator '"$*"'; do
    echo "[evaluate.sh] evaluator exited with $? at $(date); restarting in 60s" >&2
    sleep 60
  done
' >> runs/ratings/evaluator.log 2>&1 &
echo "evaluator started (log: runs/ratings/evaluator.log)"
