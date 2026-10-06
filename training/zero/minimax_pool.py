"""
A pool of TypeScript minimax processes (packages/ai/src/zero/minimax-server.ts).

Minimax is CPU-bound and single-threaded per process, so we run one process
per core and fan requests out with a thread pool. Each thread owns exactly one
process, so no locking is needed beyond the pool's queue.

    with MinimaxPool() as pool:          # default: one process per usable CPU thread
        moves = pool.best_moves([ts_state, ...], depth=4)
"""

import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER = 'packages/ai/src/zero/minimax-server.ts'


class _Worker:
    def __init__(self):
        self.proc = subprocess.Popen(
            ['npx', 'tsx', SERVER], cwd=REPO, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        line = self.proc.stdout.readline().strip()
        if line != 'READY':
            raise RuntimeError(f'minimax server failed to start: {line!r}')

    def request(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps({'id': 0, **req}) + '\n')
        line = self.proc.stdout.readline()
        if not line:
            raise ConnectionError(f'minimax server exited (code {self.proc.wait()})')
        return json.loads(line)

    def ask(self, state: dict, depth: int) -> dict:
        return self.request({'state': state, 'depth': depth})['move']

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


class MinimaxPool:
    def __init__(self, workers: int | None = None):
        n = workers or len(os.sched_getaffinity(0))   # CPUs this process may use
        with ThreadPoolExecutor(n) as ex:
            self._workers = list(ex.map(lambda _: _Worker(), range(n)))
        self._free: queue.Queue[_Worker] = queue.Queue()
        for w in self._workers:
            self._free.put(w)
        self._executor = ThreadPoolExecutor(n)

    def _ask(self, state, depth):
        return self._call(lambda w: w.ask(state, depth))

    def _call(self, fn):
        w = self._free.get()
        try:
            return fn(w)
        except (ConnectionError, OSError, ValueError) as e:
            # A server died (or answered garbage): replace it and retry once,
            # so one bad process doesn't kill a long training run.
            print(f'[minimax_pool] {e}; restarting worker', file=sys.stderr, flush=True)
            w.close()
            w = _Worker()
            self._workers.append(w)
            return fn(w)
        finally:
            self._free.put(w)

    def best_moves(self, states: list[dict], depth: int | list[int]) -> list[dict]:
        """Best move for each TS state, computed in parallel. `depth` may be per-state."""
        depths = depth if isinstance(depth, list) else [depth] * len(states)
        return list(self._executor.map(self._ask, states, depths))

    def score_moves(self, states: list[dict], depth: int) -> list[dict]:
        """Every legal move's exact score for the player to move, per state:
        {'moves': [{'move', 'score'}], 'draw': score | None}."""
        req = lambda s: self._call(lambda w: w.request({'state': s, 'depth': depth, 'scoreMoves': True}))
        return list(self._executor.map(req, states))

    def close(self):
        self._executor.shutdown()
        for w in self._workers:
            w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
