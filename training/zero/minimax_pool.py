"""
A pool of TypeScript minimax processes (packages/ai/src/zero/minimax-server.ts).

Minimax is CPU-bound and single-threaded per process, so we run one process
per core and fan requests out with a thread pool. Each thread owns exactly one
process, so no locking is needed beyond the pool's queue.

    with MinimaxPool() as pool:          # default: one process per CPU thread
        moves = pool.best_moves([ts_state, ...], depth=4)
"""

import json
import os
import queue
import subprocess
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

    def ask(self, state: dict, depth: int) -> dict:
        self.proc.stdin.write(json.dumps({'id': 0, 'state': state, 'depth': depth}) + '\n')
        reply = json.loads(self.proc.stdout.readline())
        return reply['move']

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


class MinimaxPool:
    def __init__(self, workers: int | None = None):
        n = workers or os.cpu_count() or 1
        with ThreadPoolExecutor(n) as ex:
            self._workers = list(ex.map(lambda _: _Worker(), range(n)))
        self._free: queue.Queue[_Worker] = queue.Queue()
        for w in self._workers:
            self._free.put(w)
        self._executor = ThreadPoolExecutor(n)

    def _ask(self, state, depth):
        w = self._free.get()
        try:
            return w.ask(state, depth)
        finally:
            self._free.put(w)

    def best_moves(self, states: list[dict], depth: int | list[int]) -> list[dict]:
        """Best move for each TS state, computed in parallel. `depth` may be per-state."""
        depths = depth if isinstance(depth, list) else [depth] * len(states)
        return list(self._executor.map(self._ask, states, depths))

    def close(self):
        self._executor.shutdown()
        for w in self._workers:
            w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
