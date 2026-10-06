"""Shared helpers for the zero tests.

Most tests here are small unit tests that run fine on the CPU; set
JAX_PLATFORMS=cpu to keep them off a GPU that a training run is using.
"""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.zero import env as E
from training.zero import rules as R

REPO = Path(__file__).resolve().parents[3]


def ts_oracle(requests: list[dict]) -> list[dict]:
    """Ask the TS engine about states (see ts_oracle.ts), one reply per request."""
    proc = subprocess.run(
        ['npx', 'tsx', 'training/zero/tests/ts_oracle.ts'], cwd=REPO, check=True,
        input=''.join(json.dumps(r) + '\n' for r in requests), capture_output=True, text=True)
    return [json.loads(line) for line in proc.stdout.splitlines()]


class TsOracle:
    """A persistent ts_oracle.ts process for many small queries."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ['npx', 'tsx', 'training/zero/tests/ts_oracle.ts'], cwd=REPO, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def ask(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + '\n')
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)

_step_decision = jax.jit(E.step_decision)
_step_chance = jax.jit(E.step_chance)
_init = jax.jit(E.init)


def random_game(seed: int, max_steps: int = 400, draw_bias: float = 0.3):
    """Random play from a random setup. Yields (state, action, afterstate, next)
    for every decision, with draws resolved by sampling the bag."""
    rng = np.random.default_rng(seed)
    s = _init(jax.random.key(seed))
    for _ in range(max_steps):
        if bool(s.done):
            return
        legal = np.flatnonzero(np.asarray(s.legal))
        if legal[-1] == R.DRAW_ACTION and rng.random() < draw_bias:
            a = R.DRAW_ACTION
        else:
            a = int(rng.choice(legal))
        after = _step_decision(s, a)
        probs = np.asarray(E.chance_probs(after), np.float64)
        outcome = int(rng.choice(len(probs), p=probs / probs.sum()))
        nxt = _step_chance(after, outcome)
        yield s, a, after, nxt
        s = nxt


@pytest.fixture(scope='session')
def random_states():
    """A few hundred decision states (including placing states) from random play."""
    out = []
    for seed in range(12):
        for i, (s, _, _, _) in enumerate(random_game(seed)):
            if i % 3 == 0:
                out.append(s)
    return out


def stack(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


@pytest.fixture
def short_games(monkeypatch):
    """set(n): play to a turn limit of n (to get timeouts quickly). E.MAX_PLY is
    read at trace time, so compiled functions are flushed on both sides."""
    def set_max_ply(n):
        monkeypatch.setattr(E, 'MAX_PLY', n)
        jax.clear_caches()
    yield set_max_ply
    jax.clear_caches()


def make_state(pieces, bags=None, player=0, ply=0, pending=E.NONE) -> E.State:
    """A hand-built decision state in the canonical frame of the player to move.

    pieces: (row, col, tile name, owner 0=me/1=opponent, side 0=A/1=B).
    bags: [2, 15] counts [me, opponent]; default empty bags.
    """
    tile = np.full(R.N_SQ, -1, np.int32)
    owner = np.zeros(R.N_SQ, np.int32)
    side = np.zeros(R.N_SQ, np.int32)
    for r, c, name, o, sd in pieces:
        tile[r * R.BOARD + c] = R.TILE_INDEX[name]
        owner[r * R.BOARD + c] = o
        side[r * R.BOARD + c] = sd
    s = E._empty_state()._replace(
        tile=jnp.asarray(tile), owner=jnp.asarray(owner), side=jnp.asarray(side),
        bags=jnp.asarray(np.zeros((2, R.N_TYPES), np.int32) if bags is None else bags, jnp.int32),
        player=jnp.int32(player), ply=jnp.int32(ply), pending=jnp.int32(pending))
    return jax.jit(E._with_legal)(s)
