"""
Turning played games into training samples, and storing them.

Self-play runs B games side by side for T steps at a time, so we receive
chunks of shape [T, B]. Games end at different times (and new ones start in
the same slot), so `Assembler` stitches each slot's fragments together until a
game finishes, then computes the value target for every position:

    z_t = reward_t + discount_t * z_{t+1}

where reward_t is what the mover at t got from that step (+1 for a win), and
discount_t is -1 when the turn passes to the opponent (their good is my bad),
+1 when the same player moves again (draw -> place), and 0 at the end.
So z_t is the final result from the point of view of the player to move at t.

With `nstep > 0` the value target is instead bootstrapped (as in MuZero):
the rewards of the next n steps plus the search's value estimate n steps
later, all signed for the player to move at t. Far fewer random events lie
between a position and its target, so the target is much less noisy than
the final result (which may hinge on a blunder 80 moves later). The final
result is kept as `z_mc` for diagnostics.

Samples are stored compactly (~300 bytes): the raw board instead of the 60
one-hot planes (the network input is rebuilt on the GPU), and the policy
target as its top-K entries instead of a dense 3605-vector.
"""

import numpy as np

from training.zero import env as E

POLICY_K = 32

SAMPLE_SPEC = {
    'tile': ((36,), np.int8),
    'owner': ((36,), np.int8),
    'side': ((36,), np.int8),
    'bags': ((2, 15), np.int8),
    'pending': ((), np.int8),
    'ply': ((), np.int16),
    'pol_idx': ((POLICY_K,), np.int16),
    'pol_p': ((POLICY_K,), np.float16),
    'has_policy': ((), np.bool_),
    'z': ((), np.float32),         # value target (n-step bootstrapped, or z_mc)
    'z_mc': ((), np.float32),      # final game result for the player to move
}
TARGET_KEYS = ('z', 'z_mc')
STATE_KEYS = ('tile', 'owner', 'side', 'bags', 'pending', 'ply')
# Per-step bookkeeping that the assembler consumes but doesn't store.
STEP_KEYS = ('reward', 'discount', 'done', 'valid', 'is_draw')
# Optional per-step key: 'search_value' (root value after search, for the
# player to move), needed for n-step targets.


class Assembler:
    """Collects [T, B] step chunks; emits finished games as flat samples."""

    def __init__(self, batch: int, nstep: int = 0):
        self.partial: list[list[dict]] = [[] for _ in range(batch)]
        self.nstep = nstep

    def add(self, chunk: dict) -> dict | None:
        """chunk: dict of arrays [T, B, ...] with SAMPLE_SPEC keys except the
        targets, plus STEP_KEYS (and 'search_value' for n-step targets).
        Returns samples from games that finished, or None."""
        T, B = chunk['done'].shape
        finished = []
        for b in range(B):
            col = {k: v[:, b] for k, v in chunk.items()}
            valid = col['valid']
            if not valid.any():
                continue
            col = {k: v[valid] for k, v in col.items()}
            start = 0
            for end in np.flatnonzero(col['done']):
                self.partial[b].append({k: v[start:end + 1] for k, v in col.items()})
                finished.append(_concat(self.partial[b]))
                self.partial[b] = []
                start = end + 1
            if start < len(col['done']):
                self.partial[b].append({k: v[start:] for k, v in col.items()})
        if not finished:
            return None
        return _concat([_game_samples(g, self.nstep) for g in finished])

    def drop_partial(self):
        """Forget unfinished games (e.g. at the end of a fixed-size match)."""
        self.partial = [[] for _ in self.partial]


def _concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def nstep_targets(r, d, v, n):
    """z_t = r_t + d_t r_{t+1} + ... + (d_t ... d_{t+n-1}) v_{t+n}.

    d carries the sign between consecutive movers (and is 0 at the end), so
    the product turns rewards and the bootstrap value into the point of view
    of the player to move at t. Past the end of the game nothing is added."""
    T = len(r)
    t = np.arange(T)
    z = np.zeros(T, np.float32)
    sign = np.ones(T, np.float32)
    for k in range(n):
        idx = t + k
        ok = idx < T
        z[ok] += sign[ok] * r[idx[ok]]
        sign = np.where(ok, sign * d[np.minimum(idx, T - 1)], 0.0)
    idx = t + n
    ok = idx < T
    z[ok] += sign[ok] * v[idx[ok]]
    return z


def _game_samples(game: dict, nstep: int = 0) -> dict:
    """Value targets for one finished game, plus afterstate samples for draws."""
    r, d = game['reward'], game['discount']
    z = np.zeros(len(r), np.float32)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + d[t] * acc
        z[t] = acc
    out = {k: game[k] for k in SAMPLE_SPEC if k not in TARGET_KEYS}
    out['z_mc'] = z
    if nstep > 0 and 'search_value' in game:
        out['z'] = nstep_targets(r, d, game['search_value'], nstep)
    else:
        out['z'] = z.copy()

    # After choosing to draw, the same player is still to move and nothing has
    # been revealed yet: that afterstate is worth the same z. Adding it as a
    # value-only sample teaches the value head to evaluate draw afterstates,
    # which the search queries (see search.py).
    draws = game['is_draw']
    if draws.any():
        after = {k: v[draws].copy() for k, v in out.items()}
        after['pending'][:] = E.DRAWING
        after['has_policy'][:] = False
        out = _concat([out, after])
    return out


class Replay:
    """Fixed-capacity ring buffer of samples, uniformly sampled."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = {k: np.zeros((capacity,) + shape, dtype) for k, (shape, dtype) in SAMPLE_SPEC.items()}
        self.size = 0
        self.cursor = 0
        self.total_added = 0

    def add(self, samples: dict):
        n = len(samples['z'])
        idx = (self.cursor + np.arange(n)) % self.capacity
        for k in SAMPLE_SPEC:
            self.data[k][idx] = samples[k]
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n

    def sample(self, batch: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, batch)
        return {k: v[idx] for k, v in self.data.items()}
