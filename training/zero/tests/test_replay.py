"""
replay.Assembler: value targets must equal the final result from the point of
view of the player to move, however games are split across chunks and slots.

The oracle is independent of the assembler's recursion: we log the absolute
player at every step and the winner of every game, and tag each record with a
unique id (smuggled through pol_idx[0], which the assembler passes through).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import selfplay as SP


def _random_actions(key, legal, draw_bias=0.3):
    k1, k2 = jax.random.split(key)
    logits = jnp.where(legal, 0.0, -jnp.inf)
    a = jax.random.categorical(k1, logits)
    draw = legal[:, R.DRAW_ACTION] & (jax.random.uniform(k2, a.shape) < draw_bias)
    return jnp.where(draw, R.DRAW_ACTION, a).astype(jnp.int32)


def _play(short_games, batch, steps, reset, max_ply=40, seed=0):
    """Random games in [steps] x [batch] records, plus an oracle per record."""
    short_games(max_ply)                             # get timeouts too

    def step(s, key):
        k1, k2, k3 = jax.random.split(key, 3)
        actions = _random_actions(k1, s.legal)
        after = jax.vmap(lambda s, a: E.step_decision(s, a))(s, actions)
        outcome = jax.random.categorical(k2, jnp.log(jax.vmap(E.chance_probs)(after)))
        nxt = jax.vmap(E.step_chance)(after, outcome)
        rec = {**SP.compact_state(s), **SP.compact_policy(jnp.ones((batch, R.N_ACTIONS)) / R.N_ACTIONS),
               'has_policy': jnp.ones(batch, bool), **SP.step_record(s, after, actions)}
        info = {'player': s.player, 'done_after': after.done, 'winner': after.winner}
        if reset:
            nxt = SP._reset_done(nxt, k3)
        return nxt, rec, info

    step = jax.jit(step)
    s = jax.jit(jax.vmap(lambda k: E.init(k)))(jax.random.split(jax.random.key(seed), batch))
    key = jax.random.key(seed + 1)
    recs, infos = [], []
    for _ in range(steps):
        key, k = jax.random.split(key)
        s, rec, info = step(s, k)
        recs.append(jax.device_get(rec))
        infos.append(jax.device_get(info))
    chunk = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
    info = {k: np.stack([r[k] for r in infos]) for k in infos[0]}

    # Tag every record and compute its expected z from the oracle.
    T, B = chunk['done'].shape
    ids = np.arange(T * B).reshape(T, B)
    chunk['pol_idx'][:, :, 0] = ids
    expected = {}
    for b in range(B):
        game = []
        for t in range(T):
            if not chunk['valid'][t, b]:
                continue
            game.append(t)
            if info['done_after'][t, b]:
                w = info['winner'][t, b]
                for u in game:
                    p = info['player'][u, b]
                    expected[ids[u, b]] = 0.0 if w < 0 else (1.0 if p == w else -1.0)
                game = []
    return chunk, expected, info


def _split(chunk, rng, n_parts):
    T = chunk['done'].shape[0]
    cuts = np.sort(rng.choice(np.arange(1, T), n_parts - 1, replace=False))
    bounds = [0, *cuts, T]
    return [{k: v[a:b] for k, v in chunk.items()} for a, b in zip(bounds[:-1], bounds[1:])]


def _check(samples, expected, chunk):
    ids = samples['pol_idx'][:, 0].astype(np.int64)
    pol = samples['has_policy']
    # Every finished game's record appears exactly once with a policy...
    assert sorted(ids[pol].tolist()) == sorted(expected)
    for i, z in zip(ids[pol], samples['z'][pol]):
        assert z == expected[i], (i, z, expected[i])
    # ...and every draw once more as a value-only afterstate sample.
    flat_draw = chunk['is_draw'].reshape(-1)
    draw_ids = sorted(i for i in expected if flat_draw[i])
    assert sorted(ids[~pol].tolist()) == draw_ids
    assert (samples['pending'][~pol] == E.DRAWING).all()
    for i, z in zip(ids[~pol], samples['z'][~pol]):
        assert z == expected[i]


@pytest.mark.parametrize('n_parts', [1, 3, 17])
def test_assembler_selfplay_style(short_games, n_parts):
    """Slots restart finished games in place: several games per slot per chunk,
    games spanning several chunks."""
    chunk, expected, info = _play(short_games, batch=6, steps=240, reset=True)
    winners = info['winner'][info['done_after']]
    assert (winners == 0).any() and (winners == 1).any() and (winners == -1).any(), 'want all outcomes'
    asm = RP.Assembler(6)
    parts = [asm.add(c) for c in _split(chunk, np.random.default_rng(n_parts), n_parts)]
    parts = [p for p in parts if p is not None]
    samples = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    _check(samples, expected, chunk)
    # Unfinished tails are kept for the next chunk, not emitted.
    n_tail = sum(len(p['done']) for slot in asm.partial for p in slot)
    assert n_tail + len(expected) == chunk['valid'].sum()


def test_assembler_arena_style(short_games):
    """No restarts: finished games keep producing invalid rows until all end."""
    chunk, expected, _ = _play(short_games, batch=5, steps=90, reset=False)
    assert (~chunk['valid']).any()
    asm = RP.Assembler(5)
    samples = asm.add(chunk)
    _check(samples, expected, chunk)


def test_z_hand_example():
    """P1 move, P2 move, P1 draw, P1 place, P2 wins: z from each mover's view."""
    T = 5
    chunk = {k: np.zeros((T, 1) + shape, dtype) for k, (shape, dtype) in RP.SAMPLE_SPEC.items() if k not in RP.TARGET_KEYS}
    chunk['has_policy'][:] = True
    chunk['reward'] = np.array([[0], [0], [0], [0], [1]], np.float32)
    chunk['discount'] = np.array([[-1], [-1], [1], [-1], [0]], np.float32)
    chunk['done'] = np.array([[0], [0], [0], [0], [1]], bool)
    chunk['valid'] = np.ones((T, 1), bool)
    chunk['is_draw'] = np.array([[0], [0], [1], [0], [0]], bool)
    out = RP.Assembler(1).add(chunk)
    np.testing.assert_array_equal(out['z'], [-1, 1, -1, -1, 1, -1])
    np.testing.assert_array_equal(out['has_policy'], [1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(out['pending'], [0, 0, 0, 0, 0, E.DRAWING])


def test_replay_ring_buffer():
    rp = RP.Replay(10)
    def batch(n, start):
        d = {k: np.zeros((n,) + shape, dtype) for k, (shape, dtype) in RP.SAMPLE_SPEC.items()}
        d['z'] = np.arange(start, start + n, dtype=np.float32)
        return d
    rp.add(batch(7, 0))
    assert rp.size == 7
    rp.add(batch(7, 7))
    assert rp.size == 10 and rp.cursor == 4
    assert sorted(rp.data['z'].tolist()) == list(range(4, 14))
    s = rp.sample(1000, np.random.default_rng(0))
    assert set(s['z'].tolist()) == set(range(4, 14))


def _hand_chunk(search_value):
    """The game of test_z_hand_example, with search values per step."""
    T = 5
    chunk = {k: np.zeros((T, 1) + shape, dtype) for k, (shape, dtype) in RP.SAMPLE_SPEC.items()
             if k not in RP.TARGET_KEYS}
    chunk['has_policy'][:] = True
    chunk['reward'] = np.array([[0], [0], [0], [0], [1]], np.float32)
    chunk['discount'] = np.array([[-1], [-1], [1], [-1], [0]], np.float32)
    chunk['done'] = np.array([[0], [0], [0], [0], [1]], bool)
    chunk['valid'] = np.ones((T, 1), bool)
    chunk['is_draw'] = np.array([[0], [0], [1], [0], [0]], bool)
    chunk['search_value'] = np.asarray(search_value, np.float32)[:, None]
    return chunk


def test_nstep_hand_example():
    """n=2: z_t = r_t + d_t r_{t+1} + d_t d_{t+1} v_{t+2}, signed for the mover at t."""
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = RP.Assembler(1, nstep=2).add(_hand_chunk(v))
    # t=0: d0 d1 = (+1) v2 = 0.3   t=1: d1 d2 = -1 -> -v3 = -0.4
    # t=2: d2 d3 = -1 -> -v4 = -0.5
    # t=3: r3 + d3 r4 = -1 (game ends inside the window, no bootstrap)
    # t=4: r4 = 1
    np.testing.assert_allclose(out['z'], [0.3, -0.4, -0.5, -1, 1, -0.5], atol=1e-6)
    # The real result is kept alongside, and the draw afterstate copies its step.
    np.testing.assert_array_equal(out['z_mc'], [-1, 1, -1, -1, 1, -1])


def test_nstep_long_window_is_the_game_result():
    out = RP.Assembler(1, nstep=50).add(_hand_chunk([0.9] * 5))
    np.testing.assert_array_equal(out['z'], out['z_mc'])


def test_nstep_needs_search_values():
    """Without search values (e.g. minimax games) targets fall back to results."""
    chunk = _hand_chunk([0.9] * 5)
    chunk.pop('search_value')
    out = RP.Assembler(1, nstep=2).add(chunk)
    np.testing.assert_array_equal(out['z'], out['z_mc'])


def test_nstep_matches_brute_force():
    rng = np.random.default_rng(0)
    T, n = 40, 7
    r = np.zeros(T, np.float32); r[-1] = rng.choice([-1, 1])
    d = rng.choice([-1.0, 1.0], T).astype(np.float32); d[-1] = 0
    v = rng.uniform(-1, 1, T).astype(np.float32)
    z = RP.nstep_targets(r, d, v, n)
    for t in range(T):
        acc, sign = 0.0, 1.0
        for k in range(n):
            if t + k >= T:
                sign = 0.0
                break
            acc += sign * r[t + k]
            sign *= d[t + k]
        if t + n < T:
            acc += sign * v[t + n]
        assert abs(z[t] - acc) < 1e-6, t
