import jax
import jax.numpy as jnp
import numpy as np

from conftest import load_exercise, random_states
from training.zero import env as E

ex = load_exercise('ex02_vmap_scan')


def test_material_single():
    tile = jnp.full(36, -1, jnp.int32).at[jnp.array([0, 1, 2])].set(1)
    owner = jnp.zeros(36, jnp.int32).at[2].set(1)
    assert int(ex.material(tile, owner)) == 1          # two mine, one theirs


def test_material_batch_matches_loop():
    states = random_states(16, plies=20)
    batched = np.asarray(ex.material_batch(states.tile, states.owner))
    assert batched.shape == (16,)
    for i in range(16):
        assert batched[i] == int(ex.material(states.tile[i], states.owner[i]))


def test_material_starts_even():
    s = jax.vmap(E.init)(jax.random.split(jax.random.key(0), 8))
    np.testing.assert_array_equal(ex.material_batch(s.tile, s.owner), 0)


def test_playout_is_jittable_and_vmappable():
    s = jax.vmap(E.init)(jax.random.split(jax.random.key(1), 32))
    play = jax.jit(jax.vmap(lambda k, st: ex.random_playout(k, st, 300)))
    final, moves = play(jax.random.split(jax.random.key(2), 32), s)
    assert moves.shape == (32,)
    # 300 random steps is plenty for almost every game to end.
    assert int(final.done.sum()) >= 28
    # A game can't have made more real moves than steps, and finished games stopped counting.
    assert int(moves.max()) <= 300
    assert np.all(np.asarray(moves)[np.asarray(~final.done)] == 300)


def test_playout_counts_moves():
    s = E.init(jax.random.key(3))
    final, moves = ex.random_playout(jax.random.key(4), s, 5)
    assert int(moves) == 5                      # no game ends in 5 steps from the start
    assert not bool(final.done)
