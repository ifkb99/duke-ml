import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import load_exercise, random_states
from training.zero import env as E

ex = load_exercise('ex01_board')


@pytest.fixture(scope='module')
def states():
    return random_states(64, plies=25)


def test_flip_matches_env(states):
    got = jax.vmap(ex.flip_perspective)(states.tile, states.owner, states.side, states.bags)
    want = jax.vmap(E._flip_perspective)(states.tile, states.owner, states.side, states.bags)
    for g, w, name in zip(got, want, ('tile', 'owner', 'side', 'bags')):
        np.testing.assert_array_equal(g, w, err_msg=f'{name} differs')


def test_flip_twice_is_identity(states):
    s = jax.tree.map(lambda x: x[0], states)
    once = ex.flip_perspective(s.tile, s.owner, s.side, s.bags)
    twice = ex.flip_perspective(*once)
    for a, b in zip(twice, (s.tile, s.owner, s.side, s.bags)):
        np.testing.assert_array_equal(a, b)


def test_planes_shape_and_dtype(states):
    s = jax.tree.map(lambda x: x[0], states)
    planes = ex.board_planes(s.tile, s.owner, s.side)
    assert planes.shape == (6, 6, 60)
    assert planes.dtype == jnp.float32


def test_planes_hand_example():
    tile = jnp.full(36, -1, jnp.int32).at[33].set(0).at[2].set(1)   # my Duke at (5,3), their Footman at (0,2)
    owner = jnp.zeros(36, jnp.int32).at[2].set(1)
    side = jnp.zeros(36, jnp.int32).at[2].set(1)                   # their Footman on side B
    planes = ex.board_planes(tile, owner, side)
    assert planes[5, 3, 0] == 1.0                                  # me, Duke(0), side A
    assert planes[0, 2, 30 + 1 * 2 + 1] == 1.0                     # opponent, Footman(1), side B
    assert planes.sum() == 2.0                                     # nothing else is set


def test_planes_match_env(states):
    got = jax.vmap(ex.board_planes)(states.tile, states.owner, states.side)
    want, _ = jax.vmap(E.observe)(states)
    np.testing.assert_array_equal(got, want)
