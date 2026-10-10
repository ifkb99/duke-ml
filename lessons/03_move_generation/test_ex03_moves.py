import jax
import jax.numpy as jnp
import numpy as np

from conftest import load_exercise, random_states
from training.zero import env as E
from training.zero import rules as R

ex = load_exercise('ex03_moves')


def test_shape():
    s = E.init(jax.random.key(0))
    out = ex.pseudo_moves(s.tile, s.owner, s.side)
    assert out.shape == (36, 36) and out.dtype == jnp.bool_


def test_duke_slides_are_blocked():
    # My Duke (side A slides left/right) on (5,2) with my Footman on (5,4):
    # it can reach (5,0), (5,1), (5,3) but not (5,4) or (5,5).
    tile = jnp.full(36, -1, jnp.int32).at[32].set(R.DUKE).at[34].set(R.FOOTMAN)
    owner = jnp.zeros(36, jnp.int32)
    side = jnp.zeros(36, jnp.int32)
    out = np.asarray(ex.pseudo_moves(tile, owner, side))
    assert sorted(np.flatnonzero(out[32]).tolist()) == [30, 31, 33]


def test_matches_env_on_random_positions():
    states = random_states(128, plies=40)
    got = jax.vmap(ex.pseudo_moves)(states.tile, states.owner, states.side)
    want, _ = jax.vmap(E._pseudo_legal)(states.tile, states.owner, states.side)
    np.testing.assert_array_equal(np.asarray(got).reshape(128, -1), np.asarray(want))
