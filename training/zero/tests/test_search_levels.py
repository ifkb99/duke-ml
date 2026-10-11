"""Grouped ("semantic if") selection: search.LEVELS.

With no levels the grouped code must be the flat Gumbel search exactly, so any
difference in play between 'flat' and a grouping comes from the grouping alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.zero import rules as R
from training.zero import search as S
from training.zero.tests.conftest import stack
from training.zero.tests.test_search_internals import make_random_forward


def _states(random_states, n=16):
    # fresh decisions only (no tile in hand), so drawing competes with board moves
    fresh = [s for s in random_states if int(s.pending) == -1 and bool(s.legal[R.DRAW_ACTION])]
    return stack(fresh[::max(1, len(fresh) // n)][:n])


@pytest.mark.parametrize('gumbel_scale', [0.0, 1.0])
def test_no_levels_is_the_flat_search(random_states, monkeypatch, gumbel_scale):
    monkeypatch.setitem(S.LEVELS, 'none', ())
    forward, states = make_random_forward(), _states(random_states)
    flat = jax.jit(S.make_search(forward, 24))(None, jax.random.key(1), states, gumbel_scale)
    none = jax.jit(S.make_search(forward, 24, levels='none'))(None, jax.random.key(1), states, gumbel_scale)
    np.testing.assert_array_equal(none.action, flat.action)
    np.testing.assert_allclose(none.action_weights, flat.action_weights, atol=1e-6)
    np.testing.assert_allclose(none.search_value, flat.search_value, atol=1e-6)


@pytest.mark.parametrize('levels', ['piece', 'draw', 'draw-piece', 'draw-piece/best/pi'])
def test_levels_play_legal_moves(random_states, levels):
    forward, states = make_random_forward(), _states(random_states)
    out = jax.jit(S.make_search(forward, 24, levels=levels))(None, jax.random.key(1), states, 1.0)
    assert np.asarray(states.legal)[np.arange(len(out.action)), np.asarray(out.action)].all()


def test_parts_combine_their_actions():
    # actions 0..2 in part 0, 3 in part 1; action 2 is masked out
    table = jnp.array([0, 0, 0, 1])
    mask = jnp.array([True, True, False, True])
    logits = jnp.log(jnp.array([0.2, 0.3, 0.4, 0.1]))
    visits = jnp.array([1, 3, 5, 0])
    q = jnp.array([0.5, -0.5, 9.0, 0.0])
    valid, part_logits, part_n, part_q = S._parts(table, 2, mask, logits, visits, q)
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_allclose(jax.nn.softmax(part_logits), [0.5 / 0.6, 0.1 / 0.6], rtol=1e-5)
    np.testing.assert_array_equal(part_n, [4, 0])
    np.testing.assert_allclose(part_q[0], (0.5 - 3 * 0.5) / 4)
