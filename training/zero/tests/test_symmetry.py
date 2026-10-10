"""Left-right mirror symmetry (symmetry.py): the game really is symmetric, and
the action table maps every move to its mirror image."""
import jax
import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R
from training.zero import symmetry as Y
from training.zero.tests.conftest import stack

_step_all = jax.jit(jax.vmap(E.step_decision, (None, 0)))


def mirror(s: E.State) -> E.State:
    sq = jnp.asarray(Y.SQ)
    return E._with_legal(s._replace(tile=s.tile[sq], owner=s.owner[sq], side=s.side[sq]))


def test_tables_are_involutions():
    assert sorted(Y.ACTION.tolist()) == list(range(R.N_ACTIONS))
    np.testing.assert_array_equal(Y.ACTION[Y.ACTION], np.arange(R.N_ACTIONS))
    np.testing.assert_array_equal(Y.SQ[Y.SQ], np.arange(R.N_SQ))
    assert Y.ACTION[R.DRAW_ACTION] == R.DRAW_ACTION


def test_mirrored_positions_have_the_mirrored_legal_moves(random_states):
    flipped = 0
    for s in random_states:
        legal = np.asarray(s.legal)
        m = mirror(s)
        np.testing.assert_array_equal(np.asarray(m.legal)[Y.ACTION], legal)
        np.testing.assert_array_equal(np.asarray(m.place_ok)[Y.DIR4], np.asarray(s.place_ok))
        flipped += (Y.ACTION[np.flatnonzero(legal)] != np.flatnonzero(legal)).any()
    assert flipped > len(random_states) // 2      # not vacuous: most positions aren't symmetric


def test_mirrored_moves_lead_to_mirrored_positions(random_states):
    for s in random_states[::7]:
        legal = np.flatnonzero(np.asarray(s.legal))
        after = _step_all(s, jnp.asarray(legal))
        after_m = _step_all(mirror(s), jnp.asarray(Y.ACTION[legal]))
        for k in ('tile', 'owner', 'side'):
            np.testing.assert_array_equal(np.asarray(getattr(after, k))[:, Y.SQ], np.asarray(getattr(after_m, k)), k)
        for k in ('bags', 'pending', 'ply', 'done', 'winner', 'reward', 'switched', 'player'):
            np.testing.assert_array_equal(np.asarray(getattr(after, k)), np.asarray(getattr(after_m, k)), k)
        np.testing.assert_array_equal(np.asarray(after_m.legal)[:, Y.ACTION], np.asarray(after.legal))


def test_mirror_samples(random_states):
    states = stack(random_states[:64])
    n = 64
    rng = np.random.default_rng(0)
    b = {k: np.asarray(getattr(states, k)).astype(np.int8) for k in ('tile', 'owner', 'side')}
    b['pol_idx'] = rng.integers(0, R.N_ACTIONS, (n, 8)).astype(np.int16)
    b['z'] = rng.random(n).astype(np.float32)
    assert Y.mirror_samples(b, rng, 0.0) is b
    once = Y.mirror_samples(b, np.random.default_rng(1), 1.0)
    np.testing.assert_array_equal(once['tile'], b['tile'][:, Y.SQ])
    np.testing.assert_array_equal(once['pol_idx'], Y.ACTION[b['pol_idx']])
    twice = Y.mirror_samples(once, np.random.default_rng(2), 1.0)
    for k in b:
        np.testing.assert_array_equal(twice[k], b[k])
    half = Y.mirror_samples(b, np.random.default_rng(3), 0.5)
    flipped = (half['pol_idx'] != b['pol_idx']).any(-1)
    assert 16 < flipped.sum() < 48
    np.testing.assert_array_equal(half['z'], b['z'])
