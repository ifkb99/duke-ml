"""distill.copy_from_checkpoint: a deeper student starts as the trained network."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from training.zero import env as E
from training.zero.distill import copy_from_checkpoint
from training.zero.network import DukeZeroNet, make_forward, observe_batch


def _trained(blocks, seed=1):
    """A small network with every weight random, so no block is the identity."""
    model = DukeZeroNet(8, blocks, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(seed))
    _, state = nnx.split(model)
    keys = jax.random.split(jax.random.key(seed), len(jax.tree.leaves(state)))
    leaves = [0.1 * jax.random.normal(k, x.shape) for k, x in zip(keys, jax.tree.leaves(state))]
    return nnx.to_pure_dict(jax.tree.unflatten(jax.tree.structure(state), leaves))


def _outputs(blocks, pure, states):
    graphdef, params = nnx.split(DukeZeroNet(8, blocks, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(7)))
    nnx.replace_by_pure_dict(params, pure)
    planes, feats = observe_batch(states)
    logits, value = make_forward(graphdef)(params, planes, feats)
    return np.asarray(logits), np.asarray(value)


def _states(n=8):
    # a few plies into some games, so the boards differ
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(0), n))
    for i in range(6):
        k1, k2 = jax.random.split(jax.random.key(i + 1))
        k1, k2 = jax.random.split(k1, n), jax.random.split(k2, n)
        actions = jax.vmap(lambda s, k: jax.random.choice(k, s.legal.shape[0], p=s.legal / s.legal.sum()))(states, k1)
        states = jax.vmap(E.step)(states, actions, k2)
    return states


def test_deeper_student_starts_as_the_checkpoint():
    trained, states = _trained(2), _states()
    _, student = nnx.split(DukeZeroNet(8, 3, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(3)))
    deeper = copy_from_checkpoint(student, trained)
    logits, value = _outputs(3, deeper, states)
    want_logits, want_value = _outputs(2, trained, states)
    np.testing.assert_allclose(value, want_value, atol=1e-6)
    np.testing.assert_allclose(logits, want_logits, atol=1e-5)
    # the extra block is fresh, not a copy, and can learn: its first conv is random
    assert np.any(deeper['blocks'][2]['conv1']['kernel'])


def test_same_depth_copies_everything():
    trained = _trained(2)
    _, student = nnx.split(DukeZeroNet(8, 2, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(3)))
    copied = copy_from_checkpoint(student, trained)
    jax.tree.map(np.testing.assert_array_equal, copied, trained)


def test_shallower_student_is_refused():
    _, student = nnx.split(DukeZeroNet(8, 1, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(3)))
    with pytest.raises(AssertionError):
        copy_from_checkpoint(student, _trained(2))
