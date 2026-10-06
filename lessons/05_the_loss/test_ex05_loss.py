import jax
import jax.numpy as jnp
import numpy as np

from conftest import load_exercise

ex = load_exercise('ex05_loss')


def test_dense_target_scatter():
    idx = jnp.array([[2, 0, 0], [1, 3, 0]])
    p = jnp.array([[1.0, 0.0, 0.0], [0.75, 0.25, 0.0]])        # zeros are padding
    dense = np.asarray(ex.dense_target(idx, p, 4))
    np.testing.assert_allclose(dense, [[0, 0, 1, 0], [0, 0.75, 0, 0.25]])


def test_policy_loss_one_hot_is_negative_log_prob():
    logits = jnp.array([[1.0, 2.0, 3.0, 50.0]])
    legal = jnp.array([[True, True, True, False]])               # the 50.0 is illegal: ignored
    target = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    probs = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
    got = float(ex.policy_loss(logits, legal, target, jnp.array([True])))
    assert np.isclose(got, -np.log(probs[2]), atol=1e-5)


def test_policy_loss_ignores_samples_without_policy_and_has_no_nans():
    logits = jnp.array([[0.0, 0.0], [5.0, -5.0]])
    legal = jnp.array([[True, True], [False, False]])             # an afterstate: nothing legal
    target = jnp.array([[0.5, 0.5], [0.0, 0.0]])
    got = float(ex.policy_loss(logits, legal, target, jnp.array([True, False])))
    assert np.isclose(got, np.log(2), atol=1e-5)
    assert np.isfinite(float(ex.policy_loss(logits, legal, target, jnp.array([False, False]))))


def test_policy_loss_large_logits_stable():
    logits = jnp.array([[1000.0, 999.0]])
    got = float(ex.policy_loss(logits, jnp.array([[True, True]]), jnp.array([[0.0, 1.0]]), jnp.array([True])))
    assert np.isclose(got, np.log(1 + np.e), atol=1e-4)


def test_value_loss():
    assert np.isclose(float(ex.value_loss(jnp.array([0.5, -1.0]), jnp.array([1.0, -1.0]))), 0.125)


def test_gradient_points_toward_target():
    """One gradient step on the policy loss should raise the target move's probability."""
    logits = jnp.zeros((1, 5))
    legal = jnp.ones((1, 5), bool)
    target = jax.nn.one_hot(jnp.array([3]), 5)
    g = jax.grad(lambda l: ex.policy_loss(l, legal, target, jnp.array([True])))(logits)
    new = jax.nn.softmax(logits - 1.0 * g)
    assert float(new[0, 3]) > 0.2
