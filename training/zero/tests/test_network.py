"""The spatial policy head puts every logit where rules.py numbers the action."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from training.zero import rules as R
from training.zero.network import N_CMD, DukeZeroNet

C = 8


def _spatial(seed=0):
    model = DukeZeroNet(C, 1, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(seed))
    rng = np.random.default_rng(seed)
    for layer in (model.policy_sq, model.policy_q, model.policy_k, model.policy_misc):
        layer.kernel[...] = jnp.asarray(rng.normal(size=layer.kernel.shape), jnp.float32)
        layer.bias[...] = jnp.asarray(rng.normal(size=layer.bias.shape), jnp.float32)
    return model


def _changed(model, square):
    """Actions whose logit moves when only `square`'s features change."""
    pooled = jnp.ones((1, C))
    x0 = jnp.zeros((1, R.BOARD, R.BOARD, C))
    x1 = x0.reshape(1, R.N_SQ, C).at[0, square].set(1.0).reshape(x0.shape)
    diff = model.policy_logits(x1, pooled) - model.policy_logits(x0, pooled)
    return set(np.flatnonzero(np.abs(np.asarray(diff[0])) > 1e-6).tolist())


def _moves_from(f):
    return {f * R.N_SQ + t for t in range(R.N_SQ)}


def _moves_to(t):
    return {f * R.N_SQ + t for f in range(R.N_SQ)}


def _commands_by(f):
    return {R.CMD_BASE + f * N_CMD + i for i in range(N_CMD)}


def test_per_square_logits_follow_the_action_numbering():
    model = _spatial()
    for layer in (model.policy_q, model.policy_k):           # no bilinear term
        layer.kernel[...] = 0.0
    for f in (0, 7, 20, 35):
        assert _changed(model, f) == _moves_from(f) | _commands_by(f)


def test_bilinear_term_reads_the_target_square():
    model = _spatial()
    model.policy_sq.kernel[...] = 0.0
    model.policy_q.kernel[...] = 0.0  # q = its bias everywhere
    for t in (0, 13, 35):
        assert _changed(model, t) == _moves_to(t)


def test_draw_and_placements_come_from_the_pooled_board():
    model = _spatial()
    x = jnp.zeros((1, R.BOARD, R.BOARD, C))
    a = model.policy_logits(x, jnp.zeros((1, C)))
    b = model.policy_logits(x, jnp.ones((1, C)))
    changed = np.flatnonzero(np.abs(np.asarray(b - a)[0]) > 1e-6)
    assert changed.tolist() == list(range(R.DRAW_ACTION, R.N_ACTIONS))


def test_spatial_network_starts_uniform_and_is_small():
    model = DukeZeroNet(128, 10, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(0))
    planes = jax.random.uniform(jax.random.key(1), (3, R.BOARD, R.BOARD, 60))
    logits, value = model(planes, jnp.ones((3, 47)))
    assert logits.shape == (3, R.N_ACTIONS) and value.shape == (3,)
    np.testing.assert_array_equal(np.asarray(logits), 0)      # uniform prior at init

    def size(m):
        return sum(np.size(x) for x in jax.tree.leaves(nnx.state(m, nnx.Param)))
    dense = DukeZeroNet(128, 10, dtype=jnp.float32, rngs=nnx.Rngs(0))
    head = size(model) - (size(dense) - size(dense.policy_conv) - size(dense.policy_out))
    assert head < 30_000 and size(dense.policy_out) > 4_000_000
