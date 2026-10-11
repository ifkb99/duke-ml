"""The factored policy head (network.py): a move is decided in steps (draw or
not, which piece, where), and policy_logp turns the steps into one distribution."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from training.zero import env as E
from training.zero import rules as R
from training.zero import search as S
from training.zero import train as T
from training.zero.distill import copy_from_checkpoint
from training.zero.network import DukeZeroNet, make_forward, observe_batch, policy_logp
from training.zero.tests.conftest import stack

EXTRA = R.N_SQ + 1


def _fresh(random_states, n=32):
    fresh = [s for s in random_states if int(s.pending) == -1 and bool(s.legal[R.DRAW_ACTION])]
    return stack(fresh[::max(1, len(fresh) // n)][:n])


def test_flat_logits_are_a_plain_masked_softmax(random_states):
    states = stack(random_states[:16])
    logits = jax.random.normal(jax.random.key(0), (16, R.N_ACTIONS))
    np.testing.assert_allclose(policy_logp(logits, states.legal),
                               jax.nn.log_softmax(jnp.where(states.legal, logits, -1e9)), atol=1e-6)


def test_factored_is_a_distribution_over_legal_moves(random_states):
    states = stack(random_states)           # fresh decisions and placing a drawn tile
    logits = jax.random.normal(jax.random.key(1), (len(random_states), R.N_ACTIONS + EXTRA)) * 3
    p = np.exp(np.asarray(policy_logp(logits, states.legal)))
    legal = np.asarray(states.legal)
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-5)
    assert (p[~legal] == 0).all()


def test_factored_steps(random_states):
    states = _fresh(random_states)
    n = states.tile.shape[0]
    m = jax.random.normal(jax.random.key(2), (n, R.N_ACTIONS))
    piece = jax.random.normal(jax.random.key(3), (n, R.N_SQ))
    kind = jnp.full((n, 1), 0.7)
    p = np.exp(np.asarray(policy_logp(jnp.concatenate([m, piece, kind], -1), states.legal)))
    legal = np.asarray(states.legal)
    for i in range(n):
        # draw or not: sigmoid of the kind logit (both are legal here)
        np.testing.assert_allclose(p[i, R.DRAW_ACTION], 1 / (1 + np.exp(-0.7)), rtol=1e-5)
        # which piece: softmax of the piece logits over the pieces that can act
        acting = np.unique(R.PIECE_OF[np.flatnonzero(legal[i, :R.DRAW_ACTION])])
        per_piece = np.array([p[i, (R.PIECE_OF == g) & legal[i]].sum() for g in acting])
        want = np.exp(piece[i, acting]) / np.exp(piece[i, acting]).sum() * (1 - p[i, R.DRAW_ACTION])
        np.testing.assert_allclose(per_piece, want, rtol=1e-4)
        # where: the move logits, normalized within the piece
        g = acting[0]
        moves = np.flatnonzero((R.PIECE_OF == g) & legal[i])
        np.testing.assert_allclose(p[i, moves] / p[i, moves].sum(), jax.nn.softmax(m[i, moves]), rtol=1e-4)


def test_level_loss_plus_where_is_the_policy_loss(random_states):
    # chain rule: CE(move) = CE(draw or piece) + CE(where | piece), for any target
    states = _fresh(random_states, 8)
    logp = policy_logp(jax.random.normal(jax.random.key(4), (8, R.N_ACTIONS + EXTRA)), states.legal)
    target = jax.nn.softmax(jnp.where(states.legal, jax.random.normal(jax.random.key(5), (8, R.N_ACTIONS)), -1e9))
    ce = -jnp.sum(target * logp, -1)
    t_piece, p_piece = target @ T.PIECE_MEMBER, jnp.exp(logp) @ T.PIECE_MEMBER
    within = logp - jnp.log(jnp.maximum(p_piece @ T.PIECE_MEMBER.T, 1e-30))
    where = -jnp.sum(jnp.where(target > 0, target * within, 0.0), -1)
    np.testing.assert_allclose(T.level_loss(target, logp) + where, ce, rtol=1e-4)


def test_factored_net_searches_and_starts_from_a_spatial_checkpoint(random_states):
    states = _fresh(random_states, 8)
    _, spatial = nnx.split(DukeZeroNet(8, 1, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(1)))
    graphdef, params = nnx.split(DukeZeroNet(8, 1, 'factored', dtype=jnp.float32, rngs=nnx.Rngs(2)))
    nnx.replace_by_pure_dict(params, copy_from_checkpoint(params, nnx.to_pure_dict(spatial)))
    forward = make_forward(graphdef)
    logits, _ = forward(params, *observe_batch(states))
    assert logits.shape == (8, R.N_ACTIONS + EXTRA)
    # the draw and piece steps start uniform; the spatial head's logits carry over
    np.testing.assert_array_equal(logits[:, R.N_ACTIONS:], 0)
    out = jax.jit(S.make_search(forward, 16))(params, jax.random.key(0), states, 1.0)
    assert np.asarray(states.legal)[np.arange(8), np.asarray(out.action)].all()
    np.testing.assert_allclose(np.asarray(out.action_weights).sum(-1), 1, atol=1e-5)
