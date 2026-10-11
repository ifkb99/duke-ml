"""semantic.rule_features (tensor algebra over the input planes) agrees with
the rules engine (env.py) on every square of random positions."""

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R
from training.zero import semantic as M
from training.zero.network import observe_batch
from training.zero.tests.conftest import stack


def _engine(s):
    """The same features the slow way, one square at a time."""
    occ = s.tile >= 0
    attacked = lambda tile, owner, side, t: E._is_attacked(tile, side, (tile >= 0) & (owner == 1), tile >= 0, t)
    opp = jax.vmap(lambda t: attacked(s.tile, s.owner, s.side, t))(E.SQ)
    ft, fo, fs, _ = E._flip_perspective(s.tile, s.owner, s.side, s.bags)
    mine = jax.vmap(lambda t: attacked(ft, fo, fs, E.FLIP[t]))(E.SQ)
    move_strike, commands = E._pseudo_legal(s.tile, s.owner, s.side)
    n_moves = move_strike.reshape(R.N_SQ, R.N_SQ).sum(-1) + commands.reshape(R.N_SQ, 64).sum(-1)
    my_duke = E._duke_square(s.tile, s.owner)
    opp_duke = jnp.argmax((s.tile == R.DUKE) & (s.owner == 1) & occ)
    return opp, mine, n_moves, opp[my_duke], mine[opp_duke]


def _flipped(s):
    """The same board from the other side's point of view."""
    tile, owner, side, bags = E._flip_perspective(s.tile, s.owner, s.side, s.bags)
    return E._with_legal(s._replace(tile=tile, owner=owner, side=side, bags=bags))


def test_rule_features_match_the_engine(random_states):
    # Random play rarely leaves the Duke en prise for the side to move: add
    # every position from the other side's view (a check becomes a chance to take it).
    states = stack(random_states)
    states = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), states, jax.jit(jax.vmap(_flipped))(states))
    n = states.tile.shape[0]
    planes, features = M.rule_features(observe_batch(states)[0])
    planes = np.asarray(planes).reshape(n, R.N_SQ, M.N_PLANES)
    opp, mine, n_moves, check, take = (np.asarray(x) for x in jax.jit(jax.vmap(_engine))(states))
    np.testing.assert_array_equal(planes[..., 0], opp)
    np.testing.assert_array_equal(planes[..., 1], mine)
    np.testing.assert_allclose(planes[..., 2] * 8, n_moves, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(features), np.stack([check, take], -1))
    # the positions exercise every feature
    assert opp.any() and mine.any() and check.any() and take.any() and (n_moves > 0).any()


def _net(**kw):
    from flax import nnx
    from training.zero.network import DukeZeroNet
    return nnx.split(DukeZeroNet(8, 2, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(1), **kw))


def test_semantic_inputs_start_ignored(random_states):
    # A student with semantic inputs, started from a plain network, plays exactly as it does.
    from flax import nnx
    from training.zero.distill import copy_from_checkpoint
    from training.zero.network import make_forward
    from training.zero.tests.test_distill import _trained
    plain_gd, plain = _net()
    nnx.replace_by_pure_dict(plain, _trained(2))
    gd, student = _net(semantic_inputs=True, semantic_targets=True)
    nnx.replace_by_pure_dict(student, copy_from_checkpoint(student, nnx.to_pure_dict(plain)))
    obs = observe_batch(stack(random_states[:32]))
    want = make_forward(plain_gd)(plain, *obs)
    got = make_forward(gd)(student, *obs)
    np.testing.assert_allclose(got[0], want[0], atol=1e-5)
    np.testing.assert_allclose(got[1], want[1], atol=1e-6)


def test_semantic_targets_train(random_states):
    import optax
    from training.zero import selfplay as SP
    from training.zero import train as T
    from training.zero.network import make_forward_aux
    gd, params = _net(semantic_targets=True)
    states = stack(random_states[:64])
    batch = {**SP.compact_state(states), 'pol_idx': jnp.zeros((64, 32), jnp.int16),
             'pol_p': jnp.zeros((64, 32)).at[:, 0].set(1.0), 'has_policy': jnp.zeros(64, bool),
             'z': jnp.zeros(64), 'z_mc': jnp.zeros(64)}
    loss = T.make_loss_fn(make_forward_aux(gd), aux_weight=1.0)
    tx = optax.adam(1e-2)
    opt = tx.init(params)
    first = None
    for _ in range(30):
        (_, m), g = jax.value_and_grad(loss, has_aux=True)(params, batch)
        first = first if first is not None else float(m['semantic_loss'])
        updates, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, updates)
    assert float(m['semantic_loss']) < 0.8 * first
