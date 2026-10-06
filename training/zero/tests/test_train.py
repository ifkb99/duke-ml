"""
train.py: the loss on real self-play samples (top-K targets with padding,
has_policy masking, draw-afterstate samples with no legal actions) and the
checkpoint save -> load -> resume round trip, including the optimizer state.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero import train as T
from training.zero.network import DukeZeroNet, make_forward


def _tiny():
    model = DukeZeroNet(8, 1, dtype=jnp.float32, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=1e-4))
    return graphdef, params, tx


@pytest.fixture(scope='module')
def samples():
    """Samples from real self-play (tiny net, few simulations, short games)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(E, 'MAX_PLY', 16)
    jax.clear_caches()
    try:
        graphdef, params, _ = _tiny()
        forward = make_forward(graphdef)
        selfplay = SP.make_selfplay(S.make_search(forward, 4, 4), unroll=24)
        states = jax.vmap(E.init)(jax.random.split(jax.random.key(0), 6))
        asm = RP.Assembler(6)
        parts = []
        for i in range(3):
            states, rec = selfplay(params, states, jax.random.key(i + 1))
            rec = jax.device_get(rec)
            rec.pop('search_value')
            out = asm.add(rec)
            if out is not None:
                parts.append(out)
    finally:
        mp.undo()
        jax.clear_caches()
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert (~out['has_policy']).any(), 'want draw afterstate samples'
    return out


def test_policy_targets_are_legal(samples):
    tile, owner, side, bags, pending = (jnp.asarray(samples[k], jnp.int32)
                                        for k in ('tile', 'owner', 'side', 'bags', 'pending'))
    legal, _, _ = jax.vmap(E.legal_actions)(tile, owner, side, bags, pending)
    legal = np.asarray(legal)
    pol = samples['has_policy']
    idx, p = samples['pol_idx'].astype(int), samples['pol_p'].astype(np.float32)
    rows = np.arange(len(idx))[:, None]
    assert (legal[rows, idx] | (p == 0))[pol].all(), 'policy mass on an illegal action'
    np.testing.assert_allclose(p[pol].sum(-1), 1.0, atol=2e-3)
    # Draw afterstates: no legal actions at all.
    assert not legal[~pol].any()
    assert (samples['pending'][~pol] == E.DRAWING).all()


def test_training_input_matches_search_input(random_states):
    """The learner rebuilds network inputs from compact int8/int16 samples;
    they must be exactly what the search fed the network for that state."""
    states = jax.tree.map(lambda *xs: jnp.stack(xs), *random_states[:64])
    c = SP.compact_state(states)
    p1, f1 = jax.vmap(E.observe)(states)
    p2, f2 = jax.vmap(E.observe_arrays)(c['tile'], c['owner'], c['side'], c['bags'], c['pending'], c['ply'])
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))
    np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))


def _manual_losses(forward, params, b):
    tile, owner, side, bags, pending = (jnp.asarray(b[k], jnp.int32)
                                        for k in ('tile', 'owner', 'side', 'bags', 'pending'))
    planes, feats = jax.vmap(E.observe_arrays)(tile, owner, side, bags, pending, jnp.asarray(b['ply']))
    logits, value = forward(params, planes, feats)
    legal, _, _ = jax.vmap(E.legal_actions)(tile, owner, side, bags, pending)
    logits, legal, value = np.asarray(logits, np.float64), np.asarray(legal), np.asarray(value, np.float64)
    pl = []
    for i in range(len(logits)):
        if not b['has_policy'][i]:
            continue
        lg = logits[i][legal[i]]
        lse = np.log(np.exp(lg - lg.max()).sum()) + lg.max()
        target = np.zeros(R.N_ACTIONS)
        np.add.at(target, b['pol_idx'][i].astype(int), b['pol_p'][i].astype(np.float64))
        pl.append(-(target[legal[i]] * (logits[i][legal[i]] - lse)).sum())
    return np.mean(pl), np.mean((value - b['z']) ** 2)


def test_loss_matches_manual_and_masks(samples):
    graphdef, params, tx = _tiny()
    forward = make_forward(graphdef)
    # Make the policy head non-trivial so the check means something.
    params = jax.tree.map(lambda x: x + 0.05 * jax.random.normal(jax.random.key(1), x.shape), params)
    step = T.make_train_step(forward, tx)
    b = {k: v[:256] for k, v in samples.items()}
    opt_state = tx.init(params)
    _, _, m = step(params, opt_state, b)
    pl, vl = _manual_losses(forward, params, b)
    np.testing.assert_allclose(float(m['policy_loss']), pl, rtol=1e-4)
    np.testing.assert_allclose(float(m['value_loss']), vl, rtol=1e-4)
    assert np.isfinite(float(m['grad_norm']))

    # Policy targets of has_policy=False samples don't matter at all.
    b2 = dict(b)
    junk = np.random.default_rng(0)
    nopol = ~b['has_policy']
    b2['pol_idx'] = b['pol_idx'].copy()
    b2['pol_idx'][nopol] = junk.integers(0, R.N_ACTIONS, (nopol.sum(), RP.POLICY_K))
    b2['pol_p'] = b['pol_p'].copy()
    b2['pol_p'][nopol] = 1.0 / RP.POLICY_K
    p1, _, m1 = step(params, opt_state, b)
    p2, _, m2 = step(params, opt_state, b2)
    assert float(m1['policy_loss']) == float(m2['policy_loss'])
    # GPU reductions use atomics, so even identical steps can differ in the
    # last bits (~1e-8); a real leak from junk targets would be far larger.
    for x, y in zip(jax.tree.leaves(p1), jax.tree.leaves(p2)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-7)


def test_one_hot_targets_with_padding():
    """Imitation targets: one action, 31 zero-probability (index 0) pads."""
    oh = SP.one_hot_policy(jnp.array([R.DRAW_ACTION, 5]))
    assert oh['pol_idx'].shape == (2, RP.POLICY_K)
    target = np.zeros((2, R.N_ACTIONS))
    np.add.at(target, (np.arange(2)[:, None], np.asarray(oh['pol_idx'], int)),
              np.asarray(oh['pol_p'], np.float32))
    assert target[0, R.DRAW_ACTION] == 1 and target[1, 5] == 1 and target.sum() == 2


def test_training_reduces_loss(samples):
    graphdef, params, tx = _tiny()
    step = T.make_train_step(make_forward(graphdef), tx)
    opt_state = tx.init(params)
    b = {k: v[:128] for k, v in samples.items()}
    _, _, m0 = step(params, opt_state, b)
    for _ in range(60):
        params, opt_state, m = step(params, opt_state, b)
    assert float(m['policy_loss']) < float(m0['policy_loss'])
    assert float(m['value_loss']) < float(m0['value_loss'])


def test_checkpoint_round_trip(tmp_path, samples):
    graphdef, params, tx = _tiny()
    forward = make_forward(graphdef)
    step = T.make_train_step(forward, tx)
    opt_state = tx.init(params)
    b = {k: v[:64] for k, v in samples.items()}
    for _ in range(3):
        params, opt_state, _ = step(params, opt_state, b)

    cfg = T.Config(channels=8, blocks=1)
    path = tmp_path / 'latest.pkl'
    T.save_checkpoint(path, params, opt_state, 17, cfg)
    assert not path.with_suffix('.tmp').exists()

    # Resume exactly as train.main does: graphdef from a fresh model of the
    # configured size, params and opt_state from the checkpoint.
    _, loaded, ckpt = T.load_params(path)
    assert ckpt['iteration'] == 17 and ckpt['config']['channels'] == 8
    for x, y in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    resumed_opt = jax.device_put(ckpt['opt_state'])
    assert jax.tree.structure(resumed_opt) == jax.tree.structure(opt_state)

    fresh_graphdef, _ = nnx.split(DukeZeroNet(8, 1, dtype=jnp.float32, rngs=nnx.Rngs(123)))
    step2 = T.make_train_step(make_forward(fresh_graphdef), tx)
    a_params, a_opt, a_m = step(params, opt_state, b)
    b_params, b_opt, b_m = step2(jax.device_put(loaded), resumed_opt, b)
    for k in a_m:
        np.testing.assert_allclose(float(a_m[k]), float(b_m[k]), rtol=1e-6)
    for x, y in zip(jax.tree.leaves((a_params, a_opt)), jax.tree.leaves((b_params, b_opt))):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-7)


def test_scanned_train_steps_match_sequential(samples):
    """make_train_steps (several steps in one call) == the same steps one by one."""
    graphdef, params, tx = _tiny()
    forward = make_forward(graphdef)
    one, many = T.make_train_step(forward, tx), T.make_train_steps(forward, tx)
    batches = T.sample_stacked(_replay_of(samples), 3, 64, np.random.default_rng(0))
    p1, o1, losses = params, tx.init(params), []
    for i in range(3):
        p1, o1, m = one(p1, o1, {k: v[i] for k, v in batches.items()})
        losses.append(float(m['policy_loss']))
    p3, o3, m3 = many(params, tx.init(params), batches)
    for x, y in zip(jax.tree.leaves(p1), jax.tree.leaves(p3)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
    assert np.isclose(float(m3['policy_loss']), np.mean(losses), rtol=1e-5)


def _replay_of(samples):
    replay = RP.Replay(len(samples['z']))
    replay.add(samples)
    return replay
