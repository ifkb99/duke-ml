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


def test_spatial_head_trains_and_round_trips(tmp_path, samples):
    graphdef, params = nnx.split(DukeZeroNet(8, 1, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(0)))
    _, _, tx = _tiny()
    step = T.make_train_step(make_forward(graphdef), tx)
    opt_state = tx.init(params)
    b = {k: v[:128] for k, v in samples.items()}
    _, _, m0 = step(params, opt_state, b)
    for _ in range(60):
        params, opt_state, m = step(params, opt_state, b)
    assert float(m['policy_loss']) < float(m0['policy_loss'])

    path = tmp_path / 'latest.pkl'
    T.save_checkpoint(path, params, opt_state, 3, T.Config(channels=8, blocks=1, policy_head='spatial'))
    loaded_graphdef, loaded, ckpt = T.load_params(path)
    assert ckpt['config']['policy_head'] == 'spatial'
    for x, y in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


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
    """make_train_steps (several steps in one call) == the same steps one by one.

    Runs on CPU: GPU reductions are nondeterministic in the last bits, and
    Adam's division by sqrt(v) amplifies that for near-zero gradients."""
    with jax.default_device(jax.devices('cpu')[0]):
        graphdef, params, tx = _tiny()
        forward = make_forward(graphdef)
        one, many = T.make_train_step(forward, tx), T.make_train_steps(forward, tx)
        batches = T.sample_stacked(_replay_of(samples), 3, 64, np.random.default_rng(0))
        batches = jax.device_put(batches)
        p1, o1, losses = params, tx.init(params), []
        for i in range(3):
            p1, o1, m = one(p1, o1, {k: v[i] for k, v in batches.items()})
            losses.append(float(m['policy_loss']))
        p3, o3, m3 = many(params, tx.init(params), batches)
    for x, y in zip(jax.tree.leaves(p1), jax.tree.leaves(p3)):
        assert x.devices() == {jax.devices('cpu')[0]}
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
    assert np.isclose(float(m3['policy_loss']), np.mean(losses), rtol=1e-5)


def _replay_of(samples):
    replay = RP.Replay(len(samples['z']))
    replay.add(samples)
    return replay


def test_fresh_diagnostics(samples):
    graphdef, params, _ = _tiny()
    forward = make_forward(graphdef)
    params = jax.tree.map(lambda x: x + 0.05 * jax.random.normal(jax.random.key(2), x.shape), params)
    b = {k: v[:256] for k, v in samples.items()}
    d = {k: float(v) for k, v in T.make_diagnostics(forward)(params, b).items()}
    _, m = T.make_loss_fn(forward)(params, b)
    np.testing.assert_allclose(d['fresh_policy_ce'], float(m['policy_loss']), rtol=1e-4)
    assert d['fresh_policy_kl'] >= -1e-4
    assert d['fresh_target_entropy'] >= 0 and d['fresh_prior_entropy'] >= 0
    assert -1 <= d['fresh_value_corr_mc'] <= 1
    # Without n-step targets the target is the game result itself.
    np.testing.assert_allclose(d['fresh_value_mse'], d['fresh_value_mse_mc'], rtol=1e-5)
    np.testing.assert_allclose(d['fresh_mc_baseline'], np.mean(b['z_mc'] ** 2), rtol=1e-5)


def test_game_ids(samples):
    gid = T.game_ids(samples)
    assert gid[0] == 0 and (np.diff(gid) >= 0).all()
    firsts = np.flatnonzero(np.diff(gid, prepend=-1))
    assert (samples['ply'][firsts] == 0).all()


def test_imitation_early_stopping(samples):
    graphdef, params, tx = _tiny()
    forward = make_forward(graphdef)
    cfg = T.Config(batch=64, steps_per_call=4, imitation_steps=400, imitation_check_every=8,
                   imitation_patience=2, imitation_holdout=0.3)
    logged = []
    log = lambda msg, **rec: logged.append(rec)
    loss_fn = T.make_loss_fn(forward)
    p, o = T.imitate(cfg, samples, params, tx.init(params), T.make_train_steps(forward, tx),
                     jax.jit(lambda p, b: loss_fn(p, b)[1]), jax.devices()[0],
                     np.random.default_rng(0), log)
    checks = [r for r in logged if r.get('kind') == 'imitation']
    assert checks and all('heldout_value_loss' in r for r in checks)
    # Kept parameters are those of the best held-out check.
    assert jax.tree.structure(p) == jax.tree.structure(params)


def test_imitation_lr_drops(samples):
    """On a plateau: back to the best step with the lr cut, then stop once the
    drops are used up. Held-out losses are scripted; params count the steps."""
    _, params, _ = _tiny()
    cfg = T.Config(batch=64, steps_per_call=4, imitation_steps=400, imitation_check_every=4,
                   imitation_patience=2, imitation_holdout=0.3, lr=1e-3,
                   imitation_lr_drops=2, imitation_lr_factor=0.5)
    tx = T.make_optimizer(cfg)
    held = iter([1.0, 0.9, 0.95, 0.95,          # best at check 2 (step 8), then a plateau
                 0.8, 0.85, 0.85,               # after drop 1: best at step 20, plateau
                 0.9, 0.9])                     # after drop 2: no gain, stop
    lrs, steps_in = [], []
    leaf0 = jax.tree.leaves(params)[0]

    def train_steps(p, o, batches):
        lrs.append(float(o[1].hyperparams['learning_rate']))
        steps_in.append(int(np.round(float((jax.tree.leaves(p)[0] - leaf0).ravel()[0]))) * 4)
        return jax.tree.map(lambda x: x + 1, p), o, {'policy_loss': jnp.zeros(()), 'value_loss': jnp.zeros(())}

    loss = lambda p, b: {'policy_loss': next(held), 'value_loss': 0.0}
    logged = []
    T.imitate(cfg, samples, params, tx.init(params), train_steps, loss, jax.devices()[0],
              np.random.default_rng(0), lambda msg, **rec: logged.append((msg, rec)))
    drops = [rec for _, rec in logged if rec.get('kind') == 'imitation_lr_drop']
    assert [d['back_to'] for d in drops] == [8, 20]
    assert [d['lr'] for d in drops] == pytest.approx([5e-4, 2.5e-4])
    assert lrs == pytest.approx([1e-3] * 4 + [5e-4] * 3 + [2.5e-4] * 2)   # float32 in the state
    # SGD steps in the params each call starts from (4 per call): after a drop,
    # training resumes from the best check's params. The step counter runs on,
    # so the best check after drop 1 is "step 20" with 12 steps in its params.
    assert steps_in == [0, 4, 8, 12, 8, 12, 16, 12, 16]
    assert logged[-1][0].startswith('imitation: keeping step 20')
    with pytest.raises(StopIteration):
        next(held)


# ---------------------------------------------------------------------------
# Learning-rate schedule and playout cap randomization
# ---------------------------------------------------------------------------

def test_learning_rate_schedule():
    cfg = T.Config(lr=1e-3, lr_min=1e-4, lr_decay_start=100, lr_decay_iterations=200)
    assert T.learning_rate(cfg, 50) == 1e-3
    assert T.learning_rate(cfg, 100) == pytest.approx(1e-3)
    assert T.learning_rate(cfg, 200) == pytest.approx(5.5e-4)      # halfway down the cosine
    assert T.learning_rate(cfg, 300) == pytest.approx(1e-4)
    assert T.learning_rate(cfg, 10_000) == pytest.approx(1e-4)
    assert T.learning_rate(T.Config(lr=2e-4), 10_000) == 2e-4        # no decay configured


def test_scheduled_lr_is_used_and_old_states_upgrade(samples):
    graphdef, params, old_tx = _tiny()
    forward = make_forward(graphdef)
    b = {k: v[:64] for k, v in samples.items()}
    # An optimizer state saved before the schedule existed (plain AdamW)...
    old_state = old_tx.init(params)
    for _ in range(2):
        params, old_state, _ = T.make_train_step(forward, old_tx)(params, old_state, b)
    cfg = T.Config(lr=1e-3, weight_decay=1e-4)
    tx = T.make_optimizer(cfg)
    state = T.upgrade_opt_state(jax.device_get(old_state), tx, params)
    assert jax.tree.structure(state) == jax.tree.structure(tx.init(params))
    # ...keeps its moments and step count, so the next step matches the old optimizer's.
    step = T.make_train_step(forward, tx)
    want, _, _ = T.make_train_step(forward, old_tx)(params, old_state, b)
    got, state, _ = step(params, T.with_learning_rate(state, 1e-3), b)
    for x, y in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-7)
    # A zero learning rate (weight decay is scaled by it too) leaves params unchanged.
    frozen, _, _ = step(got, T.with_learning_rate(state, 0.0), b)
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(frozen)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_chunked_search_matches_whole_batch(random_states):
    from training.zero.tests.conftest import stack
    graphdef, params, _ = _tiny()
    search = S.make_search(make_forward(graphdef), 8, 4, 8)
    states = stack(random_states[:8])
    whole = jax.jit(search)(params, jax.random.key(0), states, 0.0)
    parts = jax.jit(S.chunked(search, 4))(params, jax.random.key(0), states, 0.0)
    np.testing.assert_array_equal(np.asarray(whole.action), np.asarray(parts.action))
    np.testing.assert_allclose(np.asarray(whole.action_weights), np.asarray(parts.action_weights), atol=1e-6)


def test_fast_selfplay_gives_no_policy_targets():
    graphdef, params, _ = _tiny()
    search = S.make_search(make_forward(graphdef), 4, 4)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(0), 4))
    _, rec = SP.make_selfplay(search, 3, policy_targets=False)(params, states, jax.random.key(1))
    assert not np.asarray(rec['has_policy']).any()
    _, rec = SP.make_selfplay(search, 3)(params, states, jax.random.key(1))
    assert np.asarray(rec['has_policy']).all()
