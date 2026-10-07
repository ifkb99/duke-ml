"""MuZero pieces: action encoding, warm start, trajectories, unrolled loss, search."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from training.zero import env as E
from training.zero import muzero as MZ
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero import train as T
from training.zero.network import DukeZeroNet, make_forward, observe_batch


def _tiny_mz(seed=0):
    return nnx.split(MZ.MuZeroNet(8, 1, 'spatial', 1, dtype=jnp.float32, rngs=nnx.Rngs(seed)))


def test_action_encoding():
    planes, feats = MZ._action_tables()
    planes = planes.reshape(R.N_ACTIONS, R.N_SQ, 3)
    a = 7 * R.N_SQ + 20
    assert planes[a, 7, 0] == 1 and planes[a, 20, 1] == 1 and planes[a].sum() == 2
    cmd, sd, dd = 14, 1, 4                       # commander 14: from north to east
    a = R.CMD_BASE + cmd * 64 + sd * 8 + dd
    assert planes[a, R.NBR8[cmd, sd], 0] == 1 and planes[a, R.NBR8[cmd, dd], 1] == 1 and planes[a, cmd, 2] == 1
    assert feats[R.DRAW_ACTION].tolist() == [1, 0, 0, 0, 0]
    assert feats[R.PLACE_BASE + 2].tolist() == [0, 0, 0, 1, 0]
    assert planes[R.DRAW_ACTION].sum() == 0 and feats[:R.DRAW_ACTION].sum() == 0


def test_warm_start_reproduces_alphazero():
    az = DukeZeroNet(8, 1, 'spatial', dtype=jnp.float32, rngs=nnx.Rngs(3))
    az_gd, az_params = nnx.split(az)
    gd, params = _tiny_mz()
    params = MZ.copy_alphazero(params, az_params)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(0), 4))
    want = make_forward(az_gd)(az_params, *observe_batch(states))
    got = make_forward(gd)(params, *observe_batch(states))
    for x, y in zip(want, got):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)


@pytest.fixture(scope='module')
def trajectories():
    """Steps of finished self-play games (tiny MuZero network searching its own model)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(E, 'MAX_PLY', 12)
    jax.clear_caches()
    try:
        gd, params = _tiny_mz()
        sp = SP.make_selfplay(MZ.make_search(gd, 4, 4), unroll=16)
        states = jax.vmap(E.init)(jax.random.split(jax.random.key(0), 8))
        asm, parts = MZ.TrajectoryAssembler(8, nstep=3), []
        for i in range(3):
            states, rec = sp(params, states, jax.random.key(i + 1))
            out = asm.add(jax.device_get(rec))
            if out is not None:
                parts.append(out)
    finally:
        mp.undo()
        jax.clear_caches()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def test_trajectories_replay_through_the_rules(trajectories):
    """State t, the recorded action and draw outcome give exactly state t+1."""
    d = trajectories
    starts = np.flatnonzero(np.r_[True, d['remaining'][:-1] == 1])
    assert (d['remaining'][starts] >= 1).all() and len(starts) >= 4
    checked = 0
    for s0 in starts:
        n = int(d['remaining'][s0])
        assert d['remaining'][s0:s0 + n].tolist() == list(range(n, 0, -1))
        assert d['discount'][s0 + n - 1] == 0                       # the game ends there
        for t in range(s0, s0 + n - 1):
            arr = {k: jnp.asarray(d[k][t], jnp.int32) for k in ('tile', 'owner', 'side', 'bags', 'pending', 'ply')}
            st = E._with_legal(E._empty_state()._replace(**arr))
            after = E.step_decision(st, int(d['action'][t]))
            assert float(after.reward) == pytest.approx(float(d['reward'][t]))
            nxt = E.step_chance(after, int(d['outcome'][t]))
            for k in ('tile', 'owner', 'side', 'pending', 'ply'):
                np.testing.assert_array_equal(np.asarray(getattr(nxt, k)), d[k][t + 1], err_msg=f'step {t}: {k}')
            np.testing.assert_array_equal(np.asarray(nxt.bags), d['bags'][t + 1])
            checked += 1
    assert checked > 20


def test_unroll_windows(trajectories):
    # Positions repeat across games (every game starts alike), so mark each
    # step with a unique 'ply' to find a window's start.
    d = {**trajectories, 'ply': np.arange(len(trajectories['z']), dtype=np.int16)}
    replay = MZ.TrajectoryReplay(len(d['z']) + 7, unroll=4)
    replay.add(d)
    rng = np.random.default_rng(0)
    b = replay.sample(256, rng)
    remaining = np.asarray(b['u_valid']).sum(1)
    for i in range(256):
        t = int(b['ply'][i])
        n = min(int(d['remaining'][t]), 5)
        assert remaining[i] == n
        assert b['u_action'][i, :n].tolist() == d['action'][t:t + n].tolist()
        assert b['u_z'][i, :n].tolist() == pytest.approx(d['z'][t:t + n].tolist())
        assert (b['u_z'][i, n:] == 0).all() and (b['u_reward'][i, n:] == 0).all()
        assert not b['u_has_policy'][i, n:].any()


def test_unrolled_training_learns_the_model(trajectories):
    gd, params = _tiny_mz()
    tx = optax.adam(3e-3)
    replay = MZ.TrajectoryReplay(len(trajectories['z']), unroll=3)
    replay.add(trajectories)
    batch = jax.tree.map(jnp.asarray, replay.sample(128, np.random.default_rng(1)))
    step = T.make_train_step(None, tx, loss_fn=MZ.make_loss_fn(gd))
    opt = tx.init(params)
    _, _, m0 = step(params, opt, batch)
    for _ in range(80):
        params, opt, m = step(params, opt, batch)
    # (Short test games end by timeout, so rewards are all 0 and the zero-initialized
    # reward head starts exact: only losses that start above 0 can go down.)
    for k in ('policy_loss', 'value_loss', 'q_loss', 'reward_loss', 'chance_loss', 'policy_loss_unrolled'):
        assert float(m[k]) < float(m0[k]) or float(m0[k]) < 1e-6, k
    assert float(m0['chance_loss']) > 0.5 and float(m['chance_loss']) < float(m0['chance_loss'])
    assert float(m['legal_mass']) > float(m0['legal_mass'])


def test_search_plays_legal_moves_and_chunks():
    gd, params = _tiny_mz(2)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(5), 8))
    search = MZ.make_search(gd, 8, 4)
    out = jax.jit(search)(params, jax.random.key(0), states, 1.0)
    assert np.asarray(states.legal)[np.arange(8), np.asarray(out.action)].all()
    np.testing.assert_allclose(np.asarray(out.action_weights).sum(-1), 1, rtol=1e-5)
    assert not (np.asarray(out.action_weights) * ~np.asarray(states.legal)).any()
    whole = jax.jit(search)(params, jax.random.key(0), states, 0.0)
    parts = jax.jit(S.chunked(search, 2))(params, jax.random.key(0), states, 0.0)
    np.testing.assert_array_equal(np.asarray(whole.action), np.asarray(parts.action))


def test_checkpoint_round_trip(tmp_path):
    gd, params = _tiny_mz()
    cfg = T.Config(channels=8, blocks=1, policy_head='spatial', algo='muzero', dynamics_blocks=1)
    tx = T.make_optimizer(cfg)
    T.save_checkpoint(tmp_path / 'latest.pkl', params, tx.init(params), 3, cfg)
    _, loaded, ckpt = T.load_params(tmp_path / 'latest.pkl')
    assert ckpt['config']['algo'] == 'muzero'
    for x, y in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
