"""The TypeScript port of the environment (packages/ai/src/zero/env.ts) agrees
with the JAX one: rule tables, legal moves, transitions and network input."""

import jax
import numpy as np
import pytest

from training.zero import env as E
from training.zero import rules as R
from training.zero.tests.conftest import TsOracle, random_game

FIELDS = ('tile', 'owner', 'side', 'bags', 'player', 'pending', 'ply', 'done', 'winner', 'switched')


@pytest.fixture(scope='module')
def oracle():
    o = TsOracle('training/zero/tests/ts_zero_oracle.ts')
    yield o
    o.close()


def to_json(s: E.State) -> dict:
    s = jax.device_get(s)
    return {
        'tile': s.tile.tolist(), 'owner': s.owner.tolist(), 'side': s.side.tolist(),
        'bags': s.bags.reshape(-1).tolist(), 'player': int(s.player), 'pending': int(s.pending),
        'ply': int(s.ply), 'done': bool(s.done), 'winner': int(s.winner), 'reward': float(s.reward),
        'switched': bool(s.switched), 'legal': np.flatnonzero(s.legal).tolist(),
        'placeOk': s.place_ok.astype(int).tolist(),
    }


def assert_same(ts: dict, s: E.State, what: str):
    want = to_json(s)
    for k in FIELDS + ('legal', 'placeOk'):
        assert ts[k] == want[k], f'{what}: {k} differs'
    assert ts['reward'] == pytest.approx(want['reward']), f'{what}: reward differs'


def test_tables_match(oracle):
    t = oracle.ask({'op': 'tables'})
    assert t['TILE_NAMES'] == R.TILE_NAMES
    assert t['INITIAL_BAG'] == R.INITIAL_BAG.tolist()
    for name in ('MOVE_OK', 'NEEDS_CLEAR', 'STRIKE_OK', 'CMD_DIR', 'BETWEEN'):
        assert t[name] == np.flatnonzero(getattr(R, name)).tolist(), name


_step = jax.jit(E.step_decision)
_chance = jax.jit(E.step_chance)
_observe = jax.jit(E.observe)


def test_random_games_match(oracle):
    rng = np.random.default_rng(0)
    n = 0
    for s, a, after, nxt in (step for seed in range(100, 112) for step in random_game(seed)):
        js = to_json(s)
        legal = oracle.ask({'op': 'legal', 'state': js})
        assert legal['legal'] == js['legal'], f'ply {int(s.ply)}: legal moves differ'
        assert legal['placeOk'] == js['placeOk']

        assert_same(oracle.ask({'op': 'step', 'state': js, 'action': a})['state'], after, 'afterstate')
        outcome = int(nxt.pending) if int(after.pending) == E.DRAWING and not bool(after.done) else 0
        ch = oracle.ask({'op': 'chance', 'state': to_json(after), 'outcome': outcome})
        assert_same(ch['state'], nxt, 'after chance')
        np.testing.assert_allclose(ch['probs'], np.asarray(E.chance_probs(after)), rtol=1e-6)

        obs = oracle.ask({'op': 'observe', 'state': js})
        planes, feats = _observe(s)
        np.testing.assert_array_equal(obs['planes'], np.asarray(planes).reshape(-1))
        np.testing.assert_allclose(obs['features'], np.asarray(feats), rtol=1e-6)

        # Now and then, an illegal action (the mover loses).
        if rng.random() < 0.05:
            bad = int(rng.choice(np.flatnonzero(~np.asarray(s.legal))))
            assert_same(oracle.ask({'op': 'step', 'state': js, 'action': bad})['state'],
                        _step(s, bad), f'illegal action {bad}')
        n += 1
    assert n > 300, n


def test_benchmark_positions_match(oracle):
    """Legal moves on positions from real games (the benchmark), which reach
    further into the game than random play does."""
    from training.zero import benchmark as BM
    if not BM.BENCH.exists():
        pytest.skip('no benchmark built')
    b = BM.load_benchmark()
    states = BM.states_from_arrays({k: b[k] for k in BM.STATE_KEYS})
    for i in range(0, len(b['ply']), 3):
        s = jax.tree.map(lambda x: x[i], states)
        js = to_json(s)
        assert oracle.ask({'op': 'legal', 'state': js})['legal'] == js['legal'], f'position {i}'


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _test_net(seed=0, hidden=8):
    rng = np.random.default_rng(seed)
    n_in = R.N_SQ * E.N_PLANES + E.N_FEATURES
    return {'W': rng.normal(0, 0.3, (n_in, hidden)).astype(np.float32),
            'U': rng.normal(0, 1.5, (hidden, R.N_ACTIONS)).astype(np.float32),
            'v': rng.normal(0, 1.0, hidden).astype(np.float32)}


def _test_forward(p, planes, features):
    import jax.numpy as jnp
    x = jnp.concatenate([planes.reshape(planes.shape[0], -1), features], -1)
    h = jnp.tanh(x @ p['W'])
    return h @ p['U'], jnp.tanh(h @ p['v'])


@pytest.fixture(scope='module')
def search_positions():
    """Decision states from random play, a few of them mid-draw (placing)."""
    out = []
    for seed in range(200, 204):
        for i, (s, _, _, _) in enumerate(random_game(seed)):
            if i % 7 == 3:
                out.append(s)
    return out


@pytest.mark.parametrize('sims,considered,depth,noise', [
    (32, 16, 16, False), (32, 16, 16, True), (64, 8, 4, True), (24, 4, 16, False)])
def test_search_matches_jax(oracle, search_positions, sims, considered, depth, noise):
    from training.zero import search as S
    from training.zero.tests.conftest import stack
    p = _test_net()
    oracle.ask({'op': 'net', **{k: v.tolist() for k, v in p.items()}})
    states = stack(search_positions)
    rng = jax.random.key(7)
    out = jax.jit(S.make_search(_test_forward, sims, considered, depth), static_argnums=3)(
        p, rng, states, 1.0 if noise else 0.0)
    gumbel = np.asarray(jax.random.gumbel(jax.random.split(rng)[1], (len(search_positions), R.N_ACTIONS)))
    for b, s in enumerate(search_positions):
        req = {'op': 'search', 'state': to_json(s), 'simulations': sims, 'maxConsidered': considered,
               'maxDepth': depth}
        if noise:
            req['gumbel'] = gumbel[b].tolist()
        ts = oracle.ask(req)
        legal = np.flatnonzero(np.asarray(s.legal))
        assert ts['action'] == int(out.action[b]), f'position {b}: action'
        np.testing.assert_allclose(ts['weights'], np.asarray(out.action_weights[b])[legal], atol=2e-4,
                                   err_msg=f'position {b}: improved policy')
        assert ts['searchValue'] == pytest.approx(float(out.search_value[b]), abs=2e-4), f'position {b}'
    assert len(search_positions) > 20


def test_batching_does_not_change_the_search(oracle, search_positions):
    """One network call per simulation and the batched search give the same tree."""
    p = _test_net(1)
    oracle.ask({'op': 'net', **{k: v.tolist() for k, v in p.items()}})
    fewer_calls = 0
    for s in search_positions[:12]:
        req = {'op': 'search', 'state': to_json(s), 'simulations': 48, 'maxConsidered': 16, 'maxDepth': 16}
        one, many = oracle.ask({**req, 'maxBatch': 1}), oracle.ask(req)
        assert one['action'] == many['action'] and one['visits'] == many['visits']
        np.testing.assert_allclose(one['weights'], many['weights'], rtol=1e-12)
        fewer_calls += many['batches'] < one['batches']
    assert fewer_calls >= 6
