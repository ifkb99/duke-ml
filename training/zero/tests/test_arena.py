"""
arena.play bookkeeping, with a stand-in for the minimax pool that plays random
legal TS moves (draws resolved to `place` like minimax-server) and logs every
request. Checks that the env stays in sync with TS through draw replays, that
colors alternate, that results are counted from agent A's side, and that the
epsilon path (including a random draw mid-minimax) works.
"""

import jax
import numpy as np
import pytest

from training.zero import arena as AR
from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import ts_bridge as B
from training.zero.tests.conftest import TsOracle


class FakePool:
    def __init__(self, oracle, seed=0, draw_bias=0.5):
        self.oracle = oracle
        self.rng = np.random.default_rng(seed)
        self.draw_bias = draw_bias
        self.log = []             # (state, move, next) in TS JSON

    def best_moves(self, states, depth):
        out = []
        for st in states:
            assert st['status'] == 'active'
            legal = self.oracle.ask({'state': st})['legal']
            assert legal, 'asked for a move in a finished game'
            draws = [m for m in legal if m['type'] == 'draw']
            m = draws[self.rng.integers(len(draws))] if draws and self.rng.random() < self.draw_bias \
                else legal[self.rng.integers(len(legal))]
            if m['type'] == 'draw':
                bag = st['bags'][st['currentPlayer']]
                m = {'type': 'place', 'tileName': bag[self.rng.integers(len(bag))], 'position': m['position']}
            nxt = self.oracle.ask({'state': st, 'move': m})['next']
            self.log.append((st, m, nxt))
            out.append(m)
        return out


@pytest.fixture(scope='module')
def oracle():
    o = TsOracle()
    yield o
    o.close()


def _key(ts_state):
    a = B.state_arrays_from_ts(ts_state)
    return tuple(np.asarray(a[k]).tobytes() for k in ('tile', 'owner', 'side', 'bags', 'player', 'ply'))


def test_minimax_vs_minimax_stays_in_sync_with_ts(oracle, short_games):
    short_games(60)
    pool = FakePool(oracle, seed=1)
    a = AR.Agent('minimax', depth=1)
    b = AR.Agent('minimax', depth=1)
    res, steps = AR.play(a, b, 6, jax.random.key(0), pool=pool, record=True)
    assert res['wins'] + res['losses'] + res['draws'] == 6

    # Every position minimax is asked about is exactly what TS reaches by
    # applying the previous minimax move of that game (both sides are minimax).
    expected = {_key(nxt) for _, _, nxt in pool.log}
    n_places = 0
    for st, m, _ in pool.log:
        if st['turnNumber'] > 1:
            assert _key(st) in expected, 'env and TS diverged'
        n_places += m['type'] == 'place'
    assert n_places > 5

    # Recorded steps: draws are replayed as draw -> place with one-hot targets.
    chunk = AR.stack_steps(steps)
    samples = RP.Assembler(6).add(chunk)
    assert samples['has_policy'].sum() == chunk['valid'].sum()
    assert chunk['is_draw'].sum() == n_places


def test_colors_alternate_and_results_are_from_a(oracle, short_games):
    short_games(40)
    pool = FakePool(oracle, seed=2, draw_bias=0.2)
    n = 8
    res, steps = AR.play(AR.Agent('minimax', depth=1), AR.Agent('random'), n,
                         jax.random.key(3), pool=pool, record=True)
    chunk = AR.stack_steps(steps)
    # Step 0: P1 to move everywhere; A (minimax, the only agent with policy
    # targets here) plays P1 in even games.
    np.testing.assert_array_equal(chunk['has_policy'][0], np.arange(n) % 2 == 0)

    wins = losses = draws = 0
    for g in range(n):
        last = np.flatnonzero(chunk['valid'][:, g])[-1]
        assert chunk['done'][last, g]
        r = chunk['reward'][last, g]
        a_moved = chunk['has_policy'][last, g]
        if r == 0:
            draws += 1
        elif (r > 0) == a_moved:
            wins += 1
        else:
            losses += 1
    assert (res['wins'], res['losses'], res['draws']) == (wins, losses, draws)
    assert res['score'] == (wins + 0.5 * draws) / n


def test_epsilon_path(oracle, short_games):
    short_games(60)
    pool = FakePool(oracle, seed=4, draw_bias=0.3)
    a = AR.Agent('minimax', depth=[1, 2, 1, 2, 1, 2], epsilon=0.5)
    b = AR.Agent('minimax', depth=1, epsilon=0.5)
    res, steps = AR.play(a, b, 6, jax.random.key(5), pool=pool, record=True)
    chunk = AR.stack_steps(steps)
    valid = chunk['valid']
    act = chunk['pol_idx'][..., 0].astype(int)
    # Some epsilon moves happened (no policy target), including random draws
    # whose placement was then also random.
    eps = valid & ~chunk['has_policy']
    assert eps.any()
    rand_draw = eps[:-1] & (act[:-1] == R.DRAW_ACTION)
    assert rand_draw.any()
    assert (~chunk['has_policy'][1:][rand_draw]).all()
    assert (chunk['pending'][1:][rand_draw] >= 0).all()
    # A minimax draw is always followed by its queued placement.
    mm_draw = valid[:-1] & chunk['has_policy'][:-1] & (act[:-1] == R.DRAW_ACTION)
    assert mm_draw.any()
    assert (chunk['has_policy'][1:][mm_draw]).all()
    assert (act[1:][mm_draw] >= R.PLACE_BASE).all()
    assert RP.Assembler(6).add(chunk) is not None
