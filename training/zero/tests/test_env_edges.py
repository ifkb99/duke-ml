"""
Edge cases of the JAX environment that random-play differential tests rarely
reach: finished games, illegal actions, timeouts, stalemates, setups, the
network input encoding, chance probabilities and draw bookkeeping.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.zero import env as E
from training.zero import rules as R
from training.zero import ts_bridge as B
from training.zero.tests.conftest import make_state, random_game, stack, ts_oracle

step_decision = jax.jit(E.step_decision)
step_chance = jax.jit(E.step_chance)


def _tree_equal(a, b, skip=()):
    for name in a._fields:
        if name in skip:
            continue
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)),
                                      err_msg=name)


def _finished_states():
    out = []
    for seed in range(40):
        for s, a, after, nxt in random_game(seed, max_steps=600):
            if bool(nxt.done):
                out.append(nxt)
    return out


@pytest.fixture(scope='module')
def finished():
    states = _finished_states()
    assert len(states) >= 20
    return states


def test_stepping_finished_game_is_noop(finished):
    rng = np.random.default_rng(0)
    for s in finished[:20]:
        for a in (int(rng.integers(R.N_ACTIONS)), R.DRAW_ACTION, R.PLACE_BASE):
            nxt = step_decision(s, a)
            _tree_equal(s, nxt, skip=('reward', 'switched'))
            assert float(nxt.reward) == 0 and not bool(nxt.switched)
            ch = step_chance(nxt, 3)
            _tree_equal(nxt, ch)
            np.testing.assert_array_equal(E.chance_probs(nxt), np.eye(R.N_TYPES)[0])
        assert not np.asarray(s.legal).any()


def test_illegal_action_loses(random_states):
    rng = np.random.default_rng(1)
    for s in random_states[:60]:
        illegal = np.flatnonzero(~np.asarray(s.legal))
        a = int(rng.choice(illegal))
        nxt = step_decision(s, a)
        assert bool(nxt.done)
        assert int(nxt.winner) == 1 - int(s.player)
        assert float(nxt.reward) == -1.0
        assert not bool(nxt.switched)
        assert not np.asarray(nxt.legal).any()
        assert int(nxt.ply) == int(s.ply)


def test_timeout_at_max_ply():
    s = E.init(jax.random.key(0))
    s = s._replace(ply=jnp.int32(E.MAX_PLY - 1))
    legal = np.flatnonzero(np.asarray(s.legal))
    board = int(legal[0])
    assert board < R.DRAW_ACTION
    nxt = step_decision(s, board)
    assert bool(nxt.done) and int(nxt.winner) == -1 and float(nxt.reward) == 0.0
    assert int(nxt.ply) == E.MAX_PLY and not bool(nxt.switched)

    # A draw doesn't end the turn; the placement after it does.
    after = step_decision(s, R.DRAW_ACTION)
    assert not bool(after.done) and int(after.ply) == E.MAX_PLY - 1
    placing = step_chance(after, int(np.argmax(np.asarray(after.bags[0]) > 0)))
    place = int(np.flatnonzero(np.asarray(placing.legal))[0])
    end = step_decision(placing, place)
    assert bool(end.done) and int(end.winner) == -1 and float(end.reward) == 0.0

    # One turn earlier the game goes on.
    s2 = s._replace(ply=jnp.int32(E.MAX_PLY - 2))
    assert not bool(step_decision(s2, board).done)


# My Footman steps onto (0,1) next to the enemy Duke in the corner. Taking it
# would put that Duke next to my second Footman, so the enemy is stuck.
_STALEMATE = [
    (5, 5, 'Duke', 0, 0),
    (1, 1, 'Footman', 0, 0),
    (0, 2, 'Footman', 0, 0),
    (0, 0, 'Duke', 1, 0),
]
_STALEMATE_MOVE = (1 * 6 + 1) * R.N_SQ + (0 * 6 + 1)


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('opp_bag', [False, True])
def test_stalemate_matches_ts(player, opp_bag):
    bags = np.zeros((2, R.N_TYPES), np.int32)
    bags[0, R.TILE_INDEX['Pikeman']] = 1
    if opp_bag:            # the enemy can still draw next to its Duke
        bags[1, R.TILE_INDEX['Pikeman']] = 1
    s = make_state(_STALEMATE, bags=bags, player=player, ply=7)
    assert bool(s.legal[_STALEMATE_MOVE])
    nxt = step_decision(s, _STALEMATE_MOVE)

    move = B.action_to_move(_STALEMATE_MOVE, s)
    (reply,) = ts_oracle([{'state': B.state_to_ts(s), 'move': move}])
    ts_next = B.state_arrays_from_ts(reply['next'])
    assert bool(nxt.done) == bool(ts_next['done'])
    assert int(nxt.winner) == int(ts_next['winner'])
    if opp_bag:
        assert not bool(nxt.done)
        assert np.flatnonzero(np.asarray(nxt.legal)).tolist() == [R.DRAW_ACTION]
    else:
        assert bool(nxt.done) and int(nxt.winner) == player and float(nxt.reward) == 1.0
        assert bool(nxt.switched)


def test_init_setups_are_legal_and_cover_all():
    keys = jax.random.split(jax.random.key(0), 2000)
    s = jax.device_get(jax.jit(jax.vmap(E.init))(keys))
    seen = set()
    for i in range(len(keys)):
        tile, owner = s.tile[i], s.owner[i]
        assert s.player[i] == 0 and s.ply[i] == 0 and s.pending[i] == E.NONE and not s.done[i]
        np.testing.assert_array_equal(s.bags[i], [R.INITIAL_BAG, R.INITIAL_BAG])
        assert R.INITIAL_BAG.sum() == 16 and R.INITIAL_BAG[R.FOOTMAN] == 1
        assert (s.side[i] == 0).all()
        setup = []
        for who, home in ((0, 5), (1, 0)):
            mine = (tile >= 0) & (owner == who)
            assert mine.sum() == 3
            (duke,) = np.flatnonzero(mine & (tile == R.DUKE))
            assert duke // 6 == home and duke % 6 in R.DUKE_START_COLS
            fm = np.flatnonzero(mine & (tile == R.FOOTMAN))
            assert len(fm) == 2
            for f in fm:
                assert abs(f // 6 - duke // 6) + abs(f % 6 - duke % 6) == 1
            setup.append((duke, *fm))
        seen.add(tuple(setup))
        # Same as TS for this setup: legal moves of P1.
        assert s.legal[i].any()
    assert len(seen) == 36      # 2 Duke columns x 3 Footman pairs, per player


def test_init_matches_ts_legal_moves():
    keys = jax.random.split(jax.random.key(1), 36)
    states = [jax.device_get(E.init(k)) for k in keys]
    replies = ts_oracle([{'state': B.state_to_ts(s)} for s in states])
    for s, rep in zip(states, replies):
        duke = B.duke_square(s.tile, s.owner)
        expected = {B.move_to_actions(m, 0, duke)[0][1] for m in rep['legal']}
        assert set(np.flatnonzero(s.legal).tolist()) == expected


def test_observe_planes_and_features(random_states):
    for s in random_states[::7]:
        planes, feats = (np.asarray(x) for x in E.observe(s))
        tile, owner, side = (np.asarray(x) for x in (s.tile, s.owner, s.side))
        assert planes.shape == (6, 6, E.N_PLANES) and feats.shape == (E.N_FEATURES,)
        for sq in range(R.N_SQ):
            r, c = divmod(sq, 6)
            hot = np.flatnonzero(planes[r, c])
            if tile[sq] < 0:
                assert hot.size == 0
            else:
                assert hot.tolist() == [owner[sq] * 30 + tile[sq] * 2 + side[sq]]
        bags = np.asarray(s.bags)
        np.testing.assert_allclose(feats[:30], bags.reshape(-1) / 3.0)
        pend = int(s.pending)
        np.testing.assert_array_equal(feats[30:45], np.eye(R.N_TYPES)[pend] if pend >= 0 else 0)
        assert feats[45] == (pend == E.DRAWING)
        np.testing.assert_allclose(feats[46], int(s.ply) / E.MAX_PLY)


def test_observe_is_from_the_movers_point_of_view():
    """After a turn passes, the new mover sees the board mirrored with owners swapped."""
    for s, a, after, nxt in random_game(5, max_steps=40):
        if not bool(after.switched) or bool(nxt.done):
            continue
        p0, _ = E.observe(s)
        p1, f1 = E.observe(nxt)
        p0, p1 = np.asarray(p0), np.asarray(p1)
        # Squares the move didn't touch: same tile, mirrored row, owner swapped.
        tile0, tile1 = np.asarray(s.tile), np.asarray(nxt.tile)
        for sq in range(R.N_SQ):
            if tile0[sq] < 0 or tile1[R.FLIP[sq]] != tile0[sq]:
                continue
            r, c = divmod(sq, 6)
            fr, fc = divmod(int(R.FLIP[sq]), 6)
            if np.asarray(s.owner)[sq] == 1:     # opponent tiles never move on my turn
                np.testing.assert_array_equal(p1[fr, fc, :30], p0[r, c, 30:])
        np.testing.assert_allclose(np.asarray(f1[:15]), np.asarray(s.bags[1]) / 3.0)


def test_chance_probs(random_states):
    for s in random_states[:80]:
        p = np.asarray(E.chance_probs(s))
        np.testing.assert_array_equal(p, np.eye(R.N_TYPES)[0])     # decision states: certain
        if bool(s.legal[R.DRAW_ACTION]):
            after = step_decision(s, R.DRAW_ACTION)
            p = np.asarray(E.chance_probs(after))
            bag = np.asarray(s.bags[0], np.float64)
            np.testing.assert_allclose(p, bag / bag.sum(), rtol=1e-6)


def test_place_ok_carries_over_through_draws(random_states):
    n = 0
    for s in random_states:
        if not bool(s.legal[R.DRAW_ACTION]):
            continue
        after = step_decision(s, R.DRAW_ACTION)
        assert int(after.pending) == E.DRAWING and int(after.player) == int(s.player)
        assert not np.asarray(after.legal).any() and not bool(after.switched)
        assert float(after.reward) == 0 and not bool(after.done)
        for outcome in np.flatnonzero(np.asarray(s.bags[0]) > 0)[:3]:
            placing = step_chance(after, int(outcome))
            # Recompute legality from scratch for the revealed state.
            legal, place_ok, _ = E.legal_actions(placing.tile, placing.owner, placing.side,
                                                 placing.bags, placing.pending)
            np.testing.assert_array_equal(np.asarray(placing.legal), np.asarray(legal))
            assert np.asarray(placing.legal[R.PLACE_BASE:]).any()
            assert int(placing.bags[0, outcome]) == int(s.bags[0, outcome]) - 1
            # Placing then passes the turn with the tile on side A next to my Duke.
            a = int(np.flatnonzero(np.asarray(placing.legal))[0])
            done = step_decision(placing, a)
            assert bool(done.switched) or bool(done.done)
            n += 1
    assert n > 20


def test_step_keeps_decision_reward():
    s = make_state(_STALEMATE, player=0)
    nxt = E.step(s, _STALEMATE_MOVE, jax.random.key(0))
    assert float(nxt.reward) == 1.0 and bool(nxt.switched) and bool(nxt.done)
