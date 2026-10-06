"""
Invariants over many batched random games (the way self-play drives the env):
every unfinished state has a legal action, Dukes are present, bags and tile
counts are conserved, the turn passes exactly when `switched` says so, and the
pseudo-legal candidate buffer never overflows.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R

GAMES = int(os.environ.get('DUKE_INVARIANT_GAMES', 128))
STEPS = 260


@jax.jit
def _step(s, key):
    k1, k2, k3 = jax.random.split(key, 3)
    b = s.done.shape[0]
    a = jax.random.categorical(k1, jnp.where(s.legal, 0.0, -jnp.inf)).astype(jnp.int32)
    draw = s.legal[:, R.DRAW_ACTION] & (jax.random.uniform(k2, (b,)) < 0.3)
    a = jnp.where(draw, R.DRAW_ACTION, a)
    after = jax.vmap(E.step_decision)(s, a)
    outcome = jax.random.categorical(k3, jnp.log(jax.vmap(E.chance_probs)(after)))
    return jax.vmap(E.step_chance)(after, outcome), after, a


def _total_tiles(s):
    """Per absolute player: tiles on board + in bag (captures only decrease it)."""
    tile, owner, bags, player = (np.asarray(x) for x in (s.tile, s.owner, s.bags, s.player))
    on_board_me = ((tile >= 0) & (owner == 0)).sum(-1)
    on_board_op = ((tile >= 0) & (owner == 1)).sum(-1)
    me = on_board_me + bags[:, 0].sum(-1) + (np.asarray(s.pending) >= 0)
    op = on_board_op + bags[:, 1].sum(-1)
    p1 = np.where(player == 0, me, op)
    p2 = np.where(player == 0, op, me)
    return p1, p2


def test_random_play_invariants():
    s = jax.jit(jax.vmap(E.init))(jax.random.split(jax.random.key(0), GAMES))
    key = jax.random.key(1)
    p1_prev, p2_prev = _total_tiles(s)
    assert (p1_prev == 19).all() and (p2_prev == 19).all()
    outcomes = set()
    for _ in range(STEPS):
        prev = jax.device_get(s)
        key, k = jax.random.split(key)
        s, after, a = _step(s, k)
        cur, after = jax.device_get(s), jax.device_get(after)
        live = ~cur.done

        assert (cur.overflow == 0).all()
        assert cur.legal[live].any(-1).all(), 'unfinished game with no legal action'
        assert not cur.legal[cur.done].any()
        assert (cur.bags >= 0).all()
        assert (cur.pending[live] >= E.NONE).all()       # no DRAWING at decision points
        dukes_me = ((cur.tile == R.DUKE) & (cur.owner == 0)).sum(-1)
        dukes_op = ((cur.tile == R.DUKE) & (cur.owner == 1)).sum(-1)
        assert (dukes_me[live] == 1).all() and (dukes_op[live] == 1).all()
        # Empty squares carry no owner/side.
        empty = cur.tile < 0
        assert (cur.owner[empty] == 0).all() and (cur.side[empty] == 0).all()

        # Turn passing: the absolute player changes iff the decision passed the turn.
        stepped = ~prev.done
        np.testing.assert_array_equal((cur.player != prev.player)[stepped], after.switched[stepped])
        np.testing.assert_array_equal(cur.ply[stepped],
                                      prev.ply[stepped] + ((prev.pending[stepped] >= 0) |
                                                           (np.asarray(a)[stepped] < R.DRAW_ACTION)))

        # Tiles are only ever removed by captures; never created or duplicated.
        p1, p2 = _total_tiles(cur)
        assert (p1 <= p1_prev).all() and (p2 <= p2_prev).all()
        p1_prev, p2_prev = p1, p2
        outcomes |= set(cur.winner[cur.done].tolist())
        if cur.done.all():
            break
    assert {0, 1} <= outcomes
