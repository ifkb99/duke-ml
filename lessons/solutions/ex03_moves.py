"""Reference solution for lesson 3."""

import jax.numpy as jnp

from training.zero import rules as R

SQ = jnp.arange(R.N_SQ)
MOVE_OK = jnp.asarray(R.MOVE_OK)
NEEDS_CLEAR = jnp.asarray(R.NEEDS_CLEAR)
STRIKE_OK = jnp.asarray(R.STRIKE_OK)
BETWEEN = jnp.asarray(R.BETWEEN)


def pseudo_moves(tile, owner, side):
    occ = tile >= 0                                              # [36]
    mine = occ & (owner == 0)                                    # [36]
    opp = occ & (owner == 1)                                     # [36]
    t = jnp.maximum(tile, 0)                                     # [36] empty -> type 0 (masked by `mine`)

    move_ok = MOVE_OK[0, t, side, SQ]                            # [36 f, 36 g]
    needs_clear = NEEDS_CLEAR[0, t, side, SQ]                    # [36 f, 36 g]
    occupied_between = jnp.any(BETWEEN & occ[None, None, :], -1) # [36 f, 36 g]

    moves = mine[:, None] & move_ok & ~(needs_clear & occupied_between) & ~mine[None, :]
    strikes = mine[:, None] & STRIKE_OK[0, t, side, SQ] & opp[None, :]
    return moves | strikes
