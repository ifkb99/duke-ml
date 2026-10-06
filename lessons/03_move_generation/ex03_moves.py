"""Lesson 3 exercise: move generation with lookup tables."""

import jax.numpy as jnp

from training.zero import rules as R

SQ = jnp.arange(R.N_SQ)                      # [36]
MOVE_OK = jnp.asarray(R.MOVE_OK)             # [2, 15, 2, 36, 36] bool
NEEDS_CLEAR = jnp.asarray(R.NEEDS_CLEAR)     # [2, 15, 2, 36, 36] bool
STRIKE_OK = jnp.asarray(R.STRIKE_OK)         # [2, 15, 2, 36, 36] bool
BETWEEN = jnp.asarray(R.BETWEEN)             # [36, 36, 36] bool


def pseudo_moves(tile, owner, side):
    """All moves and strikes for the player to move, ignoring Duke safety.

    Args (single game, canonical frame):
      tile:  int32[36]  -1 = empty
      owner: int32[36]  0 = me, 1 = opponent
      side:  int32[36]  0 = A, 1 = B

    Returns bool[36, 36]: result[f, g] is True if my tile on f can
      - MOVE to g: geometrically possible (MOVE_OK), every in-between square
        empty when the move needs it (NEEDS_CLEAR + BETWEEN), and g is not my
        own tile; or
      - STRIKE g: possible per STRIKE_OK and g holds an opponent tile.
    Rows f without one of my tiles must be all False.

    Use the owner index 0 ("me") in the tables. No Python loops over squares.
    """
    raise NotImplementedError
