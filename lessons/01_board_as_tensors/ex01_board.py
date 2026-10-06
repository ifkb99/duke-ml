"""Lesson 1 exercise: the board as tensors. Fill in the two functions.

Inputs are single games (no batch dimension). All arrays are jax arrays.
"""

import jax
import jax.numpy as jnp

from training.zero import rules as R

FLIP = jnp.asarray(R.FLIP)   # [36] int: FLIP[sq] = mirrored square (row -> 5 - row)
N_TYPES = R.N_TYPES          # 15
N_PLANES = 2 * N_TYPES * 2   # owner x type x side = 60


def flip_perspective(tile, owner, side, bags):
    """Hand the turn to the other player.

    Args:
      tile:  int32[36]   tile type per square, -1 = empty
      owner: int32[36]   0 = player to move, 1 = opponent (0 on empty squares)
      side:  int32[36]   0 = side A, 1 = side B
      bags:  int32[2,15] bag counts, row 0 = player to move, row 1 = opponent

    Returns (tile, owner, side, bags) from the OTHER player's point of view:
      - every square is mirrored top-to-bottom (use FLIP as a gather index),
      - owners are swapped on occupied squares (empty squares stay owner 0),
      - the two bag rows are swapped.

    Hint: `x[FLIP]` mirrors a [36] array. `jnp.where(cond, a, b)` picks
    elementwise.
    """
    raise NotImplementedError


def board_planes(tile, owner, side):
    """One-hot board planes for the network.

    Args: tile, owner, side as above (int32[36]).

    Returns float32[6, 6, 60]: planes[r, c, owner*30 + type*2 + side] = 1.0
    for the tile on square r*6+c, and all zeros on empty squares.

    Hint: compute the channel index per square (an int array [36]), use
    jax.nn.one_hot to get [36, 60], zero out empty squares, then reshape.
    Watch out: on empty squares tile == -1, so clamp it before computing an
    index (jnp.maximum(tile, 0)) and rely on the zeroing step.
    """
    raise NotImplementedError
