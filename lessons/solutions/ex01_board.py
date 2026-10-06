"""Reference solution for lesson 1."""

import jax
import jax.numpy as jnp

from training.zero import rules as R

FLIP = jnp.asarray(R.FLIP)
N_TYPES = R.N_TYPES
N_PLANES = 2 * N_TYPES * 2


def flip_perspective(tile, owner, side, bags):
    tile, owner, side = tile[FLIP], owner[FLIP], side[FLIP]   # mirror rows: [36] each
    owner = jnp.where(tile >= 0, 1 - owner, 0)               # swap owners, keep empties at 0
    return tile, owner, side, bags[::-1]                     # swap bag rows: [2, 15]


def board_planes(tile, owner, side):
    channel = owner * (2 * N_TYPES) + jnp.maximum(tile, 0) * 2 + side   # [36]
    planes = jax.nn.one_hot(channel, N_PLANES)                         # [36, 60]
    planes = planes * (tile >= 0)[:, None]                             # empty squares -> 0
    return planes.reshape(6, 6, N_PLANES)                              # [6, 6, 60]
