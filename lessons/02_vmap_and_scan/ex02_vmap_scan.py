"""Lesson 2 exercise: vmap and scan."""

import jax
import jax.numpy as jnp

from training.zero import env as E


def material(tile, owner):
    """Single game: (number of my tiles) - (number of opponent tiles).

    tile, owner: int32[36] (tile == -1 means empty; owner 0 = me, 1 = opponent).
    Returns an int32 scalar.
    """
    raise NotImplementedError


def material_batch(tile, owner):
    """Batched material: tile, owner are int32[B, 36]; returns int32[B].

    Use jax.vmap on `material` instead of rewriting it with axis arguments.
    """
    raise NotImplementedError


def random_playout(key, state: E.State, n_steps: int):
    """Play n_steps uniformly random legal moves in ONE game using lax.scan.

    Returns (final_state, moves_made) where moves_made counts the steps taken
    while the game was still running (a finished game ignores further steps).

    Per step:
      - split the key (one part to pick the action, one for env.step's chance),
      - pick a uniformly random legal action: jax.random.categorical over
        logits that are 0 for legal actions and -inf for illegal ones
        (state.legal is bool[3605]),
      - state = E.step(state, action, chance_key).

    `n_steps` is a Python int (static), so it's fine to use it as scan's length.
    Hint: scan over `jax.random.split(key, n_steps)`, and carry the state.
    """
    raise NotImplementedError
