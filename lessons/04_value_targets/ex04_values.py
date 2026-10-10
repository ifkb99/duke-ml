"""Lesson 4 exercise: value targets for a finished game."""

import jax
import jax.numpy as jnp
import numpy as np


def value_targets_numpy(reward, discount):
    """z[t] = reward[t] + discount[t] * z[t+1], with z[T] = 0.

    reward, discount: np.ndarray float32[T] for ONE finished game.
    Returns np.ndarray float32[T].
    """
    raise NotImplementedError


def value_targets_jax(reward, discount):
    """Same as value_targets_numpy, with jax.lax.scan(..., reverse=True).

    reward, discount: jax arrays float32[T]. Returns float32[T].
    Must work under jax.jit.
    """
    raise NotImplementedError
