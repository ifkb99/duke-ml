"""Shared test helpers for the lessons.

`load_exercise("ex01_board")` imports your exercise file, or the reference
solution when LESSONS_USE_SOLUTIONS=1 is set (handy to check the tests
themselves, or to peek once you're stuck).
"""

import importlib
import os
import sys
from pathlib import Path

# Training may be using the GPUs; take memory as needed instead of grabbing
# 75% of a card up front (JAX's default).
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

LESSONS = Path(__file__).resolve().parent
sys.path.insert(0, str(LESSONS))
sys.path.insert(0, str(LESSONS.parent))   # so `import training.zero...` works


def load_exercise(name: str):
    if os.environ.get('LESSONS_USE_SOLUTIONS'):
        return importlib.import_module(f'solutions.{name}')
    return importlib.import_module(name)


def random_states(n: int, plies: int = 30, seed: int = 0):
    """n positions reached by `plies` random moves, as a batched env.State."""
    import jax
    import jax.numpy as jnp
    from training.zero import env as E

    key = jax.random.key(seed)
    s = jax.jit(jax.vmap(E.init))(jax.random.split(key, n))
    step = jax.jit(jax.vmap(E.step))
    for i in range(plies):
        k1, k2 = jax.random.split(jax.random.fold_in(key, i))
        a = jax.random.categorical(k1, jnp.where(s.legal, 0.0, -jnp.inf))
        nxt = step(s, a, jax.random.split(k2, n))
        # Keep the last unfinished position of games that end early.
        s = jax.tree.map(
            lambda old, new: jnp.where(nxt.done.reshape((-1,) + (1,) * (old.ndim - 1)), old, new), s, nxt)
    return s
