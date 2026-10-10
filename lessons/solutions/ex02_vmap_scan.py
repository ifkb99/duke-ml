"""Reference solution for lesson 2."""

import jax
import jax.numpy as jnp

from training.zero import env as E


def material(tile, owner):
    occupied = tile >= 0
    mine = jnp.sum(occupied & (owner == 0))
    theirs = jnp.sum(occupied & (owner == 1))
    return (mine - theirs).astype(jnp.int32)


material_batch = jax.vmap(material)


def random_playout(key, state: E.State, n_steps: int):
    def body(s, k):
        k_action, k_chance = jax.random.split(k)
        logits = jnp.where(s.legal, 0.0, -jnp.inf)          # [3605]
        action = jax.random.categorical(k_action, logits)
        running = ~s.done                                   # was this a real move?
        return E.step(s, action, k_chance), running

    final, running = jax.lax.scan(body, state, jax.random.split(key, n_steps))
    return final, running.sum()
