"""Reference solution for lesson 4."""

import jax
import jax.numpy as jnp
import numpy as np


def value_targets_numpy(reward, discount):
    z = np.zeros(len(reward), np.float32)
    z_next = 0.0
    for t in range(len(reward) - 1, -1, -1):
        z_next = reward[t] + discount[t] * z_next
        z[t] = z_next
    return z


def value_targets_jax(reward, discount):
    def body(z_next, rd):
        r, d = rd
        z = r + d * z_next
        return z, z

    _, z = jax.lax.scan(body, jnp.float32(0), (reward, discount), reverse=True)
    return z
