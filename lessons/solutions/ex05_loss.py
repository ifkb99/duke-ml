"""Reference solution for lesson 5."""

import jax
import jax.numpy as jnp


def dense_target(pol_idx, pol_p, n_actions):
    b = pol_idx.shape[0]
    rows = jnp.arange(b)[:, None]                               # [B, 1]
    return jnp.zeros((b, n_actions)).at[rows, pol_idx].add(pol_p)   # [B, A]


def policy_loss(logits, legal, target, has_policy):
    log_probs = jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)   # [B, A]
    per_sample = -jnp.sum(target * log_probs, axis=-1)                        # [B]
    weight = has_policy.astype(jnp.float32)
    return jnp.sum(per_sample * weight) / jnp.maximum(weight.sum(), 1.0)


def value_loss(value, z):
    return jnp.mean((value - z) ** 2)
