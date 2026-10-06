"""Lesson 5 exercise: the training loss."""

import jax
import jax.numpy as jnp


def dense_target(pol_idx, pol_p, n_actions):
    """Scatter top-K targets into a dense policy.

    pol_idx: int32[B, K] action indices; pol_p: float32[B, K] their probabilities
    (entries with probability 0 are padding and may repeat indices).
    Returns float32[B, n_actions].

    Hint: jnp.zeros(...).at[rows, cols].add(values), with rows shaped [B, 1].
    """
    raise NotImplementedError


def policy_loss(logits, legal, target, has_policy):
    """Masked cross-entropy, averaged over samples that have a policy target.

    logits: float32[B, A]; legal: bool[B, A]; target: float32[B, A] (rows sum to
    1 where has_policy); has_policy: bool[B]. Returns a float32 scalar.

    Steps: mask illegal logits with -1e9, log_softmax over the last axis,
    per-sample loss = -sum(target * log_probs), then average over rows where
    has_policy is True (return 0.0 if there are none, without dividing by zero).
    """
    raise NotImplementedError


def value_loss(value, z):
    """Mean squared error. value, z: float32[B]. Returns a float32 scalar."""
    raise NotImplementedError
