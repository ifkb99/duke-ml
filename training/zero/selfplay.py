"""
Batched self-play on the GPU.

`make_selfplay` returns a jitted function that advances B games by T steps,
running a full tree search for every move, and returns the per-step records
the replay `Assembler` needs (shape [T, B, ...]). Finished games are replaced
by fresh ones in place, so the batch stays full.
"""

import jax
import jax.numpy as jnp

from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R


def compact_state(s: E.State) -> dict:
    """Batched State -> the small integer arrays stored in the replay buffer."""
    return {
        'tile': s.tile.astype(jnp.int8),
        'owner': s.owner.astype(jnp.int8),
        'side': s.side.astype(jnp.int8),
        'bags': s.bags.astype(jnp.int8),
        'pending': s.pending.astype(jnp.int8),
        'ply': s.ply.astype(jnp.int16),
    }


def compact_policy(weights) -> dict:
    """Dense policy [B, A] -> top-K (indices, renormalized probs)."""
    p, idx = jax.lax.top_k(weights, RP.POLICY_K)
    p = p / jnp.maximum(p.sum(-1, keepdims=True), 1e-9)
    return {'pol_idx': idx.astype(jnp.int16), 'pol_p': p.astype(jnp.float16)}


def one_hot_policy(actions) -> dict:
    """A single chosen action as a top-K policy (used for imitation targets)."""
    b = actions.shape[0]
    idx = jnp.zeros((b, RP.POLICY_K), jnp.int16).at[:, 0].set(actions.astype(jnp.int16))
    p = jnp.zeros((b, RP.POLICY_K), jnp.float16).at[:, 0].set(1)
    return {'pol_idx': idx, 'pol_p': p}


def step_record(s: E.State, after: E.State, actions) -> dict:
    """Bookkeeping for one batched step from decision state s to afterstate `after`."""
    return {
        'reward': after.reward,
        'discount': jnp.where(after.done, 0.0, jnp.where(after.switched, -1.0, 1.0)),
        'done': after.done,
        'valid': ~s.done,
        'is_draw': actions == R.DRAW_ACTION,
    }


def _reset_done(s: E.State, key) -> E.State:
    fresh = jax.vmap(E.init)(jax.random.split(key, s.done.shape[0]))
    def pick(old, new):
        mask = s.done.reshape(s.done.shape + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)
    return jax.tree.map(pick, s, fresh)


def make_selfplay(search, unroll: int, policy_targets: bool = True):
    """Returns jitted selfplay(params, states, key) -> (states, records[T, B]).

    policy_targets=False marks every record has_policy=False: a cheap search
    that only moves the game along and gives value targets (playout cap
    randomization, see train.Config.full_simulations)."""
    step_decision = jax.vmap(E.step_decision)
    step_chance = jax.vmap(E.step_chance)
    chance_probs = jax.vmap(E.chance_probs)

    def one_step(params, s, key):
        k_search, k_chance, k_reset = jax.random.split(key, 3)
        out = search(params, k_search, s, 1.0)
        after = step_decision(s, out.action)
        outcome = jax.random.categorical(k_chance, jnp.log(chance_probs(after)))
        nxt = step_chance(after, outcome)
        record = {
            **compact_state(s),
            **compact_policy(out.action_weights),
            'has_policy': jnp.full_like(s.done, policy_targets),
            **step_record(s, after, out.action),
            'search_value': out.search_value,
        }
        return _reset_done(nxt, k_reset), record

    @jax.jit
    def selfplay(params, states, key):
        def body(s, k):
            return one_step(params, s, k)
        return jax.lax.scan(body, states, jax.random.split(key, unroll))

    return selfplay
