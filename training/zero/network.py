"""
DukeZeroNet: a small residual network with a policy head and a value head.

Inputs (from env.observe, batched):
  planes   [B, 6, 6, 60]  one-hot tiles: owner (me/opponent) x type x side
  features [B, 47]        bag counts, tile in hand, drawing flag, turn number

Outputs:
  logits   [B, 3605]      unnormalized log-probabilities over actions
  value    [B]            expected outcome for the player to move, in [-1, 1]

Shapes through the trunk (C = channels):
  planes --conv3x3--> [B,6,6,C] + features projected to a per-channel bias
  -> residual blocks [B,6,6,C] -> heads

Convolutions and dense layers compute in bfloat16 (tensor cores; ~2-3x faster
on Ampere) while weights stay float32; norms and the value output use float32.

LayerNorm is used instead of BatchNorm so the network behaves identically in
training and inside the search (no running statistics to manage), and the
module holds nothing but Params — `nnx.split` gives a clean (graphdef, params).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from training.zero import env as E
from training.zero import rules as R


class ResBlock(nnx.Module):
    """Pre-activation residual block: x + conv(relu(norm(conv(relu(norm(x))))))."""

    def __init__(self, ch: int, *, dtype, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(ch, rngs=rngs)
        self.conv1 = nnx.Conv(ch, ch, (3, 3), dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(ch, rngs=rngs)
        self.conv2 = nnx.Conv(ch, ch, (3, 3), dtype=dtype, rngs=rngs,
                              kernel_init=nnx.initializers.zeros)  # block starts as identity

    def __call__(self, x):
        y = self.conv1(nnx.relu(self.norm1(x)))
        y = self.conv2(nnx.relu(self.norm2(y)))
        return x + y.astype(x.dtype)


class DukeZeroNet(nnx.Module):
    def __init__(self, channels: int = 96, blocks: int = 6, *, dtype=jnp.bfloat16, rngs: nnx.Rngs):
        c = channels
        self.stem = nnx.Conv(E.N_PLANES, c, (3, 3), dtype=dtype, rngs=rngs)
        self.film = nnx.Linear(E.N_FEATURES, c, rngs=rngs)
        self.blocks = nnx.List([ResBlock(c, dtype=dtype, rngs=rngs) for _ in range(blocks)])
        self.norm = nnx.LayerNorm(c, rngs=rngs)

        sq = R.N_SQ
        self.policy_conv = nnx.Conv(c, 32, (1, 1), dtype=dtype, rngs=rngs)
        self.policy_out = nnx.Linear(sq * 32 + c, R.N_ACTIONS, dtype=dtype, rngs=rngs,
                                     kernel_init=nnx.initializers.zeros)  # uniform prior at init
        self.value_conv = nnx.Conv(c, 8, (1, 1), dtype=dtype, rngs=rngs)
        self.value_hidden = nnx.Linear(sq * 8 + c, 128, rngs=rngs)
        self.value_out = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, planes, features):
        b = planes.shape[0]
        x = self.stem(planes).astype(jnp.float32) + self.film(features)[:, None, None, :]  # [B,6,6,C]
        for block in self.blocks:
            x = block(x)
        x = nnx.relu(self.norm(x))
        pooled = x.mean(axis=(1, 2))                                      # [B,C] board summary

        p = nnx.relu(self.policy_conv(x)).reshape(b, -1)                  # [B, 36*32]
        logits = self.policy_out(jnp.concatenate([p, pooled], axis=-1)).astype(jnp.float32)  # [B, 3605]

        v = nnx.relu(self.value_conv(x)).reshape(b, -1).astype(jnp.float32)  # [B, 36*8]
        v = nnx.relu(self.value_hidden(jnp.concatenate([v, pooled], axis=-1)))
        value = jnp.tanh(self.value_out(v))[:, 0]                         # [B]
        return logits, value


def make_forward(graphdef):
    """Pure function (params, planes, features) -> (logits, value).

    Search and training pass `params` around explicitly as a pytree, which is
    what jit/vmap/mctx expect; this rebuilds the module from it on each call
    (free under jit — it all traces away).
    """
    def forward(params, planes, features):
        return nnx.merge(graphdef, params)(planes, features)
    return forward


def observe_batch(states):
    return jax.vmap(E.observe)(states)
