"""
DukeNet — dual-head neural network for The Duke.

Two outputs:
  policy: logits over 4428 actions (which move to play)
  value:  scalar in [-1, +1]      (who's winning)

Architecture overview:
  1. Board tensor (33, 6, 6) goes through a conv + residual tower
  2. Tower output is flattened and concatenated with bag features (30,)
  3. Combined features split into two "heads":
     - Policy head: Dense layers → 4428 logits
     - Value head:  Dense layers → tanh scalar

Key Flax NNX concepts:
  - nnx.Module:    base class. Define layers in __init__, forward pass in __call__.
  - nnx.Conv:      convolution. Args: (in_features, out_features, kernel_size, ...).
                   Flax uses NHWC layout (batch, height, width, channels),
                   so you'll need to transpose your (B, C, H, W) boards.
  - nnx.Linear:    fully connected layer. Args: (in_features, out_features).
  - nnx.BatchNorm: normalizes activations. In NNX, running stats are managed
                   automatically on the module. Call model.eval() for inference
                   mode and model.train() for training mode.
  - nnx.relu:      activation function (zeroes out negatives).
  - nnx.Rngs:      random number generator container passed to layers for
                   parameter initialization.

Flax NNX docs:  https://flax.readthedocs.io/en/latest/nnx_basics.html
"""

import jax.numpy as jnp
from flax import nnx

from training.load_data import TOTAL_ACTIONS, NUM_CHANNELS, BOARD_SIZE, BAG_FEATURE_SIZE


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------
#
# A residual block is the workhorse of modern deep nets. The idea:
#   output = relu(x + F(x))
# where F is two conv layers with batch norm. The skip connection (adding x
# back) means the block only needs to learn the *difference* from identity,
# which makes deeper networks much easier to train.
#
# Shape: (B, H, W, filters) → (B, H, W, filters)   [no change]
#
# Reference: "Deep Residual Learning" (He et al. 2015)
# ---------------------------------------------------------------------------

class ResBlock(nnx.Module):
    def __init__(self, filters: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(filters, filters, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=filters, rngs=rngs)
        self.conv2 = nnx.Conv(filters, filters, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=filters, rngs=rngs)

    def __call__(self, x):
        Fx = nnx.relu(self.bn1(self.conv1(x)))
        Fx = self.bn2(self.conv2(Fx))
        return nnx.relu(x + Fx)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
#
# Forward pass pseudocode:
#   1. Transpose boards from NCHW → NHWC         (Flax conv expects channels-last)
#   2. Initial Conv (3×3, SAME padding) → BatchNorm → ReLU
#   3. Pass through `num_blocks` ResBlocks
#   4. Flatten spatial dims → concat bag features
#   5. Policy head:  Linear(256) → ReLU → Linear(4428)   → return raw logits
#   6. Value head:   Linear(128) → ReLU → Linear(1) → tanh → squeeze to (B,)
#
# Tips:
#   - jnp.transpose(boards, (0, 2, 3, 1))  converts NCHW → NHWC
#   - x.reshape(x.shape[0], -1)            flattens all but batch dim
#   - jnp.concatenate([a, b], axis=-1)     joins along last axis
#   - jnp.tanh(x).squeeze(-1)             squish Linear(1) output to scalar
# ---------------------------------------------------------------------------

class DukeNet(nnx.Module):
    def __init__(self, num_filters: int = 64, num_blocks: int = 4,
                 num_actions: int = TOTAL_ACTIONS, *, rngs: nnx.Rngs):
        self.initial_conv = nnx.Conv(NUM_CHANNELS, num_filters, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.initial_bn = nnx.BatchNorm(num_features=num_filters, rngs=rngs)

        self.blocks = nnx.List([ResBlock(num_filters, rngs=rngs) for _ in range(num_blocks)])

        flat_size = BOARD_SIZE * BOARD_SIZE * num_filters + BAG_FEATURE_SIZE

        self.policy_fc1 = nnx.Linear(flat_size, 256, rngs=rngs)
        self.policy_fc2 = nnx.Linear(256, num_actions, rngs=rngs)

        self.value_fc1 = nnx.Linear(flat_size, 128, rngs=rngs)
        self.value_fc2 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, boards, bags):
        """
        Args:
            boards: (B, 33, 6, 6) float32 — board plane tensor
            bags:   (B, 30)        float32 — bag feature vector

        Returns:
            policy_logits: (B, 4428) float32 — raw scores per action
            value:         (B,)      float32 — predicted outcome in [-1, 1]
        """
        x = jnp.transpose(boards, (0, 2, 3, 1))  # → (B, 6, 6, 33)

        x = nnx.relu(self.initial_bn(self.initial_conv(x)))

        for block in self.blocks:
            x = block(x)

        flat = x.reshape(x.shape[0], -1)                      # (B, 6*6*64)
        flat = jnp.concatenate([flat, bags], axis=-1)          # (B, 6*6*64 + 30)

        p = nnx.relu(self.policy_fc1(flat))
        p = self.policy_fc2(p)

        v = nnx.relu(self.value_fc1(flat))
        v = jnp.tanh(self.value_fc2(v)).squeeze(-1)  # (B,) in [-1, 1]

        return p, v


if __name__ == '__main__':
    model = DukeNet(rngs=nnx.Rngs(0))
    boards = jnp.zeros((1, 33, 6, 6))
    bags = jnp.zeros((1, 30))
    policy, value = model(boards, bags)
    print(f'policy: {policy.shape}  value: {value.shape}')
    print(nnx.display(model))
