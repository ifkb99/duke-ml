"""
Training loop — supervised learning on minimax self-play data.

Workflow:
  1. Generate data:   npx tsx packages/ai/src/generate.ts 1000 4 training/selfplay.bin
  2. Train:           python -m training.supervised.train training/selfplay.bin
  3. Evaluate:        python -m training.supervised.evaluate training/selfplay.bin checkpoints/

Usage:
  python -m training.supervised.train <data.bin> [--epochs 50] [--batch-size 256] [--lr 1e-3]

ML concepts you'll encounter below:
  - Epoch:       one full pass through the entire training set.
  - Batch:       a small chunk of data processed at once (e.g. 256 samples).
                 You iterate over batches within each epoch.
  - Loss:        a number measuring how wrong the network is. Training = minimizing loss.
  - Gradient:    the direction to nudge each parameter to reduce loss.
                 JAX computes these automatically via `jax.grad`.
  - Optimizer:   algorithm that uses gradients to update params.
                 Adam (optax.adam) is the standard starting point.

Key JAX/Flax NNX/Optax functions:
  - nnx.Optimizer(model, optax.adam(lr))  — wraps model + optimizer together
  - nnx.value_and_grad(loss_fn, has_aux=True)
        — compute loss AND gradients w.r.t. the model in one call
  - optimizer.update(grads)               — applies gradients to model params
  - optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        — standard classification loss. `logits` are raw network outputs (B, num_classes),
          `labels` are integer class indices (B,). Returns per-sample loss (B,).
  - jnp.mean((pred - target) ** 2)       — mean squared error for value head

NNX training is simpler than Linen:
  - No TrainState subclass needed — nnx.Optimizer holds everything.
  - No batch_stats plumbing — BatchNorm state lives on the model and updates
    automatically during forward passes in training mode.
  - model.train() / model.eval() toggles BatchNorm behavior.
"""

import sys
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from training.load_data import load_selfplay
from training.supervised.model import DukeNet


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
#
# The loss function takes the model as its first argument — this is how NNX
# knows which parameters to compute gradients for.
#
# You need to compute two losses and add them together:
#
# 1. Policy loss — cross-entropy between the network's action logits
#    and the action the minimax player actually chose.
#    Use: optax.softmax_cross_entropy_with_integer_labels(logits, actions)
#    This returns a (B,) vector; take the mean.
#
# 2. Value loss — mean squared error between predicted value and
#    the actual game outcome (+1, -1, or 0).
#    Use: jnp.mean((pred_values - target_values) ** 2)
#
# Total loss = policy_loss + value_loss
#
# Pattern:
#   def compute_loss(model, boards, bags, actions, values):
#       logits, pred_values = model(boards, bags)
#       policy_loss = ...
#       value_loss = ...
#       return total_loss, {'policy_loss': ..., 'value_loss': ...}
# ---------------------------------------------------------------------------

def compute_loss(model, boards, bags, actions, values):
    """Compute combined policy + value loss.

    Args:
        model:   DukeNet instance (NNX grad differentiates w.r.t. this)
        boards:  (B, 33, 6, 6) board tensors
        bags:    (B, 30) bag features
        actions: (B,) int32 action indices the minimax player chose
        values:  (B,) float32 game outcomes from current player's POV

    Returns:
        (total_loss, aux) where aux is a dict with:
            'policy_loss', 'value_loss'
    """
    logits, pred_values = model(boards, bags)
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions).mean()
    value_loss = jnp.mean((pred_values - values) ** 2)
    total_loss = policy_loss + value_loss
    return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss}

# ---------------------------------------------------------------------------
# Single training step (JIT-compiled)
# ---------------------------------------------------------------------------
#
# This is the inner loop body. NNX can compile it with @nnx.jit.
#
# Pattern:
#   @nnx.jit
#   def train_step(model, optimizer, boards, bags, actions, values):
#       grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
#       (loss, aux), grads = grad_fn(model, boards, bags, actions, values)
#       optimizer.update(grads)
#       return loss, aux
#
# That's it — no apply_gradients, no batch_stats replace. NNX handles
# the state updates internally.
#
# @nnx.jit compiles the function for GPU/TPU, like @jax.jit but aware
# of NNX module state.
# ---------------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, boards, bags, actions, values):
    """Execute one gradient update.

    Args:
        model:     DukeNet instance
        optimizer: nnx.Optimizer wrapping the model
        boards, bags, actions, values: one batch of training data

    Returns:
        (loss, metrics_dict)
    """
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(model, boards, bags, actions, values)
    optimizer.update(grads)
    return loss, aux


# ---------------------------------------------------------------------------
# Data batching helper
# ---------------------------------------------------------------------------
#
# Shuffle the data each epoch, then yield chunks of `batch_size`.
# Use jax.random for shuffling to keep things reproducible:
#
#   perm = jax.random.permutation(rng, num_samples)
#   for i in range(0, num_samples, batch_size):
#       idx = perm[i : i + batch_size]
#       yield boards[idx], bags[idx], actions[idx], values[idx]
# ---------------------------------------------------------------------------

def iter_batches(rng, boards, bags, actions, values, batch_size: int):
    """Yield shuffled batches for one epoch."""
    perm = jax.random.permutation(rng, len(boards))
    for i in range(0, len(boards), batch_size):
        idx = perm[i : i + batch_size]
        yield boards[idx], bags[idx], actions[idx], values[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
#
# Pseudocode:
#   model = DukeNet(rngs=nnx.Rngs(seed))
#   optimizer = nnx.Optimizer(model, optax.adam(lr))
#
#   for epoch in range(num_epochs):
#       model.train()    # set BatchNorm to training mode
#       for batch in iter_batches(...):
#           loss, metrics = train_step(model, optimizer, *batch)
#           accumulate metrics
#       print epoch summary (avg policy_loss, avg value_loss)
#       optionally save checkpoint every N epochs
#
# Orbax can't serialize NNX RNG keys directly, so we split them out
# before saving and let them reinitialize on restore.
#
# Save:
#   graphdef, rng_state, other_state = nnx.split(model, nnx.RngState, ...)
#   checkpointer.save(path, other_state)
#
# Restore:
#   model = DukeNet(rngs=nnx.Rngs(0))
#   graphdef, rng_state, abstract_state = nnx.split(model, nnx.RngState, ...)
#   restored = checkpointer.restore(path, abstract_state)
#   model = nnx.merge(graphdef, rng_state, restored)
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.9
CHECKPOINT_EVERY = 5


def save_checkpoint(checkpointer: ocp.StandardCheckpointer, model: DukeNet, ckpt_dir: Path, step: int):
    """Save model state with Orbax, filtering out non-serializable RNG keys."""
    _, _, saveable_state = nnx.split(model, nnx.RngState, ...)
    checkpointer.save(ckpt_dir / f'epoch_{step}', saveable_state, force=True)
    checkpointer.wait_until_finished()


def load_checkpoint(ckpt_dir: Path, step: int) -> DukeNet:
    """Restore a DukeNet from an Orbax checkpoint."""
    ckpt_dir = ckpt_dir.resolve()
    model = DukeNet(rngs=nnx.Rngs(0))
    graphdef, rng_state, abstract_state = nnx.split(model, nnx.RngState, ...)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(ckpt_dir / f'epoch_{step}', abstract_state)
    return nnx.merge(graphdef, rng_state, restored_state)


def train(data_path: str, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
    data = load_selfplay(data_path)

    n = len(data['boards'])
    split = int(n * TRAIN_SPLIT)

    rng = jax.random.PRNGKey(42)
    rng, perm_rng = jax.random.split(rng)
    perm = jax.random.permutation(perm_rng, n)

    train_boards, val_boards = data['boards'][perm[:split]], data['boards'][perm[split:]]
    train_bags, val_bags = data['bags'][perm[:split]], data['bags'][perm[split:]]
    train_actions, val_actions = data['actions'][perm[:split]], data['actions'][perm[split:]]
    train_values, val_values = data['values'][perm[:split]], data['values'][perm[split:]]

    print(f'Loaded {n} samples: {split} train, {n - split} val')

    model = DukeNet(rngs=nnx.Rngs(0))
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(lr))

    ckpt_dir = Path('checkpoints').resolve()
    ckpt_dir.mkdir(exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        rng, epoch_rng = jax.random.split(rng)
        train_policy, train_value, train_batches = 0.0, 0.0, 0
        for batch in iter_batches(epoch_rng, train_boards, train_bags, train_actions, train_values, batch_size):
            _, aux = train_step(model, optimizer, *batch)
            train_policy += aux['policy_loss']
            train_value += aux['value_loss']
            train_batches += 1

        # --- Validation ---
        model.eval()
        rng, val_rng = jax.random.split(rng)
        val_policy, val_value, val_batches = 0.0, 0.0, 0
        for batch in iter_batches(val_rng, val_boards, val_bags, val_actions, val_values, batch_size):
            _, aux = compute_loss(model, *batch)
            val_policy += aux['policy_loss']
            val_value += aux['value_loss']
            val_batches += 1

        print(
            f'Epoch {epoch:3d} | '
            f'train policy: {train_policy / train_batches:.4f}  value: {train_value / train_batches:.4f} | '
            f'val policy: {val_policy / val_batches:.4f}  value: {val_value / val_batches:.4f}'
        )

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpointer, model, ckpt_dir, epoch + 1)
            print(f'  Saved checkpoint: epoch_{epoch + 1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DukeNet on minimax self-play data')
    parser.add_argument('data', help='Path to selfplay .bin file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
