"""
Offline evaluation — measure how well the trained network imitates minimax.

Usage:
  python -m training.supervised.evaluate training/selfplay.bin checkpoints/epoch_49.pkl

Metrics to compute:
  - Policy accuracy (top-1): does argmax of logits match the minimax action?
    Even 30-40% is decent — the branching factor is ~15-30 moves, so random
    would score ~3-7%. Matching minimax's exact move 1/3 of the time means
    the network has learned real positional understanding.

  - Policy top-5 accuracy: is the correct action in the top 5 predictions?
    Should be significantly higher than top-1. This tells you the network
    has the right "neighborhood" of good moves even when it doesn't nail
    the exact one.

  - Value MSE: mean squared error between predicted and actual game outcome.
    Target values are +1 (win), -1 (loss), 0 (draw). MSE < 0.5 means the
    network is doing better than always predicting 0.

  - Value sign accuracy: does sign(prediction) match sign(target)?
    i.e. does the network at least know who's winning? This is the most
    intuitive value metric.

Key JAX bits:
  - jnp.argmax(logits, axis=-1)  — top-1 prediction
  - jax.lax.top_k(logits, k=5)   — top-k values and indices
  - For NNX inference, call model.eval() first, then model(boards, bags).
    model.eval() tells BatchNorm to use running averages instead of
    batch statistics.
"""

import sys
import argparse
import pickle

import jax
import jax.numpy as jnp
from flax import nnx

from training.load_data import load_selfplay
from training.supervised_v1_baseline.model import DukeNet


def load_checkpoint(path: str, model: DukeNet) -> None:
    """Load saved state into an existing model.

    If you saved with pickle + nnx.state in train.py:
      state = pickle.load(open(path, 'rb'))
      nnx.update(model, state)
    """
    raise NotImplementedError  # TODO


def evaluate(data_path: str, checkpoint_path: str, batch_size: int = 512):
    """Run evaluation on the full dataset and print metrics.

    Steps:
      1. Load data with load_selfplay(data_path)
      2. Create DukeNet, load checkpoint into it
      3. model.eval() — set BatchNorm to inference mode
      4. Run forward pass in batches, collect all logits and values
      5. Compute and print metrics:
         - Policy accuracy (top-1)
         - Policy accuracy (top-5)
         - Value MSE
         - Value sign accuracy
    """
    raise NotImplementedError  # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained DukeNet')
    parser.add_argument('data', help='Path to selfplay .bin file')
    parser.add_argument('checkpoint', help='Path to checkpoint .pkl file')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    evaluate(args.data, args.checkpoint, batch_size=args.batch_size)
