"""
Inference server — reads JSON requests from stdin, writes responses to stdout.

Protocol (one JSON object per line):
  Request:  {"boards": [...], "bags": [...], "legal_mask": [...]}
    boards:     flat list of 33*6*6 = 1188 floats
    bags:       flat list of 30 floats
    legal_mask: flat list of 4428 floats (1.0 for legal actions, 0.0 otherwise)

  Response: {"action": int, "value": float}

Usage:
  python -m training.supervised.infer_server checkpoints/epoch_50
"""

import sys
import json
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

from training.supervised.model import DukeNet
from training.supervised.train import load_checkpoint
from training.load_data import NUM_CHANNELS, BOARD_SIZE, BAG_FEATURE_SIZE, TOTAL_ACTIONS


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m training.supervised.infer_server <checkpoint_dir> [epoch]', file=sys.stderr)
        sys.exit(1)

    ckpt_dir = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f'Loading checkpoint: {ckpt_dir} epoch {step}', file=sys.stderr)
    model = load_checkpoint(ckpt_dir, step)
    model.eval()
    print('Model loaded, ready for inference.', file=sys.stderr)
    print('READY', flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            boards = jnp.array(req['boards']).reshape(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            bags = jnp.array(req['bags']).reshape(1, BAG_FEATURE_SIZE)
            legal_mask = jnp.array(req['legal_mask']).reshape(TOTAL_ACTIONS)

            logits, value = model(boards, bags)
            logits = logits[0]
            value = float(value[0])

            masked_logits = jnp.where(legal_mask > 0, logits, -1e9)
            action = int(jnp.argmax(masked_logits))

            resp = json.dumps({'action': action, 'value': value})
            print(resp, flush=True)

        except Exception as e:
            print(json.dumps({'error': str(e)}), flush=True)


if __name__ == '__main__':
    main()
