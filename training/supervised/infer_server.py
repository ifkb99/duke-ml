"""
Inference server — reads JSON requests from stdin, writes responses to stdout.

Two request types are supported (selected via the optional "type" field):

  type="action" (default — used by raw policy mode)
    Request:  {"boards": [...], "bags": [...], "legal_mask": [...]}
      boards:     flat list of NUM_CHANNELS*BOARD_SIZE*BOARD_SIZE floats
      bags:       flat list of BAG_FEATURE_SIZE floats
      legal_mask: flat list of TOTAL_ACTIONS floats (1.0 for legal, else 0.0)
    Response: {"action": int, "value": float}

  type="value_batch" (used by 1-ply search)
    Request:  {"type": "value_batch", "n": N, "boards": [...], "bags": [...]}
      boards: N * NUM_CHANNELS*BOARD_SIZE*BOARD_SIZE floats (flat, row-major)
      bags:   N * BAG_FEATURE_SIZE floats (flat)
    Response: {"values": [N floats]}

Usage:
  python -m training.supervised.infer_server <checkpoint_dir> [epoch_or_best]
"""

import sys
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from training.supervised.model import DukeNet
from training.supervised.train import load_checkpoint
from training.load_data import NUM_CHANNELS, BOARD_SIZE, BAG_FEATURE_SIZE, TOTAL_ACTIONS


def _parse_step(arg: str) -> int | str:
    try:
        return int(arg)
    except ValueError:
        return arg


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m training.supervised.infer_server <checkpoint_dir> [epoch_or_best]', file=sys.stderr)
        sys.exit(1)

    ckpt_dir = Path(sys.argv[1])
    step = _parse_step(sys.argv[2]) if len(sys.argv) > 2 else 50

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
            req_type = req.get('type', 'action')

            if req_type == 'value_batch':
                n = int(req['n'])
                boards = jnp.asarray(req['boards'], dtype=jnp.float32).reshape(
                    n, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE
                )
                bags = jnp.asarray(req['bags'], dtype=jnp.float32).reshape(n, BAG_FEATURE_SIZE)
                _, values = model(boards, bags)
                values_list = np.asarray(values).reshape(-1).tolist()
                print(json.dumps({'values': values_list}), flush=True)

            else:
                boards = jnp.asarray(req['boards'], dtype=jnp.float32).reshape(
                    1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE
                )
                bags = jnp.asarray(req['bags'], dtype=jnp.float32).reshape(1, BAG_FEATURE_SIZE)
                legal_mask = jnp.asarray(req['legal_mask'], dtype=jnp.float32).reshape(TOTAL_ACTIONS)

                logits, value = model(boards, bags)
                logits = logits[0]
                value = float(value[0])

                masked_logits = jnp.where(legal_mask > 0, logits, -1e9)
                action = int(jnp.argmax(masked_logits))

                print(json.dumps({'action': action, 'value': value}), flush=True)

        except Exception as e:
            print(json.dumps({'error': str(e)}), flush=True)


if __name__ == '__main__':
    main()
