"""Held-out policy loss split by decision level: the coarse decision (draw, or
which piece acts) and where it goes (policy = coarse + where, chain rule).

Same held-out games as distill.py on the az6-data{0..3} samples (the last 10%
of games), subsampled:

    level_losses.py runs/az7-drop/imitation.pkl runs/az8-levelw/imitation.pkl ... [--samples 200000]
"""
import argparse
from pathlib import Path

import jax
import numpy as np

from training.zero import train as T
from training.zero.network import make_forward

p = argparse.ArgumentParser()
p.add_argument('ckpts', nargs='+')
p.add_argument('--samples', type=int, default=200_000)
a = p.parse_args()

parts = [dict(np.load(f'runs/az6-data{i}/distill_samples.npz')) for i in range(4)]
data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
gid = T.game_ids(data)
held = np.flatnonzero(gid >= int(round((gid.max() + 1) * 0.9)))
idx = np.sort(np.random.default_rng(0).choice(held, min(a.samples, len(held)), replace=False))
test = {k: v[idx] for k, v in data.items()}

for path in a.ckpts:
    graphdef, params, _ = T.load_params(Path(path))
    loss = jax.jit(lambda prm, b: T.make_loss_fn(make_forward(graphdef))(prm, b)[1])
    ms = [jax.device_get(loss(params, {k: v[i:i + 4096] for k, v in test.items()}))
          for i in range(0, len(test['z']), 4096)]
    m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    print(f'{path:>40}: policy {m["policy_loss"]:.4f} = draw/piece {m["level_loss"]:.4f} + where '
          f'{m["policy_loss"] - m["level_loss"]:.4f} | top-1 {m["policy_top1"]:.3f} | value {m["value_loss"]:.4f}',
          flush=True)
