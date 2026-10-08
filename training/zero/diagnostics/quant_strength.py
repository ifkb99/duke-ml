"""How much quantizing the weights (quantize.py) changes a network: on the
benchmark positions, in float32, against the unquantized network. Also writes
the quantized checkpoints to runs/quant/<run>-q<bits>/ for head-to-head
matches with ladder2.py.

    .venv/bin/python training/zero/diagnostics/quant_strength.py runs/az5/ckpt_001600.pkl [positions]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from training.zero import benchmark as BM
from training.zero import quantize as Q
from training.zero import train as T
from training.zero.network import DukeZeroNet, make_forward, observe_batch

path = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
_, _, ckpt = T.load_params(path)
c = ckpt['config']
graphdef, ref = nnx.split(DukeZeroNet(c['channels'], c['blocks'], c.get('policy_head', 'dense'),
                                      dtype=jnp.float32, rngs=nnx.Rngs(0)))
forward = jax.jit(make_forward(graphdef))

b = BM.load_benchmark()
idx = np.random.default_rng(0).choice(len(b['ply']), min(n, len(b['ply'])), replace=False)
states = BM.states_from_arrays({k: b[k][idx] for k in BM.STATE_KEYS})
planes, feats = observe_batch(states)
legal = np.asarray(states.legal)


def run(pure):
    nnx.replace_by_pure_dict(ref, jax.tree.map(jnp.asarray, pure))
    outs = [forward(ref, planes[i:i + 1024], feats[i:i + 1024]) for i in range(0, len(idx), 1024)]
    logits = np.concatenate([np.asarray(o[0]) for o in outs])
    value = np.concatenate([np.asarray(o[1]) for o in outs])
    logits = np.where(legal, logits, -np.inf)
    logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
    return logp, value


base_logp, base_value = run(ckpt['params'])
out_dir = Path('runs/quant')
for bits in (8, 4):
    t = time.time()
    qp = Q.quantize_params(ckpt['params'], bits)
    logp, value = run(qp)
    p = np.exp(base_logp)
    kl = np.where(legal, p * (base_logp - np.where(legal, logp, 0)), 0).sum(1)
    rec = {'checkpoint': str(path), 'iteration': ckpt['iteration'], 'bits': bits, 'block': Q.BLOCK,
           'positions': len(idx),
           'top1_agrees': float((logp.argmax(1) == base_logp.argmax(1)).mean()),
           'policy_kl': float(kl.mean()), 'policy_kl_max': float(kl.max()),
           'value_mae': float(np.abs(value - base_value).mean()),
           'value_max_err': float(np.abs(value - base_value).max())}
    print(json.dumps(rec), f'({time.time() - t:.0f}s)', flush=True)
    d = out_dir / f'{path.parent.name}-q{bits}'
    d.mkdir(parents=True, exist_ok=True)
    with (d / path.name).open('wb') as f:
        pickle.dump({**ckpt, 'params': qp, 'opt_state': None}, f)
    with (out_dir / 'results.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
