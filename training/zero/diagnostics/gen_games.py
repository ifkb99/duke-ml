"""Generate fresh self-play samples with a checkpoint; save with a time-ordered game split."""
import sys
from pathlib import Path
import jax, numpy as np
from training.zero import env as E, search as S, selfplay as SP, train as T, replay as RP
from training.zero.network import make_forward

graphdef, params, ckpt = T.load_params(Path(f'runs/az1/{sys.argv[1]}.pkl'))
params = jax.device_put(params); forward = make_forward(graphdef)
B, calls = 1024, int(sys.argv[2])
sp = SP.make_selfplay(S.make_search(forward, 32), unroll=32)
states = jax.vmap(E.init)(jax.random.split(jax.random.key(10), B))
asm, parts = RP.Assembler(B), []
for i in range(calls):
    states, rec = sp(params, states, jax.random.key(1000 + i))
    rec = jax.device_get(rec); rec.pop('search_value')
    out = asm.add(rec)
    if out is not None:
        out['call'] = np.full(len(out['z']), i, np.int32)
        parts.append(out)
    print(i, flush=True)
d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
np.savez(sys.argv[3], **d)
print('saved', len(d['z']))
