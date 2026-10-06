"""Value loss on fresh self-play games (never trained on) vs the logged training loss."""
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from training.zero import env as E, search as S, selfplay as SP, train as T, replay as RP
from training.zero.network import make_forward

graphdef, params, ckpt = T.load_params(Path(f'runs/az1/{sys.argv[1]}.pkl'))
params = jax.device_put(params); forward = make_forward(graphdef)
B = 256
sp = SP.make_selfplay(S.make_search(forward, 32, 16, 16), unroll=32)
states = jax.vmap(E.init)(jax.random.split(jax.random.key(10), B))
asm, parts = RP.Assembler(B), []
for i in range(12):
    states, rec = sp(params, states, jax.random.key(100 + i))
    rec = jax.device_get(rec); rec.pop('search_value')
    out = asm.add(rec)
    if out is not None: parts.append(out)
d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
obs = jax.jit(jax.vmap(E.observe_arrays))
vs = []
for i in range(0, len(d['z']), 1024):
    sl = slice(i, i + 1024)
    a = [jnp.asarray(d[k][sl], jnp.int32) for k in ('tile', 'owner', 'side', 'bags', 'pending')]
    _, v = forward(params, *obs(*a, jnp.asarray(d['ply'][sl])))
    vs.append(np.asarray(v, np.float32))
v, z = np.concatenate(vs), d['z']
print(f"{sys.argv[1]} (iter {ckpt['iteration']}): {len(z)} fresh samples")
print(f"  held-out value MSE {np.mean((v - z) ** 2):.3f}   predict-0 baseline {np.mean(z ** 2):.3f}   predict-mean baseline {np.var(z):.3f}")
print(f"  corr(v, z) {np.corrcoef(v, z)[0, 1]:.3f}   mean v {v.mean():+.3f}  mean z {z.mean():+.3f}")
