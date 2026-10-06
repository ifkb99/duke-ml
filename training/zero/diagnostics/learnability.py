"""Is the value learnable from this data? Game-level split, fresh net, held-out curve."""
import sys, json, time
import jax, jax.numpy as jnp, numpy as np, optax
from flax import nnx
from training.zero import env as E
from training.zero.network import DukeZeroNet, make_forward

import os
ABLATE = os.environ.get('ABLATE', '')
print('ablate:', ABLATE or 'none')
d = dict(np.load(sys.argv[1]))
test = d['call'] >= int(sys.argv[2])
tr = {k: v[~test] for k, v in d.items()}; te = {k: v[test] for k, v in d.items()}
print(f"train {len(tr['z'])} samples, test {len(te['z'])} samples (separate games)")

def material(x):
    own = ((x['tile'] >= 0) & (x['owner'] == 0)).sum(-1); opp = ((x['tile'] >= 0) & (x['owner'] == 1)).sum(-1)
    return (own - opp).astype(np.float32)
m_tr, m_te = material(tr), material(te)
w = np.polyfit(m_tr, tr['z'], 1)
pred = np.clip(np.polyval(w, m_te), -1, 1)
print(f"material baseline: corr(material, z) test {np.corrcoef(m_te, te['z'])[0,1]:.3f}, MSE {np.mean((pred - te['z'])**2):.3f}, predict-0 {np.mean(te['z']**2):.3f}")

model = DukeZeroNet(96, 6, rngs=nnx.Rngs(0)); graphdef, params = nnx.split(model)
forward = make_forward(graphdef)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=1e-4))
opt = tx.init(params)
obs = jax.vmap(E.observe_arrays)

def inputs(b):
    a = [jnp.asarray(b[k], jnp.int32) for k in ('tile', 'owner', 'side', 'bags', 'pending')]
    planes, feats = obs(*a, jnp.asarray(b['ply']))
    if ABLATE == 'features': feats = jnp.zeros_like(feats)
    if ABLATE == 'planes': planes = jnp.zeros_like(planes)
    return planes, feats

@jax.jit
def step(params, opt, b):
    def loss(p):
        _, v = forward(p, *inputs(b))
        return jnp.mean((v - b['z']) ** 2)
    l, g = jax.value_and_grad(loss)(params)
    u, opt = tx.update(g, opt, params)
    return optax.apply_updates(params, u), opt, l

@jax.jit
def predict(params, b):
    return forward(params, *inputs(b))[1]

def test_mse(params, data, n=60000):
    idx = np.arange(min(n, len(data['z'])))
    vs = [np.asarray(predict(params, {k: v[idx[i:i + 4096]] for k, v in data.items()})) for i in range(0, len(idx), 4096)]
    v = np.concatenate(vs); z = data['z'][idx]
    return float(np.mean((v - z) ** 2)), float(np.corrcoef(v, z)[0, 1])

rng = np.random.default_rng(0)
n, bs = len(tr['z']), 1024
steps_per_epoch = n // bs
log = []
for s in range(int(float(sys.argv[3]) * steps_per_epoch) + 1):
    if s % (steps_per_epoch // 4) == 0:
        mse_te, c_te = test_mse(params, te); mse_tr, c_tr = test_mse(params, tr)
        rec = dict(epoch=s / steps_per_epoch, train_mse=mse_tr, test_mse=mse_te, test_corr=c_te)
        log.append(rec); print(json.dumps(rec), flush=True)
    idx = rng.integers(0, n, bs)
    params, opt, _ = step(params, opt, {k: v[idx] for k, v in tr.items()})
json.dump(log, open(sys.argv[4], 'w'))
