"""One-step consistency of the value head: v(s) vs the values of its children.

q(a) = reward if the move ends the game, else -v(next) if the turn passes,
else v(afterstate) (draws: same player still to move). For a sensible value
function, v(s) should track max_a q(a) (or the policy-weighted average).
"""
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from training.zero import env as E, search as S, selfplay as SP, train as T
from training.zero.network import make_forward, observe_batch

graphdef, params, ckpt = T.load_params(Path(f'runs/az1/{sys.argv[1]}.pkl'))
params = jax.device_put(params); forward = make_forward(graphdef)
sp = SP.make_selfplay(S.make_search(forward, 32), unroll=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
states = jax.vmap(E.init)(jax.random.split(jax.random.key(1), 256))
states, _ = sp(params, states, jax.random.key(2))

def net(s, chunk=256):
    n = s.tile.shape[0]; ls, vs = [], []
    for i in range(0, n, chunk):
        part = jax.tree.map(lambda x: x[i:i + chunk], s)
        logits, v = forward(params, *observe_batch(part))
        ls.append(np.asarray(logits.astype(jnp.float32))); vs.append(np.asarray(v.astype(jnp.float32)))
    return np.concatenate(ls), np.concatenate(vs)

logits, v = net(states)
legal = np.asarray(states.legal); pending = np.asarray(states.pending); done = np.asarray(states.done)
rows = [i for i in range(len(v)) if pending[i] == E.NONE and not done[i]]
pairs = [(i, a) for i in rows for a in np.flatnonzero(legal[i])]
si = jnp.asarray([p[0] for p in pairs]); acts = jnp.asarray([p[1] for p in pairs])
parent = jax.tree.map(lambda x: x[si], states)
after = jax.jit(jax.vmap(E.step_decision))(parent, acts)
_, va = net(after)
rew, dn, sw = (np.asarray(x) for x in (after.reward, after.done, after.switched))
q = np.where(dn, rew, np.where(sw, -va, va))

qmax, qavg, qmin = [], [], []
for i in rows:
    m = np.array([p[0] == i for p in pairs])
    qi = q[m]; acts_i = np.asarray(acts)[m]
    pr = np.exp(logits[i, acts_i] - logits[i, acts_i].max()); pr /= pr.sum()
    qmax.append(qi.max()); qmin.append(qi.min()); qavg.append((pr * qi).sum())
vr = v[rows]; qmax, qavg, qmin = map(np.array, (qmax, qavg, qmin))
print(f"{sys.argv[1]} (iter {ckpt['iteration']}): {len(rows)} positions, {len(pairs)} children")
print(f"  mean v(s) {vr.mean():+.3f}  mean max q {qmax.mean():+.3f}  mean policy-avg q {qavg.mean():+.3f}  mean min q {qmin.mean():+.3f}")
for name, x in [('max q', qmax), ('policy-avg q', qavg), ('-max q (sign-flipped)', -qmax)]:
    print(f"  corr(v(s), {name}) = {np.corrcoef(vr, x)[0, 1]:+.3f}   mean |v - x| = {np.abs(vr - x).mean():.3f}")
print(f"  spread max q - min q: mean {(qmax - qmin).mean():.3f}; positions with a winning move: {(qmax >= 0.999).sum()}, v there {vr[qmax >= 0.999].mean() if (qmax >= 0.999).any() else float('nan'):+.3f}")
