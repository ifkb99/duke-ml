"""Do Assembler value targets equal the true result from the mover's point of view?"""
import jax, jax.numpy as jnp, numpy as np
from training.zero import env as E, selfplay as SP, replay as RP

B, T = 128, 400
key = jax.random.key(0)
s = jax.vmap(E.init)(jax.random.split(key, B))
step_dec = jax.jit(jax.vmap(E.step_decision)); step_ch = jax.jit(jax.vmap(E.step_chance))
probs = jax.jit(jax.vmap(E.chance_probs))
asm = RP.Assembler(B)
players, games_out = [[] for _ in range(B)], []
recs = []
rng = np.random.default_rng(0)
for t in range(T):
    legal = np.asarray(s.legal)
    acts = np.array([rng.choice(np.flatnonzero(l)) if l.any() else 0 for l in legal], np.int32)
    after = step_dec(s, jnp.asarray(acts))
    key, k = jax.random.split(key)
    nxt = step_ch(after, jax.random.categorical(k, jnp.log(probs(after))))
    rec = {**SP.compact_state(s), **SP.one_hot_policy(jnp.asarray(acts)), 'has_policy': jnp.ones(B, bool),
           **SP.step_record(s, after, jnp.asarray(acts)), 'player': s.player, 'winner': after.winner}
    recs.append(jax.device_get(rec))
    key, k = jax.random.split(key)
    s = SP._reset_done(nxt, k)
chunk = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
player, winner = chunk.pop('player'), chunk.pop('winner')
# Ground truth per (t, b): find the game's final winner.
truth = np.full(player.shape, np.nan)
for b in range(B):
    start = 0
    for end in list(np.flatnonzero(chunk['done'][:, b])):
        w = winner[end, b]
        p = player[start:end + 1, b]
        truth[start:end + 1, b] = np.where(w < 0, 0, np.where(p == w, 1, -1))
        start = end + 1
# Assembler z in the same order: rebuild by running each column separately.
mism = tot = 0
for b in range(B):
    a = RP.Assembler(1)
    out = a.add({k: v[:, b:b + 1] for k, v in chunk.items()})
    if out is None: continue
    keep = out['has_policy']           # drop the extra draw-afterstate samples
    z = out['z'][keep]
    tr = truth[:, b][~np.isnan(truth[:, b])]
    n = min(len(z), len(tr))
    mism += int((z[:n] != tr[:n]).sum()); tot += n
print(f"checked {tot} positions, mismatches {mism}")
