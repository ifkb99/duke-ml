"""Does search beat the raw policy? And how different are the training targets from the prior?"""
import sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from training.zero import arena as AR, search as S, train as T, env as E, selfplay as SP
from training.zero.network import make_forward, observe_batch

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
games = int(sys.argv[1])
for name in sys.argv[2:]:
    graphdef, params, ckpt = T.load_params(Path(f'runs/az1/{name}.pkl'))
    params = jax.device_put(params); forward = make_forward(graphdef)
    search32 = S.make_search(forward, 32, 16, 16)

    # 1. Target statistics on self-play positions.
    sp = SP.make_selfplay(search32, unroll=48)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(1), 512))
    states, _ = sp(params, states, jax.random.key(2))          # advance into mid-game
    states = jax.tree.map(lambda x: x, states)
    dec = np.asarray(states.pending == E.NONE) & ~np.asarray(states.done)
    out = search32(params, jax.random.key(3), states, 1.0)
    logits, value = forward(params, *observe_batch(states))
    legal = np.asarray(states.legal)
    prior = np.asarray(jax.nn.softmax(jnp.where(states.legal, logits.astype(jnp.float32), -1e9)))
    w = np.asarray(out.action_weights)
    eps = 1e-12
    kl = (w * (np.log(w + eps) - np.log(prior + eps))).sum(-1)
    ent_w = -(w * np.log(w + eps)).sum(-1); ent_p = -(prior * np.log(prior + eps)).sum(-1)
    agree = w.argmax(-1) == prior.argmax(-1)
    act_agree = np.asarray(out.action) == prior.argmax(-1)
    nleg = legal.sum(-1)
    m = dec
    print(f"[{name}] iter {ckpt['iteration']}: {m.sum()} positions, legal median {np.median(nleg[m]):.0f}", flush=True)
    print(f"  KL(target||prior) mean {kl[m].mean():.3f} median {np.median(kl[m]):.3f}; "
          f"entropy target {ent_w[m].mean():.2f} prior {ent_p[m].mean():.2f}; "
          f"argmax agree {agree[m].mean():.2f}; chosen==prior argmax {act_agree[m].mean():.2f}", flush=True)
    print(f"  root value |v| {np.abs(np.asarray(value))[m].mean():.3f}, search value |v| {np.abs(np.asarray(out.search_value))[m].mean():.3f}, "
          f"corr {np.corrcoef(np.asarray(value)[m], np.asarray(out.search_value)[m])[0,1]:.3f}", flush=True)

    # 2. Search (32 sims) vs raw policy (one candidate = the prior's argmax).
    t = time.time()
    a = AR.Agent('net', name='search32', params=params, search=search32)
    b = AR.Agent('net', name='policy', params=params, search=S.make_search(forward, 2, 1, 4))
    r, _ = AR.play(a, b, games, jax.random.key(5))
    print(f"  search32 vs raw policy: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}) {time.time()-t:.0f}s", flush=True)
