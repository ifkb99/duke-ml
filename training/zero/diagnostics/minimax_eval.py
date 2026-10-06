"""A checkpoint vs minimax at chosen depths, with timing.

    .venv/bin/python training/zero/diagnostics/minimax_eval.py runs/az2/latest.pkl 32 6 [8 ...]
"""
import sys, time
from pathlib import Path
import jax
from training.zero import arena as AR, search as S, train as T
from training.zero.minimax_pool import MinimaxPool
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
graphdef, params, ckpt = T.load_params(Path(sys.argv[1]))
games, depths = int(sys.argv[2]), [int(d) for d in sys.argv[3:]]
params = jax.device_put(params)
search = jax.jit(S.make_search(make_forward(graphdef), 32, 16, 16))
print(f"iteration {ckpt['iteration']}", flush=True)
with MinimaxPool(16) as pool:
    for d in depths:
        t = time.time()
        r, _ = AR.play(AR.Agent('net', params=params, search=search),
                       AR.Agent('minimax', name=f'minimax{d}', depth=d), games, jax.random.key(3), pool=pool)
        print(f"minimax{d}: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}), "
              f"{r['mean_plies']:.0f} plies, {time.time() - t:.0f}s", flush=True)
