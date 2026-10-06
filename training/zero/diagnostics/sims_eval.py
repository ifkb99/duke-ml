"""Same checkpoint vs minimax4 at different search budgets (identical seeds)."""
import sys, time
from pathlib import Path
import jax
from training.zero import arena as AR, search as S, train as T
from training.zero.minimax_pool import MinimaxPool
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
ckpt_path = Path(sys.argv[1]); games = int(sys.argv[2])
graphdef, params, ckpt = T.load_params(ckpt_path)
params = jax.device_put(params)
forward = make_forward(graphdef)
print(f"checkpoint iteration {ckpt['iteration']}", flush=True)
with MinimaxPool(16) as pool:
    for sims, depth in [(32, 16), (200, 40), (800, 80)]:
        t = time.time()
        net = AR.Agent('net', params=params, search=S.make_search(forward, sims, 16, depth))
        res, _ = AR.play(net, AR.Agent('minimax', name='minimax4', depth=4), games, jax.random.key(7), pool=pool)
        print(f"sims {sims:4d}: score {res['score']:.3f}  (W{res['wins']} D{res['draws']} L{res['losses']})  {time.time()-t:.0f}s", flush=True)
