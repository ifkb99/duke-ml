"""One checkpoint at several simulation counts vs TypeScript minimax, all levels in parallel
(one thread per level, sharing one minimax pool).

    .venv/bin/python training/zero/diagnostics/sims_vs_minimax.py runs/az4/ckpt_001400.pkl 8 32 32 128 512 1024
"""
import sys, time, threading, jax
from pathlib import Path
from training.zero import arena as AR, search as S, train as T
from training.zero.network import make_forward
from training.zero.minimax_pool import MinimaxPool

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
ckpt_path, depth, games = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
levels = [int(x) for x in sys.argv[4:]]
graphdef, params, ckpt = T.load_params(ckpt_path)
params = jax.device_put(params)
print(f"{ckpt_path.parent.name}:{ckpt['iteration']} vs minimax{depth}, {games} games per level", flush=True)
# A tree holds (sims + 1) x 3,620 children per game: split big searches so they fit.
chunks = lambda sims: max(1, min(games, games * sims // 8192))
results = {}

def run(sims, pool):
    search = jax.jit(S.chunked(S.make_search(make_forward(graphdef), sims), chunks(sims)))
    net = AR.Agent('net', name=f"{ckpt_path.parent.name}@{sims}", params=params, search=search)
    mm = AR.Agent('minimax', name=f'minimax{depth}', depth=depth)
    t = time.time()
    r, _ = AR.play(net, mm, games, jax.random.key(7), pool=pool)
    results[sims] = r
    print(f"{sims:5d} sims: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}), "
          f"{r['mean_plies']:.0f} plies, {time.time() - t:.0f}s", flush=True)

with MinimaxPool(28) as pool:
    threads = [threading.Thread(target=run, args=(s, pool)) for s in levels]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
