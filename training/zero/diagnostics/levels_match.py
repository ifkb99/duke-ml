"""One network, two searches: grouped ("semantic if") levels vs the flat search,
at equal simulations (search.LEVELS), or B with its own count (is the grouped
search as strong with fewer simulations, i.e. faster?):

    levels_match.py runs/az7-drop/imitation.pkl draw-piece 128 1024 [seed] [levels_b] [sims_b]
"""
import sys, time
from pathlib import Path

import jax

from training.zero import arena as AR, search as S, train as T
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
path, levels_a, sims, games = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 21
levels_b = sys.argv[6] if len(sys.argv) > 6 else 'flat'
sims_b = int(sys.argv[7]) if len(sys.argv) > 7 else sims

graphdef, params, ckpt = T.load_params(path)
params = jax.device_put(params)


def agent(lv, n):
    chunks = next(c for c in range(1, games + 1) if games % c == 0 and games * n // c <= 8192)
    return AR.Agent('net', name=f'{path.parent.name}@{n}:{lv}', params=params,
                    search=jax.jit(S.chunked(S.make_search(make_forward(graphdef), n, levels=lv), chunks)))


agents = [agent(levels_a, sims), agent(levels_b, sims_b)]
t = time.time()
r, _ = AR.play(agents[0], agents[1], games, jax.random.key(seed))
print(f"{agents[0].name} vs {agents[1].name}: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}), "
      f"{r['mean_plies']:.0f} plies, {time.time() - t:.0f}s", flush=True)
