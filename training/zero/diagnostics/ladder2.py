"""Head-to-head between two checkpoints (any runs), same search as training evals.

    .venv/bin/python training/zero/diagnostics/ladder2.py runs/az2/ckpt_001000.pkl runs/az2/ckpt_000800.pkl 256
"""
import sys, time
from pathlib import Path
import jax
from training.zero import arena as AR, search as S, train as T
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
a_path, b_path, games = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
agents = []
for p in (a_path, b_path):
    graphdef, params, ckpt = T.load_params(p)
    search = jax.jit(S.make_search(make_forward(graphdef), 32, 16, 16))
    agents.append(AR.Agent('net', name=f"{p.parent.name}:{ckpt['iteration']}", params=jax.device_put(params), search=search))
t = time.time()
r, _ = AR.play(agents[0], agents[1], games, jax.random.key(int(sys.argv[4]) if len(sys.argv) > 4 else 21))
print(f"{agents[0].name} vs {agents[1].name}: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}), "
      f"{r['mean_plies']:.0f} plies, {time.time() - t:.0f}s", flush=True)
