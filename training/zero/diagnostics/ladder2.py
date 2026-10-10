"""Head-to-head between two checkpoints (any runs), same search as training evals
(32 simulations each), or each with its own simulation count:

    .venv/bin/python training/zero/diagnostics/ladder2.py runs/az2/ckpt_001000.pkl runs/az2/ckpt_000800.pkl 256
    ladder2.py <a.pkl> <b.pkl> <games> [seed] [sims_a sims_b]   # e.g. equal phone time
"""
import sys, time
from pathlib import Path
import jax
from training.zero import arena as AR, search as S, train as T
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
a_path, b_path, games = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sims = [int(x) for x in sys.argv[5:7]] or [32, 32]
agents = []
for p, n in zip((a_path, b_path), sims):
    graphdef, params, ckpt = T.load_params(p)
    # A tree holds (sims + 1) x 3,620 children per game: split deep searches.
    chunks = next(c for c in range(1, games + 1) if games % c == 0 and games * n // c <= 8192)
    search = jax.jit(S.chunked(S.make_search(make_forward(graphdef), n), chunks))
    name = f"{p.parent.name}:{ckpt['iteration']}" + (f'@{n}' if n != 32 else '')
    agents.append(AR.Agent('net', name=name, params=jax.device_put(params), search=search))
t = time.time()
r, _ = AR.play(agents[0], agents[1], games, jax.random.key(int(sys.argv[4]) if len(sys.argv) > 4 else 21))
print(f"{agents[0].name} vs {agents[1].name}: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}), "
      f"{r['mean_plies']:.0f} plies, {time.time() - t:.0f}s", flush=True)
