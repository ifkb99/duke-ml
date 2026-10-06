"""Checkpoint vs checkpoint, same search settings as training evals."""
import sys, time
import jax
from pathlib import Path
from training.zero import arena as AR, search as S, train as T
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
run = Path('runs/az1')
names = ['imitation', 'ckpt_000200', 'ckpt_001000', 'ckpt_002200']
agents, search = {}, None
for n in names:
    graphdef, params, _ = T.load_params(run / f'{n}.pkl')
    if search is None:
        search = S.make_search(make_forward(graphdef), 32)
    agents[n] = AR.Agent('net', name=n, params=jax.device_put(params), search=search)
games = int(sys.argv[1])
for a, b in [('ckpt_002200', 'ckpt_000200'), ('ckpt_001000', 'ckpt_000200'),
             ('ckpt_002200', 'ckpt_001000'), ('ckpt_000200', 'imitation'), ('ckpt_002200', 'imitation')]:
    t = time.time()
    r, _ = AR.play(agents[a], agents[b], games, jax.random.key(11))
    print(f"{a} vs {b}: score {r['score']:.3f} (W{r['wins']} D{r['draws']} L{r['losses']}) {time.time()-t:.0f}s", flush=True)
