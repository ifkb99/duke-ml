"""A checkpoint vs minimax with and without quiescence search, at chosen depths and simulations.

    .venv/bin/python training/zero/diagnostics/quiescence_eval.py runs/az7-drop/imitation.pkl 64 32,128 4 4q 6 6q

Depths ending in `q` turn on minimax's quiescence search. Every pairing uses the
same game seed, so plain and quiescence minimax face the same openings and draws
(the comparison is paired). Each minimax configuration gets a fresh process pool:
the servers keep their transposition tables between requests, and entries from a
quiescence search would leak into a plain one.
"""
import json, sys, time
from pathlib import Path
import jax
from training.zero import arena as AR, search as S, train as T
from training.zero.minimax_pool import MinimaxPool
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')
graphdef, params, ckpt = T.load_params(Path(sys.argv[1]))
games = int(sys.argv[2])
sims_list = [int(x) for x in sys.argv[3].split(',')]
configs = [(int(d.rstrip('q')), d.endswith('q')) for d in sys.argv[4:]]
params = jax.device_put(params)
forward = make_forward(graphdef)
print(f"checkpoint {sys.argv[1]} (iteration {ckpt.get('iteration', '?')}), {games} games per pairing", flush=True)

for sims in sims_list:
    search = jax.jit(S.make_search(forward, sims))
    for depth, q in configs:
        name = f"minimax{depth}{'q' if q else ''}"
        t = time.time()
        with MinimaxPool(16) as pool:
            r, _ = AR.play(AR.Agent('net', params=params, search=search),
                           AR.Agent('minimax', name=name, depth=depth, quiescence=q),
                           games, jax.random.key(3), pool=pool)
        row = {'sims': sims, 'opponent': name, 'score': round(r['score'], 4), 'wins': r['wins'],
               'draws': r['draws'], 'losses': r['losses'], 'mean_plies': round(r['mean_plies'], 1),
               'seconds': round(time.time() - t)}
        print(json.dumps(row), flush=True)
