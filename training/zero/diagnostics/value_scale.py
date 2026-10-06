"""How much should search values count against the prior? (search value_scale)

    python -m training.zero.diagnostics.value_scale runs/az3-net/latest.pkl --scales 0.1 0.2 0.3 0.5 1.0

For one network and each value_scale: the tactics benchmark (tactics.py) at
`--sims` simulations, and a match against the same network at the baseline
scale (the first one), both sides searching with `--sims` simulations.
Writes runs/review/value_scale.json.
"""

import argparse
import json
from pathlib import Path

import jax

from training.zero import arena as AR
from training.zero import search as S
from training.zero import tactics as TA
from training.zero import train as T
from training.zero.network import make_forward


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--scales', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5, 1.0])
    p.add_argument('--sims', type=int, default=32)
    p.add_argument('--games', type=int, default=256)
    a = p.parse_args()

    graphdef, params, ckpt = T.load_params(a.checkpoint)
    forward = make_forward(graphdef)
    params = jax.device_put(params)
    tactics = TA.load()
    searches = {v: jax.jit(S.make_search(forward, a.sims, value_scale=v)) for v in a.scales}
    base = a.scales[0]
    out = {'checkpoint': str(a.checkpoint), 'iteration': ckpt['iteration'], 'sims': a.sims, 'results': []}
    for v in a.scales:
        r = {'value_scale': v, **TA.score_network(forward, params, tactics, (a.sims,), searches={a.sims: searches[v]})}
        if v != base:
            res, _ = AR.play(AR.Agent('net', name=f'vs{v}', params=params, search=searches[v]),
                             AR.Agent('net', name=f'vs{base}', params=params, search=searches[base]),
                             a.games, jax.random.key(int(v * 1000)))
            r['score_vs_baseline'] = res['score']
        out['results'].append(r)
        print(json.dumps(r), flush=True)
    path = T.RUNS / 'review' / 'value_scale.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
