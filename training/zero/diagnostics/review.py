"""Review checkpoints before deciding what to train next.

    python -m training.zero.diagnostics.review NAME=CHECKPOINT ... [--sims 32 128]

For each network, on the benchmark (2,100 positions, every move scored by
depth-8 minimax; see benchmark.py) and on its own self-play:
  - move quality of the raw policy and of the search at each simulation count
    (acc / regret / blunder / missed win / walked into loss / puzzles),
    overall and by game phase and by the kind of move minimax prefers
  - rules: policy probability on illegal moves (before masking)
  - value vs deep minimax: does the value head know who is winning?
  - value calibration on self-play games (evaluator.calibration)
Writes runs/review/<NAME>.json and prints a summary.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import benchmark as BM
from training.zero import rules as R
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero import train as T
from training.zero.network import make_forward, observe_batch

OUT = T.RUNS / 'review'
PHASES = (('opening', 0, 20), ('middle', 20, 60), ('late', 60, 999))


def kind_of(action):
    return np.where(action < R.N_MOVE_ACTIONS, 'move/strike',
                    np.where(action < R.DRAW_ACTION, 'command', 'draw'))


def review(name, path, sims, bench):
    graphdef, params, ckpt = T.load_params(path)
    forward = make_forward(graphdef)
    params = jax.device_put(params)
    n = len(bench['ply'])
    scores = bench['scores']
    best_action = np.nanargmax(scores, 1)
    best = np.nanmax(scores, 1)

    # Raw network on every benchmark position.
    pol, value, illegal_mass = np.zeros(n, np.int64), np.zeros(n), np.zeros(n)
    for i in range(0, n, 256):
        states = BM.states_from_arrays({k: bench[k][i:i + 256] for k in BM.STATE_KEYS})
        logits, v = forward(params, *observe_batch(states))
        logits = logits.astype(jnp.float32)
        p = jax.nn.softmax(logits)
        illegal_mass[i:i + 256] = np.asarray(jnp.sum(jnp.where(states.legal, 0.0, p), -1))
        pol[i:i + 256] = np.asarray(jnp.argmax(jnp.where(states.legal, logits, -jnp.inf), -1))
        value[i:i + 256] = np.asarray(v)

    chosen = {'policy': pol}
    for s in sims:
        search = jax.jit(S.make_search(forward, s))
        acts = np.zeros(n, np.int64)
        for i in range(0, n, 256):
            states = BM.states_from_arrays({k: bench[k][i:i + 256] for k in BM.STATE_KEYS})
            acts[i:i + 256] = np.asarray(search(params, jax.random.key(i), states, 0.0).action)
        chosen[f'search{s}'] = acts

    out = {'name': name, 'checkpoint': str(path), 'iteration': ckpt['iteration'],
           'config': {k: ckpt['config'].get(k) for k in ('channels', 'blocks', 'policy_head')}}
    ply = bench['ply']
    for who, acts in chosen.items():
        out[who] = BM.move_quality(scores, acts)
        for ph, lo, hi in PHASES:
            m = (ply >= lo) & (ply < hi)
            out[who][f'acc_{ph}'] = float(BM.move_quality(scores[m], acts[m])['acc'])
        # Accuracy when minimax's best move is of each kind, and how often we pick that kind.
        for k in ('move/strike', 'command', 'draw'):
            m = kind_of(best_action) == k
            if m.sum() >= 30:
                out[who][f'acc_when_best_is_{k}'] = float(BM.move_quality(scores[m], acts[m])['acc'])
            out[who][f'chooses_{k}'] = float((kind_of(acts) == k).mean())
    out['minimax_best_kind'] = {k: float((kind_of(best_action) == k).mean()) for k in ('move/strike', 'command', 'draw')}
    out['phase_positions'] = {ph: int(((ply >= lo) & (ply < hi)).sum()) for ph, lo, hi in PHASES}

    # Rules: probability on illegal moves before masking.
    out['illegal_mass_mean'] = float(illegal_mass.mean())
    out['illegal_mass_p90'] = float(np.quantile(illegal_mass, 0.9))

    # Value vs depth-8 minimax (best score for the player to move).
    clipped = np.clip(best, -30, 30)
    decisive = np.abs(best) >= 5
    out['value_vs_minimax'] = {
        'spearman': float(np.corrcoef(np.argsort(np.argsort(value)), np.argsort(np.argsort(clipped)))[0, 1]),
        'sign_agree_decisive': float((np.sign(value[decisive]) == np.sign(best[decisive])).mean()),
        'decisive_positions': int(decisive.sum()),
        'mean_value_forced_win': float(value[best >= BM.WIN].mean()) if (best >= BM.WIN).any() else None,
        'mean_value_forced_loss': float(value[best <= -BM.WIN].mean()) if (best <= -BM.WIN).any() else None,
    }

    # Calibration on its own self-play.
    from training.zero.evaluator import calibration
    sp = SP.make_selfplay(S.make_search(forward, 32), unroll=32)
    cal, table = calibration(forward, params, jax.random.key(11), sp)
    out['calibration'] = cal
    out['calibration_table'] = table
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('nets', nargs='+', help='NAME=CHECKPOINT')
    p.add_argument('--sims', type=int, nargs='+', default=[32, 128])
    a = p.parse_args()
    bench = BM.load_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in a.nets:
        name, path = spec.split('=', 1)
        r = review(name, Path(path), a.sims, bench)
        (OUT / f'{name}.json').write_text(json.dumps(r, indent=1))
        line = ' | '.join(f"{w} acc {r[w]['acc']:.3f} blunder {r[w]['blunder']:.3f}"
                          for w in ['policy'] + [f'search{s}' for s in a.sims])
        print(f"{name} (iteration {r['iteration']}): {line} | illegal mass {r['illegal_mass_mean']:.4f} "
              f"| value~minimax {r['value_vs_minimax']['spearman']:.2f} "
              f"| calibration error {r['calibration'].get('calibration_error', float('nan')):.3f}", flush=True)


if __name__ == '__main__':
    main()
