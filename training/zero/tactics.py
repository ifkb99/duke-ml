"""
Tactics benchmark: positions with a forced outcome, which minimax proves
whatever its evaluation heuristic thinks of the rest of the position.

  win   a forced win exists: does the network play a move that keeps it?
  loss  some moves walk into a forced loss (at least DANGER of them): does
        the network avoid them?

benchmark.py's accuracy/blunder metrics compare moves by minimax's heuristic,
which over-rates drawing (EXPERIMENTS.md, "Network review"); forced results
(scores of ±100,000) don't depend on it.

Build (fresh self-play from a checkpoint, scanned and scored at --depth):

    python -m training.zero.tactics build runs/az3-net/latest.pkl
    python -m training.zero.tactics score runs/az3-net/latest.pkl

Positions are new games the networks we compare never trained on.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from training.zero import benchmark as BM
from training.zero import env as E
from training.zero import rules as R
from training.zero import search as S
from training.zero import ts_bridge as B
from training.zero.network import make_forward, observe_batch

TACTICS = BM.RUNS / 'benchmark' / 'tactics.npz'
WIN = BM.WIN
DANGER = 0.3        # loss positions: at least this fraction of legal moves lose by force
KEYS = ('tile', 'owner', 'side', 'bags', 'ply')


def fresh_positions(ckpt: Path, n: int, seed: int) -> dict:
    """Decision positions (no tile in hand) from new self-play games, deduplicated."""
    from training.zero import selfplay as SP
    from training.zero.train import load_params
    graphdef, params, _ = load_params(ckpt)
    params = jax.device_put(params)
    sp = SP.make_selfplay(S.make_search(make_forward(graphdef), 32, 16, 16), unroll=8)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(seed), 1024))
    parts, seen, total, t = [], set(), 0, time.time()
    for i in range(10_000):
        states, _ = sp(params, states, jax.random.key(seed * 100_000 + i))
        host = jax.device_get(states)
        keep = []
        for j in np.flatnonzero((host.pending == E.NONE) & ~host.done):
            key = host.tile[j].tobytes() + host.owner[j].tobytes() + host.side[j].tobytes() + host.bags[j].tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(j)
        parts.append({k: np.asarray(getattr(host, k))[keep] for k in KEYS})
        total += len(keep)
        if i % 20 == 0:
            print(f'  {total} positions ({time.time() - t:.0f}s)', flush=True)
        if total >= n:
            break
    d = {k: np.concatenate([p[k] for p in parts])[:n] for k in KEYS}
    d['player'] = np.ones(len(d['ply']), np.int32)     # canonical frame = P2's frame
    return d


def scan(d: dict, depth: int, workers: int | None) -> np.ndarray:
    """scores[i, action]: every legal move's minimax score (NaN = illegal)."""
    from training.zero.minimax_pool import MinimaxPool
    host = jax.device_get(BM.states_from_arrays(d))
    n = len(d['ply'])
    scores = np.full((n, R.N_ACTIONS), np.nan, np.float32)
    t = time.time()
    with MinimaxPool(workers) as pool:
        for start in range(0, n, 2048):
            idx = range(start, min(n, start + 2048))
            singles = [jax.tree.map(lambda x: x[i], host) for i in idx]
            for i, st, rep in zip(idx, singles, pool.score_moves([B.state_to_ts(s) for s in singles], depth)):
                duke = B.duke_square(st.tile, st.owner)
                for m in rep['moves']:
                    scores[i, B.move_to_actions(m['move'], int(st.player), duke)[0][1]] = m['score']
                if rep['draw'] is not None:
                    scores[i, R.DRAW_ACTION] = rep['draw']
            print(f'  scanned {min(n, start + 2048)}/{n} ({time.time() - t:.0f}s)', flush=True)
    return scores


def classify(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    legal = ~np.isnan(scores)
    best = np.nanmax(np.where(legal, scores, -np.inf), 1)
    losing = (np.nan_to_num(scores, nan=0) <= -WIN) & legal
    frac = losing.sum(1) / np.maximum(legal.sum(1), 1)
    win = best >= WIN
    loss = ~win & (best > -WIN) & (frac >= DANGER)
    return win, loss


def build(a):
    d = fresh_positions(a.checkpoint, a.positions, a.seed)
    print(f'{len(d["ply"])} positions; scoring every move at depth {a.depth}', flush=True)
    scores = scan(d, a.depth, a.workers)
    win, loss = classify(scores)
    keep = np.flatnonzero(win | loss)
    out = {k: v[keep] for k, v in d.items()}
    out['scores'] = scores[keep]
    out['kind'] = np.where(win[keep], 'win', 'loss')
    TACTICS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TACTICS, depth=a.depth, source=str(a.checkpoint), scanned=len(d['ply']), **out)
    print(f'saved {TACTICS}: {win.sum()} win positions, {loss.sum()} loss positions '
          f'(of {len(d["ply"])} scanned)')


def load() -> dict:
    with np.load(TACTICS) as f:
        return {k: f[k] for k in f.files}


def score_network(forward, params, t: dict, simulations=(32, 128), chunk=256, searches=None) -> dict:
    """win_kept / loss_avoided for the raw policy and for the search at each
    simulation count (`searches`: already-jitted searches by simulation count)."""
    n = len(t['ply'])
    chosen = {'policy': np.zeros(n, np.int64), **{f'search{s}': np.zeros(n, np.int64) for s in simulations}}
    searches = {s: (searches or {}).get(s) or jax.jit(S.make_search(forward, s, 16, 16)) for s in simulations}
    for i in range(0, n, chunk):
        states = BM.states_from_arrays({k: t[k][i:i + chunk] for k in KEYS + ('player',)})
        logits, _ = forward(params, *observe_batch(states))
        chosen['policy'][i:i + chunk] = np.asarray(
            jax.numpy.argmax(jax.numpy.where(states.legal, logits.astype(jax.numpy.float32), -jax.numpy.inf), -1))
        for s, search in searches.items():
            chosen[f'search{s}'][i:i + chunk] = np.asarray(search(params, jax.random.key(i), states, 0.0).action)
    win, loss = t['kind'] == 'win', t['kind'] == 'loss'
    out = {}
    for who, acts in chosen.items():
        got = t['scores'][np.arange(n), acts]
        out[f'{who}_win_kept'] = float((got[win] >= WIN).mean())
        out[f'{who}_loss_avoided'] = float((got[loss] > -WIN).mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('checkpoint', type=Path)
    b.add_argument('--positions', type=int, default=120_000)
    b.add_argument('--depth', type=int, default=6)
    b.add_argument('--workers', type=int, default=None)
    b.add_argument('--seed', type=int, default=77)
    s = sub.add_parser('score')
    s.add_argument('checkpoint', type=Path)
    s.add_argument('--sims', type=int, nargs='+', default=[32, 128])
    a = p.parse_args()
    if a.cmd == 'build':
        build(a)
    else:
        from training.zero.train import load_params
        graphdef, params, ckpt = load_params(a.checkpoint)
        print(json.dumps({'iteration': ckpt['iteration'], **score_network(
            make_forward(graphdef), jax.device_put(params), load(), a.sims)}, indent=1))


if __name__ == '__main__':
    main()
