"""
A fixed test set of positions with every move scored by deep minimax.

Game results are noisy (±0.12 for a 64-game match). A few thousand fixed
positions where we know how good every move is give an almost noise-free
measure of move quality, and show *what kind* of play is improving.

Build once (positions from az2 self-play at three strengths, every legal move
scored by minimax; ~10 min of sampling), then score the moves deeper:

    uv run python -m training.zero.benchmark build            # depth 6
    uv run python -m training.zero.benchmark relabel --depth 8

Score a checkpoint:

    uv run python -m training.zero.benchmark score runs/az2/ckpt_001000.pkl

Metrics (each for the raw policy's top move and for the 32-simulation search):
  acc            chose a move within 0.25 points of the best (≈ a best move;
                 several moves often tie)
  regret         average points lost vs the best move (capped at 10 per position)
  blunder        lost 3+ points (roughly a tile) vs the best move
  missed_win     a forced win existed and the chosen move doesn't keep it
  walked_into_loss  the chosen move allows a forced loss that was avoidable
  puzzle_acc     accuracy on "puzzles": positions where the best move is 3+
                 points better than every other move
Scores are minimax evaluation points for the player to move (a typical tile
is worth a few points; ±99,999 is a forced win/loss).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero import ts_bridge as B
from training.zero.network import make_forward, observe_batch

RUNS = Path(__file__).resolve().parents[2] / 'runs'
BENCH = RUNS / 'benchmark' / 'positions.npz'
STATE_KEYS = ('tile', 'owner', 'side', 'bags', 'player', 'ply')
WIN = 90_000
TOL, BLUNDER, PUZZLE_GAP, REGRET_CAP = 0.25, 3.0, 3.0, 10.0


def states_from_arrays(d: dict) -> E.State:
    """Batched decision States (pending NONE) from stored arrays."""
    n = len(d['ply'])
    s = E.State(
        tile=jnp.asarray(d['tile'], jnp.int32), owner=jnp.asarray(d['owner'], jnp.int32),
        side=jnp.asarray(d['side'], jnp.int32), bags=jnp.asarray(d['bags'], jnp.int32),
        player=jnp.asarray(d['player'], jnp.int32), pending=jnp.full(n, E.NONE, jnp.int32),
        ply=jnp.asarray(d['ply'], jnp.int32), done=jnp.zeros(n, bool),
        winner=jnp.full(n, -1, jnp.int32), reward=jnp.zeros(n, jnp.float32),
        switched=jnp.zeros(n, bool), legal=jnp.zeros((n, R.N_ACTIONS), bool),
        place_ok=jnp.zeros((n, 4), bool), overflow=jnp.zeros(n, jnp.int32))
    return jax.jit(jax.vmap(E._with_legal))(s)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def sample_positions(ckpt: Path, n: int, seed: int) -> dict:
    """Decision positions from self-play of a checkpoint, spread over the game."""
    from training.zero.train import load_params
    graphdef, params, _ = load_params(ckpt)
    params = jax.device_put(params)
    sp = SP.make_selfplay(S.make_search(make_forward(graphdef), 32, 16, 16), unroll=8)
    states = jax.vmap(E.init)(jax.random.split(jax.random.key(seed), 512))
    rng = np.random.default_rng(seed)
    pool = {k: [] for k in STATE_KEYS}
    for i in range(40):           # 320 moves per slot: covers whole games
        states, _ = sp(params, states, jax.random.key(seed * 1000 + i))
        if i < 2:
            continue
        host = jax.device_get(states)
        ok = (host.pending == E.NONE) & ~host.done
        take = np.flatnonzero(ok)
        take = rng.choice(take, min(len(take), 48), replace=False)
        for k in STATE_KEYS:
            pool[k].append(np.asarray(getattr(host, k))[take])
    d = {k: np.concatenate(v) for k, v in pool.items()}
    # Stratify by game phase so endgames aren't drowned out by openings.
    buckets = [(0, 20), (20, 50), (50, 100), (100, E.MAX_PLY)]
    keep = []
    for lo, hi in buckets:
        idx = np.flatnonzero((d['ply'] >= lo) & (d['ply'] < hi))
        keep.append(rng.choice(idx, min(len(idx), n // len(buckets)), replace=False))
    keep = np.concatenate(keep)
    return {k: v[keep] for k, v in d.items()}


def label(d: dict, depth: int, workers: int | None) -> np.ndarray:
    """scores[i, action] = minimax score of that move for the player to move (NaN = not scored)."""
    from training.zero.minimax_pool import MinimaxPool
    states = states_from_arrays(d)
    host = jax.device_get(states)
    n = len(d['ply'])
    singles = [jax.tree.map(lambda x: x[i], host) for i in range(n)]
    scores = np.full((n, R.N_ACTIONS), np.nan, np.float32)
    with MinimaxPool(workers) as pool:
        t = time.time()
        for start in range(0, n, 256):
            chunk = range(start, min(n, start + 256))
            replies = pool.score_moves([B.state_to_ts(singles[i]) for i in chunk], depth)
            for i, rep in zip(chunk, replies):
                st = singles[i]
                duke = B.duke_square(st.tile, st.owner)
                for m in rep['moves']:
                    a = B.move_to_actions(m['move'], int(st.player), duke)[0][1]
                    scores[i, a] = m['score']
                if rep['draw'] is not None:
                    scores[i, R.DRAW_ACTION] = rep['draw']
            print(f'  scored {min(n, start + 256)}/{n} ({time.time() - t:.0f}s)', flush=True)
    legal = np.asarray(host.legal)
    missing = legal & np.isnan(scores)
    print(f'legal moves without a score: {missing.sum()} of {legal.sum()}')
    return scores


def build(args):
    parts = []
    for i, it in enumerate(args.iterations):
        print(f'sampling from {args.run} iteration {it}', flush=True)
        d = sample_positions(RUNS / args.run / f'ckpt_{it:06d}.pkl', args.per_checkpoint, seed=i + 1)
        d['source_iteration'] = np.full(len(d['ply']), it, np.int32)
        parts.append(d)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f'{len(d["ply"])} positions; scoring every move at depth {args.depth}', flush=True)
    d['scores'] = label(d, args.depth, args.workers)
    BENCH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BENCH, depth=args.depth, **d)
    print(f'saved {BENCH}')
    summarize(d)


def relabel(args):
    """Re-score the same positions at another depth."""
    d = load_benchmark()
    print(f'{len(d["ply"])} positions; re-scoring every move at depth {args.depth} (was {int(d["depth"])})', flush=True)
    d['scores'] = label(d, args.depth, args.workers)
    d['depth'] = np.int32(args.depth)
    np.savez_compressed(BENCH, **d)
    print(f'saved {BENCH}')
    summarize(d)


def summarize(d):
    s = d['scores']
    best = np.nanmax(s, 1)
    srt = -np.sort(-np.nan_to_num(s, nan=-np.inf), 1)
    gap = srt[:, 0] - srt[:, 1]
    print(f"positions {len(best)}; legal moves per position: median {np.median((~np.isnan(s)).sum(1)):.0f}; "
          f"forced win available: {(best >= WIN).mean():.1%}; forced loss: {(best <= -WIN).mean():.1%}; "
          f"puzzles (one move {PUZZLE_GAP}+ better): {(gap >= PUZZLE_GAP).mean():.1%}")


# ---------------------------------------------------------------------------
# Scoring a network
# ---------------------------------------------------------------------------

def load_benchmark(path: Path = BENCH) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def move_quality(scores, actions) -> dict:
    """Metrics for one chosen action per position."""
    n = len(actions)
    best = np.nanmax(scores, 1)
    chosen = scores[np.arange(n), actions]
    scored = ~np.isnan(chosen)
    chosen = np.where(scored, chosen, -np.inf)
    regret = np.clip(best - chosen, 0, REGRET_CAP)
    srt = -np.sort(-np.nan_to_num(scores, nan=-np.inf), 1)
    puzzle = (srt[:, 0] - srt[:, 1]) >= PUZZLE_GAP
    acc = regret <= TOL
    win_avail, loss_avoidable = best >= WIN, best > -WIN
    return {
        'acc': float(acc.mean()),
        'regret': float(regret.mean()),
        'blunder': float((regret >= BLUNDER).mean()),
        'missed_win': float((chosen[win_avail] < WIN).mean()) if win_avail.any() else float('nan'),
        'walked_into_loss': float((chosen[loss_avoidable] <= -WIN).mean()),
        'puzzle_acc': float(acc[puzzle].mean()) if puzzle.any() else float('nan'),
        'unscored': float((~scored).mean()),
    }


def score_network(forward, params, bench: dict, simulations=32, chunk=256, search=None) -> dict:
    """Benchmark metrics for the policy's top move and for the search's move."""
    search = search or jax.jit(S.make_search(forward, simulations, 16, 16))
    n = len(bench['ply'])
    pol, srch = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for i in range(0, n, chunk):
        part = {k: bench[k][i:i + chunk] for k in STATE_KEYS}
        states = states_from_arrays(part)
        logits, _ = forward(params, *observe_batch(states))
        pol[i:i + chunk] = np.asarray(jnp.argmax(jnp.where(states.legal, logits.astype(jnp.float32), -jnp.inf), -1))
        srch[i:i + chunk] = np.asarray(search(params, jax.random.key(i), states, 0.0).action)
    out = {}
    for name, acts in (('policy', pol), ('search', srch)):
        out.update({f'{name}_{k}': v for k, v in move_quality(bench['scores'], acts).items()})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--run', default='az2')
    b.add_argument('--iterations', type=int, nargs='+', default=[200, 600, 1000])
    b.add_argument('--per-checkpoint', type=int, default=700)
    b.add_argument('--depth', type=int, default=6)
    b.add_argument('--workers', type=int, default=None)
    r = sub.add_parser('relabel')
    r.add_argument('--depth', type=int, default=8)
    r.add_argument('--workers', type=int, default=None)
    s = sub.add_parser('score')
    s.add_argument('checkpoint', type=Path)
    a = p.parse_args()
    if a.cmd == 'build':
        build(a)
    elif a.cmd == 'relabel':
        relabel(a)
    else:
        from training.zero.train import load_params
        graphdef, params, ckpt = load_params(a.checkpoint)
        print(json.dumps({'iteration': ckpt['iteration'],
                          **score_network(make_forward(graphdef), jax.device_put(params), load_benchmark())}, indent=1))


if __name__ == '__main__':
    main()
