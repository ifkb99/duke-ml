"""Where does a network's search go wrong: drawing or not, which piece, or where?

Takes fresh decision points (no tile in hand) from held-out teacher games, has
a reference (by default big1:1525 at 512 simulations) and each tested
network/simulation count choose a move, and counts agreement at three levels:

  draw   - draw a tile vs. play a board move
  piece  - draw, or which tile acts (the mover; for a command, the commander)
  move   - the exact action

    decision_levels.py runs/az7-drop/imitation.pkl 1,32,128,400 [positions] [ref.pkl ref_sims]

Simulation counts can name a grouped search (search.LEVELS), e.g. 128:piece.
"""
import sys, time
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np

from training.zero import env as E, rules as R, search as S, train as T
from training.zero.network import make_forward

jax.config.update('jax_compilation_cache_dir', 'runs/.jax_cache')

# The acting tile's square for every action; draw = 36, place = 37.
GROUP = np.concatenate([np.arange(R.N_MOVE_ACTIONS) // R.N_SQ,
                        np.arange(R.N_CMD_ACTIONS) // 64, [36], [37] * 4])


def positions(n, seed=0, data='runs/az6-data3/distill_samples.npz'):
    """n random fresh decision states from the last tenth of the games (never trained on
    by the distilled students: imitate holds out the last games of the concatenated data)."""
    d = np.load(data)
    gid = T.game_ids(d)
    ok = np.flatnonzero((d['pending'] == E.NONE) & (gid >= 0.9 * (gid.max() + 1)))
    idx = np.sort(np.random.default_rng(seed).choice(ok, n, replace=False))

    def build(tile, owner, side, bags, ply):
        s = E._empty_state()._replace(tile=tile.astype(jnp.int32), owner=owner.astype(jnp.int32),
                                      side=side.astype(jnp.int32), bags=bags.astype(jnp.int32),
                                      ply=ply.astype(jnp.int32))
        return E._with_legal(s)
    return jax.jit(jax.vmap(build))(*(jnp.asarray(d[k][idx]) for k in ('tile', 'owner', 'side', 'bags', 'ply')))


def choose(path, sims, states, chunk_games=8192, levels='flat'):
    graphdef, params, _ = T.load_params(Path(path))
    n = states.tile.shape[0]
    chunks = next(c for c in range(1, n + 1) if n % c == 0 and n * max(sims, 2) // c <= chunk_games)
    if sims <= 1:   # the raw policy: one candidate, the prior's argmax
        search = S.make_search(make_forward(graphdef), 2, 1, 4)
    else:
        search = S.make_search(make_forward(graphdef), sims, levels=levels)
    out = jax.jit(S.chunked(search, chunks))(jax.device_put(params), jax.random.key(0), states, 0.0)
    return np.asarray(out.action)


def levels(a, ref):
    ga, gr = GROUP[a], GROUP[ref]
    return {'draw': np.mean((ga == 36) == (gr == 36)), 'piece': np.mean(ga == gr), 'move': np.mean(a == ref)}


def main():
    path, sims = sys.argv[1], [(x + ':flat').split(':')[:2] for x in sys.argv[2].split(',')]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    ref_path = sys.argv[4] if len(sys.argv) > 4 else 'runs/big1/ckpt_001525.pkl'
    ref_sims = int(sys.argv[5]) if len(sys.argv) > 5 else 512

    states = positions(n)
    legal = np.asarray(states.legal)
    groups = np.array([len(np.unique(GROUP[np.flatnonzero(l)])) for l in legal])
    can_draw = legal[:, R.DRAW_ACTION]
    print(f'{n} positions: legal moves median {np.median(legal.sum(1)):.0f} (p10 {np.percentile(legal.sum(1), 10):.0f}, '
          f'p90 {np.percentile(legal.sum(1), 90):.0f}); acting groups median {np.median(groups):.0f}; '
          f'draw legal in {can_draw.mean():.2f}', flush=True)

    t = time.time()
    ref = choose(ref_path, ref_sims, states)
    print(f'reference {ref_path}@{ref_sims}: draws in {np.mean(ref == R.DRAW_ACTION):.3f} '
          f'of positions ({np.mean(ref[can_draw] == R.DRAW_ACTION):.3f} where legal), {time.time() - t:.0f}s', flush=True)

    rows = [(f'{ref_path}@{ref_sims // 2}', choose(ref_path, ref_sims // 2, states))]
    rows += [(f'{path}@{s}' + (f':{lv}' if lv != 'flat' else ''), choose(path, int(s), states, levels=lv))
             for s, lv in sims]
    for name, a in rows:
        lv = levels(a, ref)
        d, rd = a == R.DRAW_ACTION, ref == R.DRAW_ACTION
        # Of the positions where the board move's piece is right, how often is the square too?
        same_piece = (GROUP[a] == GROUP[ref]) & ~rd
        print(f'{name:>40}: draw {lv["draw"]:.3f}  piece {lv["piece"]:.3f}  move {lv["move"]:.3f} | '
              f'draws {d.mean():.3f}, extra draws {np.mean(d & ~rd):.3f}, missed draws {np.mean(~d & rd):.3f}, '
              f'right square given right piece {np.mean(a[same_piece] == ref[same_piece]):.3f}', flush=True)


if __name__ == '__main__':
    main()
