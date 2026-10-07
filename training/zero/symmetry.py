"""
Left-right mirror symmetry, for data augmentation.

Every tile's movement pattern (tiles.json) and the starting setup are
symmetric under flipping the board's columns (c -> 5 - c), so a position and
its mirror image are equally valid, with the same value, and the mirror of
each move is a legal move there. Training on both doubles the variety of the
data for free (as AlphaZero does with Go's 8 symmetries). The board rows are
already canonical (the player to move plays "up"), so only columns flip.

    SQ[s]      mirror of square s
    ACTION[a]  mirror of action a: a move's two squares, a command's
               commander square and its two directions, a placement's
               direction (draws are unchanged)
"""

import numpy as np

from training.zero import rules as R

SQ = np.array([R.sq(s // R.BOARD, R.BOARD - 1 - s % R.BOARD) for s in range(R.N_SQ)], np.int32)
DIR8 = np.array([R.DIRS8.index((dr, -dc)) for dr, dc in R.DIRS8], np.int32)
DIR4 = np.array([R.DIRS4.index((dr, -dc)) for dr, dc in R.DIRS4], np.int32)


def _action_table() -> np.ndarray:
    a = np.arange(R.N_ACTIONS)
    out = a.copy()
    move = a < R.N_MOVE_ACTIONS
    out[move] = SQ[a[move] // R.N_SQ] * R.N_SQ + SQ[a[move] % R.N_SQ]
    cmd = (a >= R.CMD_BASE) & (a < R.DRAW_ACTION)
    c = a[cmd] - R.CMD_BASE
    out[cmd] = R.CMD_BASE + SQ[c // 64] * 64 + DIR8[(c // 8) % 8] * 8 + DIR8[c % 8]
    place = a >= R.PLACE_BASE
    out[place] = R.PLACE_BASE + DIR4[a[place] - R.PLACE_BASE]
    return out.astype(np.int32)


ACTION = _action_table()


def mirror_samples(b: dict, rng: np.random.Generator, p: float) -> dict:
    """Replay samples (replay.SAMPLE_SPEC keys) with each one mirrored with
    probability p: board squares and policy-target actions flip; bags, the
    pending tile, ply and value targets don't change."""
    if p <= 0:
        return b
    flip = rng.random(len(b['z'])) < p
    out = dict(b)
    for k in ('tile', 'owner', 'side'):
        out[k] = np.where(flip[:, None], b[k][:, SQ], b[k])
    out['pol_idx'] = np.where(flip[:, None], ACTION[b['pol_idx'].astype(np.int32)], b['pol_idx']).astype(b['pol_idx'].dtype)
    return out
