"""
Static rule tables for the JAX Duke environment, precomputed once with NumPy.

The idea: instead of looping over tiles and walking slides at runtime (slow and
awkward under `jit`), we precompute *everything that depends only on geometry*
into lookup tables. At runtime, move generation is then just gathers and
boolean algebra over fixed-shape arrays — exactly what GPUs are good at.

Coordinates
-----------
Squares are flattened: sq = row * 6 + col  (0..35).

Everything here is in the *canonical frame*: the board as seen by the player
to move ("me"). In this frame my home row is row 5 and my "forward" is toward
row 0, so tile offsets from tiles.json (engine convention: negative dRow =
forward) apply to me unchanged. The opponent's offsets are mirrored by
negating dRow — equivalently, mirror the board rows (sq -> FLIP[sq]).

Table shapes (T = 15 tile types, S = 2 sides, N = 36 squares)
-------------------------------------------------------------
  MOVE_OK   [2, T, S, N, N] bool   owner, type, side, from, to: a move/capture
                                   to `to` is geometrically possible
  NEEDS_CLEAR [2, T, S, N, N] bool that move also needs every square strictly
                                   between `from` and `to` to be empty (steps
                                   and slides; jumps ignore blockers)
  BETWEEN   [N, N, N] bool         geometry only: squares strictly between two
                                   squares on a shared row/column/diagonal
  STRIKE_OK [2, T, S, N, N] bool   a strike from `from` can hit `to`
  CMD_DIR   [2, T, S, 8] bool      which of the 8 neighbor directions are
                                   command squares
The leading axis is the owner relative to the side to move: 0 = me, 1 = opponent.
"""

import json
from pathlib import Path

import numpy as np

_DATA = json.loads((Path(__file__).parent / 'tiles.json').read_text())

BOARD = _DATA['boardSize']                     # 6
N_SQ = BOARD * BOARD                           # 36
TILE_NAMES: list[str] = [t['name'] for t in _DATA['tiles']]
N_TYPES = len(TILE_NAMES)                      # 15
TILE_INDEX = {name: i for i, name in enumerate(TILE_NAMES)}
DUKE = TILE_INDEX['Duke']
FOOTMAN = TILE_INDEX['Footman']

# Bag after setup: the full bag minus the two Footmen placed on the board.
_bag = list(_DATA['bag'])
_bag.remove('Footman')
_bag.remove('Footman')
INITIAL_BAG = np.zeros(N_TYPES, np.int32)
for _name in _bag:
    INITIAL_BAG[TILE_INDEX[_name]] += 1

# The Duke starts on one of the two center squares of its home row.
DUKE_START_COLS = (BOARD // 2 - 1, BOARD // 2)

# Neighbor directions, in a fixed order used by the command action encoding.
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Orthogonal directions, in the order used by the place action encoding.
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def sq(r: int, c: int) -> int:
    return r * BOARD + c


def on_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD and 0 <= c < BOARD


# FLIP[s] mirrors a square top-to-bottom (row r -> 5 - r). Columns are NOT
# mirrored: the engine only negates dRow when switching perspective.
FLIP = np.array([sq(BOARD - 1 - s // BOARD, s % BOARD) for s in range(N_SQ)], np.int32)

# NBR8[s, d] / NBR4[s, d]: neighbor square in direction d, or -1 if off board.
NBR8 = np.full((N_SQ, 8), -1, np.int32)
NBR4 = np.full((N_SQ, 4), -1, np.int32)
for _s in range(N_SQ):
    _r, _c = divmod(_s, BOARD)
    for _d, (_dr, _dc) in enumerate(DIRS8):
        if on_board(_r + _dr, _c + _dc):
            NBR8[_s, _d] = sq(_r + _dr, _c + _dc)
    for _d, (_dr, _dc) in enumerate(DIRS4):
        if on_board(_r + _dr, _c + _dc):
            NBR4[_s, _d] = sq(_r + _dr, _c + _dc)


# ---------------------------------------------------------------------------
# Building the tables
# ---------------------------------------------------------------------------

def _add_move(ok, path, f, g, between):
    """Record that f -> g is reachable when every square in `between` is empty.

    If several patterns reach the same target, keep the least-blocked one
    (a jump beats a step through the same square). In the base tile set this
    never happens, but the rule is the safe one if it ever does.
    """
    new = np.zeros(N_SQ, bool)
    new[list(between)] = True
    if not ok[f, g] or new.sum() < path[f, g].sum():
        ok[f, g] = True
        path[f, g] = new


def _build_side(patterns):
    ok = np.zeros((N_SQ, N_SQ), bool)
    path = np.zeros((N_SQ, N_SQ, N_SQ), bool)
    strike = np.zeros((N_SQ, N_SQ), bool)
    cmd = np.zeros(8, bool)

    for p in patterns:
        kind = p['type']
        for dr, dc in p['offsets']:
            if kind == 'command':
                assert (dr, dc) in DIRS8, 'command squares must be adjacent'
                cmd[DIRS8.index((dr, dc))] = True
                continue
            for f in range(N_SQ):
                r, c = divmod(f, BOARD)
                if kind in ('step', 'jump', 'strike'):
                    if not on_board(r + dr, c + dc):
                        continue
                    g = sq(r + dr, c + dc)
                    if kind == 'strike':
                        strike[f, g] = True
                    elif kind == 'jump':
                        _add_move(ok, path, f, g, [])
                    else:
                        # A multi-square step (e.g. F2) can't pass through
                        # pieces: the intermediate squares on the straight
                        # line must be empty. Offsets like (2,1) have none.
                        k = np.gcd(abs(dr), abs(dc))
                        ur, uc = dr // k, dc // k
                        between = [sq(r + ur * i, c + uc * i) for i in range(1, k)]
                        _add_move(ok, path, f, g, between)
                elif kind in ('slide', 'jump_slide'):
                    between = []
                    i = 1
                    while on_board(r + dr * i, c + dc * i):
                        g = sq(r + dr * i, c + dc * i)
                        _add_move(ok, path, f, g, between if kind == 'slide' else [])
                        between = between + [g]
                        i += 1
                else:
                    raise ValueError(f'unknown move type {kind}')

    # The action encoding uses one index per (from, to) for both moves and
    # strikes, so a tile side must never be able to do both to one square.
    assert not (ok & strike).any(), 'move and strike targets overlap'
    return ok, path, strike, cmd


def _build_tables():
    ok = np.zeros((2, N_TYPES, 2, N_SQ, N_SQ), bool)
    path = np.zeros((2, N_TYPES, 2, N_SQ, N_SQ, N_SQ), bool)
    strike = np.zeros((2, N_TYPES, 2, N_SQ, N_SQ), bool)
    cmd = np.zeros((2, N_TYPES, 2, 8), bool)

    # Mirror for the opponent: a move f -> g for them is FLIP[f] -> FLIP[g]
    # for "me", and their command direction (dr, dc) is my (-dr, dc).
    mirror_dir8 = [DIRS8.index((-dr, dc)) for dr, dc in DIRS8]

    for t, tile in enumerate(_DATA['tiles']):
        for s, patterns in enumerate(tile['sides']):
            o, p, st, cm = _build_side(patterns)
            ok[0, t, s], path[0, t, s], strike[0, t, s], cmd[0, t, s] = o, p, st, cm
            ok[1, t, s] = o[np.ix_(FLIP, FLIP)]
            path[1, t, s] = p[np.ix_(FLIP, FLIP, FLIP)]
            strike[1, t, s] = st[np.ix_(FLIP, FLIP)]
            cmd[1, t, s] = cm[mirror_dir8]
    return ok, path, strike, cmd


MOVE_OK, _MOVE_PATH, STRIKE_OK, CMD_DIR = _build_tables()


def _build_between():
    between = np.zeros((N_SQ, N_SQ, N_SQ), bool)
    for f in range(N_SQ):
        r, c = divmod(f, BOARD)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                passed = []
                i = 1
                while on_board(r + dr * i, c + dc * i):
                    g = sq(r + dr * i, c + dc * i)
                    between[f, g, passed] = True
                    passed.append(g)
                    i += 1
    return between


# Path blocking only ever depends on the squares between `from` and `to`, never
# on which tile is moving. That lets move generation count blockers for ALL
# (from, to) pairs with one small matmul per position instead of gathering a
# per-tile path table (see env._pseudo_legal). Verify the equivalence here.
BETWEEN = _build_between()
NEEDS_CLEAR = _MOVE_PATH.any(-1)
_paths = _MOVE_PATH[NEEDS_CLEAR]
_expected = np.broadcast_to(BETWEEN, _MOVE_PATH.shape)[NEEDS_CLEAR]
assert (_paths == _expected).all(), 'a move path is not the straight line between its squares'
del _MOVE_PATH, _paths, _expected


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
#
#   [0, 1296)      move or strike: from * 36 + to
#                  (which one is determined by the tables — they never overlap)
#   [1296, 3600)   command: 1296 + commander * 64 + src_dir * 8 + dst_dir
#                  (src/dst are the commander's neighbors in DIRS8 order)
#   3600           draw a tile from the bag
#   [3601, 3605)   place the drawn tile next to my Duke, direction in DIRS4 order
#
# All squares are in the canonical frame of the player to move.

N_MOVE_ACTIONS = N_SQ * N_SQ                        # 1296
CMD_BASE = N_MOVE_ACTIONS                           # 1296
N_CMD_ACTIONS = N_SQ * 8 * 8                        # 2304
DRAW_ACTION = CMD_BASE + N_CMD_ACTIONS              # 3600
PLACE_BASE = DRAW_ACTION + 1                        # 3601
N_ACTIONS = PLACE_BASE + 4                          # 3605
