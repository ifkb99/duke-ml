"""
Conversions between the TypeScript engine's JSON (SerializedGameState,
GameMove) and the JAX environment's canonical arrays / action indices.

TS uses absolute coordinates: P1's home row is 0, P2's is 5. The canonical
frame of the player to move is the TS frame for P2 and the row-mirrored frame
for P1 (see rules.py), so `canon(sq, player)` is FLIP[sq] for P1 and sq for P2.
FLIP is its own inverse, so the same mapping converts back.
"""

import numpy as np
import jax.numpy as jnp

from training.zero import env as E
from training.zero import rules as R

PLAYERS = ('P1', 'P2')


def canon(sq: int, player: int) -> int:
    return int(R.FLIP[sq]) if player == 0 else sq


def _sq(coord) -> int:
    return coord['row'] * R.BOARD + coord['col']


def _coord(sq: int) -> dict:
    return {'row': sq // R.BOARD, 'col': sq % R.BOARD}


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

def state_arrays_from_ts(ser: dict) -> dict:
    """SerializedGameState -> dict of numpy arrays matching env.State fields
    (except `legal`/`overflow`, which the env computes)."""
    player = PLAYERS.index(ser['currentPlayer'])
    tile = np.full(R.N_SQ, -1, np.int32)
    owner = np.zeros(R.N_SQ, np.int32)
    side = np.zeros(R.N_SQ, np.int32)
    for _id, t in ser['tiles']:
        s = canon(_sq(t['position']), player)
        tile[s] = R.TILE_INDEX[t['defName']]
        owner[s] = 0 if PLAYERS.index(t['owner']) == player else 1
        side[s] = 0 if t['side'] == 'A' else 1

    bags = np.zeros((2, R.N_TYPES), np.int32)
    for rel, p in ((0, player), (1, 1 - player)):
        for name in ser['bags'][PLAYERS[p]]:
            bags[rel, R.TILE_INDEX[name]] += 1

    status = ser['status']
    return dict(
        tile=tile, owner=owner, side=side, bags=bags,
        player=np.int32(player), pending=np.int32(E.NONE),
        ply=np.int32(ser['turnNumber'] - 1),
        done=np.bool_(status in ('P1_wins', 'P2_wins')),
        winner=np.int32({'P1_wins': 0, 'P2_wins': 1}.get(status, -1)),
    )


def state_to_ts(s) -> dict:
    """A single (unbatched) env.State -> SerializedGameState for the TS engine."""
    if int(s.pending) != E.NONE:
        raise ValueError('TS has no mid-draw states; convert before drawing')
    player = int(s.player)
    tile, owner, side = (np.asarray(x) for x in (s.tile, s.owner, s.side))
    board = [[None] * R.BOARD for _ in range(R.BOARD)]
    tiles = []
    for c in range(R.N_SQ):
        if tile[c] < 0:
            continue
        a = canon(c, player)
        abs_owner = PLAYERS[player if owner[c] == 0 else 1 - player]
        name = R.TILE_NAMES[tile[c]]
        tid = f'{abs_owner}-{name}-{a}'
        board[a // R.BOARD][a % R.BOARD] = tid
        tiles.append([tid, {'defName': name, 'owner': abs_owner,
                            'side': 'AB'[side[c]], 'position': _coord(a), 'id': tid}])
    bags_rel = np.asarray(s.bags)
    bags = {}
    for rel, p in ((0, player), (1, 1 - player)):
        bags[PLAYERS[p]] = [R.TILE_NAMES[t] for t in range(R.N_TYPES) for _ in range(bags_rel[rel, t])]
    return {
        'board': board, 'tiles': tiles, 'bags': bags,
        'currentPlayer': PLAYERS[player], 'turnNumber': int(s.ply) + 1,
        'status': 'active',
    }


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

def _dir(frm: int, to: int, dirs) -> int:
    d = (to // R.BOARD - frm // R.BOARD, to % R.BOARD - frm % R.BOARD)
    return dirs.index(d)


def duke_square(tile, owner) -> int:
    return int(np.flatnonzero((np.asarray(tile) == R.DUKE) & (np.asarray(owner) == 0))[0])


def move_to_actions(move: dict, player: int, duke_sq: int) -> list:
    """TS GameMove -> list of env steps for the player to move.

    Each element is ('action', index) or ('chance', tile_index). A resolved
    `place` becomes draw -> chance(tile) -> place. A bare `draw` (tile not
    chosen yet) becomes draw -> place, with the chance step left to the caller.
    `duke_sq` is the mover's Duke in the canonical frame.
    """
    c = lambda coord: canon(_sq(coord), player)
    kind = move['type']
    if kind in ('move', 'strike'):
        frm = c(move['from'])
        to = c(move['to'] if kind == 'move' else move['target'])
        return [('action', frm * R.N_SQ + to)]
    if kind == 'command':
        cmd, src, dst = c(move['commander']), c(move['target']), c(move['targetTo'])
        idx = R.CMD_BASE + cmd * 64 + _dir(cmd, src, R.DIRS8) * 8 + _dir(cmd, dst, R.DIRS8)
        return [('action', idx)]
    place = ('action', R.PLACE_BASE + _dir(duke_sq, c(move['position']), R.DIRS4))
    if kind == 'place':
        return [('action', R.DRAW_ACTION), ('chance', R.TILE_INDEX[move['tileName']]), place]
    if kind == 'draw':
        return [('action', R.DRAW_ACTION), place]
    raise ValueError(f'unknown move type {kind}')


def action_to_move(action: int, s, pending_tile: int | None = None) -> dict:
    """Env action for single state `s` -> TS GameMove.

    Draw/place collapse into one TS `place`: pass the place action together
    with the tile that was drawn (`pending_tile`).
    """
    player = int(s.player)
    a = canon  # canonical <-> absolute is the same mapping
    if action < R.N_MOVE_ACTIONS:
        frm, to = divmod(action, R.N_SQ)
        t, side = int(s.tile[frm]), int(s.side[frm])
        if R.STRIKE_OK[0, t, side, frm, to]:
            return {'type': 'strike', 'from': _coord(a(frm, player)), 'target': _coord(a(to, player))}
        return {'type': 'move', 'from': _coord(a(frm, player)), 'to': _coord(a(to, player))}
    if action < R.DRAW_ACTION:
        c = action - R.CMD_BASE
        cmd, sd, dd = c // 64, (c // 8) % 8, c % 8
        src, dst = int(R.NBR8[cmd, sd]), int(R.NBR8[cmd, dd])
        return {'type': 'command', 'commander': _coord(a(cmd, player)),
                'target': _coord(a(src, player)), 'targetTo': _coord(a(dst, player))}
    if action >= R.PLACE_BASE and pending_tile is not None:
        sq = int(R.NBR4[duke_square(s.tile, s.owner), action - R.PLACE_BASE])
        return {'type': 'place', 'tileName': R.TILE_NAMES[pending_tile], 'position': _coord(a(sq, player))}
    raise ValueError('draw/place must be converted together with the drawn tile')


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def stack_states(dicts: list[dict]) -> E.State:
    """Stack per-game array dicts into a batched env.State (legal not yet computed)."""
    n = len(dicts)
    batch = {k: jnp.asarray(np.stack([d[k] for d in dicts])) for k in dicts[0]}
    return E.State(
        **batch,
        reward=jnp.zeros(n, jnp.float32),
        switched=jnp.zeros(n, bool),
        legal=jnp.zeros((n, R.N_ACTIONS), bool),
        place_ok=jnp.zeros((n, 4), bool),
        overflow=jnp.zeros(n, jnp.int32),
    )
