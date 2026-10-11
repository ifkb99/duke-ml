"""
Semantic board features computed from the network's input planes: who
attacks which square, how many moves each of my pieces has, am I in check,
can I take the Duke.

A small convolutional network has to rediscover these from one-hot tiles,
which is hard in The Duke (jumps, slides, strikes, commands, and every tile
flips side after moving). The rules already know them exactly, so they can be
handed to the network as extra inputs, or used as auxiliary training targets
that teach the trunk to represent them.

Everything here is fixed tensor algebra over the planes and constant rule
tables (matmuls, comparisons, max): it batches like the rest of the network,
so it can run inside the exported model on the phone (WebGPU or wasm SIMD)
instead of in TypeScript, and train and phone compute it identically.

The attack maps follow env._is_attacked: "could this side capture whatever
stood on square t" (moves and slides with clear paths, strikes, commands), so
my attack map at the opponent's Duke is "I can take the Duke".
"""

import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R

N_PLANES = 3      # opponent attacks, my attacks, my moves per piece / 8
N_FEATURES = 2    # my Duke is attacked (in check), the opponent's Duke is attacked

_K = 2 * R.N_TYPES                                                   # type x side, as in the planes
_MOVE_OK = R.MOVE_OK.reshape(2, _K, R.N_SQ, R.N_SQ).astype(np.float32)       # [owner, k, from, to]
_NEEDS_CLEAR = R.NEEDS_CLEAR.reshape(2, _K, R.N_SQ, R.N_SQ).astype(np.float32)
_STRIKE_OK = R.STRIKE_OK.reshape(2, _K, R.N_SQ, R.N_SQ).astype(np.float32)
_CMD_DIR = R.CMD_DIR.reshape(2, _K, 8).astype(np.float32)          # [owner, k, dir]
_BETWEEN = R.BETWEEN.astype(np.float32)                            # [from, to, square between]
# _NBR[x, d, s] = 1 if s is x's neighbour in direction d (off-board directions have none)
_NBR = np.zeros((R.N_SQ, 8, R.N_SQ), np.float32)
for _x in range(R.N_SQ):
    for _d in range(8):
        if R.NBR8[_x, _d] >= 0:
            _NBR[_x, _d, R.NBR8[_x, _d]] = 1
_DUKE = [R.DUKE * 2, R.DUKE * 2 + 1]                               # the Duke's k, both sides


def _side(p, o, occ, between):
    """Attack map [B, 36] and per-piece move counts [B, 36] for owner o (0 = me)."""
    mine = p[:, :, o]                                                # [B, x, k]
    alive = mine.sum(-1)                                             # [B, 36]
    reach = jnp.einsum('bxk,kxt->bxt', mine, _MOVE_OK[o])
    needs_clear = jnp.einsum('bxk,kxt->bxt', mine, _NEEDS_CLEAR[o])
    strike = jnp.einsum('bxk,kxt->bxt', mine, _STRIKE_OK[o])
    clear = 1 - needs_clear * between
    moves = reach * clear                                            # [B, from, to]

    cmd = jnp.einsum('bxk,kd->bxd', mine, _CMD_DIR[o]) * _NBR.sum(-1)   # on-board command squares
    cmd_sq = jnp.einsum('bxd,xds->bxs', cmd, _NBR)                   # [B, x, s]: s is a command square of x
    own_on_cmd = jnp.einsum('bxs,bs->bx', cmd_sq, alive)             # x's command squares holding my tiles
    # x commands one of its tiles onto t from another command square
    cmd_attack = cmd_sq * (own_on_cmd[:, :, None] - cmd_sq * alive[:, None, :] > 0)
    attack = jnp.max(jnp.maximum(jnp.maximum(moves, strike), cmd_attack), axis=1)

    other = occ - alive                                              # the other side's tiles
    n_moves = (moves * (1 - alive[:, None, :])).sum(-1) + (strike * other[:, None, :]).sum(-1)
    n_commands = own_on_cmd * jnp.einsum('bxs,bs->bx', cmd_sq, 1 - alive)
    return attack, n_moves + n_commands


def rule_features(planes):
    """planes [B, 6, 6, 60] (env.observe) -> (planes [B, 6, 6, 3], features [B, 2])."""
    b = planes.shape[0]
    p = planes.astype(jnp.float32).reshape(b, R.N_SQ, 2, _K)         # [B, sq, owner, type x side]
    occ = p.sum((2, 3))
    between = (jnp.einsum('bs,xts->bxt', occ, _BETWEEN) > 0).astype(jnp.float32)
    opp_attack, _ = _side(p, 1, occ, between)
    my_attack, my_moves = _side(p, 0, occ, between)
    my_duke = p[:, :, 0, _DUKE].sum(-1)
    opp_duke = p[:, :, 1, _DUKE].sum(-1)
    out_planes = jnp.stack([opp_attack, my_attack, my_moves / 8], -1).reshape(b, R.BOARD, R.BOARD, N_PLANES)
    out_features = jnp.stack([(my_duke * opp_attack).sum(-1), (opp_duke * my_attack).sum(-1)], -1)
    return out_planes, out_features
