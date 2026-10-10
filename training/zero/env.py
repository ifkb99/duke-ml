"""
The Duke as a pure-functional JAX environment.

Every function here operates on ONE game. Batching is done from the outside
with `jax.vmap`, which turns e.g. `step(state, action)` into a function over
`State`s whose arrays all carry an extra leading batch axis. Writing the
single-game version and letting vmap add the batch axis is much less
error-prone than aligning batch dimensions by hand.

Turn structure (stochastic-MuZero style)
----------------------------------------
A decision is one of: move/strike, command, draw. Applying it with
`step_decision` produces an *afterstate*:
  - for move/strike/command, the afterstate is simply the next state
    (opponent to move, board already flipped to their perspective);
  - for draw, the afterstate is "I asked to draw, tile not yet known"
    (pending == DRAWING).
`step_chance` then resolves randomness: for a draw afterstate it reveals the
tile (pending = tile type, bag decremented) and the SAME player chooses where
to place it with a place action; for every other afterstate it is a no-op.
This mirrors the physical game: draw, look at the tile, then place it.

Perspective
-----------
The board is always stored from the point of view of the player to move
(see rules.py). After every completed turn we mirror the board rows and swap
owners, so "me" is always owner 0 with home row 5. `player` remembers who
"me" is in absolute terms (0 = P1, 1 = P2) for bookkeeping and for talking
to the TypeScript engine.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import rules as R

MAX_PLY = R.SPEC['env']['maxPly']             # turns before the game is declared a draw
DRAWING = R.SPEC['env']['pendingDrawing']      # `pending` value for a draw afterstate
NONE = R.SPEC['env']['pendingNone']            # `pending` value when no tile is in hand
BAG_SCALE = float(R.SPEC['env']['bagFeatureScale'])
MAX_CANDIDATES = 128   # pseudo-legal moves checked for Duke safety per state

# Tables as device constants.
SQ = jnp.arange(R.N_SQ)
FLIP = jnp.asarray(R.FLIP)
NBR8 = jnp.asarray(R.NBR8)
NBR4 = jnp.asarray(R.NBR4)
MOVE_OK = jnp.asarray(R.MOVE_OK)
NEEDS_CLEAR = jnp.asarray(R.NEEDS_CLEAR)
BETWEEN = jnp.asarray(R.BETWEEN)
STRIKE_OK = jnp.asarray(R.STRIKE_OK)
CMD_DIR = jnp.asarray(R.CMD_DIR)
NOT_EYE8 = ~jnp.eye(8, dtype=bool)


class State(NamedTuple):
    tile: jax.Array      # int32[36]   tile type, -1 = empty
    owner: jax.Array     # int32[36]   0 = player to move, 1 = opponent (0 if empty)
    side: jax.Array      # int32[36]   0 = side A, 1 = side B
    bags: jax.Array      # int32[2,15] tile counts left in bag: [me, opponent]
    player: jax.Array    # int32       absolute player to move: 0 = P1, 1 = P2
    pending: jax.Array   # int32       NONE, DRAWING, or the drawn tile type
    ply: jax.Array       # int32       completed turns
    done: jax.Array      # bool
    winner: jax.Array    # int32       absolute winner, -1 = none / draw
    reward: jax.Array    # float32     last transition's reward for whoever acted
    switched: jax.Array  # bool        did the last transition pass the turn?
    legal: jax.Array     # bool[3605]  legal actions for the player to move
    place_ok: jax.Array  # bool[4]     legal squares for a drawn tile (board-only,
                         #             so a draw/reveal never has to recompute it)
    overflow: jax.Array  # int32       pseudo-legal moves beyond MAX_CANDIDATES (should be 0)


# ---------------------------------------------------------------------------
# Attack detection
# ---------------------------------------------------------------------------

def _is_attacked(tile, side, opp_alive, occ, target):
    """Can any live opponent tile capture whatever stands on `target`?

    tile, side: int32[36]; opp_alive, occ: bool[36]; target: int32 scalar.
    Uses the owner=1 (opponent, mirrored) tables. Each piece of the formula is
    a [36]-vector over attacker squares x; we OR them and test any().
    """
    t = jnp.maximum(tile, 0)
    reach = MOVE_OK[1, t, side, SQ, target]                     # [36] per attacker x
    blocked = jnp.any(BETWEEN[:, target] & occ[None, :], axis=-1)  # [36]
    clear = ~(NEEDS_CLEAR[1, t, side, SQ, target] & blocked)
    strike = STRIKE_OK[1, t, side, SQ, target]                  # [36]

    # Command: `target` is one of x's command squares, and some *other*
    # command square of x holds an opponent tile that could be moved onto it.
    cmd = CMD_DIR[1, t, side]                                   # [36, 8]
    nb_valid = NBR8 >= 0                                        # [36, 8]
    nb = jnp.maximum(NBR8, 0)
    target_is_cmd = jnp.any(cmd & nb_valid & (NBR8 == target), axis=-1)
    has_source = jnp.any(cmd & nb_valid & opp_alive[nb] & (NBR8 != target), axis=-1)

    return jnp.any(opp_alive & ((reach & clear) | strike | (target_is_cmd & has_source)))


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

def _duke_square(tile, owner):
    return jnp.argmax((tile == R.DUKE) & (owner == 0)).astype(jnp.int32)


def _pseudo_legal(tile, owner, side):
    """Moves/strikes [1296] and commands [2304], ignoring Duke safety."""
    occ = tile >= 0
    mine = occ & (owner == 0)
    opp = occ & (owner == 1)
    t = jnp.maximum(tile, 0)

    # Moves and strikes: [36 from, 36 to]
    move_ok = MOVE_OK[0, t, side, SQ]                           # [36, 36]
    # Pieces strictly between every (from, to) pair; the same for all tiles.
    occupied_between = jnp.any(BETWEEN & occ[None, None, :], axis=-1)   # [36, 36]
    blocked = NEEDS_CLEAR[0, t, side, SQ] & occupied_between            # [36, 36]
    moves = mine[:, None] & move_ok & ~blocked & ~mine[None, :]
    strikes = mine[:, None] & STRIKE_OK[0, t, side, SQ] & opp[None, :]
    move_strike = (moves | strikes).reshape(-1)

    # Commands: [36 commander, 8 src dir, 8 dst dir]
    cmd = CMD_DIR[0, t, side] & (NBR8 >= 0)                     # [36, 8]
    nb = jnp.maximum(NBR8, 0)
    src_ok = cmd & mine[nb]
    dst_ok = cmd & ~mine[nb]
    commands = mine[:, None, None] & src_ok[:, :, None] & dst_ok[:, None, :] & NOT_EYE8
    return move_strike, commands.reshape(-1)


def _decode_board_action(tile, side, a):
    """Squares touched by a move/strike/command action a in [0, 3600).

    Returns (from_sq, to_sq, is_strike, is_command, commander).
    """
    is_ms = a < R.N_MOVE_ACTIONS
    f, g = a // R.N_SQ, a % R.N_SQ
    c = jnp.maximum(a - R.CMD_BASE, 0)
    commander, src_dir, dst_dir = c // 64, (c // 8) % 8, c % 8
    src = jnp.maximum(NBR8[commander, src_dir], 0)
    dst = jnp.maximum(NBR8[commander, dst_dir], 0)
    from_sq = jnp.where(is_ms, f, src)
    to_sq = jnp.where(is_ms, g, dst)
    is_strike = is_ms & STRIKE_OK[0, jnp.maximum(tile[f], 0), side[f], f, g]
    return from_sq, to_sq, is_strike, ~is_ms, commander


class _DukeThreats(NamedTuple):
    """Per-position precomputation for checking many candidate moves cheaply.

    Most candidates leave my Duke on square D and change occupancy on at most
    two squares, so each opponent tile x's ability to hit D can be updated
    incrementally instead of recomputed. All fields are [36] over x unless noted.
    """
    reach: jax.Array        # x can move/capture onto D (ignoring blockers)
    needs_clear: jax.Array  # ...and that move needs the in-between squares empty
    between: jax.Array      # [36 x, 36 sq] sq lies strictly between x and D
    n_between: jax.Array    # occupied squares between x and D right now
    strike: jax.Array       # x can strike D
    cmd_onto: jax.Array     # D is one of x's command squares
    cmd_src: jax.Array      # [36 x, 36 sq] sq is another command square of x
    n_src: jax.Array        # opponent tiles on those other command squares now


def _duke_threats(tile, owner, side) -> _DukeThreats:
    occ = tile >= 0
    opp = occ & (owner == 1)
    duke = _duke_square(tile, owner)
    t = jnp.maximum(tile, 0)
    between = BETWEEN[:, duke]
    cmd = CMD_DIR[1, t, side] & (NBR8 >= 0)                                   # [36, 8]
    cmd_sq = jnp.zeros((R.N_SQ, R.N_SQ), bool).at[SQ[:, None], jnp.maximum(NBR8, 0)].max(cmd)
    cmd_src = cmd_sq & (SQ != duke)[None, :]
    return _DukeThreats(
        reach=MOVE_OK[1, t, side, SQ, duke],
        needs_clear=NEEDS_CLEAR[1, t, side, SQ, duke],
        between=between,
        n_between=jnp.sum(between & occ[None, :], axis=-1),
        strike=STRIKE_OK[1, t, side, SQ, duke],
        cmd_onto=cmd_sq[:, duke],
        cmd_src=cmd_src,
        n_src=jnp.sum(cmd_src & opp[None, :], axis=-1),
    )


def _attacked_after(th: _DukeThreats, opp, occ, vacated, filled, filled_now):
    """Is my (unmoved) Duke attacked after `vacated` empties (-1 = none) and
    square `filled` ends up occupied iff `filled_now`? Any opponent tile on
    `filled` is captured either way."""
    vac = (vacated >= 0) & occ[jnp.maximum(vacated, 0)]
    d_between = (filled_now.astype(jnp.int32) - occ[filled]) * th.between[:, filled] \
        - vac * th.between[:, jnp.maximum(vacated, 0)]
    killed = opp[filled]
    alive = opp & ~((SQ == filled) & killed)
    clear = ~th.needs_clear | (th.n_between + d_between == 0)
    sources = th.n_src - (th.cmd_src[:, filled] & killed)
    hits = (th.reach & clear) | th.strike | (th.cmd_onto & (sources > 0))
    return jnp.any(alive & hits)


def _duke_move_safety(tile, owner, side):
    """safe[g]: would my Duke be safe standing on g after leaving its square?
    Covers both Duke moves and commands that move the Duke."""
    occ = tile >= 0
    opp = occ & (owner == 1)
    duke = _duke_square(tile, owner)
    base_occ = occ.at[duke].set(False)
    return ~jax.vmap(lambda g: _is_attacked(
        tile, side, opp.at[g].set(False), base_occ.at[g].set(True), g))(SQ)


def _safe_after(tile, owner, side, a, th, duke_safe):
    """Is the mover's Duke safe after board action `a` (move/strike/command)?"""
    occ = tile >= 0
    opp = occ & (owner == 1)
    from_sq, to_sq, is_strike, _, _ = _decode_board_action(tile, side, a)

    # Capturing the enemy Duke ends the game: always allowed.
    takes_duke = opp[to_sq] & (tile[to_sq] == R.DUKE)
    moved_duke = ~is_strike & (tile[from_sq] == R.DUKE) & (owner[from_sq] == 0)
    attacked = _attacked_after(
        th, opp, occ,
        vacated=jnp.where(is_strike, -1, from_sq),
        filled=to_sq,
        filled_now=~is_strike,       # a strike leaves its target square empty
    )
    return takes_duke | jnp.where(moved_duke, duke_safe[to_sq], ~attacked)


def legal_actions(tile, owner, side, bags, pending):
    """Full legal-action mask [N_ACTIONS] for the player to move."""
    occ = tile >= 0
    opp = occ & (owner == 1)
    move_strike, commands = _pseudo_legal(tile, owner, side)
    board_pseudo = jnp.concatenate([move_strike, commands])     # [3600]

    # Duke safety: only check the (few) pseudo-legal actions. nonzero needs a
    # static size under jit, so we take the first MAX_CANDIDATES indices.
    n_pseudo = board_pseudo.sum()
    (cand,) = jnp.nonzero(board_pseudo, size=MAX_CANDIDATES, fill_value=0)
    th = _duke_threats(tile, owner, side)
    duke_safe = _duke_move_safety(tile, owner, side)
    cand_ok = jax.vmap(lambda a: _safe_after(tile, owner, side, a, th, duke_safe))(cand)
    cand_ok &= jnp.arange(MAX_CANDIDATES) < n_pseudo
    board_legal = jnp.zeros(R.DRAW_ACTION, bool).at[cand].max(cand_ok)

    # Placement squares next to my Duke: empty, and a blocker there must not
    # leave my Duke attacked (tile-independent, so it also decides draw legality).
    duke = _duke_square(tile, owner)
    place_sq = NBR4[duke]                                        # [4]
    place_sq_c = jnp.maximum(place_sq, 0)
    place_ok = (place_sq >= 0) & ~occ[place_sq_c]
    place_safe = jax.vmap(lambda s: ~_attacked_after(
        th, opp, occ, vacated=jnp.int32(-1), filled=s, filled_now=jnp.bool_(True)))(place_sq_c)
    place_legal = place_ok & place_safe

    can_draw = (bags[0].sum() > 0) & jnp.any(place_legal)

    normal = jnp.concatenate([board_legal, can_draw[None], jnp.zeros(4, bool)])
    placing = jnp.concatenate([jnp.zeros(R.PLACE_BASE, bool), place_legal])
    legal = jnp.where(pending == NONE, normal, jnp.where(pending >= 0, placing, False))
    overflow = jnp.maximum(n_pseudo - MAX_CANDIDATES, 0).astype(jnp.int32)
    return legal, place_legal, overflow


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _with_legal(s: State) -> State:
    legal, place_ok, overflow = legal_actions(s.tile, s.owner, s.side, s.bags, s.pending)
    return s._replace(legal=legal, place_ok=place_ok, overflow=overflow)


def _empty_state() -> State:
    return State(
        tile=jnp.full(R.N_SQ, -1, jnp.int32),
        owner=jnp.zeros(R.N_SQ, jnp.int32),
        side=jnp.zeros(R.N_SQ, jnp.int32),
        bags=jnp.asarray(np.stack([R.INITIAL_BAG, R.INITIAL_BAG])),
        player=jnp.int32(0),
        pending=jnp.int32(NONE),
        ply=jnp.int32(0),
        done=jnp.bool_(False),
        winner=jnp.int32(-1),
        reward=jnp.float32(0),
        switched=jnp.bool_(False),
        legal=jnp.zeros(R.N_ACTIONS, bool),
        place_ok=jnp.zeros(4, bool),
        overflow=jnp.int32(0),
    )


# The 3 ways to place two Footmen next to a Duke on its home row:
# (left, right), (left, in front), (right, in front).
_FOOTMAN_PAIRS = jnp.array([[0, 1], [0, 2], [1, 2]])


def _setup_squares(duke_col, pair, home_row, forward):
    duke = home_row * R.BOARD + duke_col
    spots = jnp.array([duke - 1, duke + 1, duke + forward * R.BOARD])
    return duke, spots[_FOOTMAN_PAIRS[pair]]


def init(key) -> State:
    """A new game after a random legal setup, P1 to move."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    cols = jnp.array(R.DUKE_START_COLS)
    # P1 moves first, so we build directly in P1's frame: my home row is 5,
    # the opponent's home row is 0.
    my_duke, my_fm = _setup_squares(jax.random.choice(k1, cols), jax.random.randint(k2, (), 0, 3), 5, -1)
    op_duke, op_fm = _setup_squares(jax.random.choice(k3, cols), jax.random.randint(k4, (), 0, 3), 0, 1)

    s = _empty_state()
    tile = s.tile.at[my_duke].set(R.DUKE).at[my_fm].set(R.FOOTMAN)
    tile = tile.at[op_duke].set(R.DUKE).at[op_fm].set(R.FOOTMAN)
    owner = s.owner.at[op_duke].set(1).at[op_fm].set(1)
    return _with_legal(s._replace(tile=tile, owner=owner))


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def _flip_perspective(tile, owner, side, bags):
    tile, owner, side = tile[FLIP], owner[FLIP], side[FLIP]
    owner = jnp.where(tile >= 0, 1 - owner, 0)
    return tile, owner, side, bags[::-1]


def _clear(arrs, sq):
    tile, owner, side = arrs
    return tile.at[sq].set(-1), owner.at[sq].set(0), side.at[sq].set(0)


def step_decision(s: State, action) -> State:
    """Apply a decision; returns the afterstate (see module docstring).

    An illegal action ends the game as a loss for the mover. Search can stumble
    onto one in rare numerical corner cases; making it a loss teaches the
    search to avoid it instead of corrupting the board.
    """
    action = jnp.asarray(action, jnp.int32)
    tile, owner, side = s.tile, s.owner, s.side
    is_draw = action == R.DRAW_ACTION
    is_place = action >= R.PLACE_BASE
    is_board = action < R.DRAW_ACTION

    # --- move / strike / command ---
    a = jnp.where(is_board, action, 0)
    from_sq, to_sq, is_strike, is_cmd, commander = _decode_board_action(tile, side, a)
    takes_duke = is_board & (owner[to_sq] == 1) & (tile[to_sq] == R.DUKE)

    moved_tile, moved_side = tile[from_sq], side[from_sq]
    b_tile, b_owner, b_side = _clear((tile, owner, side), to_sq)           # capture
    b_tile, b_owner, b_side = jax.lax.cond(
        is_strike,
        # strike: target removed above; the striker flips in place
        lambda: (b_tile, b_owner, b_side.at[from_sq].set(1 - side[from_sq])),
        # move/command: relocate. A moved tile flips; a commanded one doesn't,
        # but its commander flips instead.
        lambda: _relocate(b_tile, b_owner, b_side, from_sq, to_sq, moved_tile,
                          moved_side, is_cmd, commander),
    )

    # --- place ---
    duke = _duke_square(tile, owner)
    place_sq = jnp.maximum(NBR4[duke, jnp.clip(action - R.PLACE_BASE, 0, 3)], 0)
    p_tile = tile.at[place_sq].set(jnp.maximum(s.pending, 0))
    p_owner = owner.at[place_sq].set(0)
    p_side = side.at[place_sq].set(0)

    tile = jnp.where(is_board, b_tile, jnp.where(is_place, p_tile, tile))
    owner = jnp.where(is_board, b_owner, jnp.where(is_place, p_owner, owner))
    side = jnp.where(is_board, b_side, jnp.where(is_place, p_side, side))
    pending = jnp.where(is_draw, DRAWING, NONE)

    # --- outcome ---
    legal_a = s.legal[action]
    illegal = ~legal_a
    win = legal_a & takes_duke
    turn_ends = legal_a & ~win & ~is_draw
    ply = s.ply + turn_ends.astype(jnp.int32)
    timeout = turn_ends & (ply >= MAX_PLY)
    passes = turn_ends & ~timeout

    f_tile, f_owner, f_side, f_bags = _flip_perspective(tile, owner, side, s.bags)
    tile = jnp.where(passes, f_tile, tile)
    owner = jnp.where(passes, f_owner, owner)
    side = jnp.where(passes, f_side, side)
    bags = jnp.where(passes, f_bags, s.bags)
    player = jnp.where(passes, 1 - s.player, s.player)

    nxt = _with_legal(State(
        tile=tile, owner=owner, side=side, bags=bags, player=player,
        pending=pending, ply=ply, done=jnp.bool_(False), winner=jnp.int32(-1),
        reward=jnp.float32(0), switched=passes,
        legal=s.legal, place_ok=s.place_ok, overflow=jnp.int32(0),
    ))

    # The new player to move with no legal move loses (checkmate/stalemate).
    stuck = passes & ~jnp.any(nxt.legal)
    mover_wins = win | stuck
    done = illegal | win | timeout | stuck
    winner = jnp.where(mover_wins, s.player, jnp.where(illegal, 1 - s.player, -1))
    reward = jnp.where(mover_wins, 1.0, jnp.where(illegal, -1.0, 0.0)).astype(jnp.float32)

    nxt = nxt._replace(
        done=done, winner=winner, reward=reward,
        legal=jnp.where(done | is_draw, False, nxt.legal),
        # A draw leaves the board untouched, so placement legality carries over.
        place_ok=jnp.where(is_draw, s.place_ok, nxt.place_ok),
    )
    # Stepping a finished game is a no-op with zero reward.
    return jax.tree.map(
        lambda old, new: jnp.where(s.done, old, new),
        s._replace(reward=jnp.float32(0), switched=jnp.bool_(False)), nxt,
    )


def _relocate(tile, owner, side, from_sq, to_sq, moved_tile, moved_side, is_cmd, commander):
    tile, owner, side = _clear((tile, owner, side), from_sq)
    tile = tile.at[to_sq].set(moved_tile)
    owner = owner.at[to_sq].set(0)
    side = side.at[to_sq].set(jnp.where(is_cmd, moved_side, 1 - moved_side))
    side = jnp.where(is_cmd, side.at[commander].set(1 - side[commander]), side)
    return tile, owner, side


def chance_probs(s: State):
    """Probability of each chance outcome [N_TYPES] from an afterstate.

    Draw afterstates: the drawn tile type, proportional to bag counts.
    Every other afterstate is deterministic: outcome 0 with probability 1.
    """
    counts = s.bags[0].astype(jnp.float32)
    drawing = (s.pending == DRAWING) & ~s.done
    return jnp.where(drawing, counts / jnp.maximum(counts.sum(), 1),
                     jax.nn.one_hot(0, R.N_TYPES))


def step_chance(s: State, outcome) -> State:
    """Resolve an afterstate's randomness (reveal the drawn tile)."""
    drawing = (s.pending == DRAWING) & ~s.done
    outcome = jnp.asarray(outcome, jnp.int32)
    drawn = s._replace(
        pending=outcome,
        bags=s.bags.at[0, outcome].add(-1),
        reward=jnp.float32(0),
        switched=jnp.bool_(False),
        legal=jnp.concatenate([jnp.zeros(R.PLACE_BASE, bool), s.place_ok]),
    )
    return jax.tree.map(lambda a, b: jnp.where(drawing, b, a), s, drawn)


def step(s: State, action, key) -> State:
    """Decision followed by sampled chance: one real step of the game."""
    after = step_decision(s, action)
    outcome = jax.random.categorical(key, jnp.log(chance_probs(after)))
    nxt = step_chance(after, outcome)
    # Keep the decision's reward/turn-passing flags on the real transition.
    return nxt._replace(reward=after.reward, switched=after.switched)


# ---------------------------------------------------------------------------
# Network input
# ---------------------------------------------------------------------------

N_PLANES = 2 * R.N_TYPES * 2            # owner x type x side one-hot = 60
N_FEATURES = 2 * R.N_TYPES + R.N_TYPES + 2  # bags, pending tile, drawing flag, ply = 47


def observe(s: State):
    """Network input: (planes float32[6, 6, 60], features float32[47])."""
    return observe_arrays(s.tile, s.owner, s.side, s.bags, s.pending, s.ply)


def observe_arrays(tile, owner, side, bags, pending, ply):
    """observe() from raw arrays (single game), e.g. compact replay samples.

    planes[r, c, owner*30 + type*2 + side] = 1 for each tile (canonical frame).
    """
    tile, owner, side = (x.astype(jnp.int32) for x in (tile, owner, side))
    pending = pending.astype(jnp.int32)
    idx = owner * (2 * R.N_TYPES) + jnp.maximum(tile, 0) * 2 + side
    planes = jax.nn.one_hot(idx, N_PLANES) * (tile >= 0)[:, None]
    planes = planes.reshape(R.BOARD, R.BOARD, N_PLANES)
    features = jnp.concatenate([
        bags.reshape(-1).astype(jnp.float32) / BAG_SCALE,
        jax.nn.one_hot(pending, R.N_TYPES),          # all-zero when pending < 0
        (pending == DRAWING)[None].astype(jnp.float32),
        (ply / MAX_PLY)[None].astype(jnp.float32),
    ])
    return planes, features
