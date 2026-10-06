/**
 * The Duke environment the network was trained on, ported from
 * training/zero/env.py for running the search in the browser.
 *
 * Same state, frame, turn structure and action numbering as the JAX version
 * (see its docstring): the board is stored from the point of view of the
 * player to move; a decision gives an afterstate; a draw afterstate is
 * resolved by a chance step that reveals the tile, then the same player
 * places it. Parity with the JAX environment is tested from Python
 * (training/zero/tests/test_ts_zero.py).
 *
 * JAX computes legality with fixed-shape tricks; here it is direct: apply
 * each pseudo-legal action to a scratch board and check my Duke.
 */

import {
  BETWEEN, bIdx, CMD_BASE, CMD_DIR, cIdx, DRAW_ACTION, DUKE, FLIP, MOVE_OK, N_ACTIONS,
  N_MOVE_ACTIONS, N_SQ, N_TYPES, NBR4, NBR8, NEEDS_CLEAR, PLACE_BASE, STRIKE_OK, tIdx,
  SPEC,
} from './rules.js';

export const MAX_PLY = SPEC.env.maxPly;           // turns before the game is a draw
export const DRAWING = SPEC.env.pendingDrawing;   // `pending` for a draw afterstate
export const NONE = SPEC.env.pendingNone;         // `pending` with no tile in hand
const BAG_SCALE = SPEC.env.bagFeatureScale;

export interface State {
  tile: Int8Array;      // [36] tile type, -1 = empty
  owner: Uint8Array;    // [36] 0 = player to move, 1 = opponent (0 if empty)
  side: Uint8Array;     // [36] 0 = side A, 1 = side B
  bags: Int32Array;     // [2 * 15] tile counts left: me, then opponent
  player: number;       // absolute player to move: 0 = P1, 1 = P2
  pending: number;      // NONE, DRAWING, or the drawn tile type
  ply: number;          // completed turns
  done: boolean;
  winner: number;       // absolute winner, -1 = none / draw
  reward: number;       // last transition's reward for whoever acted
  switched: boolean;    // did the last transition pass the turn?
  legal: Uint8Array;    // [3605] legal actions for the player to move
  placeOk: Uint8Array;  // [4] legal squares for a drawn tile
}

export function copyState(s: State): State {
  return {
    ...s, tile: s.tile.slice(), owner: s.owner.slice(), side: s.side.slice(), bags: s.bags.slice(),
    legal: s.legal.slice(), placeOk: s.placeOk.slice(),
  };
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------

/** Can any opponent tile capture whatever stands on `target`? */
export function isAttacked(tile: Int8Array, owner: Uint8Array, side: Uint8Array, target: number): boolean {
  for (let x = 0; x < N_SQ; x++) {
    const t = tile[x];
    if (t < 0 || owner[x] !== 1 || x === target) continue;
    const sd = side[x];
    if (STRIKE_OK[tIdx(1, t, sd, x, target)]) return true;
    if (MOVE_OK[tIdx(1, t, sd, x, target)]) {
      let clear = true;
      if (NEEDS_CLEAR[tIdx(1, t, sd, x, target)]) {
        for (let b = 0; b < N_SQ; b++) {
          if (BETWEEN[bIdx(x, target, b)] && tile[b] >= 0) { clear = false; break; }
        }
      }
      if (clear) return true;
    }
    // Command: target is one of x's command squares, and another of them
    // holds an opponent tile that could be moved onto it.
    let targetIsCmd = false, hasSource = false;
    for (let d = 0; d < 8; d++) {
      const nb = NBR8[x * 8 + d];
      if (nb < 0 || !CMD_DIR[cIdx(1, t, sd, d)]) continue;
      if (nb === target) targetIsCmd = true;
      else if (tile[nb] >= 0 && owner[nb] === 1) hasSource = true;
    }
    if (targetIsCmd && hasSource) return true;
  }
  return false;
}

function dukeSquare(tile: Int8Array, owner: Uint8Array): number {
  for (let s = 0; s < N_SQ; s++) if (tile[s] === DUKE && owner[s] === 0) return s;
  return 0;   // as jnp.argmax of all-false
}

// ---------------------------------------------------------------------------
// Board actions
// ---------------------------------------------------------------------------

interface Decoded { from: number; to: number; isStrike: boolean; isCmd: boolean; commander: number }

function decodeBoardAction(tile: Int8Array, side: Uint8Array, a: number): Decoded {
  if (a < N_MOVE_ACTIONS) {
    const f = Math.floor(a / N_SQ), g = a % N_SQ;
    const isStrike = STRIKE_OK[tIdx(0, Math.max(tile[f], 0), side[f], f, g)] === 1;   // as JAX, even if f is empty
    return {from: f, to: g, isStrike, isCmd: false, commander: 0};
  }
  const c = a - CMD_BASE;
  const commander = Math.floor(c / 64), src = Math.floor(c / 8) % 8, dst = c % 8;
  return {from: Math.max(NBR8[commander * 8 + src], 0), to: Math.max(NBR8[commander * 8 + dst], 0),
    isStrike: false, isCmd: true, commander};
}

/** Apply a board action in place (no perspective flip). */
function applyBoardAction(tile: Int8Array, owner: Uint8Array, side: Uint8Array, d: Decoded) {
  const movedTile = tile[d.from], movedSide = side[d.from];
  tile[d.to] = -1; owner[d.to] = 0; side[d.to] = 0;                  // capture
  if (d.isStrike) {
    side[d.from] = 1 - side[d.from];                                  // striker flips in place
    return;
  }
  tile[d.from] = -1; owner[d.from] = 0; side[d.from] = 0;
  tile[d.to] = movedTile; owner[d.to] = 0;
  // A moved tile flips; a commanded one doesn't, but its commander does.
  side[d.to] = d.isCmd ? movedSide : 1 - movedSide;
  if (d.isCmd) side[d.commander] = 1 - side[d.commander];
}

// ---------------------------------------------------------------------------
// Legal moves
// ---------------------------------------------------------------------------

const scratch = {tile: new Int8Array(N_SQ), owner: new Uint8Array(N_SQ), side: new Uint8Array(N_SQ)};

/** Legal-action mask [3605] and placement mask [4] for the player to move. */
export function legalActions(tile: Int8Array, owner: Uint8Array, side: Uint8Array, bags: Int32Array,
                             pending: number): {legal: Uint8Array; placeOk: Uint8Array} {
  const legal = new Uint8Array(N_ACTIONS);
  const placeOk = new Uint8Array(4);
  const duke = dukeSquare(tile, owner);
  const mine = (s: number) => tile[s] >= 0 && owner[s] === 0;
  const opp = (s: number) => tile[s] >= 0 && owner[s] === 1;

  // Placement squares next to my Duke: empty, and a blocker there must not
  // leave my Duke attacked (tile-independent, so it also decides draw legality).
  let anyPlace = false;
  for (let d = 0; d < 4; d++) {
    const s = NBR4[duke * 4 + d];
    if (s < 0 || tile[s] >= 0) continue;
    scratch.tile.set(tile); scratch.owner.set(owner); scratch.side.set(side);
    scratch.tile[s] = 0; scratch.owner[s] = 0;
    if (!isAttacked(scratch.tile, scratch.owner, scratch.side, duke)) { placeOk[d] = 1; anyPlace = true; }
  }
  if (pending >= 0) {
    for (let d = 0; d < 4; d++) legal[PLACE_BASE + d] = placeOk[d];
    return {legal, placeOk};
  }
  if (pending !== NONE) return {legal, placeOk};   // DRAWING: wait for the chance step

  const safe = (a: number) => {
    const d = decodeBoardAction(tile, side, a);
    if (opp(d.to) && tile[d.to] === DUKE) return true;   // taking the enemy Duke ends the game
    scratch.tile.set(tile); scratch.owner.set(owner); scratch.side.set(side);
    applyBoardAction(scratch.tile, scratch.owner, scratch.side, d);
    return !isAttacked(scratch.tile, scratch.owner, scratch.side, dukeSquare(scratch.tile, scratch.owner));
  };

  for (let f = 0; f < N_SQ; f++) {
    if (!mine(f)) continue;
    const t = tile[f], sd = side[f];
    // Moves and strikes.
    for (let g = 0; g < N_SQ; g++) {
      const k = tIdx(0, t, sd, f, g);
      let ok = false;
      if (MOVE_OK[k] && !mine(g)) {
        ok = true;
        if (NEEDS_CLEAR[k]) {
          for (let b = 0; b < N_SQ; b++) if (BETWEEN[bIdx(f, g, b)] && tile[b] >= 0) { ok = false; break; }
        }
      }
      if (!ok && STRIKE_OK[k] && opp(g)) ok = true;
      if (ok && safe(f * N_SQ + g)) legal[f * N_SQ + g] = 1;
    }
    // Commands: move one of my tiles between two of f's command squares.
    for (let sdir = 0; sdir < 8; sdir++) {
      const src = NBR8[f * 8 + sdir];
      if (src < 0 || !CMD_DIR[cIdx(0, t, sd, sdir)] || !mine(src)) continue;
      for (let ddir = 0; ddir < 8; ddir++) {
        const dst = NBR8[f * 8 + ddir];
        if (ddir === sdir || dst < 0 || !CMD_DIR[cIdx(0, t, sd, ddir)] || mine(dst)) continue;
        const a = CMD_BASE + f * 64 + sdir * 8 + ddir;
        if (safe(a)) legal[a] = 1;
      }
    }
  }

  let bagCount = 0;
  for (let i = 0; i < N_TYPES; i++) bagCount += bags[i];
  legal[DRAW_ACTION] = bagCount > 0 && anyPlace ? 1 : 0;
  return {legal, placeOk};
}

/** Fill in `legal` and `placeOk` for a state built by hand. */
export function withLegal(s: State): State {
  const {legal, placeOk} = legalActions(s.tile, s.owner, s.side, s.bags, s.pending);
  return {...s, legal, placeOk};
}

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

function flipPerspective(tile: Int8Array, owner: Uint8Array, side: Uint8Array, bags: Int32Array) {
  const t = new Int8Array(N_SQ), o = new Uint8Array(N_SQ), sd = new Uint8Array(N_SQ);
  for (let s = 0; s < N_SQ; s++) {
    const m = FLIP[s];
    t[s] = tile[m]; sd[s] = side[m];
    o[s] = tile[m] >= 0 ? 1 - owner[m] : 0;
  }
  const b = new Int32Array(2 * N_TYPES);
  b.set(bags.subarray(N_TYPES), 0);
  b.set(bags.subarray(0, N_TYPES), N_TYPES);
  return {tile: t, owner: o, side: sd, bags: b};
}

/** Apply a decision; returns the afterstate. An illegal action loses for the mover. */
export function stepDecision(s: State, action: number): State {
  if (s.done) return {...copyState(s), reward: 0, switched: false};
  let tile = s.tile.slice(), owner = s.owner.slice(), side = s.side.slice(), bags = s.bags.slice();
  const isDraw = action === DRAW_ACTION;
  const isPlace = action >= PLACE_BASE;
  const isBoard = action < DRAW_ACTION;

  let takesDuke = false;
  if (isBoard) {
    const d = decodeBoardAction(tile, side, action);
    takesDuke = owner[d.to] === 1 && tile[d.to] === DUKE;
    applyBoardAction(tile, owner, side, d);
  } else if (isPlace) {
    const duke = dukeSquare(tile, owner);
    const sq = Math.max(NBR4[duke * 4 + Math.min(Math.max(action - PLACE_BASE, 0), 3)], 0);
    tile[sq] = Math.max(s.pending, 0); owner[sq] = 0; side[sq] = 0;
  }
  const pending = isDraw ? DRAWING : NONE;

  const legalA = s.legal[action] === 1;
  const illegal = !legalA;
  const win = legalA && takesDuke;
  const turnEnds = legalA && !win && !isDraw;
  const ply = s.ply + (turnEnds ? 1 : 0);
  const timeout = turnEnds && ply >= MAX_PLY;
  const passes = turnEnds && !timeout;

  let player = s.player;
  if (passes) {
    ({tile, owner, side, bags} = flipPerspective(tile, owner, side, bags));
    player = 1 - s.player;
  }
  const {legal, placeOk} = legalActions(tile, owner, side, bags, pending);
  let anyLegal = false;
  for (let a = 0; a < N_ACTIONS; a++) if (legal[a]) { anyLegal = true; break; }

  // The new player to move with no legal move loses.
  const stuck = passes && !anyLegal;
  const moverWins = win || stuck;
  const done = illegal || win || timeout || stuck;
  return {
    tile, owner, side, bags, player, pending, ply, done,
    winner: moverWins ? s.player : illegal ? 1 - s.player : -1,
    reward: moverWins ? 1 : illegal ? -1 : 0,
    switched: passes,
    legal: done || isDraw ? new Uint8Array(N_ACTIONS) : legal,
    // A draw leaves the board untouched, so placement legality carries over.
    placeOk: isDraw ? s.placeOk.slice() : placeOk,
  };
}

/** Probability of each chance outcome [15] from an afterstate. */
export function chanceProbs(s: State): Float64Array {
  const p = new Float64Array(N_TYPES);
  if (s.pending === DRAWING && !s.done) {
    let total = 0;
    for (let i = 0; i < N_TYPES; i++) total += s.bags[i];
    for (let i = 0; i < N_TYPES; i++) p[i] = s.bags[i] / Math.max(total, 1);
  } else {
    p[0] = 1;   // deterministic afterstates have the single outcome 0
  }
  return p;
}

/** Resolve an afterstate's randomness (reveal the drawn tile). */
export function stepChance(s: State, outcome: number): State {
  if (!(s.pending === DRAWING && !s.done)) return s;
  const bags = s.bags.slice();
  bags[outcome] -= 1;
  const legal = new Uint8Array(N_ACTIONS);
  for (let d = 0; d < 4; d++) legal[PLACE_BASE + d] = s.placeOk[d];
  return {...copyState(s), pending: outcome, bags, reward: 0, switched: false, legal};
}

// ---------------------------------------------------------------------------
// Network input
// ---------------------------------------------------------------------------

export const N_PLANES = 2 * N_TYPES * 2;                  // 60
export const N_FEATURES = 2 * N_TYPES + N_TYPES + 2;      // 47
export const PLANES_SIZE = N_SQ * N_PLANES;

/** Write the network input for `s` into planes[off..] (NHWC) and features[off..]. */
export function observe(s: State, planes: Float32Array, features: Float32Array,
                        planesOff = 0, featuresOff = 0) {
  planes.fill(0, planesOff, planesOff + PLANES_SIZE);
  for (let q = 0; q < N_SQ; q++) {
    if (s.tile[q] < 0) continue;
    planes[planesOff + q * N_PLANES + s.owner[q] * 2 * N_TYPES + s.tile[q] * 2 + s.side[q]] = 1;
  }
  const f = featuresOff;
  for (let i = 0; i < 2 * N_TYPES; i++) features[f + i] = s.bags[i] / BAG_SCALE;
  for (let i = 0; i < N_TYPES; i++) features[f + 2 * N_TYPES + i] = s.pending === i ? 1 : 0;
  features[f + 3 * N_TYPES] = s.pending === DRAWING ? 1 : 0;
  features[f + 3 * N_TYPES + 1] = s.ply / MAX_PLY;
}
