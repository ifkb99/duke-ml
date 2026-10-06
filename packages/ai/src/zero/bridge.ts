/**
 * Conversions between the engine's game state / moves and the zero
 * environment's canonical state / action numbers (training/zero/ts_bridge.py).
 *
 * The engine uses absolute coordinates: P1's home row is 0, P2's is 5. The
 * canonical frame of the player to move is the engine frame for P2 and the
 * row-mirrored frame for P1, so canon(sq) is FLIP[sq] for P1 and sq for P2
 * (its own inverse, so the same mapping converts back).
 */

import type {Coord, GameMove, GameState, SerializedGameState} from '@the-duke/engine';
import {NONE, type State, withLegal} from './env.js';
import {
  BOARD, CMD_BASE, DIRS4, DIRS8, DRAW_ACTION, DUKE, FLIP, N_MOVE_ACTIONS, N_SQ, N_TYPES, NBR4, NBR8,
  PLACE_BASE, STRIKE_OK, TILE_INDEX, TILE_NAMES, tIdx,
} from './rules.js';

const PLAYERS = ['P1', 'P2'] as const;

const canon = (sq: number, player: number) => (player === 0 ? FLIP[sq] : sq);
const sqOf = (c: Coord) => c.row * BOARD + c.col;
const coordOf = (sq: number): Coord => ({row: Math.floor(sq / BOARD), col: sq % BOARD});

/** The zero state for an engine state (live or serialized), with legal moves. */
export function fromGameState(game: GameState | SerializedGameState): State {
  const player = PLAYERS.indexOf(game.currentPlayer as 'P1' | 'P2');
  const tile = new Int8Array(N_SQ).fill(-1), owner = new Uint8Array(N_SQ), side = new Uint8Array(N_SQ);
  const tiles = game.tiles instanceof Map ? [...game.tiles.values()] : game.tiles.map(([, t]) => t);
  for (const t of tiles) {
    const s = canon(sqOf(t.position), player);
    tile[s] = TILE_INDEX[t.defName];
    owner[s] = PLAYERS.indexOf(t.owner as 'P1' | 'P2') === player ? 0 : 1;
    side[s] = t.side === 'A' ? 0 : 1;
  }
  const bags = new Int32Array(2 * N_TYPES);
  [player, 1 - player].forEach((p, rel) => {
    for (const name of game.bags[PLAYERS[p]]) bags[rel * N_TYPES + TILE_INDEX[name]]++;
  });
  const status = game.status as string;
  return withLegal({
    tile, owner, side, bags, player, pending: NONE, ply: game.turnNumber - 1,
    done: status === 'P1_wins' || status === 'P2_wins',
    winner: status === 'P1_wins' ? 0 : status === 'P2_wins' ? 1 : -1,
    reward: 0, switched: false, legal: new Uint8Array(0), placeOk: new Uint8Array(4),
  });
}

function dir(from: number, to: number, dirs: [number, number][]): number {
  const dr = Math.floor(to / BOARD) - Math.floor(from / BOARD), dc = (to % BOARD) - (from % BOARD);
  return dirs.findIndex(([a, b]) => a === dr && b === dc);
}

function dukeSquare(s: State): number {
  for (let q = 0; q < N_SQ; q++) if (s.tile[q] === DUKE && s.owner[q] === 0) return q;
  throw new Error('no Duke');
}

/**
 * An action for the player to move in `s` as an engine move. A place action
 * needs the tile in hand (`s.pending`) and becomes the engine's resolved
 * `place`; the draw action itself has no engine move (draw, then place).
 */
export function actionToMove(action: number, s: State): GameMove {
  const p = s.player;
  const abs = (sq: number) => coordOf(canon(sq, p));
  if (action < N_MOVE_ACTIONS) {
    const from = Math.floor(action / N_SQ), to = action % N_SQ;
    if (STRIKE_OK[tIdx(0, s.tile[from], s.side[from], from, to)]) {
      return {type: 'strike', from: abs(from), target: abs(to)};
    }
    return {type: 'move', from: abs(from), to: abs(to)};
  }
  if (action < DRAW_ACTION) {
    const c = action - CMD_BASE;
    const cmd = Math.floor(c / 64), src = NBR8[cmd * 8 + (Math.floor(c / 8) % 8)], dst = NBR8[cmd * 8 + (c % 8)];
    return {type: 'command', commander: abs(cmd), target: abs(src), targetTo: abs(dst)};
  }
  if (action >= PLACE_BASE && s.pending >= 0) {
    const sq = NBR4[dukeSquare(s) * 4 + action - PLACE_BASE];
    return {type: 'place', tileName: TILE_NAMES[s.pending], position: abs(sq)};
  }
  throw new Error('the draw action has no engine move: draw a tile, then search the placement');
}

/** An engine move (not draw/place) as an action number for the player to move in `s`. */
export function moveToAction(move: GameMove, s: State): number {
  const c = (coord: Coord) => canon(sqOf(coord), s.player);
  switch (move.type) {
    case 'move': return c(move.from) * N_SQ + c(move.to);
    case 'strike': return c(move.from) * N_SQ + c(move.target);
    case 'command': {
      const cmd = c(move.commander);
      return CMD_BASE + cmd * 64 + dir(cmd, c(move.target), DIRS8) * 8 + dir(cmd, c(move.targetTo), DIRS8);
    }
    default:
      // draw/place: DRAW_ACTION, then (after the tile is known) the place action.
      return PLACE_BASE + dir(dukeSquare(s), c(move.position), DIRS4);
  }
}

