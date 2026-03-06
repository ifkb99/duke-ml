import type {GameMove, GameState} from '@the-duke/engine';
import {ALL_TILES, BOARD_SIZE} from '@the-duke/engine';

// ── Tile index lookup ──

const NUM_TILE_TYPES = ALL_TILES.length; // 15
const _tileIdx = new Map<string, number>();
for (let i = 0; i < NUM_TILE_TYPES; i++) _tileIdx.set(ALL_TILES[i].name, i);

function ti(name: string): number { return _tileIdx.get(name)!; }

/** Ordered tile names matching channel indices. Export for Python-side decoding. */
export const TILE_NAMES: readonly string[] = ALL_TILES.map(t => t.name);

// ── Board tensor layout ──
//
// CHW order: 33 channels × 6 × 6
//
//   Channels  0–14:  P1 tile presence (one-hot by tile type)
//   Channels 15–29:  P2 tile presence (one-hot by tile type)
//   Channel  30:     side-B indicator
//   Channel  31:     current player (uniform 1.0 when P1 to move)
//   Channel  32:     normalized turn number (turnNumber / 200, clamped to 1.0)

const BS = BOARD_SIZE;
const CELLS = BS * BS; // 36

export const NUM_CHANNELS = 33;
export const BOARD_TENSOR_SIZE = NUM_CHANNELS * CELLS; // 1188
export const BAG_FEATURE_SIZE = NUM_TILE_TYPES * 2;    // 30
export const STATE_TENSOR_SIZE = BOARD_TENSOR_SIZE + BAG_FEATURE_SIZE; // 1218

/**
 * Encode a GameState into a flat Float32Array region.
 * Writes STATE_TENSOR_SIZE values starting at `out[offset]`.
 * The region MUST already be zeroed or will be zeroed by this call.
 */
export function encodeState(state: GameState, out: Float32Array, offset: number): void {
  out.fill(0, offset, offset + STATE_TENSOR_SIZE);

  for (const tile of state.tiles.values()) {
    const cell = tile.position.row * BS + tile.position.col;
    const base = tile.owner === 'P1' ? ti(tile.defName) : NUM_TILE_TYPES + ti(tile.defName);
    out[offset + base * CELLS + cell] = 1;
    if (tile.side === 'B') {
      out[offset + 30 * CELLS + cell] = 1;
    }
  }

  if (state.currentPlayer === 'P1') {
    out.fill(1, offset + 31 * CELLS, offset + 32 * CELLS);
  }

  const t = Math.min(state.turnNumber / 200, 1);
  out.fill(t, offset + 32 * CELLS, offset + 33 * CELLS);

  const bagOff = offset + BOARD_TENSOR_SIZE;
  for (const name of state.bags.P1) out[bagOff + ti(name)] += 1;
  for (const name of state.bags.P2) out[bagOff + NUM_TILE_TYPES + ti(name)] += 1;
}

/** Convenience: encode into a fresh Float32Array (allocates). */
export function encodeStateFresh(state: GameState): Float32Array {
  const out = new Float32Array(STATE_TENSOR_SIZE);
  encodeState(state, out, 0);
  return out;
}

// ── Action space ──
//
// Fixed discrete action space of 4428 actions:
//
//   move(from, to):            from*36 + to               [0,    1296)
//   strike(from, target):      1296 + from*36 + target     [1296, 2592)
//   command(target, targetTo): 2592 + target*36 + targetTo [2592, 3888)
//   place(tileIdx, pos):       3888 + tileIdx*36 + pos     [3888, 4428)
//
// Coordinates encoded as row * 6 + col.
// For command, the commander is inferred from the game state during decoding.

const STRIKE_BASE  = CELLS * CELLS;                       // 1296
const COMMAND_BASE = STRIKE_BASE + CELLS * CELLS;          // 2592
const PLACE_BASE   = COMMAND_BASE + CELLS * CELLS;         // 3888
export const TOTAL_ACTIONS = PLACE_BASE + NUM_TILE_TYPES * CELLS; // 4428

function sq(r: number, c: number): number { return r * BS + c; }

/** Map a GameMove to its integer action index. */
export function encodeMoveIndex(move: GameMove): number {
  switch (move.type) {
    case 'move':
      return sq(move.from.row, move.from.col) * CELLS + sq(move.to.row, move.to.col);
    case 'strike':
      return STRIKE_BASE + sq(move.from.row, move.from.col) * CELLS + sq(move.target.row, move.target.col);
    case 'command':
      return COMMAND_BASE + sq(move.target.row, move.target.col) * CELLS + sq(move.targetTo.row, move.targetTo.col);
    case 'place':
      return PLACE_BASE + ti(move.tileName) * CELLS + sq(move.position.row, move.position.col);
  }
}

/**
 * Find the legal move corresponding to an action index.
 * Pass the pre-computed legal moves to avoid redundant generation.
 */
export function decodeMoveIndex(index: number, legalMoves: GameMove[]): GameMove | null {
  for (const m of legalMoves) {
    if (encodeMoveIndex(m) === index) return m;
  }
  return null;
}

/**
 * Write a binary legal-move mask into `out` at `offset`.
 * Sets 1.0 for each legal action index, 0.0 elsewhere.
 */
export function encodeLegalMask(moves: GameMove[], out: Float32Array, offset: number): void {
  out.fill(0, offset, offset + TOTAL_ACTIONS);
  for (const m of moves) out[offset + encodeMoveIndex(m)] = 1;
}
