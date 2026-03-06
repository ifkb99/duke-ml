import type {GameState, UndoRecord} from '@the-duke/engine';
import {ALL_TILES} from '@the-duke/engine';

// ---------------------------------------------------------------------------
// Shared tile name → index mapping (used by evaluate.ts too)
// ---------------------------------------------------------------------------

const TILE_NAMES = ALL_TILES.map(t => t.name);
export const TILE_NAME_INDEX: ReadonlyMap<string, number> = new Map(
  TILE_NAMES.map((n, i) => [n, i]),
);

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

function makeZobristKeys() {
  let seed = 0xDEAD_BEEF;
  function rand32(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  }

  // board[row][col][tileNameIdx][combo] where combo = ownerBit*2 + sideBit
  // 6 * 6 * 15 tiles * 4 combos = 2160 keys
  const board: number[][][][] = [];
  for (let r = 0; r < 6; r++) {
    board[r] = [];
    for (let c = 0; c < 6; c++) {
      board[r][c] = [];
      for (let t = 0; t < TILE_NAMES.length; t++) {
        board[r][c][t] = [rand32(), rand32(), rand32(), rand32()];
      }
    }
  }

  const turnKey = rand32();

  // bagKeys[tileNameIdx][playerIdx][count] — max 3 copies
  const bagKeys: number[][][] = [];
  for (let t = 0; t < TILE_NAMES.length; t++) {
    bagKeys[t] = [];
    for (let p = 0; p < 2; p++) {
      bagKeys[t][p] = [];
      for (let c = 0; c <= 3; c++) {
        bagKeys[t][p][c] = rand32();
      }
    }
  }

  return {board, turnKey, bagKeys};
}

const Z = makeZobristKeys();

// Pre-allocated array for bag counting
const _bagCounts = new Int8Array(ALL_TILES.length * 2);

export function hashState(state: GameState): number {
  let h = 0;
  for (const tile of state.tiles.values()) {
    const ti = TILE_NAME_INDEX.get(tile.defName);
    if (ti === undefined) continue;
    const combo = (tile.owner === 'P1' ? 0 : 2) + (tile.side === 'A' ? 0 : 1);
    h ^= Z.board[tile.position.row][tile.position.col][ti][combo];
  }
  if (state.currentPlayer === 'P2') h ^= Z.turnKey;

  _bagCounts.fill(0);
  for (const name of state.bags.P1) {
    const ti = TILE_NAME_INDEX.get(name);
    if (ti !== undefined) _bagCounts[ti * 2]++;
  }
  for (const name of state.bags.P2) {
    const ti = TILE_NAME_INDEX.get(name);
    if (ti !== undefined) _bagCounts[ti * 2 + 1]++;
  }
  const numTiles = ALL_TILES.length;
  for (let ti = 0; ti < numTiles; ti++) {
    h ^= Z.bagKeys[ti][0][_bagCounts[ti * 2]];
    h ^= Z.bagKeys[ti][1][_bagCounts[ti * 2 + 1]];
  }

  return h;
}

// ---------------------------------------------------------------------------
// Incremental hash update — call after makeMove, returns new hash.
// XOR is self-inverse, so the same delta undoes itself.
// ---------------------------------------------------------------------------

function countTileInBag(bag: string[], tileName: string): number {
  let c = 0;
  for (let i = 0; i < bag.length; i++) {
    if (bag[i] === tileName) c++;
  }
  return c;
}

/**
 * Compute the new hash after makeMove was applied.
 * `undo` is the UndoRecord returned by makeMove; `state` is the post-move state.
 */
export function hashAfterMove(oldHash: number, state: GameState, undo: UndoRecord): number {
  let h = oldHash ^ Z.turnKey; // turn always switches

  const move = undo.move;

  switch (move.type) {
    case 'place': {
      const tile = state.tiles.get(undo.placedTileId!)!;
      const ti = TILE_NAME_INDEX.get(tile.defName)!;
      const combo = (tile.owner === 'P1' ? 0 : 2); // side always A → +0
      h ^= Z.board[tile.position.row][tile.position.col][ti][combo];

      const playerIdx = undo.removedFromBag!.player === 'P1' ? 0 : 1;
      const newCount = countTileInBag(state.bags[undo.removedFromBag!.player], tile.defName);
      h ^= Z.bagKeys[ti][playerIdx][newCount + 1]; // XOR out old count
      h ^= Z.bagKeys[ti][playerIdx][newCount];      // XOR in new count
      break;
    }
    case 'move': {
      const tile = state.tiles.get(undo.movedTileId!)!;
      const ti = TILE_NAME_INDEX.get(tile.defName)!;
      const ownerBase = tile.owner === 'P1' ? 0 : 2;
      const oldSideBit = undo.prevSide === 'A' ? 0 : 1;
      const newSideBit = tile.side === 'A' ? 0 : 1;
      const oldPos = undo.prevPosition!;

      h ^= Z.board[oldPos.row][oldPos.col][ti][ownerBase + oldSideBit];
      h ^= Z.board[tile.position.row][tile.position.col][ti][ownerBase + newSideBit];

      if (undo.capturedTile) {
        const cap = undo.capturedTile;
        const capTi = TILE_NAME_INDEX.get(cap.defName)!;
        const capCombo = (cap.owner === 'P1' ? 0 : 2) + (cap.side === 'A' ? 0 : 1);
        h ^= Z.board[cap.position.row][cap.position.col][capTi][capCombo];
      }
      break;
    }
    case 'strike': {
      const striker = state.tiles.get(undo.movedTileId!)!;
      const ti = TILE_NAME_INDEX.get(striker.defName)!;
      const ownerBase = striker.owner === 'P1' ? 0 : 2;
      const oldSideBit = undo.prevSide === 'A' ? 0 : 1;
      const newSideBit = striker.side === 'A' ? 0 : 1;

      // Striker stays in place but flips side
      h ^= Z.board[striker.position.row][striker.position.col][ti][ownerBase + oldSideBit];
      h ^= Z.board[striker.position.row][striker.position.col][ti][ownerBase + newSideBit];

      const cap = undo.capturedTile!;
      const capTi = TILE_NAME_INDEX.get(cap.defName)!;
      const capCombo = (cap.owner === 'P1' ? 0 : 2) + (cap.side === 'A' ? 0 : 1);
      h ^= Z.board[cap.position.row][cap.position.col][capTi][capCombo];
      break;
    }
    case 'command': {
      const commanded = state.tiles.get(undo.commandedTileId!)!;
      const ti = TILE_NAME_INDEX.get(commanded.defName)!;
      const ownerBase = commanded.owner === 'P1' ? 0 : 2;
      const oldSideBit = undo.commandedPrevSide === 'A' ? 0 : 1;
      const newSideBit = commanded.side === 'A' ? 0 : 1;
      const oldPos = undo.commandedPrevPosition!;

      h ^= Z.board[oldPos.row][oldPos.col][ti][ownerBase + oldSideBit];
      h ^= Z.board[commanded.position.row][commanded.position.col][ti][ownerBase + newSideBit];

      if (undo.commandCapturedTile) {
        const cap = undo.commandCapturedTile;
        const capTi = TILE_NAME_INDEX.get(cap.defName)!;
        const capCombo = (cap.owner === 'P1' ? 0 : 2) + (cap.side === 'A' ? 0 : 1);
        h ^= Z.board[cap.position.row][cap.position.col][capTi][capCombo];
      }
      break;
    }
  }

  return h;
}

// ---------------------------------------------------------------------------
// Transposition table — flat typed arrays, GC-free, SharedArrayBuffer-ready
// ---------------------------------------------------------------------------

export const TT_EXACT = 0;
export const TT_LOWER = 1;
export const TT_UPPER = 2;

const TT_SIZE = 1 << 20; // ~1M entries
const TT_MASK = TT_SIZE - 1;

const TT_HASH  = new Int32Array(TT_SIZE);
const TT_DEPTH = new Int16Array(TT_SIZE);
const TT_SCORE = new Float32Array(TT_SIZE);
const TT_FLAG  = new Uint8Array(TT_SIZE);
const TT_MOVE  = new Int32Array(TT_SIZE); // encoded best move

// Sentinel: depth -1 means empty slot
TT_DEPTH.fill(-1);

export function ttProbe(hash: number, depth: number, alpha: number, beta: number): number | null {
  const idx = hash & TT_MASK;
  if (TT_DEPTH[idx] < 0 || TT_HASH[idx] !== hash || TT_DEPTH[idx] < depth) return null;

  const flag = TT_FLAG[idx];
  const score = TT_SCORE[idx];
  if (flag === TT_EXACT) return score;
  if (flag === TT_LOWER && score >= beta) return score;
  if (flag === TT_UPPER && score <= alpha) return score;
  return null;
}

/** Retrieve the best move encoded in the TT for move ordering. Returns 0 if none. */
export function ttProbeMove(hash: number): number {
  const idx = hash & TT_MASK;
  if (TT_HASH[idx] !== hash || TT_DEPTH[idx] < 0) return 0;
  return TT_MOVE[idx];
}

export function ttStore(hash: number, depth: number, score: number, flag: number, encodedMove: number): void {
  const idx = hash & TT_MASK;
  if (TT_DEPTH[idx] < 0 || TT_DEPTH[idx] <= depth) {
    TT_HASH[idx] = hash;
    TT_DEPTH[idx] = depth;
    TT_SCORE[idx] = score;
    TT_FLAG[idx] = flag;
    TT_MOVE[idx] = encodedMove;
  }
}

export function ttClear(): void {
  TT_DEPTH.fill(-1);
  TT_HASH.fill(0);
  TT_SCORE.fill(0);
  TT_FLAG.fill(0);
  TT_MOVE.fill(0);
}

// ---------------------------------------------------------------------------
// Move encoding — pack a GameMove into a 32-bit integer for TT storage
// Type (3 bits) | row1 (3) | col1 (3) | row2 (3) | col2 (3) | row3 (3) | col3 (3) | tileIdx (4)
// ---------------------------------------------------------------------------

export function encodeMove(move: import('@the-duke/engine').GameMove): number {
  switch (move.type) {
    case 'place': {
      const ti = TILE_NAME_INDEX.get(move.tileName) ?? 0;
      return 1 | (move.position.row << 3) | (move.position.col << 6) | (ti << 21);
    }
    case 'move':
      return 2 | (move.from.row << 3) | (move.from.col << 6) | (move.to.row << 9) | (move.to.col << 12);
    case 'strike':
      return 3 | (move.from.row << 3) | (move.from.col << 6) | (move.target.row << 9) | (move.target.col << 12);
    case 'command':
      return 4 | (move.commander.row << 3) | (move.commander.col << 6)
        | (move.target.row << 9) | (move.target.col << 12)
        | (move.targetTo.row << 15) | (move.targetTo.col << 18);
  }
}

export function movesMatch(move: import('@the-duke/engine').GameMove, encoded: number): boolean {
  if (encoded === 0) return false;
  return encodeMove(move) === encoded;
}
