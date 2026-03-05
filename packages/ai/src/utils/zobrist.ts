import type {GameState} from '@the-duke/engine';
import {ALL_TILES} from '@the-duke/engine';

// ---------------------------------------------------------------------------
// Zobrist hashing for transposition table
// ---------------------------------------------------------------------------

// Generate deterministic pseudo-random 32-bit keys using a simple xorshift PRNG
// (32-bit is fine — collisions are rare enough for a 6×6 board)
function makeZobristKeys() {
  let seed = 0xDEAD_BEEF;
  function rand32(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  }

  const tileNames = ALL_TILES.map(t => t.name);
  const tileIndex = new Map(tileNames.map((n, i) => [n, i]));

  // board[row][col][tileNameIdx][ownerIdx][sideIdx]
  // 6 * 6 * 15 tiles * 2 owners * 2 sides = 2160 keys
  const board: number[][][][] = [];
  for (let r = 0; r < 6; r++) {
    board[r] = [];
    for (let c = 0; c < 6; c++) {
      board[r][c] = [];
      for (let t = 0; t < tileNames.length; t++) {
        board[r][c][t] = [rand32(), rand32(), rand32(), rand32()];
        // index: owner(0=P1,1=P2) * 2 + side(0=A,1=B)
      }
    }
  }

  const turnKey = rand32(); // XOR when currentPlayer === 'P2'

  // Bag keys: bagKeys[tileNameIdx][playerIdx][count]
  // Max 3 copies of any tile in bag (Footman x3, Pikeman x3)
  const bagKeys: number[][][] = [];
  for (let t = 0; t < tileNames.length; t++) {
    bagKeys[t] = [];
    for (let p = 0; p < 2; p++) {
      bagKeys[t][p] = [];
      for (let c = 0; c <= 3; c++) {
        bagKeys[t][p][c] = rand32();
      }
    }
  }

  return {board, turnKey, tileIndex, bagKeys};
}

const ZOBRIST = makeZobristKeys();

// Pre-allocated array for bag counting — avoids Map allocation per call.
// Indexed by [tileNameIdx * 2 + playerIdx], stores count.
const _bagCounts = new Int8Array(ALL_TILES.length * 2);

export function hashState(state: GameState): number {
  let h = 0;
  for (const tile of state.tiles.values()) {
    const ti = ZOBRIST.tileIndex.get(tile.defName);
    if (ti === undefined) continue;
    const ownerBit = tile.owner === 'P1' ? 0 : 1;
    const sideBit = tile.side === 'A' ? 0 : 1;
    h ^= ZOBRIST.board[tile.position.row][tile.position.col][ti][ownerBit * 2 + sideBit];
  }
  if (state.currentPlayer === 'P2') h ^= ZOBRIST.turnKey;

  // Hash bag contents using pre-allocated count array (no Map allocation)
  _bagCounts.fill(0);
  for (const name of state.bags.P1) {
    const ti = ZOBRIST.tileIndex.get(name);
    if (ti !== undefined) _bagCounts[ti * 2]++;
  }
  for (const name of state.bags.P2) {
    const ti = ZOBRIST.tileIndex.get(name);
    if (ti !== undefined) _bagCounts[ti * 2 + 1]++;
  }
  const numTiles = ALL_TILES.length;
  for (let ti = 0; ti < numTiles; ti++) {
    h ^= ZOBRIST.bagKeys[ti][0][_bagCounts[ti * 2]];
    h ^= ZOBRIST.bagKeys[ti][1][_bagCounts[ti * 2 + 1]];
  }

  return h;
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

export const TT_EXACT = 0;
export const TT_LOWER = 1; // alpha cutoff — score is a lower bound
export const TT_UPPER = 2; // beta cutoff  — score is an upper bound

interface TTEntry {
  hash: number;
  depth: number;
  score: number;
  flag: number;
}

// Fixed-size table (power of 2 for fast masking)
const TT_SIZE = 1 << 20; // ~1M entries
const TT_MASK = TT_SIZE - 1;
const tt: (TTEntry | null)[] = new Array(TT_SIZE).fill(null);

export function ttProbe(hash: number, depth: number, alpha: number, beta: number): number | null {
  const entry = tt[hash & TT_MASK];
  if (entry === null || entry.hash !== hash || entry.depth < depth) return null;

  if (entry.flag === TT_EXACT) return entry.score;
  if (entry.flag === TT_LOWER && entry.score >= beta) return entry.score;
  if (entry.flag === TT_UPPER && entry.score <= alpha) return entry.score;
  return null;
}

export function ttStore(hash: number, depth: number, score: number, flag: number): void {
  const idx = hash & TT_MASK;
  const existing = tt[idx];
  // Replace if empty or if new entry has >= depth (depth-preferred replacement)
  if (existing === null || existing.depth <= depth) {
    tt[idx] = {hash, depth, score, flag};
  }
}
