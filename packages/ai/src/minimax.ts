import type { GameMove, GameState } from '@the-duke/engine';
import { generateAllMoves, applyMove, ALL_TILES } from '@the-duke/engine';
import { evaluate } from './evaluate.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
}

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

  return { board, turnKey, tileIndex };
}

const ZOBRIST = makeZobristKeys();

function hashState(state: GameState): number {
  let h = 0;
  for (const tile of state.tiles.values()) {
    const ti = ZOBRIST.tileIndex.get(tile.defName);
    if (ti === undefined) continue;
    const ownerBit = tile.owner === 'P1' ? 0 : 1;
    const sideBit = tile.side === 'A' ? 0 : 1;
    h ^= ZOBRIST.board[tile.position.row][tile.position.col][ti][ownerBit * 2 + sideBit];
  }
  if (state.currentPlayer === 'P2') h ^= ZOBRIST.turnKey;
  return h;
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

const TT_EXACT = 0;
const TT_LOWER = 1; // alpha cutoff — score is a lower bound
const TT_UPPER = 2; // beta cutoff  — score is an upper bound

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

function ttProbe(hash: number, depth: number, alpha: number, beta: number): number | null {
  const entry = tt[hash & TT_MASK];
  if (entry === null || entry.hash !== hash || entry.depth < depth) return null;

  if (entry.flag === TT_EXACT) return entry.score;
  if (entry.flag === TT_LOWER && entry.score >= beta) return entry.score;
  if (entry.flag === TT_UPPER && entry.score <= alpha) return entry.score;
  return null;
}

function ttStore(hash: number, depth: number, score: number, flag: number): void {
  const idx = hash & TT_MASK;
  const existing = tt[idx];
  // Replace if empty or if new entry has >= depth (depth-preferred replacement)
  if (existing === null || existing.depth <= depth) {
    tt[idx] = { hash, depth, score, flag };
  }
}

// ---------------------------------------------------------------------------
// Move ordering — captures and strikes first
// ---------------------------------------------------------------------------

function orderMoves(moves: GameMove[], state: GameState): GameMove[] {
  return moves.sort((a, b) => moveScore(b, state) - moveScore(a, state));
}

function moveScore(move: GameMove, state: GameState): number {
  if (move.type === 'strike') {
    const targetId = state.board[move.target.row][move.target.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 1000;
    }
    return 10;
  }
  if (move.type === 'move') {
    const targetId = state.board[move.to.row][move.to.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 1000;
      return 10;
    }
    return 0;
  }
  if (move.type === 'command') {
    const targetId = state.board[move.targetTo.row][move.targetTo.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 1000;
      return 10;
    }
    return 0;
  }
  // place moves — lowest priority
  return -1;
}

// ---------------------------------------------------------------------------
// Minimax with alpha-beta pruning + transposition table
// ---------------------------------------------------------------------------

function minimax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  stats: { nodes: number },
): number {
  stats.nodes++;

  if (depth === 0 || state.status !== 'active') {
    return evaluate(state);
  }

  const hash = hashState(state);

  // Transposition table lookup
  const ttHit = ttProbe(hash, depth, alpha, beta);
  if (ttHit !== null) return ttHit;

  const moves = orderMoves(generateAllMoves(state), state);

  if (moves.length === 0) {
    // No legal moves = loss for current player
    return maximizing ? -99_999 : 99_999;
  }

  let bestScore: number;
  let flag: number;

  if (maximizing) {
    bestScore = -Infinity;
    for (const move of moves) {
      const child = applyMove(state, move);
      bestScore = Math.max(bestScore, minimax(child, depth - 1, alpha, beta, false, stats));
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) break;
    }
    flag = bestScore >= beta ? TT_LOWER : (bestScore <= alpha ? TT_UPPER : TT_EXACT);
  } else {
    bestScore = Infinity;
    for (const move of moves) {
      const child = applyMove(state, move);
      bestScore = Math.min(bestScore, minimax(child, depth - 1, alpha, beta, true, stats));
      beta = Math.min(beta, bestScore);
      if (alpha >= beta) break;
    }
    flag = bestScore <= alpha ? TT_UPPER : (bestScore >= beta ? TT_LOWER : TT_EXACT);
  }

  ttStore(hash, depth, bestScore, flag);
  return bestScore;
}

/**
 * Find the best move using minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 */
export function findBestMove(state: GameState, depth = 6): MinimaxResult {
  // TODO: ensure that draw move "reward" is sum(bag_piece_scores) / n_pieces_in_bag
  const moves = orderMoves(generateAllMoves(state), state);
  if (moves.length === 0) return { move: null, score: 0, nodesSearched: 0 };

  const maximizing = state.currentPlayer === 'P1';
  const stats = { nodes: 0 };

  let bestMove: GameMove = moves[0];
  let bestScore = maximizing ? -Infinity : Infinity;
  let alpha = -Infinity;
  let beta = Infinity;

  for (const move of moves) {
    const child = applyMove(state, move);
    const score = minimax(child, depth - 1, alpha, beta, !maximizing, stats);

    if (maximizing) {
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
      alpha = Math.max(alpha, bestScore);
    } else {
      if (score < bestScore) {
        bestScore = score;
        bestMove = move;
      }
      beta = Math.min(beta, bestScore);
    }
  }

  return { move: bestMove, score: bestScore, nodesSearched: stats.nodes };
}
