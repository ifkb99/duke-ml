import type {GameMove, GameState, Player} from '@the-duke/engine';
import {findDukePosition, generatePseudoLegalMoves, isPlayerInCheck, isSquareAttackedBy, makeMove, unmakeMove} from '@the-duke/engine';
import {evaluate, evaluatePlacement} from './evaluate.js';
import {
  TT_EXACT, TT_LOWER, TT_UPPER,
  hashState, hashAfterMove, ttStore, ttProbe, ttProbeMove, ttClear,
  encodeMove, movesMatch,
} from './utils/zobrist.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_DEPTH = 64;
const WIN_SCORE = 99_999;
const MAX_PLACEMENTS_PER_TILE = 4;

// ---------------------------------------------------------------------------
// Killer move table — 2 slots per depth
// ---------------------------------------------------------------------------

const killerMoves: Int32Array = new Int32Array(MAX_DEPTH * 2);

function clearKillers(): void {
  killerMoves.fill(0);
}

function storeKiller(depth: number, encoded: number): void {
  const base = depth * 2;
  if (killerMoves[base] !== encoded) {
    killerMoves[base + 1] = killerMoves[base];
    killerMoves[base] = encoded;
  }
}

function isKiller(depth: number, encoded: number): boolean {
  const base = depth * 2;
  return killerMoves[base] === encoded || killerMoves[base + 1] === encoded;
}

// ---------------------------------------------------------------------------
// Inline legality check
// ---------------------------------------------------------------------------

function isLegalAfterMake(state: GameState, mover: Player): boolean {
  if (state.status !== 'active') return true;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const dukePos = findDukePosition(state, mover);
  return dukePos ? !isSquareAttackedBy(state, dukePos, opponent) : false;
}

// ---------------------------------------------------------------------------
// Move ordering — PV move first, then captures/strikes, then killers, then quiet
// ---------------------------------------------------------------------------

function isCapture(move: GameMove, state: GameState): boolean {
  if (move.type === 'strike') return true;
  if (move.type === 'move') return state.board[move.to.row][move.to.col] !== null;
  if (move.type === 'command') return state.board[move.targetTo.row][move.targetTo.col] !== null;
  return false;
}

function moveScore(move: GameMove, state: GameState, pvEncoded: number, depth: number): number {
  const encoded = encodeMove(move);

  // PV move from TT gets highest priority
  if (pvEncoded !== 0 && encoded === pvEncoded) return 100_000;

  if (move.type === 'strike') {
    const targetId = state.board[move.target.row][move.target.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
    }
    return 10_000;
  }
  if (move.type === 'move') {
    const targetId = state.board[move.to.row][move.to.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
      return 10_000;
    }
    if (isKiller(depth, encoded)) return 5_000;
    return 0;
  }
  if (move.type === 'command') {
    const targetId = state.board[move.targetTo.row][move.targetTo.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
      return 10_000;
    }
    if (isKiller(depth, encoded)) return 5_000;
    return 0;
  }
  // Place moves — lowest priority
  return -1;
}

function orderMoves(moves: GameMove[], state: GameState, pvEncoded: number, depth: number): GameMove[] {
  return moves.sort((a, b) => moveScore(b, state, pvEncoded, depth) - moveScore(a, state, pvEncoded, depth));
}

// ---------------------------------------------------------------------------
// Expectimax: split moves into board moves and draw outcomes
// ---------------------------------------------------------------------------

interface DrawOutcome {
  tileName: string;
  probability: number;
  placements: GameMove[];
}

function splitMoves(
  moves: GameMove[],
  state: GameState,
): {boardMoves: GameMove[]; drawOutcomes: DrawOutcome[]} {
  const boardMoves: GameMove[] = [];
  const placeByTile = new Map<string, GameMove[]>();

  for (const move of moves) {
    if (move.type === 'place') {
      const list = placeByTile.get(move.tileName);
      if (list) list.push(move);
      else placeByTile.set(move.tileName, [move]);
    } else {
      boardMoves.push(move);
    }
  }

  if (placeByTile.size === 0) return {boardMoves, drawOutcomes: []};

  const bag = state.bags[state.currentPlayer];
  const bagTotal = bag.length;
  if (bagTotal === 0) return {boardMoves, drawOutcomes: []};

  const bagCounts = new Map<string, number>();
  for (const name of bag) {
    bagCounts.set(name, (bagCounts.get(name) ?? 0) + 1);
  }

  const drawOutcomes: DrawOutcome[] = [];
  for (const [tileName, placements] of placeByTile) {
    const count = bagCounts.get(tileName) ?? 0;
    if (count > 0) {
      drawOutcomes.push({tileName, probability: count / bagTotal, placements});
    }
  }

  return {boardMoves, drawOutcomes};
}

// ---------------------------------------------------------------------------
// Apply + search helper with incremental hashing
// ---------------------------------------------------------------------------

function applyAndSearch(
  state: GameState,
  move: GameMove,
  mover: Player,
  depth: number,
  alpha: number,
  beta: number,
  childMaximizing: boolean,
  hash: number,
  stats: {nodes: number},
  allowNull: boolean,
): number | null {
  const undo = makeMove(state, move);
  if (!isLegalAfterMake(state, mover)) {
    unmakeMove(state, undo);
    return null;
  }
  const newHash = hashAfterMove(hash, state, undo);
  const score = minimax(state, depth, alpha, beta, childMaximizing, newHash, stats, allowNull);
  unmakeMove(state, undo);
  return score;
}

// ---------------------------------------------------------------------------
// Evaluate chance node with static placement pruning
// Only fully searches top-N placements per drawn tile (sorted by static eval).
// ---------------------------------------------------------------------------

function evaluateChanceNode(
  drawOutcomes: DrawOutcome[],
  state: GameState,
  mover: Player,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  hash: number,
  stats: {nodes: number},
): number {
  let expectedValue = 0;

  for (const {probability, placements} of drawOutcomes) {
    let bestPlacementScore = maximizing ? -Infinity : Infinity;
    let innerAlpha = alpha;
    let innerBeta = beta;

    // Static pruning: sort placements by tile value, only search top N
    const sorted = placements.length > MAX_PLACEMENTS_PER_TILE
      ? placements
          .map(m => ({move: m, val: m.type === 'place' ? evaluatePlacement(state, m.tileName) : 0}))
          .sort((a, b) => b.val - a.val)
          .slice(0, MAX_PLACEMENTS_PER_TILE)
          .map(x => x.move)
      : placements;

    for (const move of sorted) {
      const score = applyAndSearch(state, move, mover, depth - 1, innerAlpha, innerBeta, !maximizing, hash, stats, true);
      if (score === null) continue;

      if (maximizing) {
        bestPlacementScore = Math.max(bestPlacementScore, score);
        innerAlpha = Math.max(innerAlpha, bestPlacementScore);
      } else {
        bestPlacementScore = Math.min(bestPlacementScore, score);
        innerBeta = Math.min(innerBeta, bestPlacementScore);
      }
      if (innerAlpha >= innerBeta) break;
    }

    expectedValue += probability * bestPlacementScore;
  }

  return expectedValue;
}

// ---------------------------------------------------------------------------
// Minimax with alpha-beta + TT + null-move pruning + LMR + expectimax draws
// ---------------------------------------------------------------------------

function minimax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  hash: number,
  stats: {nodes: number},
  allowNull: boolean,
): number {
  stats.nodes++;

  if (depth <= 0 || state.status !== 'active') {
    return evaluate(state);
  }

  // Transposition table probe
  const ttHit = ttProbe(hash, depth, alpha, beta);
  if (ttHit !== null) return ttHit;

  const mover = state.currentPlayer;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const inCheck = isPlayerInCheck(state, mover);

  // --- Null-move pruning ---
  // Skip if in check, if null-move is disallowed (already in null-move), or at shallow depth
  if (allowNull && !inCheck && depth >= 3) {
    const R = 3;
    state.currentPlayer = opponent;
    const nullHash = hash ^ 0; // turn key XOR handled by passing !maximizing
    // Null move = opponent gets an extra turn. Flip hash turn key manually.
    const nullScore = minimax(state, depth - 1 - R, alpha, beta, !maximizing, nullHash ^ hashNullMoveDelta(), stats, false);
    state.currentPlayer = mover;

    if (maximizing ? nullScore >= beta : nullScore <= alpha) {
      return nullScore;
    }
  }

  const allMoves = generatePseudoLegalMoves(state);

  if (allMoves.length === 0) {
    return maximizing ? -WIN_SCORE : WIN_SCORE;
  }

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const pvEncoded = ttProbeMove(hash);
  const orderedBoardMoves = orderMoves(boardMoves, state, pvEncoded, depth);

  let bestScore = maximizing ? -Infinity : Infinity;
  let bestEncoded = 0;
  let flag: number;
  let hasLegalMove = false;
  const origAlpha = alpha;
  const origBeta = beta;

  for (let i = 0; i < orderedBoardMoves.length; i++) {
    const move = orderedBoardMoves[i];
    const isCap = isCapture(move, state);

    // --- Late Move Reductions ---
    let searchDepth = depth - 1;
    const doLMR = i >= 3 && depth >= 3 && !isCap && !inCheck;
    if (doLMR) {
      searchDepth = depth - 2; // reduced search
    }

    let score = applyAndSearch(state, move, mover, searchDepth, alpha, beta, !maximizing, hash, stats, true);
    if (score === null) continue;

    // Re-search at full depth if LMR found a promising score
    if (doLMR) {
      const promising = maximizing ? score > alpha : score < beta;
      if (promising) {
        score = applyAndSearch(state, move, mover, depth - 1, alpha, beta, !maximizing, hash, stats, true)!;
      }
    }

    hasLegalMove = true;
    const encoded = encodeMove(move);

    if (maximizing) {
      if (score > bestScore) {
        bestScore = score;
        bestEncoded = encoded;
      }
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) {
        if (!isCap) storeKiller(depth, encoded);
        break;
      }
    } else {
      if (score < bestScore) {
        bestScore = score;
        bestEncoded = encoded;
      }
      beta = Math.min(beta, bestScore);
      if (alpha >= beta) {
        if (!isCap) storeKiller(depth, encoded);
        break;
      }
    }
  }

  // Chance nodes — search 3 plies shallower than board moves
  if (drawOutcomes.length > 0 && alpha < beta) {
    const drawDepth = Math.max(depth - 3, 0);
    const drawScore = evaluateChanceNode(drawOutcomes, state, mover, drawDepth, alpha, beta, maximizing, hash, stats);
    hasLegalMove = true;
    if (maximizing) {
      bestScore = Math.max(bestScore, drawScore);
      alpha = Math.max(alpha, bestScore);
    } else {
      bestScore = Math.min(bestScore, drawScore);
      beta = Math.min(beta, bestScore);
    }
  }

  if (!hasLegalMove) return maximizing ? -WIN_SCORE : WIN_SCORE;

  // Determine TT flag from original bounds
  if (bestScore <= origAlpha) {
    flag = TT_UPPER;
  } else if (bestScore >= origBeta) {
    flag = TT_LOWER;
  } else {
    flag = TT_EXACT;
  }

  ttStore(hash, depth, bestScore, flag, bestEncoded);
  return bestScore;
}

// Null-move hash delta: just the turn key. Reuse zobrist internals.
// We import hashState but need the turn key separately. We derive it by
// hashing an empty-ish state toggle. Instead, hardcode the known turn key
// from the PRNG sequence. For correctness, we compute it at init.
let _nullMoveTurnDelta = 0;
{
  // The turn key is deterministic from the PRNG. Recompute it here.
  let seed = 0xDEAD_BEEF;
  function rand32(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  }
  const numTiles = 15; // ALL_TILES.length
  // Skip board keys: 6 * 6 * numTiles * 4 = 2160 rand32 calls
  for (let i = 0; i < 6 * 6 * numTiles * 4; i++) rand32();
  _nullMoveTurnDelta = rand32(); // the turnKey
}

function hashNullMoveDelta(): number {
  return _nullMoveTurnDelta;
}

// ---------------------------------------------------------------------------
// Public API: findBestMove with iterative deepening
// ---------------------------------------------------------------------------

/**
 * Find the best move using iterative deepening minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 * Draw moves are treated as chance nodes (expectimax).
 */
export function findBestMove(state: GameState, maxDepth = 6): MinimaxResult {
  const allMoves = generatePseudoLegalMoves(state);
  if (allMoves.length === 0) return {move: null, score: 0, nodesSearched: 0};

  const maximizing = state.currentPlayer === 'P1';
  const mover = state.currentPlayer;
  const stats = {nodes: 0};
  const rootHash = hashState(state);

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);

  let bestMove: GameMove | null = null;
  let bestScore = maximizing ? -Infinity : Infinity;

  // --- Iterative deepening ---
  for (let d = 1; d <= maxDepth; d++) {
    clearKillers();

    let iterBestMove: GameMove | null = null;
    let iterBestScore = maximizing ? -Infinity : Infinity;
    let alpha = -Infinity;
    let beta = Infinity;

    // Aspiration window (from depth 3+, use prior iteration score)
    if (d >= 3 && bestMove !== null && Number.isFinite(bestScore)) {
      alpha = bestScore - 50;
      beta = bestScore + 50;
    }

    const pvEncoded = ttProbeMove(rootHash);
    const orderedBoardMoves = orderMoves(boardMoves, state, pvEncoded, d);

    for (const move of orderedBoardMoves) {
      const score = applyAndSearch(state, move, mover, d - 1, alpha, beta, !maximizing, rootHash, stats, true);
      if (score === null) continue;

      if (iterBestMove === null) iterBestMove = move;

      if (maximizing) {
        if (score > iterBestScore) {
          iterBestScore = score;
          iterBestMove = move;
        }
        alpha = Math.max(alpha, iterBestScore);
      } else {
        if (score < iterBestScore) {
          iterBestScore = score;
          iterBestMove = move;
        }
        beta = Math.min(beta, iterBestScore);
      }
    }

    // Evaluate draw outcomes
    if (drawOutcomes.length > 0) {
      const drawScore = evaluateChanceNode(drawOutcomes, state, mover, d, alpha, beta, maximizing, rootHash, stats);
      const drawIsBetter = maximizing ? drawScore > iterBestScore : drawScore < iterBestScore;

      if (drawIsBetter) {
        iterBestScore = drawScore;
        // Pick the best placement for a random draw
        const bag = state.bags[state.currentPlayer];
        const randomTile = bag[Math.floor(Math.random() * bag.length)];
        const tilePlacements = drawOutcomes.find(o => o.tileName === randomTile)?.placements ?? [];

        let bestPlacement: GameMove | null = null;
        let bestPlacementScore = maximizing ? -Infinity : Infinity;

        for (const move of tilePlacements) {
          const score = applyAndSearch(state, move, mover, d - 1, -Infinity, Infinity, !maximizing, rootHash, stats, true);
          if (score === null) continue;
          if (maximizing ? score > bestPlacementScore : score < bestPlacementScore) {
            bestPlacementScore = score;
            bestPlacement = move;
          }
        }

        if (bestPlacement) iterBestMove = bestPlacement;
      }
    }

    // Aspiration window fail — re-search with full window
    if ((alpha !== -Infinity || beta !== Infinity) &&
        (iterBestScore <= alpha - 50 || iterBestScore >= beta + 50)) {
      // Fall-through: the result from this iteration is still the best we have,
      // but next iteration will use full window since bestScore will be way off
    }

    if (iterBestMove !== null) {
      bestMove = iterBestMove;
      bestScore = iterBestScore;
    }
  }

  return {move: bestMove, score: bestScore, nodesSearched: stats.nodes};
}

export { ttClear };
