import type {Coord, GameMove, GameState, Player} from '@the-duke/engine';
import {findDukePosition, generatePseudoLegalMoves, isSquareAttackedBy, makeMove, unmakeMove} from '@the-duke/engine';
import {evaluate} from './evaluate.js';
import {TT_EXACT, TT_LOWER, TT_UPPER, hashState, ttStore, ttProbe} from './utils/zobrist.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
}

// ---------------------------------------------------------------------------
// Inline legality check — avoids the double make/unmake of generateAllMoves
// ---------------------------------------------------------------------------

/**
 * After makeMove has been called, check whether the move was legal
 * (i.e. the mover's Duke is not left in check).
 * `mover` is the player who made the move (before makeMove switched currentPlayer).
 */
function isLegalAfterMake(state: GameState, mover: Player): boolean {
  if (state.status !== 'active') return true;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const dukePos = findDukePosition(state, mover);
  return dukePos ? !isSquareAttackedBy(state, dukePos, opponent) : false;
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
// Evaluate a chance node: weighted average over random tile draws,
// with player choosing best placement position for each drawn tile.
// ---------------------------------------------------------------------------

/** Apply makeMove, check legality, call minimax if legal, then unmakeMove. Returns null if illegal. */
function applyAndSearch(
  state: GameState,
  move: GameMove,
  mover: Player,
  depth: number,
  alpha: number,
  beta: number,
  childMaximizing: boolean,
  stats: {nodes: number},
): number | null {
  const undo = makeMove(state, move);
  if (!isLegalAfterMake(state, mover)) {
    unmakeMove(state, undo);
    return null; // illegal move
  }
  const score = minimax(state, depth, alpha, beta, childMaximizing, stats);
  unmakeMove(state, undo);
  return score;
}

function evaluateChanceNode(
  drawOutcomes: DrawOutcome[],
  state: GameState,
  mover: Player,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  stats: {nodes: number},
): number {
  let expectedValue = 0;

  for (const {probability, placements} of drawOutcomes) {
    let bestPlacementScore = maximizing ? -Infinity : Infinity;

    let innerAlpha = alpha;
    let innerBeta = beta;

    for (const move of placements) {
      const score = applyAndSearch(state, move, mover, depth - 1, innerAlpha, innerBeta, !maximizing, stats);
      if (score === null) continue; // illegal

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
// Minimax with alpha-beta pruning + transposition table + expectimax draws
// Uses pseudo-legal moves with inline legality checking to avoid double work.
// ---------------------------------------------------------------------------

function minimax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  stats: {nodes: number},
): number {
  stats.nodes++;

  if (depth <= 0 || state.status !== 'active') {
    return evaluate(state);
  }

  const hash = hashState(state);

  // Transposition table lookup
  const ttHit = ttProbe(hash, depth, alpha, beta);
  if (ttHit !== null) return ttHit;

  // Use pseudo-legal moves — legality is checked inline in applyAndSearch
  const allMoves = generatePseudoLegalMoves(state);

  if (allMoves.length === 0) {
    // No pseudo-legal moves = loss for current player
    return maximizing ? -99_999 : 99_999;
  }

  const mover = state.currentPlayer;
  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const orderedBoardMoves = orderMoves(boardMoves, state);

  let bestScore: number;
  let flag: number;
  let hasLegalMove = false;

  if (maximizing) {
    bestScore = -Infinity;

    for (const move of orderedBoardMoves) {
      const score = applyAndSearch(state, move, mover, depth - 1, alpha, beta, false, stats);
      if (score === null) continue; // illegal
      hasLegalMove = true;
      bestScore = Math.max(bestScore, score);
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) break;
    }

    // Chance nodes search 2 plies shallower than board moves to limit
    // the exponential branching of expectimax (14 tile types × 3-4 placements).
    if (drawOutcomes.length > 0 && alpha < beta) {
      const drawDepth = Math.max(depth - 2, 0);
      const drawScore = evaluateChanceNode(drawOutcomes, state, mover, drawDepth, alpha, beta, true, stats);
      hasLegalMove = true; // placements are always legal
      bestScore = Math.max(bestScore, drawScore);
      alpha = Math.max(alpha, bestScore);
    }

    if (!hasLegalMove) return -99_999; // no legal moves = loss

    flag = bestScore >= beta ? TT_LOWER : (bestScore <= alpha ? TT_UPPER : TT_EXACT);
  } else {
    bestScore = Infinity;

    for (const move of orderedBoardMoves) {
      const score = applyAndSearch(state, move, mover, depth - 1, alpha, beta, true, stats);
      if (score === null) continue; // illegal
      hasLegalMove = true;
      bestScore = Math.min(bestScore, score);
      beta = Math.min(beta, bestScore);
      if (alpha >= beta) break;
    }

    if (drawOutcomes.length > 0 && alpha < beta) {
      const drawDepth = Math.max(depth - 2, 0);
      const drawScore = evaluateChanceNode(drawOutcomes, state, mover, drawDepth, alpha, beta, false, stats);
      hasLegalMove = true; // placements are always legal
      bestScore = Math.min(bestScore, drawScore);
      beta = Math.min(beta, bestScore);
    }

    if (!hasLegalMove) return 99_999; // no legal moves = loss

    flag = bestScore <= alpha ? TT_UPPER : (bestScore >= beta ? TT_LOWER : TT_EXACT);
  }

  ttStore(hash, depth, bestScore, flag);
  return bestScore;
}

/**
 * Find the best move using minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 * Draw moves are treated as chance nodes (expectimax).
 */
export function findBestMove(state: GameState, depth = 6): MinimaxResult {
  // Use pseudo-legal moves — legality checked inline
  const allMoves = generatePseudoLegalMoves(state);
  if (allMoves.length === 0) return {move: null, score: 0, nodesSearched: 0};

  const maximizing = state.currentPlayer === 'P1';
  const mover = state.currentPlayer;
  const stats = {nodes: 0};

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const orderedBoardMoves = orderMoves(boardMoves, state);

  let bestMove: GameMove | null = null;
  let bestScore = maximizing ? -Infinity : Infinity;
  let alpha = -Infinity;
  let beta = Infinity;

  // Evaluate board moves with standard alpha-beta
  for (const move of orderedBoardMoves) {
    const score = applyAndSearch(state, move, mover, depth - 1, alpha, beta, !maximizing, stats);
    if (score === null) continue; // illegal

    if (bestMove === null) bestMove = move;

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

  // Evaluate draw — at the root we always do the full evaluation
  if (drawOutcomes.length > 0) {
    const drawScore = evaluateChanceNode(drawOutcomes, state, mover, depth, alpha, beta, maximizing, stats);
    const drawIsBetter = maximizing ? drawScore > bestScore : drawScore < bestScore;

    if (drawIsBetter) {
      bestScore = drawScore;

      // Drawing is best action — simulate the real game: randomly pick a tile
      // from the bag, then return the best placement for that tile.
      const bag = state.bags[state.currentPlayer];
      const randomTile = bag[Math.floor(Math.random() * bag.length)];
      const tilePlacements = drawOutcomes.find(d => d.tileName === randomTile)?.placements ?? [];

      let bestPlacement: GameMove | null = null;
      let bestPlacementScore = maximizing ? -Infinity : Infinity;

      for (const move of tilePlacements) {
        const score = applyAndSearch(state, move, mover, depth - 1, -Infinity, Infinity, !maximizing, stats);
        if (score === null) continue; // illegal

        if (maximizing ? score > bestPlacementScore : score < bestPlacementScore) {
          bestPlacementScore = score;
          bestPlacement = move;
        }
      }

      if (bestPlacement) bestMove = bestPlacement;
    }
  }

  return {move: bestMove, score: bestScore, nodesSearched: stats.nodes};
}
