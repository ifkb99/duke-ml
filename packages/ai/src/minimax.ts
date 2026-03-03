import type {GameMove, GameState} from '@the-duke/engine';
import {generateAllMoves, applyMove} from '@the-duke/engine';
import {evaluate} from './evaluate.js';
import {TT_EXACT, TT_LOWER, TT_UPPER, hashState, ttStore, ttProbe} from './utils/zobrist.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
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

function evaluateChanceNode(
  drawOutcomes: DrawOutcome[],
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  stats: {nodes: number},
): number {
  let expectedValue = 0;

  for (const {probability, placements} of drawOutcomes) {
    let bestPlacementScore = maximizing ? -Infinity : Infinity;

    // Player chooses where to place — this is a deterministic max/min decision.
    // Use local copies of alpha/beta for pruning within placements.
    let innerAlpha = alpha;
    let innerBeta = beta;

    for (const move of placements) {
      const child = applyMove(state, move);
      const score = minimax(child, depth - 1, innerAlpha, innerBeta, !maximizing, stats);

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

  if (depth === 0 || state.status !== 'active') {
    return evaluate(state);
  }

  const hash = hashState(state);

  // Transposition table lookup
  const ttHit = ttProbe(hash, depth, alpha, beta);
  if (ttHit !== null) return ttHit;

  const allMoves = generateAllMoves(state);

  if (allMoves.length === 0) {
    // No legal moves = loss for current player
    return maximizing ? -99_999 : 99_999;
  }

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const orderedBoardMoves = orderMoves(boardMoves, state);

  let bestScore: number;
  let flag: number;

  if (maximizing) {
    bestScore = -Infinity;

    // Evaluate board moves first (captures first → tight bounds early)
    for (const move of orderedBoardMoves) {
      const child = applyMove(state, move);
      bestScore = Math.max(bestScore, minimax(child, depth - 1, alpha, beta, false, stats));
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) break;
    }

    // Evaluate draw as a single chance node competing with board moves
    if (drawOutcomes.length > 0 && alpha < beta) {
      const drawScore = evaluateChanceNode(drawOutcomes, state, depth, alpha, beta, true, stats);
      bestScore = Math.max(bestScore, drawScore);
      alpha = Math.max(alpha, bestScore);
    }

    flag = bestScore >= beta ? TT_LOWER : (bestScore <= alpha ? TT_UPPER : TT_EXACT);
  } else {
    bestScore = Infinity;

    for (const move of orderedBoardMoves) {
      const child = applyMove(state, move);
      bestScore = Math.min(bestScore, minimax(child, depth - 1, alpha, beta, true, stats));
      beta = Math.min(beta, bestScore);
      if (alpha >= beta) break;
    }

    if (drawOutcomes.length > 0 && alpha < beta) {
      const drawScore = evaluateChanceNode(drawOutcomes, state, depth, alpha, beta, false, stats);
      bestScore = Math.min(bestScore, drawScore);
      beta = Math.min(beta, bestScore);
    }

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
  const allMoves = generateAllMoves(state);
  if (allMoves.length === 0) return {move: null, score: 0, nodesSearched: 0};

  const maximizing = state.currentPlayer === 'P1';
  const stats = {nodes: 0};

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const orderedBoardMoves = orderMoves(boardMoves, state);

  let bestMove: GameMove = allMoves[0];
  let bestScore = maximizing ? -Infinity : Infinity;
  let alpha = -Infinity;
  let beta = Infinity;

  // Evaluate board moves with standard alpha-beta
  for (const move of orderedBoardMoves) {
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

  // Evaluate draw as a single chance node
  if (drawOutcomes.length > 0) {
    const drawScore = evaluateChanceNode(drawOutcomes, state, depth, alpha, beta, maximizing, stats);
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
        const child = applyMove(state, move);
        const score = minimax(child, depth - 1, -Infinity, Infinity, !maximizing, stats);

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
