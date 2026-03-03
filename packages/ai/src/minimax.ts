import type { GameMove, GameState } from '@the-duke/engine';
import { generateAllMoves, applyMove } from '@the-duke/engine';
import { evaluate } from './evaluate.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
}

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

  const moves = generateAllMoves(state);

  if (moves.length === 0) {
    // No legal moves = loss for current player
    return maximizing ? -99_999 : 99_999;
  }

  if (maximizing) {
    let value = -Infinity;
    moves.forEach((move) => {
      const child = applyMove(state, move);
      value = Math.max(value, minimax(child, depth - 1, alpha, beta, false, stats));
      alpha = Math.max(alpha, value);
      if (alpha >= beta) return;
    });
    return value;
  } else {
    let value = Infinity;
    moves.forEach((move) => {
      const child = applyMove(state, move);
      value = Math.min(value, minimax(child, depth - 1, alpha, beta, true, stats));
      beta = Math.min(beta, value);
      if (alpha >= beta) return;
    });
    return value;
  }
}

/**
 * Find the best move using minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 */
export function findBestMove(state: GameState, depth = 6): MinimaxResult {
  // TODO: ensure that draw move "reward" is sum(bag_piece_scores) / n_pieces_in_bag
  const moves = generateAllMoves(state);
  if (moves.length === 0) return { move: null, score: 0, nodesSearched: 0 };

  const maximizing = state.currentPlayer === 'P1';
  const stats = { nodes: 0 };

  let bestMove: GameMove = moves[0];
  let bestScore = maximizing ? -Infinity : Infinity;

  moves.forEach((move) => {
    const child = applyMove(state, move);
    const score = minimax(child, depth - 1, -Infinity, Infinity, !maximizing, stats);

    if (maximizing ? score > bestScore : score < bestScore) {
      bestScore = score;
      bestMove = move;
    }
  });

  return { move: bestMove, score: bestScore, nodesSearched: stats.nodes };
}
