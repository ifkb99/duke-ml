import type { GameMove, GameState, SerializedGameState } from '@the-duke/engine';
import { generateAllMoves, applyMove, serialize } from '@the-duke/engine';
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
  movesAtStateDict: Map<SerializedGameState, GameMove[]>,
): number {
  stats.nodes++;

  if (depth === 0 || state.status !== 'active') {
    return evaluate(state);
  }

  const serializedState = serialize(state);
  let moves = movesAtStateDict.get(serializedState);
  if (moves === undefined) {
    moves = generateAllMoves(state);
    movesAtStateDict.set(serializedState, moves);
  }

  if (moves.length === 0) {
    // No legal moves = loss for current player
    return maximizing ? -99_999 : 99_999;
  }

  if (maximizing) {
    let value = -Infinity;
    moves.forEach((move) => {
      const child = applyMove(state, move);
      value = Math.max(value, minimax(child, depth - 1, alpha, beta, false, stats, movesAtStateDict));
      alpha = Math.max(alpha, value);
      if (alpha >= beta) return value;
    });
    return value;
  } else {
    let value = Infinity;
    moves.forEach((move) => {
      const child = applyMove(state, move);
      value = Math.min(value, minimax(child, depth - 1, alpha, beta, true, stats, movesAtStateDict));
      beta = Math.min(beta, value);
      if (alpha >= beta) return value;
    });
    return value;
  }
}

/**
 * Find the best move using minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 */
export function findBestMove(state: GameState, depth = 4): MinimaxResult {
  const movesAtStateDict = new Map<SerializedGameState, GameMove[]>();
  const moves = generateAllMoves(state);
  movesAtStateDict.set(serialize(state), moves);
  if (moves.length === 0) return { move: null, score: 0, nodesSearched: 0 };

  const maximizing = state.currentPlayer === 'P1';
  const stats = { nodes: 0 };

  let bestMove: GameMove = moves[0];
  let bestScore = maximizing ? -Infinity : Infinity;

  moves.forEach((move) => {
    const child = applyMove(state, move);
    const score = minimax(child, depth - 1, -Infinity, Infinity, !maximizing, stats, movesAtStateDict);

    if (maximizing ? score > bestScore : score < bestScore) {
      bestScore = score;
      bestMove = move;
    }
  });

  return { move: bestMove, score: bestScore, nodesSearched: stats.nodes };
}
