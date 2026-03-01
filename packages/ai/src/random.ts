import type { GameMove, GameState } from '@the-duke/engine';
import { generateAllMoves } from '@the-duke/engine';

export function pickRandomMove(state: GameState): GameMove | null {
  const moves = generateAllMoves(state);
  if (moves.length === 0) return null;
  return moves[Math.floor(Math.random() * moves.length)];
}
