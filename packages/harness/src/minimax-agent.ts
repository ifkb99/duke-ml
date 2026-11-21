/**
 * Minimax agent: wraps findBestMove from @the-duke/ai at a fixed depth.
 *
 * The AI's transposition table is module-global, so two MinimaxAgents in one
 * game (or a previous game) would otherwise leak search state into each
 * other. Clearing it before every search (~0.4 ms) makes each move depend
 * only on the position and Math.random, which keeps seeded matches
 * reproducible.
 *
 * Note: findBestMove resolves a tile draw itself (uniformly from the bag via
 * Math.random, then picks the best square for the revealed tile) and returns
 * a concrete 'place' move rather than the engine's 'draw' move.
 */
import type { GameMove, GameState } from '@the-duke/engine';
import { findBestMove, ttClear } from '@the-duke/ai';
import type { Agent, MoveResult } from './runner';

export class MinimaxAgent implements Agent {
  readonly name: string;
  readonly depth: number;

  constructor(depth: number, name?: string) {
    if (!Number.isInteger(depth) || depth < 1) {
      throw new Error(`MinimaxAgent depth must be a positive integer, got ${depth}`);
    }
    this.depth = depth;
    this.name = name ?? `minimax-d${depth}`;
  }

  async getMove(state: GameState, legalMoves: GameMove[]): Promise<MoveResult> {
    ttClear();
    const result = findBestMove(state, this.depth);
    // findBestMove only returns null when there are no legal moves, which the
    // runner never asks about; fall back to the first legal move regardless.
    return { move: result.move ?? legalMoves[0] };
  }
}
