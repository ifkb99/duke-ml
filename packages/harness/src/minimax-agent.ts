/**
 * Minimax agent: wraps findBestMove from @the-duke/ai at a fixed depth.
 *
 * The AI's transposition table is module-global, so two MinimaxAgents in one
 * game (or a previous game) would otherwise leak search state into each
 * other. Clearing it before every search (~0.4 ms) makes each move depend
 * only on the position and Math.random, which keeps seeded matches
 * reproducible. It also keeps a quiescence agent and a plain one from reading
 * each other's TT scores.
 *
 * Note: findBestMove resolves a tile draw itself (uniformly from the bag via
 * Math.random, then picks the best square for the revealed tile) and returns
 * a concrete 'place' move rather than the engine's 'draw' move.
 */
import type { GameMove, GameState } from '@the-duke/engine';
import { findBestMove, ttClear, type SearchOptions } from '@the-duke/ai';
import type { Agent, MoveResult } from './runner';

/** Node counts accumulated over every search an agent (or a shared group of agents) ran. */
export interface SearchTally {
  searches: number;
  nodes: number;
  quiescenceNodes: number;
}

export function emptyTally(): SearchTally {
  return { searches: 0, nodes: 0, quiescenceNodes: 0 };
}

export interface MinimaxAgentOptions {
  name?: string;
  /** Passed to findBestMove. Omitted fields use the AI's defaults. */
  search?: SearchOptions;
  /** If given, node counts of each search are added to it. */
  tally?: SearchTally;
}

export class MinimaxAgent implements Agent {
  readonly name: string;
  readonly depth: number;
  private readonly search: SearchOptions;
  private readonly tally?: SearchTally;

  constructor(depth: number, nameOrOptions?: string | MinimaxAgentOptions) {
    if (!Number.isInteger(depth) || depth < 1) {
      throw new Error(`MinimaxAgent depth must be a positive integer, got ${depth}`);
    }
    const opts = typeof nameOrOptions === 'string' ? { name: nameOrOptions } : nameOrOptions ?? {};
    this.depth = depth;
    this.search = opts.search ?? {};
    this.tally = opts.tally;
    const q = this.search.quiescence;
    this.name = opts.name ?? `minimax${q === undefined ? '' : q ? '-q' : '-noq'}-d${depth}`;
  }

  async getMove(state: GameState, legalMoves: GameMove[]): Promise<MoveResult> {
    ttClear();
    const result = findBestMove(state, this.depth, Math.random, this.search);
    if (this.tally) {
      this.tally.searches++;
      this.tally.nodes += result.nodesSearched;
      this.tally.quiescenceNodes += result.quiescenceNodes;
    }
    // findBestMove only returns null when there are no legal moves, which the
    // runner never asks about; fall back to the first legal move regardless.
    return { move: result.move ?? legalMoves[0] };
  }
}
