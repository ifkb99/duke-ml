/**
 * Agent interface + game runner.
 *
 * All agents — random, linear, minimax, LLM — implement Agent. The runner
 * handles setup placement, the main loop, a ply cap (no draw rule exists in
 * the engine), per-move metrics, and JSONL game logs for replay.
 */
import type { GameMove, GameState, Player } from '@the-duke/engine';
import { createInitialState, applyMove, applySetupPlacement, getStandardSetupTargets, resetInstanceCounter, generateAllMoves, serialize } from '@the-duke/engine';
import { moveToString } from './notation';
import * as fs from 'node:fs';

// ---------------------------------------------------------------------------

export interface MoveResult {
  move: GameMove;
  /** Diagnostics an agent may report (LLM agents fill these in). */
  meta?: {
    illegalAttempts?: number;
    parseFailures?: number;
    fellBackToRandom?: boolean;
    latencyMs?: number;
    rawResponse?: string;
  };
}

export interface Agent {
  readonly name: string;
  /** Choose a move. legalMoves is always non-empty. */
  getMove(state: GameState, legalMoves: GameMove[]): Promise<MoveResult>;
  /** Optional: choose a setup placement. Defaults to random if absent. */
  getSetupPlacement?(state: GameState, targets: { row: number; col: number }[]): Promise<{ row: number; col: number }>;
}

// ---------------------------------------------------------------------------

export function makeRng(seed: number): () => number {
  // mulberry32 — deterministic, good enough for game sampling
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RandomAgent implements Agent {
  readonly name: string;
  private rng: () => number;
  constructor(seed = 42, name = 'random') {
    this.rng = makeRng(seed);
    this.name = name;
  }
  async getMove(_state: GameState, legalMoves: GameMove[]): Promise<MoveResult> {
    return { move: legalMoves[Math.floor(this.rng() * legalMoves.length)] };
  }
}

// ---------------------------------------------------------------------------

export interface GameRecord {
  gameId: string;
  p1: string;
  p2: string;
  winner: 'P1' | 'P2' | 'draw_ply_cap';
  plies: number;
  moves: string[];
  /** Aggregated per-player agent metrics. */
  metrics: Record<Player, { illegalAttempts: number; parseFailures: number; randomFallbacks: number }>;
}

const PLY_CAP = 300;

export async function runGame(
  p1: Agent,
  p2: Agent,
  opts: { gameId?: string; seed?: number; logStream?: fs.WriteStream } = {},
): Promise<GameRecord> {
  resetInstanceCounter(0); // reproducible tile IDs per game
  const rng = makeRng(opts.seed ?? 1);
  let state = createInitialState();

  // --- Setup phase ---
  while (state.status === 'setup') {
    const targets = getStandardSetupTargets(state);
    const agent = state.currentPlayer === 'P1' ? p1 : p2;
    const pick = agent.getSetupPlacement
      ? await agent.getSetupPlacement(state, targets)
      : targets[Math.floor(rng() * targets.length)];
    state = applySetupPlacement(state, pick);
  }

  // --- Main loop ---
  const moves: string[] = [];
  const metrics: GameRecord['metrics'] = {
    P1: { illegalAttempts: 0, parseFailures: 0, randomFallbacks: 0 },
    P2: { illegalAttempts: 0, parseFailures: 0, randomFallbacks: 0 },
  };

  let plies = 0;
  while (state.status === 'active' && plies < PLY_CAP) {
    const legal = generateAllMoves(state);
    // Engine flags checkmate/stalemate inside applyMove, so legal.length > 0 here;
    // guard anyway in case a cap-adjacent edge slips through.
    if (legal.length === 0) break;

    const player = state.currentPlayer;
    const agent = player === 'P1' ? p1 : p2;
    const { move, meta } = await agent.getMove(state, legal);

    if (meta) {
      metrics[player].illegalAttempts += meta.illegalAttempts ?? 0;
      metrics[player].parseFailures += meta.parseFailures ?? 0;
      if (meta.fellBackToRandom) metrics[player].randomFallbacks++;
    }

    moves.push(moveToString(move));
    state = applyMove(state, move);
    plies++;
  }

  const winner: GameRecord['winner'] =
    state.status === 'P1_wins' ? 'P1'
    : state.status === 'P2_wins' ? 'P2'
    : 'draw_ply_cap';

  const record: GameRecord = {
    gameId: opts.gameId ?? `g${Date.now()}`,
    p1: p1.name,
    p2: p2.name,
    winner,
    plies,
    moves,
    metrics,
  };

  opts.logStream?.write(JSON.stringify({ ...record, finalState: serialize(state) }) + '\n');
  return record;
}

// ---------------------------------------------------------------------------

export interface MatchStats {
  a: string;
  b: string;
  games: number;
  aWins: number;
  bWins: number;
  draws: number;
  aWinrate: number;
  /** 95% Wilson score interval on a's winrate (draws counted as half). */
  ci95: [number, number];
  meanPlies: number;
}

export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = successes / n;
  const d = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/**
 * Run a match of n games, alternating colors each game (a plays P1 on even
 * game indices). Color alternation matters: P1 moves first.
 */
export async function runMatch(
  agentA: () => Agent,
  agentB: () => Agent,
  n: number,
  opts: { logPath?: string; label?: string } = {},
): Promise<MatchStats> {
  const logStream = opts.logPath ? fs.createWriteStream(opts.logPath, { flags: 'a' }) : undefined;
  let aWins = 0, bWins = 0, draws = 0, totalPlies = 0;
  const nameA = agentA().name, nameB = agentB().name;

  for (let i = 0; i < n; i++) {
    const aIsP1 = i % 2 === 0;
    const rec = await runGame(
      aIsP1 ? agentA() : agentB(),
      aIsP1 ? agentB() : agentA(),
      { gameId: `${opts.label ?? 'match'}-${i}`, seed: 1000 + i, logStream },
    );
    totalPlies += rec.plies;
    if (rec.winner === 'draw_ply_cap') draws++;
    else if ((rec.winner === 'P1') === aIsP1) aWins++;
    else bWins++;
  }
  logStream?.end();

  const effective = aWins + 0.5 * draws;
  return {
    a: nameA, b: nameB, games: n, aWins, bWins, draws,
    aWinrate: effective / n,
    ci95: wilson(Math.round(effective), n),
    meanPlies: totalPlies / n,
  };
}
