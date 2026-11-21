/**
 * Match CLI: play two agents against each other and print a short report.
 *
 *   npx tsx packages/harness/src/match.ts --a minimax:3 --b random --games 20 --seed 1
 *
 * Agent specs: `random`, `minimax:<depth>`.
 * Sides alternate: A moves first (plays P1) in games 1, 3, 5, ...
 *
 * --seed makes a run reproducible. It seeds the setup placements, the random
 * agents, and Math.random for the duration of the match (the engine resolves
 * tile draws and findBestMove picks the drawn tile with Math.random).
 */
import { pathToFileURL } from 'node:url';
import type { GameMove, GameState } from '@the-duke/engine';
import { RandomAgent, makeRng, runMatch, type Agent, type MatchStats, type MoveResult } from './runner';
import { MinimaxAgent } from './minimax-agent';

export const USAGE = `Usage: npx tsx packages/harness/src/match.ts --a <spec> --b <spec> [--games N] [--seed S]

  --a <spec>     agent A (required)
  --b <spec>     agent B (required)
  --games N      number of games, positive integer (default 20)
  --seed S       integer seed (default 1)
  --help         show this message

Agent specs:
  random             uniform random legal move
  minimax:<depth>    findBestMove from @the-duke/ai at the given depth (1-12)

A moves first in odd-numbered games, B in even-numbered games.`;

export class UsageError extends Error {}

// ---------------------------------------------------------------------------
// Argument parsing

export type AgentSpec = { kind: 'random' } | { kind: 'minimax'; depth: number };

export interface MatchArgs {
  a: AgentSpec;
  b: AgentSpec;
  games: number;
  seed: number;
}

const MAX_DEPTH = 12;

export function parseAgentSpec(spec: string): AgentSpec {
  if (spec === 'random') return { kind: 'random' };
  const m = /^minimax:(\d+)$/.exec(spec);
  if (m) {
    const depth = Number(m[1]);
    if (depth >= 1 && depth <= MAX_DEPTH) return { kind: 'minimax', depth };
    throw new UsageError(`minimax depth must be between 1 and ${MAX_DEPTH}, got ${m[1]}`);
  }
  throw new UsageError(`unknown agent spec '${spec}'`);
}

function parseInteger(flag: string, value: string, min: number): number {
  if (!/^-?\d+$/.test(value)) throw new UsageError(`${flag} must be an integer, got '${value}'`);
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n < min) throw new UsageError(`${flag} must be an integer >= ${min}, got '${value}'`);
  return n;
}

/** Parse CLI arguments. Returns 'help' for --help; throws UsageError on bad input. */
export function parseArgs(argv: string[]): MatchArgs | 'help' {
  const raw: Record<string, string> = {};
  const known = new Set(['--a', '--b', '--games', '--seed']);

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') return 'help';
    const eq = arg.indexOf('=');
    const flag = eq >= 0 ? arg.slice(0, eq) : arg;
    if (!known.has(flag)) throw new UsageError(`unknown argument '${arg}'`);
    let value: string | undefined;
    if (eq >= 0) value = arg.slice(eq + 1);
    else {
      value = argv[i + 1];
      if (value === undefined || value.startsWith('--')) throw new UsageError(`${flag} needs a value`);
      i++;
    }
    if (flag in raw) throw new UsageError(`${flag} given more than once`);
    raw[flag] = value;
  }

  if (raw['--a'] === undefined) throw new UsageError('--a is required');
  if (raw['--b'] === undefined) throw new UsageError('--b is required');

  return {
    a: parseAgentSpec(raw['--a']),
    b: parseAgentSpec(raw['--b']),
    games: raw['--games'] === undefined ? 20 : parseInteger('--games', raw['--games'], 1),
    seed: raw['--seed'] === undefined ? 1 : parseInteger('--seed', raw['--seed'], 0),
  };
}

// ---------------------------------------------------------------------------
// Running a match

/** Accumulated move timing for one side across all games. */
export interface MoveTiming {
  moves: number;
  totalMs: number;
}

class TimedAgent implements Agent {
  readonly name: string;
  readonly getSetupPlacement?: Agent['getSetupPlacement'];

  constructor(private inner: Agent, private timing: MoveTiming) {
    this.name = inner.name;
    if (inner.getSetupPlacement) this.getSetupPlacement = inner.getSetupPlacement.bind(inner);
  }

  async getMove(state: GameState, legalMoves: GameMove[]): Promise<MoveResult> {
    const t0 = performance.now();
    const result = await this.inner.getMove(state, legalMoves);
    this.timing.totalMs += performance.now() - t0;
    this.timing.moves++;
    return result;
  }
}

/** Build a fresh-agent factory for a spec. Random agents get a new seed each game. */
function agentFactory(spec: AgentSpec, seed: number): () => Agent {
  if (spec.kind === 'minimax') return () => new MinimaxAgent(spec.depth);
  let game = 0;
  return () => new RandomAgent(seed + 7919 * game++);
}

export interface MatchReport {
  args: MatchArgs;
  stats: MatchStats;
  timing: { a: MoveTiming; b: MoveTiming };
}

/**
 * Run a match from parsed args. Math.random is replaced by a seeded RNG for
 * the duration and restored afterwards.
 */
export async function playMatch(args: MatchArgs): Promise<MatchReport> {
  const timing = { a: { moves: 0, totalMs: 0 }, b: { moves: 0, totalMs: 0 } };
  const makeA = agentFactory(args.a, args.seed * 1_000_003 + 1);
  const makeB = agentFactory(args.b, args.seed * 1_000_003 + 2);

  const savedRandom = Math.random;
  Math.random = makeRng(args.seed * 2_654_435_761 + 12345);
  try {
    const stats = await runMatch(
      () => new TimedAgent(makeA(), timing.a),
      () => new TimedAgent(makeB(), timing.b),
      args.games,
      { label: 'match', seed: args.seed * 100_000 },
    );
    return { args, stats, timing };
  } finally {
    Math.random = savedRandom;
  }
}

function fmtMs(t: MoveTiming): string {
  if (t.moves === 0) return 'n/a (0 moves)';
  const avg = t.totalMs / t.moves;
  const s = avg >= 100 ? avg.toFixed(0) : avg >= 1 ? avg.toFixed(1) : avg.toFixed(3);
  return `${s} ms (${t.moves} moves)`;
}

export function formatReport(r: MatchReport): string {
  const { stats, timing, args } = r;
  const [lo, hi] = stats.ci95;
  return [
    `A = ${stats.a}, B = ${stats.b}: ${stats.games} games, seed ${args.seed} (A moves first in odd-numbered games)`,
    `A record:       ${stats.aWins} W / ${stats.bWins} L / ${stats.draws} D (draws = ply cap)`,
    `A score:        ${stats.aWinrate.toFixed(3)}  95% Wilson CI [${lo.toFixed(3)}, ${hi.toFixed(3)}] (draws count half)`,
    `Average plies:  ${stats.meanPlies.toFixed(1)}`,
    `Avg time/move:  A ${fmtMs(timing.a)}, B ${fmtMs(timing.b)}`,
  ].join('\n');
}

// ---------------------------------------------------------------------------

async function main(argv: string[]): Promise<number> {
  let args: MatchArgs | 'help';
  try {
    args = parseArgs(argv);
  } catch (e) {
    if (e instanceof UsageError) {
      console.error(`error: ${e.message}\n\n${USAGE}`);
      return 2;
    }
    throw e;
  }
  if (args === 'help') {
    console.log(USAGE);
    return 0;
  }
  console.log(formatReport(await playMatch(args)));
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exitCode = await main(process.argv.slice(2));
}
