/**
 * Evaluate trained neural network against minimax or random players.
 *
 * Usage:
 *   npx tsx packages/ai/src/neural-eval.ts [options]
 *
 * Options:
 *   --games N         Number of games to play (default: 20)
 *   --opponent TYPE   "minimax" or "random" (default: "minimax")
 *   --depth N         Minimax search depth (default: 2)
 *   --checkpoint DIR  Checkpoint directory (default: "training/checkpoints")
 *   --epoch N         Checkpoint epoch (default: 50)
 *   --net-player P    Which side the net plays: "P1", "P2", or "both" (default: "both")
 */

import {spawn, type ChildProcess} from 'node:child_process';
import {createInterface, type Interface} from 'node:readline';
import {parseArgs} from 'node:util';

import {
  createInitialState, applySetupPlacement, applyMove,
  generateAllMoves, getSetupTargets,
  type GameState, type GameMove,
} from '@the-duke/engine';

import {findBestMove, ttClear} from './minimax.js';
import {pickRandomMove} from './random.js';
import {
  encodeStateFresh, decodeMoveIndex, encodeLegalMask,
  STATE_TENSOR_SIZE, BOARD_TENSOR_SIZE, TOTAL_ACTIONS,
} from './encoding.js';
import {evaluate} from './evaluate.js';

// ── Python bridge ──

class NeuralPlayer {
  private proc: ChildProcess;
  private rl: Interface;
  private pending: ((resp: {action: number; value: number}) => void) | null = null;

  constructor(checkpointDir: string, epoch: number) {
    this.proc = spawn('uv', [
      'run', '-m', 'training.supervised.infer_server',
      checkpointDir, String(epoch),
    ], {
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'inherit'],
    });

    this.rl = createInterface({input: this.proc.stdout!});
    this.rl.on('line', (line: string) => {
      if (line === 'READY') return;
      if (this.pending) {
        const cb = this.pending;
        this.pending = null;
        cb(JSON.parse(line));
      }
    });
  }

  async pickMove(state: GameState, legalMoves: GameMove[]): Promise<{move: GameMove | null; value: number}> {
    const encoded = encodeStateFresh(state);
    const boards = Array.from(encoded.subarray(0, BOARD_TENSOR_SIZE));
    const bags = Array.from(encoded.subarray(BOARD_TENSOR_SIZE, STATE_TENSOR_SIZE));

    const maskBuf = new Float32Array(TOTAL_ACTIONS);
    encodeLegalMask(legalMoves, maskBuf, 0);
    const legal_mask = Array.from(maskBuf);

    const req = JSON.stringify({boards, bags, legal_mask});
    this.proc.stdin!.write(req + '\n');

    const resp = await new Promise<{action: number; value: number}>((resolve) => {
      this.pending = resolve;
    });

    const move = decodeMoveIndex(resp.action, legalMoves);
    return {move, value: resp.value};
  }

  close() {
    this.proc.stdin!.end();
    this.proc.kill();
  }
}

// ── Setup phase (reuse minimax setup from selfplay.ts) ──

const SETUP_MAXIMIZING: Record<string, boolean> = {
  p1_duke: true, p1_footman1: true, p1_footman2: true,
  p2_duke: false, p2_footman1: false, p2_footman2: false,
};

function setupMinimax(state: GameState): {score: number; coord: {row: number; col: number} | null} {
  if (state.status === 'active') return {score: evaluate(state), coord: null};
  const targets = getSetupTargets(state);
  if (targets.length === 0) return {score: evaluate(state), coord: null};

  const maximizing = SETUP_MAXIMIZING[state.setupPhase!] ?? true;
  let bestScore = maximizing ? -Infinity : Infinity;
  let bestCoord = targets[0];

  for (const coord of targets) {
    const next = applySetupPlacement(state, coord);
    const {score} = setupMinimax(next);
    if (maximizing ? score > bestScore : score < bestScore) {
      bestScore = score;
      bestCoord = coord;
    }
  }
  return {score: bestScore, coord: bestCoord};
}

function setupGame(): GameState {
  let state = createInitialState();
  while (state.status === 'setup') {
    const {coord} = setupMinimax(state);
    if (!coord) break;
    state = applySetupPlacement(state, coord);
  }
  return state;
}

// ── Game loop ──

type Player = 'P1' | 'P2';

async function playGame(
  neural: NeuralPlayer,
  netSide: Player,
  opponent: 'minimax' | 'random',
  depth: number,
): Promise<{outcome: number; turns: number}> {
  ttClear();
  let state = setupGame();
  let turns = 0;
  const maxTurns = 200;

  while (state.status === 'active' && turns < maxTurns) {
    const moves = generateAllMoves(state);
    if (moves.length === 0) break;

    let chosenMove: GameMove | null = null;

    if (state.currentPlayer === netSide) {
      const result = await neural.pickMove(state, moves);
      chosenMove = result.move;
    } else if (opponent === 'minimax') {
      const result = findBestMove(state, depth);
      chosenMove = result.move;
    } else {
      chosenMove = pickRandomMove(state);
    }

    if (!chosenMove) break;
    state = applyMove(state, chosenMove);
    turns++;
  }

  let outcome = 0;
  if (state.status === 'P1_wins') outcome = 1;
  else if (state.status === 'P2_wins') outcome = -1;
  return {outcome, turns};
}

// ── Main ──

async function main() {
  const {values} = parseArgs({
    options: {
      games: {type: 'string', default: '100'},
      opponent: {type: 'string', default: 'minimax'},
      depth: {type: 'string', default: '2'},
      checkpoint: {type: 'string', default: 'training/checkpoints'},
      epoch: {type: 'string', default: '50'},
      'net-player': {type: 'string', default: 'both'},
    },
  });

  const numGames = parseInt(values.games!);
  const opponent = values.opponent as 'minimax' | 'random';
  const depth = parseInt(values.depth!);
  const checkpointDir = values.checkpoint!;
  const epoch = parseInt(values.epoch!);
  const netPlayerArg = values['net-player'] as 'P1' | 'P2' | 'both';

  console.log(`Neural eval: ${numGames} games, net vs ${opponent} (depth ${depth})`);
  console.log(`Checkpoint: ${checkpointDir} epoch ${epoch}`);
  console.log(`Net plays: ${netPlayerArg}\n`);

  const neural = new NeuralPlayer(checkpointDir, epoch);

  // Wait for Python to signal readiness
  await new Promise<void>((resolve) => {
    const check = setInterval(() => {
      // The READY signal is consumed by the readline handler
      // Give the process a moment to start
    }, 100);
    setTimeout(() => { clearInterval(check); resolve(); }, 3000);
  });

  const sides: Player[] = netPlayerArg === 'both' ? ['P1', 'P2'] : [netPlayerArg];
  const gamesPerSide = netPlayerArg === 'both' ? Math.ceil(numGames / 2) : numGames;

  let netWins = 0, netLosses = 0, draws = 0, totalGames = 0;

  for (const side of sides) {
    console.log(`--- Net as ${side} ---`);
    for (let g = 0; g < gamesPerSide; g++) {
      const {outcome, turns} = await playGame(neural, side, opponent, depth);

      const netSign = side === 'P1' ? 1 : -1;
      const netResult = outcome * netSign;

      if (netResult > 0) netWins++;
      else if (netResult < 0) netLosses++;
      else draws++;

      totalGames++;
      const tag = netResult > 0 ? 'WIN' : netResult < 0 ? 'LOSS' : 'DRAW';
      console.log(`  Game ${totalGames}: ${tag} (${turns} turns)`);
    }
  }

  console.log(`\n=== Results ===`);
  console.log(`Net wins: ${netWins}/${totalGames} (${(100 * netWins / totalGames).toFixed(1)}%)`);
  console.log(`Net losses: ${netLosses}/${totalGames} (${(100 * netLosses / totalGames).toFixed(1)}%)`);
  console.log(`Draws: ${draws}/${totalGames} (${(100 * draws / totalGames).toFixed(1)}%)`);

  neural.close();
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
