/**
 * Evaluate trained neural network against minimax or random players.
 *
 * Usage:
 *   npx tsx packages/ai/src/neural-eval.ts [options]
 *
 * Options:
 *   --games N         Number of games to play (default: 100)
 *   --opponent TYPE   "minimax" or "random" (default: "minimax")
 *   --depth N         Minimax search depth (default: 2)
 *   --checkpoint DIR  Checkpoint directory (default: "training/checkpoints")
 *   --epoch N|best    Checkpoint epoch (default: 50)
 *   --net-player P    Which side the net plays: "P1", "P2", or "both" (default: "both")
 *   --mode MODE       "policy" (raw network argmax) or "1ply" (one-ply value search)
 *                     (default: "policy")
 */

import {spawn, type ChildProcess} from 'node:child_process';
import {createInterface, type Interface} from 'node:readline';
import {parseArgs} from 'node:util';

import {
  createInitialState, applySetupPlacement, applyMove,
  generateAllMoves, getSetupTargets, enumerateDrawOutcomes,
  type GameState, type GameMove,
} from '@the-duke/engine';

import {findBestMove, ttClear} from './minimax.js';
import {pickRandomMove} from './random.js';
import {
  encodeState, encodeStateFresh, decodeMoveIndex, encodeLegalMask,
  STATE_TENSOR_SIZE, BOARD_TENSOR_SIZE, BAG_FEATURE_SIZE, TOTAL_ACTIONS,
} from './encoding.js';
import {evaluate} from './evaluate.js';

// ── Python bridge ──

type ActionResp = {action: number; value: number};
type ValueBatchResp = {values: number[]};
type ErrResp = {error: string};

class NeuralPlayer {
  private proc: ChildProcess;
  private rl: Interface;
  private pending: ((resp: ActionResp | ValueBatchResp) => void) | null = null;

  constructor(checkpointDir: string, epoch: string) {
    this.proc = spawn('uv', [
      'run', '-m', 'training.supervised.infer_server',
      checkpointDir, epoch,
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
        const parsed = JSON.parse(line) as ActionResp | ValueBatchResp | ErrResp;
        if ('error' in parsed) {
          throw new Error(`Inference server error: ${parsed.error}`);
        }
        cb(parsed);
      }
    });
  }

  /** Raw policy argmax: ask the network for a single action. */
  async pickByPolicy(state: GameState, legalMoves: GameMove[]): Promise<{move: GameMove | null; value: number}> {
    // The policy head's action space is (tile, square) placements, so expand
    // draws into per-tile placement candidates for the legal mask. The tile the
    // net "picks" is discarded below — only its square preference is honored,
    // with the tile drawn randomly — so the net cannot peek into the bag.
    const boardMoves = legalMoves.filter(m => m.type !== 'draw');
    const placeCandidates: GameMove[] = [];
    for (const o of enumerateDrawOutcomes(state)) {
      for (const p of o.placements) placeCandidates.push(p);
    }
    const candidates = [...boardMoves, ...placeCandidates];

    const encoded = encodeStateFresh(state);
    const boards = Array.from(encoded.subarray(0, BOARD_TENSOR_SIZE));
    const bags = Array.from(encoded.subarray(BOARD_TENSOR_SIZE, STATE_TENSOR_SIZE));

    const maskBuf = new Float32Array(TOTAL_ACTIONS);
    encodeLegalMask(candidates, maskBuf, 0);
    const legal_mask = Array.from(maskBuf);

    const req = JSON.stringify({type: 'action', boards, bags, legal_mask});
    this.proc.stdin!.write(req + '\n');
    const resp = await new Promise<ActionResp | ValueBatchResp>((resolve) => { this.pending = resolve; }) as ActionResp;

    const chosen = decodeMoveIndex(resp.action, candidates);
    if (chosen && chosen.type === 'place') {
      const bag = state.bags[state.currentPlayer];
      const drawn = bag[Math.floor(Math.random() * bag.length)];
      return {move: {type: 'place', tileName: drawn, position: chosen.position}, value: resp.value};
    }
    return {move: chosen, value: resp.value};
  }

  /** Batch value evaluation: ask the network for V(s_i) for each state. */
  async evaluateBatch(states: GameState[]): Promise<number[]> {
    const n = states.length;
    if (n === 0) return [];

    const boardsArr = new Float32Array(n * BOARD_TENSOR_SIZE);
    const bagsArr = new Float32Array(n * BAG_FEATURE_SIZE);

    const scratch = new Float32Array(STATE_TENSOR_SIZE);
    for (let i = 0; i < n; i++) {
      encodeState(states[i], scratch, 0);
      boardsArr.set(scratch.subarray(0, BOARD_TENSOR_SIZE), i * BOARD_TENSOR_SIZE);
      bagsArr.set(scratch.subarray(BOARD_TENSOR_SIZE, STATE_TENSOR_SIZE), i * BAG_FEATURE_SIZE);
    }

    const req = JSON.stringify({
      type: 'value_batch',
      n,
      boards: Array.from(boardsArr),
      bags: Array.from(bagsArr),
    });
    this.proc.stdin!.write(req + '\n');
    const resp = await new Promise<ActionResp | ValueBatchResp>((resolve) => { this.pending = resolve; }) as ValueBatchResp;
    return resp.values;
  }

  close() {
    this.proc.stdin!.end();
    this.proc.kill();
  }
}

// ── 1-ply lookahead with expectimax draws ──
//
// For each candidate "decision" the player can make, ask the network for V of
// the resulting state. After our move, the resulting state's currentPlayer is
// the opponent, so the value head reports from the opponent's perspective —
// we minimize.
//
// Crucially, "place" moves are NOT free choices: in real play the player
// commits to drawing and the engine reveals a random tile from the bag. To
// avoid the player cheating by hand-picking a killer tile (e.g. forcing mate
// with a specific Assassin draw), placements are treated as a chance node:
//
//   value(draw) = Σ over tile_t in bag  p(t) · min_pos V(place(t, pos))
//
// We then compare the best deterministic board move against value(draw) and
// pick whichever is better. If "draw" wins, we sample a random tile from the
// bag and play its best placement — matching how minimax handles the same
// decision when generating training data.
//
// Terminal-state shortcuts:
//   - A board move that ends in our favor → take it (deterministic).
//   - A placement that ends in our favor → only contributes value=-1 inside
//     the chance node, never forces an immediate pick.
//   - Any move that ends in our loss → never picked.

async function pickByOnePly(
  neural: NeuralPlayer,
  state: GameState,
  legalMoves: GameMove[],
): Promise<{move: GameMove | null; value: number}> {
  if (legalMoves.length === 0) return {move: null, value: 0};

  const us = state.currentPlayer;
  const winStatus = us === 'P1' ? 'P1_wins' : 'P2_wins';
  const lossStatus = us === 'P1' ? 'P2_wins' : 'P1_wins';

  // States needing a network value; each candidate references a slot here, or
  // carries a forced value (terminal shortcut) and slot = -1.
  const evalStates: GameState[] = [];
  interface Cand { move: GameMove; forced: number | null; slot: number; }

  // 1. Board moves. An immediate win is taken deterministically; an immediate
  //    loss is discarded; the rest are queued for value evaluation.
  const boardCands: Cand[] = [];
  for (const m of legalMoves) {
    if (m.type === 'draw') continue;
    const s = applyMove(state, m);
    if (s.status === winStatus) return {move: m, value: -1};
    if (s.status === lossStatus) continue;
    boardCands.push({move: m, forced: null, slot: evalStates.length});
    evalStates.push(s);
  }

  // 2. Draw decision, modeled as a chance node (expectimax). For each drawable
  //    tile we look at the resolved placements from the engine and later take
  //    the best square. A placement that wins outright gets value -1 without
  //    consulting the net (it never saw such terminal states clearly).
  const drawOutcomes = enumerateDrawOutcomes(state);
  const placeByTile = new Map<string, Cand[]>();
  for (const o of drawOutcomes) {
    const list: Cand[] = [];
    for (const p of o.placements) {
      const s = applyMove(state, p);
      if (s.status === lossStatus) continue;
      if (s.status === winStatus) { list.push({move: p, forced: -1, slot: -1}); continue; }
      list.push({move: p, forced: null, slot: evalStates.length});
      evalStates.push(s);
    }
    if (list.length > 0) placeByTile.set(o.tileName, list);
  }

  if (boardCands.length === 0 && placeByTile.size === 0) {
    // Every option loses immediately. Resign-grade fallback.
    return {move: legalMoves[0], value: 1};
  }

  // 3. One batched value query for all non-forced states.
  const netValues = evalStates.length > 0 ? await neural.evaluateBatch(evalStates) : [];
  const valOf = (c: Cand) => (c.forced !== null ? c.forced : netValues[c.slot]);

  // 4. Best board move (minimize the opponent's value).
  let bestBoardMove: GameMove | null = null;
  let bestBoardValue = Infinity;
  for (const c of boardCands) {
    const v = valOf(c);
    if (v < bestBoardValue) { bestBoardValue = v; bestBoardMove = c.move; }
  }

  // 5. Best placement per tile, then expected value of drawing:
  //    E_tile[ min_square V ]  — matches minimax's chance-node model.
  const bestPerTile = new Map<string, {value: number; move: GameMove}>();
  for (const [tile, list] of placeByTile) {
    let best = Infinity;
    let bestMove = list[0].move;
    for (const c of list) {
      const v = valOf(c);
      if (v < best) { best = v; bestMove = c.move; }
    }
    bestPerTile.set(tile, {value: best, move: bestMove});
  }

  const bag = state.bags[us];
  let drawExpected = Infinity;
  if (bag.length > 0 && bestPerTile.size > 0) {
    const counts = new Map<string, number>();
    for (const t of bag) counts.set(t, (counts.get(t) ?? 0) + 1);
    drawExpected = 0;
    for (const [tile, count] of counts) {
      const p = count / bag.length;
      const best = bestPerTile.get(tile);
      // Squares are tile-independent, so every bag tile has placements unless
      // none are legal at all (handled by the empty-map guard above).
      drawExpected += best ? p * best.value : p; // 1.0 ≈ loss if somehow none
    }
  }

  // 6. Choose the better of "best board move" vs "draw". If drawing wins, we
  //    sample a tile (blind) and play its best square — genuine random draw.
  if (drawExpected < bestBoardValue && bestPerTile.size > 0) {
    const drawn = bag[Math.floor(Math.random() * bag.length)];
    const choice = bestPerTile.get(drawn);
    if (choice) return {move: choice.move, value: drawExpected};
  }
  if (bestBoardMove) return {move: bestBoardMove, value: bestBoardValue};

  // Only draws were available.
  const drawn = bag[Math.floor(Math.random() * bag.length)];
  const choice = bestPerTile.get(drawn);
  if (choice) return {move: choice.move, value: drawExpected};
  return {move: legalMoves[0], value: 1};
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
type Mode = 'policy' | '1ply';

async function playGame(
  neural: NeuralPlayer,
  netSide: Player,
  opponent: 'minimax' | 'random',
  depth: number,
  mode: Mode,
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
      const result = mode === '1ply'
        ? await pickByOnePly(neural, state, moves)
        : await neural.pickByPolicy(state, moves);
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
      mode: {type: 'string', default: 'policy'},
    },
  });

  const numGames = parseInt(values.games!);
  const opponent = values.opponent as 'minimax' | 'random';
  const depth = parseInt(values.depth!);
  const checkpointDir = values.checkpoint!;
  const epoch = values.epoch!;
  const netPlayerArg = values['net-player'] as 'P1' | 'P2' | 'both';
  const mode = values.mode as Mode;

  if (mode !== 'policy' && mode !== '1ply') {
    console.error(`Invalid --mode "${mode}". Must be "policy" or "1ply".`);
    process.exit(1);
  }

  console.log(`Neural eval: ${numGames} games, net (${mode}) vs ${opponent} (depth ${depth})`);
  console.log(`Checkpoint: ${checkpointDir} epoch ${epoch}`);
  console.log(`Net plays: ${netPlayerArg}\n`);

  const neural = new NeuralPlayer(checkpointDir, epoch);

  // Wait for Python to signal readiness.
  await new Promise<void>((resolve) => setTimeout(resolve, 3000));

  const sides: Player[] = netPlayerArg === 'both' ? ['P1', 'P2'] : [netPlayerArg];
  const gamesPerSide = netPlayerArg === 'both' ? Math.ceil(numGames / 2) : numGames;

  let netWins = 0, netLosses = 0, draws = 0, totalGames = 0;

  for (const side of sides) {
    console.log(`--- Net as ${side} ---`);
    for (let g = 0; g < gamesPerSide; g++) {
      const {outcome, turns} = await playGame(neural, side, opponent, depth, mode);

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
