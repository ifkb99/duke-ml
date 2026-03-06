import type {Coord, GameMove, GameState} from '@the-duke/engine';
import {applySetupPlacement, createInitialState, generateAllMoves, applyMove, getSetupTargets} from '@the-duke/engine';
import {evaluate} from './evaluate.js';
import {findBestMove, ttClear} from './minimax.js';
import {encodeState, encodeMoveIndex, STATE_TENSOR_SIZE} from './encoding.js';

// ── Public constants ──

/** Float32 values per training sample: state tensor + action index + value target */
export const SAMPLE_FLOATS = STATE_TENSOR_SIZE + 2; // 1220

// ── Types ──

export interface SelfPlayConfig {
  /** Minimax search depth (default 4) */
  depth?: number;
  /** Maximum game length in turns (default 200) */
  maxTurns?: number;
}

export interface SelfPlayResult {
  /** Flat buffer: numSamples × SAMPLE_FLOATS float32 values */
  data: Float32Array;
  numSamples: number;
  /** +1 P1 win, −1 P2 win, 0 draw/stalemate */
  outcome: number;
}

// ── Setup phase — minimax over placement tree ──
// Branching factor: Duke ~6, Footmen ~4 each → max ~5K leaves, instant to search.

const SETUP_MAXIMIZING: Record<string, boolean> = {
  p1_duke: true, p1_footman1: true, p1_footman2: true,
  p2_duke: false, p2_footman1: false, p2_footman2: false,
};

function setupMinimax(state: GameState): {score: number; coord: Coord | null} {
  if (state.status === 'active') return {score: evaluate(state), coord: null};

  const targets = getSetupTargets(state);
  if (targets.length === 0) return {score: evaluate(state), coord: null};

  const maximizing = SETUP_MAXIMIZING[state.setupPhase!] ?? true;
  let bestScore = maximizing ? -Infinity : Infinity;
  let bestCoord: Coord = targets[0];

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

function setupGameWithMinimax(): GameState {
  let state = createInitialState();
  while (state.status === 'setup') {
    const {coord} = setupMinimax(state);
    if (!coord) break;
    state = applySetupPlacement(state, coord);
  }
  return state;
}

// ── Self-play ──

/**
 * Run one self-play game using minimax for both sides.
 *
 * Returns a flat Float32Array of training samples laid out as:
 *   [state tensor (1218f)] [action index (1f)] [value target (1f)]
 * repeated for each move in the game.
 *
 * Value targets are from the current-player's perspective at each step:
 *   +1 = current player ultimately won, −1 = lost, 0 = draw.
 */
export function selfPlay(config: SelfPlayConfig = {}): SelfPlayResult {
  const depth = config.depth ?? 4;
  const maxTurns = config.maxTurns ?? 200;

  ttClear();

  let state = setupGameWithMinimax();
  const buf = new Float32Array(maxTurns * SAMPLE_FLOATS);
  const playerSigns = new Int8Array(maxTurns);
  let n = 0;

  for (let turn = 0; turn < maxTurns; turn++) {
    if (state.status !== 'active') break;

    const moves = generateAllMoves(state);
    if (moves.length === 0) break;

    const result = findBestMove(state, depth);
    if (!result.move) break;

    const off = n * SAMPLE_FLOATS;
    encodeState(state, buf, off);
    buf[off + STATE_TENSOR_SIZE] = encodeMoveIndex(result.move);
    playerSigns[n] = state.currentPlayer === 'P1' ? 1 : -1;
    n++;

    state = applyMove(state, result.move);
  }

  let outcome = 0;
  if (state.status === 'P1_wins') outcome = 1;
  else if (state.status === 'P2_wins') outcome = -1;

  // Fill value targets from each position's current-player perspective
  for (let i = 0; i < n; i++) {
    buf[i * SAMPLE_FLOATS + STATE_TENSOR_SIZE + 1] = outcome * playerSigns[i];
  }

  return {
    data: buf.subarray(0, n * SAMPLE_FLOATS),
    numSamples: n,
    outcome,
  };
}

/**
 * Run N self-play games, concatenating all samples into one contiguous buffer.
 *
 * The returned Float32Array can be written directly to disk and loaded in
 * Python/JAX with:
 *   `np.fromfile(path, dtype=np.float32).reshape(-1, 1220)`
 */
export function selfPlayBatch(
  numGames: number,
  config: SelfPlayConfig = {},
): {data: Float32Array; totalSamples: number; outcomes: number[]} {
  const chunks: Float32Array[] = [];
  const outcomes: number[] = [];
  let totalSamples = 0;

  for (let g = 0; g < numGames; g++) {
    const result = selfPlay(config);
    if (result.numSamples > 0) {
      chunks.push(Float32Array.from(result.data));
      totalSamples += result.numSamples;
    }
    outcomes.push(result.outcome);
  }

  const out = new Float32Array(totalSamples * SAMPLE_FLOATS);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }

  return {data: out, totalSamples, outcomes};
}
