import type {Coord, GameMove, GameState, SerializedGameState} from '@the-duke/engine';
import {applySetupPlacement, createInitialState, generateAllMoves, applyMove, getSetupTargets, serialize} from '@the-duke/engine';
import {evaluate} from './evaluate.js';
import {findBestMove, ttClear} from './minimax.js';

export interface TrainingSample {
  state: SerializedGameState;
  move: GameMove;
  /** +1 = P1 win, -1 = P2 win, 0 = draw/stalemate */
  outcome: number;
}

export interface SelfPlayResult {
  samples: TrainingSample[];
  outcome: number;
  turns: number;
}

export interface SelfPlayOptions {
  depth?: number;
  maxTurns?: number;
}

// ---------------------------------------------------------------------------
// Setup phase — minimax over placement tree
// P1 maximizes, P2 minimizes. Uses static eval at the leaf (all 6 pieces placed).
// Branching factor: Duke ~6, Footmen ~4 each → max ~5K leaves, instant to search.
// ---------------------------------------------------------------------------

const SETUP_MAXIMIZING: Record<string, boolean> = {
  p1_duke: true, p1_footman1: true, p1_footman2: true,
  p2_duke: false, p2_footman1: false, p2_footman2: false,
};

function setupMinimax(state: GameState): {score: number; coord: Coord | null} {
  if (state.status === 'active') {
    return {score: evaluate(state), coord: null};
  }

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

/**
 * Play a complete self-play game using minimax for both sides.
 * Setup placements are chosen by minimax over the setup tree.
 * Returns training samples with retroactively-filled outcomes.
 */
export function selfPlay(options: SelfPlayOptions = {}): SelfPlayResult {
  const depth = options.depth ?? 4;
  const maxTurns = options.maxTurns ?? 200;

  ttClear();

  let state = setupGameWithMinimax();
  const pendingSamples: {state: SerializedGameState; move: GameMove}[] = [];

  for (let turn = 0; turn < maxTurns; turn++) {
    if (state.status !== 'active') break;

    const moves = generateAllMoves(state);
    if (moves.length === 0) break;

    const result = findBestMove(state, depth);
    if (!result.move) break;

    pendingSamples.push({state: serialize(state), move: result.move});
    state = applyMove(state, result.move);
  }

  let outcome = 0;
  if (state.status === 'P1_wins') outcome = 1;
  else if (state.status === 'P2_wins') outcome = -1;

  const samples: TrainingSample[] = pendingSamples.map(s => ({
    ...s,
    outcome,
  }));

  return {samples, outcome, turns: pendingSamples.length};
}
