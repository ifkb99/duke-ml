import type {GameMove, GameState, SerializedGameState} from '@the-duke/engine';
import {applySetupPlacement, createInitialState, generateAllMoves, applyMove, serialize} from '@the-duke/engine';
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
  /** Fixed setup positions [p1Duke, p1Foot1, p1Foot2, p2Duke, p2Foot1, p2Foot2] */
  setupCoords?: {row: number; col: number}[];
}

const DEFAULT_SETUPS: {row: number; col: number}[][] = [
  [{row: 0, col: 2}, {row: 0, col: 1}, {row: 1, col: 2}, {row: 5, col: 3}, {row: 5, col: 4}, {row: 4, col: 3}],
  [{row: 0, col: 3}, {row: 0, col: 2}, {row: 1, col: 3}, {row: 5, col: 2}, {row: 5, col: 3}, {row: 4, col: 2}],
  [{row: 0, col: 1}, {row: 0, col: 0}, {row: 1, col: 1}, {row: 5, col: 4}, {row: 5, col: 5}, {row: 4, col: 4}],
  [{row: 0, col: 4}, {row: 0, col: 3}, {row: 1, col: 4}, {row: 5, col: 1}, {row: 5, col: 2}, {row: 4, col: 1}],
];

function setupGame(coords?: {row: number; col: number}[]): GameState {
  const setup = coords ?? DEFAULT_SETUPS[Math.floor(Math.random() * DEFAULT_SETUPS.length)];
  let state = createInitialState();
  for (const coord of setup) {
    state = applySetupPlacement(state, coord);
  }
  return state;
}

/**
 * Play a complete self-play game using minimax for both sides.
 * Returns training samples with retroactively-filled outcomes.
 */
export function selfPlay(options: SelfPlayOptions = {}): SelfPlayResult {
  const depth = options.depth ?? 4;
  const maxTurns = options.maxTurns ?? 200;

  ttClear();

  let state = setupGame(options.setupCoords);
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
